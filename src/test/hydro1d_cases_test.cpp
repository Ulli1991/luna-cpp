// The 1D test cases of LUNA (and two it lacks) against LUNA and against their
// exact solutions, static and moving, one table: hydro::problems_1d.
//
// luna_1d_gates.hpp is generated (scripts/ladder_1d.py gates) from one ladder of
// runs of both codes, HLLC at LUNA's settings, in the norm of LUNA's tests:
// the sum over cells of V |W - exact cell average|, for rho, v, P. On a uniform
// line the static scheme is LUNA's, so static L1 must equal LUNA's, per limiter
// (MC is our face limiter), within 1e-6, 1e-2 for the linear waves, whose errors at
// amplitude 1e-6 are near rounding (ratio 1.000). Moving has no LUNA run to
// equal: L1 within 1.3 x measured, at the two smallest N (CI has 2 cores, with
// sanitizers). Orders within +-0.2 of the measured value: bands from measurement, per variable, never a nominal
// order. A moving run with a known failure must still fail as measured; when it
// stops failing this test says so, and the gate gets its numbers. The rest
// are LUNA's own checks with LUNA's bounds, on a static uniform line: fronts,
// budgets, positivity, wall heating and each case's symptom (test files named below).

#include "check.hpp"
#include "exact_riemann.hpp"
#include "finite_volume_1d.hpp"
#include "luna_1d_gates.hpp"
#include "problems_1d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace {

using Exec = Kokkos::DefaultExecutionSpace;
using Memory = Exec::memory_space;
using Mesh = voronoi::Mesh<1, Exec>;
using State = hydro::State<1>;
using hydro::Limiter;
using hydro::Motion;
namespace pr = hydro::problems_1d;
using Three = std::array<double, 3>;

constexpr long steps_per_cell = 200;  // a run this long has stalled (the ladder's rule)

struct Cell {
    double a, b;
    State W;
};
struct Run {
    std::vector<Cell> cells;
    Three before{}, after{};  // extensive totals, long double sums
    long steps = 0;
    double t = 0;
    bool reached = false;
};

// The problem on n uniform cells to t (default its t_end), LUNA's rule at 0.8
// on a static line, AREPO's at 0.4 on a moving one, the bench's defaults.
template <class Riemann = hydro::HLLC>
Run simulate(const std::string& name, int n, Motion motion, Limiter limiter = Limiter::face_centroid, double amp = 0, double stop = 0) {
    const Exec exec;
    const pr::Problem pb = pr::problem(name, amp);
    const double t_end = stop > 0 ? stop : pb.t_end;
    Mesh mesh(exec, pb.L, pb.ends);
    {
        Kokkos::View<double* [1], Memory> x("x", n);
        auto h = Kokkos::create_mirror_view(x);
        for (int i = 0; i < n; ++i) h(i, 0) = pb.L * (i + 0.5) / n;
        Kokkos::deep_copy(x, h);
        mesh.update(x);
    }
    hydro::FiniteVolume<1, Exec, Riemann> fv(exec, mesh, hydro::Gas{pb.gamma}, 1, limiter, motion, pb.walls);
    auto x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.positions());
    auto c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.centroid());
    auto V = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.volume());
    Run r;
    {
        auto q = Kokkos::create_mirror_view(fv.state());
        std::array<long double, 3> sum{0, 0, 0};
        for (int i = 0; i < n; ++i) {
            const double a = x(i, 0) + c(i, 0) - 0.5 * V(i), b = a + V(i);
            const State u = pb.average(a + pb.origin, b + pb.origin);
            for (int v = 0; v < 3; ++v) {
                q(i, v) = u[v] * V(i);
                sum[v] += q(i, v);
            }
        }
        for (int v = 0; v < 3; ++v) r.before[v] = double(sum[v]);
        Kokkos::deep_copy(fv.state(), q);
    }
    const bool moving = motion == Motion::moving;
    const double courant = moving ? hydro::courant_moving_1d : hydro::courant_static_1d;
    const hydro::TimeStep rule = moving ? hydro::TimeStep::arepo : hydro::TimeStep::luna;
    while (r.t < t_end * (1 - 1e-12) && r.steps < steps_per_cell * n) {
        const double dt = std::min(fv.time_step(courant, rule), t_end - r.t);
        if (!(dt > 1e-15 * t_end)) break;
        fv.step(dt);
        r.t += dt;
        ++r.steps;
    }
    exec.fence();
    r.reached = r.t >= t_end * (1 - 1e-12);
    mesh.update_cells();
    x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.positions());
    c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.centroid());
    V = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.volume());
    const auto q = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, fv.state());
    std::array<long double, 3> sum{0, 0, 0};
    for (int i = 0; i < n; ++i) {
        const double a = x(i, 0) + c(i, 0) - 0.5 * V(i) + pb.origin;
        r.cells.push_back({a, a + V(i), hydro::primitive(State{q(i, 0) / V(i), q(i, 1) / V(i), q(i, 2) / V(i)}, fv.gas())});
        for (int v = 0; v < 3; ++v) sum[v] += q(i, v);
    }
    for (int v = 0; v < 3; ++v) r.after[v] = double(sum[v]);
    std::sort(r.cells.begin(), r.cells.end(), [](const Cell& p, const Cell& s) { return p.a < s.a; });
    return r;
}

State primitive_of(const State& U, double gamma) {
    const double u = U[0] > 0 ? U[1] / U[0] : 0;
    return {U[0], u, (gamma - 1) * (U[2] - 0.5 * U[0] * u * u)};
}

// The isentropic simple wave at (x, t): along its characteristics (scripts/problems_1d.py).
State simple_at(double x, double t, double amp) {
    const double two_pi = 2 * pr::pi;
    double x0 = x - t - std::floor(x - t);
    for (int k = 0; k < 60; ++k) {
        const double c = 1 + amp * std::sin(two_pi * x0);
        double f = x0 + (6 * c - 5) * t - x;
        f -= std::round(f);
        x0 -= f / (1 + 6 * t * amp * two_pi * std::cos(two_pi * x0));
    }
    const double c = 1 + amp * std::sin(two_pi * x0), rho = std::pow(c, 5);
    return {rho, 5 * (c - 1), std::pow(rho, 1.4) / 1.4};
}

// The exact primitives of a cell [a, b] at t_end (conservative average).
struct Exact {
    std::string name;
    pr::Problem pb;
    exact_riemann::Solver solver;
    Exact(const std::string& problem, double eps)
        : name(problem), pb(pr::problem(problem, eps)), solver({pb.left[0], pb.left[1], pb.left[2]}, {pb.right[0], pb.right[1], pb.right[2]}, pb.gamma) {}
    State operator()(double a, double b) const;
};
State Exact::operator()(double a, double b) const {
    if (name == "noh") {
        constexpr double g = pr::monatomic, D = (g - 1) / 2, rho_s = (g + 1) / (g - 1), p_s = (g - 1) / 2 * rho_s;
        const double f = std::clamp((D * pb.t_end - a) / (b - a), 0.0, 1.0);
        return {f * rho_s + (1 - f), -(1 - f), f * p_s + (1 - f) * 1e-6};
    }
    if (name == "sound-right" || name == "sound-left" || name == "entropy") return primitive_of(pb.average(a, b), pb.gamma);
    if (name == "sound-advected") {
        const double x = 0.5 * (a + b), t = pb.t_end, k = 2 * pr::pi, c = std::sqrt(pb.gamma);
        const double amp = pb.average(0.25, 0.25)[0] - 1;  // of the advected wave: its value at x = 1/4
        const double s0 = amp * std::sin(k * (x - t)), sp = amp * std::sin(k * (x - (1 + c) * t)), sm = amp * std::sin(k * (x - (1 - c) * t));
        const double ap = 0.5 * (1 + c) / (c * c), am = 0.5 * (1 - c) / (c * c);
        return {1 + (1 - 1 / (c * c)) * s0 + ap * sp + am * sm, 1 + c * (ap * sp - am * sm), 1 + c * c * (ap * sp + am * sm)};
    }
    if (name == "simple-wave") {
        const State U = pr::gauss_average(a, b, [&](double x) { return pr::conserved(simple_at(x, pb.t_end, 0.1), 1.4); });
        return primitive_of(U, 1.4);
    }
    const exact_riemann::Primitive w = solver.average(a, b, pb.x0, pb.t_end);
    return {w.rho, w.u, w.p};
}

bool is_linear(const std::string& name) {
    return name == "sound-right" || name == "sound-left" || name == "entropy" || name == "sound-advected";
}

// e_i per variable with the rules of LUNA's tests, into L1.
Three l1_of(const std::string& name, const std::vector<Cell>& cells, const std::vector<Cell>* minus = nullptr) {
    const double eps = pr::linear_amplitude;
    const Exact plus(name, is_linear(name) ? eps : 0), negative(name, -eps);
    Three sum{0, 0, 0};
    const int n = static_cast<int>(cells.size());
    for (int i = 0; i < n; ++i) {
        const Cell& c = cells[i];
        const double V = c.b - c.a;
        if (name == "noh" && (i < 4 || 0.5 * (c.a + c.b) >= 0.3)) continue;
        const State e = plus(c.a, c.b);
        double err[3] = {c.W[0] - e[0], c.W[1] - e[1], c.W[2] - e[2]};
        if (minus) {
            const Cell& m = (*minus)[i];
            const State em = negative(m.a, m.b);
            for (int v = 0; v < 3; ++v) err[v] = (err[v] - (m.W[v] - em[v])) / (2 * eps);
        }
        for (int v = 0; v < 3; ++v) {
            if (v == 1 && name == "la-rcvcr" && !(e[0] > 0)) continue;
            if (v == 1 && (name == "la-near-vacuum" || name == "la-vacuum") && !(e[0] > 1e-3)) continue;
            sum[v] += V * std::abs(err[v]);
        }
    }
    return sum;
}

// rho of the run on the bins of a stored reference, L1 against it.
Three reference_l1(const std::vector<Cell>& cells, const double* ref, int bins) {
    std::vector<double> avg(bins, 0.0);
    const double lo = cells.front().a, hi = cells.back().b, h = (hi - lo) / bins;
    for (const Cell& c : cells) {  // a run's N is a multiple of the bins on a uniform line
        const int k = std::min(bins - 1, static_cast<int>((0.5 * (c.a + c.b) - lo) / h));
        avg[k] += (c.b - c.a) * c.W[0] / h;
    }
    double sum = 0;
    for (int k = 0; k < bins; ++k) sum += h * std::abs(avg[k] - ref[k]);
    return {sum, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
}

// L1 of a run, per gate: linear waves take both signs of the amplitude.
struct Measured {
    Three l1{};
    bool reached = false;
    double t = 0;
    long steps = 0;
};
Measured measure(const std::string& name, int n, Motion motion, Limiter limiter = Limiter::face_centroid) {
    Measured out;
    if (is_linear(name)) {
        const Run plus = simulate(name, n, motion, limiter, pr::linear_amplitude), minus = simulate(name, n, motion, limiter, -pr::linear_amplitude);
        out = {l1_of(name, plus.cells, &minus.cells), plus.reached && minus.reached, plus.t, plus.steps};
        return out;
    }
    const Run r = simulate(name, n, motion, limiter);
    out.reached = r.reached;
    out.t = r.t;
    out.steps = r.steps;
    for (const auto& ref : luna_gates::references)
        if (name == ref.problem) {
            out.l1 = reference_l1(r.cells, ref.rho, ref.bins);
            return out;
        }
    out.l1 = l1_of(name, r.cells);
    return out;
}

// test_hydro_accuracy_shocktube.py, locate(): where rho crosses (a + b) / 2 nearest xs.
double locate(const std::vector<Cell>& cells, double xs, double a, double b, double half) {
    const double m = 0.5 * (a + b), h = cells[0].b - cells[0].a;
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i + 1 < cells.size(); ++i) {
        const double x = 0.5 * (cells[i].a + cells[i].b), y = 0.5 * (cells[i + 1].a + cells[i + 1].b), p = cells[i].W[0], q = cells[i + 1].W[0];
        if (std::abs(x - xs) > std::max(half, 2 * h) || std::abs(y - xs) > std::max(half, 2 * h) || (p - m) * (q - m) > 0) continue;
        const double at = x + (m - p) / (q - p) * (y - x);
        if (std::abs(at - xs) < std::abs(best - xs)) best = at;
    }
    return best;
}

// max |U - mirror| / max |U| about the box's centre (check_mirror).
double mirror_error(const std::vector<Cell>& c) {
    double worst = 0, top = 0;
    const std::size_t n = c.size();
    for (std::size_t i = 0; i < n; ++i) {
        const State& p = c[i].W;
        const State& q = c[n - 1 - i].W;
        worst = std::max({worst, std::abs(p[0] - q[0]), std::abs(p[1] + q[1]), std::abs(p[2] - q[2])});
        top = std::max({top, std::abs(p[0]), std::abs(p[1]), std::abs(p[2])});
    }
    return worst / top;
}

double fit(const std::vector<double>& ns, const std::vector<double>& es) {
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (std::size_t i = 0; i < ns.size(); ++i) {
        const double x = std::log(ns[i]), y = std::log(es[i]);
        sx += x, sy += y, sxx += x * x, sxy += x * y;
    }
    const double m = double(ns.size());
    return -(m * sxy - sx * sy) / (m * sxx - sx * sx);
}

}  // namespace

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    Check check;
    const char* names[3] = {"rho", "v", "P"};

    // L1 against LUNA, per limiter, and moving; the orders.
    std::map<std::pair<std::string, int>, std::vector<double>> ladder;  // (problem + series, variable): L1 over the N of the gates
    std::map<std::string, std::vector<double>> sizes;
    for (const luna_gates::Row& row : luna_gates::rows) {
        const std::string name = row.problem;
        const std::string at = name + " N=" + std::to_string(row.n);
        const auto against = [&](const Three& got, const double* luna, const std::string& series) {
            bool ok = true;
            double worst = 0;
            for (int v = 0; v < 3; ++v) {
                if (std::isnan(luna[v])) continue;
                const double ratio = std::abs(got[v] / luna[v] - 1);
                worst = std::max(worst, ratio);
                ok = ok && (ratio < row.tolerance || luna[v] < row.floor);
                ladder[{name + series, v}].push_back(got[v]);
            }
            sizes[name + series].push_back(row.n);
            check.expect(ok, at + " static " + series + ": L1 " + sci(got[0]) + " " + sci(got[1]) + " " + sci(got[2]) + " equals LUNA's (largest |ratio - 1| " + sci(worst) + ")");
        };
        if (row.run_luna) {
            const Measured mc = measure(name, row.n, Motion::static_mesh);
            against(mc.l1, row.luna, "");
            if (!std::isnan(row.minmod[0])) against(measure(name, row.n, Motion::static_mesh, Limiter::minmod).l1, row.minmod, "minmod");
            if (!std::isnan(row.vanleer[0])) against(measure(name, row.n, Motion::static_mesh, Limiter::van_leer).l1, row.vanleer, "vanleer");
        }
        if (row.issue == luna_gates::waits) continue;
        const Measured m = measure(name, row.n, Motion::moving);
        if (row.issue > 0) {
            check.expect(!m.reached || m.steps < 16, at + " moving: the known failure of luna-cpp#" + std::to_string(row.issue) + " (stops at t = " + sci(m.t) + "); if it now reaches t_end, update the gate");
            continue;
        }
        bool ok = m.reached;
        for (int v = 0; v < 3; ++v) {
            if (std::isnan(row.moving[v])) continue;
            ok = ok && m.l1[v] <= 1.3 * row.moving[v];
            ladder[{name + "moving", v}].push_back(m.l1[v]);
        }
        sizes[name + "moving"].push_back(row.n);
        check.expect(ok, at + " moving: L1 " + sci(m.l1[0]) + " " + sci(m.l1[1]) + " " + sci(m.l1[2]) + " within 1.3 x measured");
    }
    for (const luna_gates::Band& band : luna_gates::bands)
        for (int v = 0; v < 3; ++v) {
            if (std::isnan(band.lo[v])) continue;
            const std::string key = std::string(band.problem) + band.series;
            const double order = fit(sizes[key], ladder[{key, v}]);
            check.expect(order >= band.lo[v] && order <= band.hi[v], std::string(band.problem) + " " + (band.series[0] ? band.series : "MC") + " " + names[v] + " order " +
                                                                        std::to_string(order) + " in [" + std::to_string(band.lo[v]) + ", " + std::to_string(band.hi[v]) + "]");
        }

    // Toro: the fronts at N = 512 (shocktube.py POS_BOUND), the exact budget (BUDGET_TOL), minmod as LUNA's bounds.
    {
        const double offset[5] = {0.4, 0.0, 0.92, 0.94, 0.73};
        struct Front { double x, a, b, half; };
        for (int k = 1; k <= 5; ++k) {
            const std::string name = "toro-" + std::to_string(k);
            const Run r = simulate(name, 512, Motion::static_mesh, Limiter::minmod);
            const pr::Problem pb = pr::problem(name);
            // The density jumps (fronts()): the two shocks if there are any, the contact.
            const exact_riemann::Solver s({pb.left[0], pb.left[1], pb.left[2]}, {pb.right[0], pb.right[1], pb.right[2]}, pb.gamma);
            const double g = pb.gamma, us = s.star_velocity(), ps = s.star_pressure(), t = pb.t_end, x0 = pb.x0;
            const auto star_rho = [&](const State& W) {
                const double r0 = ps / W[2];
                return ps > W[2] ? W[0] * (r0 + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * r0 + 1) : W[0] * std::pow(r0, 1 / g);
            };
            const double rl = star_rho(pb.left), rr = star_rho(pb.right), cl = std::sqrt(g * pb.left[2] / pb.left[0]), cr = std::sqrt(g * pb.right[2] / pb.right[0]);
            const double shock_l = pb.left[1] - cl * std::sqrt((g + 1) / (2 * g) * ps / pb.left[2] + (g - 1) / (2 * g));
            const double shock_r = pb.right[1] + cr * std::sqrt((g + 1) / (2 * g) * ps / pb.right[2] + (g - 1) / (2 * g));
            const double tail_l = ps > pb.left[2] ? shock_l : us - cl * std::pow(ps / pb.left[2], (g - 1) / (2 * g));
            const double tail_r = ps > pb.right[2] ? shock_r : us + cr * std::pow(ps / pb.right[2], (g - 1) / (2 * g));
            const double head_l = ps > pb.left[2] ? shock_l : pb.left[1] - cl, head_r = ps > pb.right[2] ? shock_r : pb.right[1] + cr;
            std::vector<double> edges;
            for (const double e : {head_l, tail_l, us, tail_r, head_r}) edges.push_back(x0 + e * t);
            std::vector<Front> fronts;
            if (ps > pb.left[2]) fronts.push_back({edges[0], pb.left[0], rl, 0});
            if (std::abs(rl - rr) > 0.05 * std::max(rl, rr)) fronts.push_back({edges[2], rl, rr, 0});
            if (ps > pb.right[2]) fronts.push_back({edges[3], rr, pb.right[0], 0});
            double worst = 0;
            for (Front f : fronts) {
                f.half = std::numeric_limits<double>::infinity();
                for (const double e : edges)
                    if (std::abs(f.x - e) > 1e-12) f.half = std::min(f.half, 0.5 * std::abs(f.x - e));
                worst = std::max(worst, std::abs(locate(r.cells, f.x, f.a, f.b, f.half) - f.x) * 512);
            }
            check.expect(worst <= offset[k - 1], name + " N=512 minmod: front offset " + sci(worst) + " cells within LUNA's bound");
        }
    }
    // Budgets: every total changes by t (F(W_L) - F(W_R)) while the waves are inside (check_tube_budget), static and moving.
    {
        struct Budget { const char* name; double t, tol_static; };
        const Budget budgets[] = {{"toro-1", 0.25, 1e-12}, {"toro-2", 0.15, 1e-9}, {"toro-3", 0.012, 3e-9}, {"toro-4", 0.035, 1e-12}, {"toro-5", 0.035, 1e-12},
                                  {"einfeldt-strong", 0.15, 1e-9}, {"einfeldt-shear", 0.15, 1e-12}, {"la-peak", 3.9e-3, 1e-12}, {"la-rcvcr", 0.09, 1e-12},
                                  {"la-stationary", 0.012, 1e-12}, {"la-slow", 0.1, 1e-12}, {"la-streams", 0.8, 1e-12}, {"la-near-vacuum", 0.375, 1e-12}, {"leblanc", 0.5, 1e-12}};
        for (const Budget& b : budgets) {
            const pr::Problem pb = pr::problem(b.name);
            const auto flux = [&](const State& W) {
                const double E = W[2] / (pb.gamma - 1) + 0.5 * W[0] * W[1] * W[1];
                return Three{W[0] * W[1], W[0] * W[1] * W[1] + W[2], W[1] * (E + W[2])};
            };
            const Three fl = flux(pb.left), fr = flux(pb.right);
            const Run r = simulate(b.name, 512, Motion::static_mesh, Limiter::face_centroid, 0, b.t);
            const double M = r.before[0], E = r.before[2], scale[3] = {M, std::sqrt(M * E), E};
            double worst = 0;
            for (int v = 0; v < 3; ++v) worst = std::max(worst, std::abs(r.after[v] - r.before[v] - b.t * (fl[v] - fr[v])) / scale[v]);
            check.expect(worst <= b.tol_static, std::string(b.name) + " N=512 static: budget closed to " + sci(worst));
        }
    }
    // Positivity, every problem of LUNA's at N = 512, static.
    for (const char* name : {"toro-1", "toro-2", "toro-3", "toro-4", "toro-5", "einfeldt-strong", "einfeldt-shear", "la-peak", "la-rcvcr", "la-stationary", "la-slow", "la-streams", "la-near-vacuum", "noh"}) {
        const Run r = simulate(name, 512, Motion::static_mesh);
        double rho = 1e300, p = 1e300;
        for (const Cell& c : r.cells) rho = std::min(rho, c.W[0]), p = std::min(p, c.W[2]);
        check.expect(r.reached && rho > 0 && p > 0, std::string(name) + " N=512 static: min rho " + sci(rho) + ", min P " + sci(p));
    }
    // LeBlanc (test_hydro_hard_leblanc.py): positivity against the right state, the contact overshoot, MC bounds.
    for (const auto& [n, bound] : std::array<std::pair<int, double>, 3>{{{256, 0.34}, {512, 0.32}, {1024, 0.32}}}) {
        const Run r = simulate("leblanc", n, Motion::static_mesh);
        const pr::Problem pb = pr::problem("leblanc");
        const Exact solution("leblanc", 0);
        double rho = 1e300, p = 1e300, e = 0, ex = 0;
        for (const Cell& c : r.cells) {
            rho = std::min(rho, c.W[0]), p = std::min(p, c.W[2]);
            e = std::max(e, c.W[2] / ((pb.gamma - 1) * c.W[0]));
            const State w = solution(c.a, c.b);
            ex = std::max(ex, w[2] / ((pb.gamma - 1) * w[0]));
        }
        check.expect(rho / pb.right[0] >= 1 - 1e-9 && p / pb.right[2] >= 1 - 1e-6 && e / ex - 1 <= bound,
                     "LeBlanc N=" + std::to_string(n) + " static: min rho / rho_R " + sci(rho / pb.right[0]) + ", min P / P_R " + sci(p / pb.right[2]) + ", e overshoot " + sci(e / ex - 1));
    }
    // Noh (test_hydro_hard_noh_planar.py), MC bounds: the wall's min rho and its 2 cells, the shock, the undisturbed inflow, the impulse.
    for (const int n : {256, 512, 1024}) {
        constexpr double G = pr::monatomic, D = (G - 1) / 2, rho_s = (G + 1) / (G - 1), p_s = (G - 1) / 2 * rho_s, T = 0.6;
        const Run r = simulate("noh", 2 * n, Motion::static_mesh);
        const std::vector<Cell>& c = r.cells;
        double wall = 1e300, undisturbed = 0;
        for (int i = 0; i < 4; ++i) wall = std::min(wall, c[i].W[0]);
        int extent = 0, inner = static_cast<int>(D * T * n) - 10;
        for (int i = 0; i < inner; ++i) extent += std::abs(c[i].W[0] / rho_s - 1) > 0.02;
        double shock = 0;
        const double half = 0.5 * (1 + rho_s);
        for (std::size_t i = 0; i + 1 < c.size(); ++i)
            if (c[i].W[0] >= half && c[i + 1].W[0] < half) shock = (double(i) + 0.5 + (c[i].W[0] - half) / (c[i].W[0] - c[i + 1].W[0])) / n;
        for (const Cell& k : c) {
            const double x = 0.5 * (k.a + k.b);
            if (x > D * T + 0.05 && x < 0.3) undisturbed = std::max({undisturbed, std::abs(k.W[0] - 1), std::abs(k.W[1] + 1)});
        }
        const double impulse = (r.after[1] - r.before[1]) / (p_s * T) - 1;
        const double offset = std::abs(shock - D * T) * n;
        check.expect(wall >= 3.5407 && wall <= 3.5768 && extent == 2 && offset <= 0.38 && undisturbed <= 1e-10 && std::abs(impulse) * n / 256 <= 0.0017 &&
                         std::abs(r.after[0] - r.before[0]) / r.before[0] < 1e-13 && std::abs(r.after[2] - r.before[2]) / r.before[2] < 1e-13,
                     "Noh n=" + std::to_string(n) + " static: wall min rho " + sci(wall) + ", " + std::to_string(extent) + " defect cells, shock off by " + sci(offset) + " cells, inflow " + sci(undisturbed) + ", impulse " + sci(impulse));
    }
    // Mirror symmetry of the symmetric problems (la14379.py, test_mirror_symmetry; 1e-13 there).
    for (const char* name : {"la-rcvcr", "la-streams"})
        for (const Limiter limiter : {Limiter::minmod, Limiter::face_centroid, Limiter::van_leer}) {
            const double e = mirror_error(simulate(name, 512, Motion::static_mesh, limiter).cells);
            check.expect(e < 1e-13, std::string(name) + " N=512: mirror symmetric to " + sci(e));
        }
    // LA-14379 symptoms, MC bounds of test_hydro_hard_la14379.py at N = 1024 (Peak 1600).
    {
        const pr::Problem peak = pr::problem("la-peak");
        const exact_riemann::Solver s({peak.left[0], peak.left[1], peak.left[2]}, {peak.right[0], peak.right[1], peak.right[2]}, 1.4);
        const double ps = s.star_pressure(), r0 = ps / peak.right[2], k = 0.4 / 2.4, rho_r = peak.right[0] * (r0 + k) / (k * r0 + 1);
        double top = 0;
        for (const Cell& c : simulate("la-peak", 1600, Motion::static_mesh).cells) top = std::max(top, c.W[0]);
        check.expect(std::abs(top / rho_r - 1) <= 0.0044, "la-peak N=1600 static: peak error " + sci(std::abs(top / rho_r - 1)));
        double centre = 0;
        for (const Cell& c : simulate("la-rcvcr", 1024, Motion::static_mesh).cells)
            if (std::abs(0.5 * (c.a + c.b) - 0.5) < 0.02) centre = std::max(centre, c.W[0]);
        check.expect(centre <= 0.00094, "la-rcvcr N=1024 static: centre rho " + sci(centre));
    }
    // Linear waves: left- and right-going errors equal (linear_waves.py, LR_TOL).
    for (const int n : {256, 512}) {
        const auto r = measure("sound-right", n, Motion::static_mesh).l1, l = measure("sound-left", n, Motion::static_mesh).l1;
        check.expect(std::abs(r[0] - l[0]) / r[0] < 1e-5 && std::abs(r[2] - l[2]) / r[2] < 1e-5, "sound N=" + std::to_string(n) + ": right and left errors agree (" + sci(std::abs(r[0] - l[0]) / r[0]) + ")");
    }
    // A true vacuum on the left: LUNA stops at t = 0 on a NaN dt (la14379.py, test_vacuum_expansion_true_vacuum, strict xfail).
    {
        const Run r = simulate("la-vacuum", 768, Motion::static_mesh);
        bool finite = true;
        for (const Cell& c : r.cells) finite = finite && std::isfinite(c.W[0]) && std::isfinite(c.W[1]) && std::isfinite(c.W[2]);
        check.expect(r.reached && finite, "la-vacuum N=768 static: rho = 0 runs to t_end, finite, " + std::to_string(r.steps) + " steps");
    }
    // Teeth: a run stopped at 0.9 t misses the bound (LUNA's test_bound_catches_a_short_run).
    {
        const Run r = simulate("toro-1", 512, Motion::static_mesh, Limiter::face_centroid, 0, 0.9 * 0.25);
        const Three short_run = l1_of("toro-1", r.cells);
        double bound = 0;
        for (const luna_gates::Row& row : luna_gates::rows)
            if (std::string(row.problem) == "toro-1" && row.n == 512) bound = row.luna[0];
        check.expect(short_run[0] > 1.3 * bound, "toro-1 N=512 stopped at 0.9 t: L1 rho " + sci(short_run[0]) + " misses the bound " + sci(1.3 * bound));
    }
    return check.failed() != 0;
}
