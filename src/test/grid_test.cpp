// Checks the fixed-grid step (grid.hpp, grid_finite_volume.hpp, grid_cuda.hpp): LUNA's per-axis Riemann cores against HLLC::flux and
// HLL::flux, uniform states on every boundary, conservation, the Kokkos and CUDA kernels' bits, and LUNA's own steps
// (src/test/data/luna_grid, made by scripts/luna_grid_bits.py on an A100): bit for bit where no FMA is contracted, else to rounding.

#include "check.hpp"
#include "grid_finite_volume.hpp"
#include "npy.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifndef LUNA_GRID_DATA
#define LUNA_GRID_DATA "src/test/data/luna_grid"
#endif

namespace {

namespace g = hydro::grid;
using Exec = Kokkos::DefaultExecutionSpace;

// LUNA's bits are expected where nothing contracts a multiply and an add: host code without FMA instructions, or a device build with
// LUNA_BIT_EXACT (-fmad=false). Elsewhere the states agree to rounding.
#if defined(__FMA__) || (defined(KOKKOS_ENABLE_CUDA) && !defined(LUNA_BIT_EXACT))
constexpr bool exact = false;
#else
constexpr bool exact = true;
#endif

// The initial primitives of scripts/luna_grid_bits.py (initial_primitives): exact operations only, so the same bits as NumPy's.
// Used by: luna_cases, kernel_forms, conservation.
template <int D>
hydro::State<D> initial_primitives(int i, int j, int k, int n0, bool floors) {
    const int c[3] = {i, j, k}, weight[3] = {7, 13, 29};
    long base = 0;
    for (int a = 0; a < D; ++a) base += static_cast<long>(weight[a]) * c[a];
    auto h = [&](int s) { return static_cast<double>((base + s) % 17) / 16.0; };
    const bool left = i < n0 / 2;
    hydro::State<D> W;
    W[0] = 1.0 + h(1) + (left ? 1.0 : 0.0);
    for (int a = 0; a < D; ++a) W[1 + a] = 0.5 * (h(3 + a) - 0.5);
    W[D + 1] = 1.0 + h(7) + (left ? 2.0 : 0.0);
    if (floors && !left) {
        W[0] = 1e-6 * W[0];
        W[D + 1] = 1e-8 * W[D + 1];
    }
    if (floors) W[1] = left ? -1.5 : 1.5;
    return W;
}

// Sets fv's state to U(i, j, k) of every cell and refreshes it. Used by: every check below.
template <int D, class FV, class F>
void set_state(FV& fv, const F& U) {
    auto h = Kokkos::create_mirror_view(fv.state());
    const int n0 = fv.extent(0), n1 = fv.extent(1), n2 = fv.extent(2);
    for (int i = 0; i < n0; ++i)
        for (int j = 0; j < n1; ++j)
            for (int k = 0; k < n2; ++k) {
                const hydro::State<D> u = U(i, j, k);
                for (int v = 0; v < D + 2; ++v) h(v, (static_cast<long>(i) * n1 + j) * n2 + k) = u[v];
            }
    Kokkos::deep_copy(fv.state(), h);
    fv.refresh();
}

// The state as one host vector, variable-major. Used by: the comparisons.
template <class FV>
std::vector<double> host_state(const FV& fv) {
    const auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fv.state());
    return std::vector<double>(h.data(), h.data() + h.size());
}

// Values that differ (NaN equals NaN, -0 equals 0) and the largest difference over the largest magnitude of b. Used by: luna_cases, kernel_forms.
struct Difference {
    long differ = 0;
    double relative = 0;
};
Difference compare(const std::vector<double>& a, const std::vector<double>& b) {
    Difference d;
    double scale = 0, worst = 0;
    for (std::size_t m = 0; m < a.size(); ++m) {
        const bool same = a[m] == b[m] || (std::isnan(a[m]) && std::isnan(b[m]));
        d.differ += !same;
        if (std::isfinite(b[m])) scale = std::max(scale, std::abs(b[m]));
        if (!same) worst = std::max(worst, std::isfinite(a[m] - b[m]) ? std::abs(a[m] - b[m]) : INFINITY);
    }
    d.relative = scale > 0 ? worst / scale : worst;
    return d;
}

// HLLC::axis_core and HLL::axis_core (through grid::face_flux, no slopes) against flux() with n = e_d, w = 0: rounding only.
template <int D, class Riemann>
void cores(Check& check, const char* name) {
    std::mt19937_64 rng(5);
    std::uniform_real_distribution<double> rho(0.05, 3.0), v(-3.0, 3.0), P(0.01, 5.0);
    const hydro::Gas gas(1.4);
    const double dx[3] = {1, 1, 1};
    const g::Params p = g::make_params(1.4, dx, 0.8, g::Limiter::mc);
    double worst = 0;
    for (int sample = 0; sample < 20000; ++sample) {
        hydro::State<D> L, R, zero{};
        L[0] = rho(rng);
        R[0] = rho(rng);
        for (int a = 0; a < D; ++a) {
            L[1 + a] = v(rng);
            R[1 + a] = v(rng);
        }
        L[D + 1] = P(rng);
        R[D + 1] = P(rng);
        for (int d = 0; d < D; ++d) {
            hydro::Vec<D> n{}, w{};
            n[d] = 1;
            const hydro::State<D> a = g::face_flux<D, Riemann>(L, zero, L, R, zero, R, d, p);
            const hydro::State<D> b = Riemann::flux(L, R, n, w, gas);
            double scale = 0, diff = 0;
            for (int m = 0; m < D + 2; ++m) {
                scale = std::max(scale, std::abs(b[m]));
                diff = std::max(diff, std::abs(a[m] - b[m]));
            }
            worst = std::max(worst, diff / scale);
        }
    }
    check.expect(worst < 1e-13, std::string(name) + " axis_core against flux(n = e_d, w = 0), " + std::to_string(D) + "D, 20000 pairs: largest difference " +
                                    sci(worst) + " of the flux (< 1e-13)");
}

// A uniform state stays the same bits on every boundary (moving on open and periodic ones, at rest on walls and diodes).
template <int D>
void uniform(Check& check) {
    const std::array<int, 3> sizes = {16, 12, 8};
    std::array<int, D> n;
    std::array<double, D> length;
    for (int a = 0; a < D; ++a) {
        n[a] = sizes[a];
        length[a] = 1.0 + 0.25 * a;
    }
    for (g::Boundary b : {g::Boundary::periodic, g::Boundary::outflow, g::Boundary::reflective, g::Boundary::diode}) {
        if (b == g::Boundary::diode && D < 3) continue;
        g::Options o;
        o.boundary = b;
        g::GridFiniteVolume<D> fv(Exec(), n, length, 1.4, o);
        const bool rest = b == g::Boundary::reflective || b == g::Boundary::diode;
        hydro::State<D> W;
        W[0] = 1.3;
        for (int a = 0; a < D; ++a) W[1 + a] = rest ? 0.0 : 0.3 - 0.2 * a;
        W[D + 1] = 0.7;
        const hydro::State<D> U = g::to_conserved<D>(W, fv.params());
        set_state<D>(fv, [&](int, int, int) { return U; });
        const std::vector<double> before = host_state(fv);
        for (int s = 0; s < 5; ++s) fv.step(fv.time_step());
        const Difference d = compare(host_state(fv), before);
        check.expect(d.differ == 0, std::to_string(D) + "D uniform state, boundary " + std::to_string(static_cast<int>(b)) + ", 5 steps: " +
                                        std::to_string(d.differ) + " values changed");
    }
}

// Periodic: mass, momentum and energy are conserved to rounding over 20 steps of the bits test's state.
template <int D>
void conservation(Check& check) {
    const std::array<int, 3> sizes = {16, 12, 8};
    std::array<int, D> n;
    std::array<double, D> length;
    for (int a = 0; a < D; ++a) {
        n[a] = sizes[a];
        length[a] = 1.0;
    }
    g::Options o;
    o.boundary = g::Boundary::periodic;
    g::GridFiniteVolume<D> fv(Exec(), n, length, 1.4, o);
    set_state<D>(fv, [&](int i, int j, int k) { return g::to_conserved<D>(initial_primitives<D>(i, j, k, n[0], false), fv.params()); });
    auto totals = [&] {
        const std::vector<double> u = host_state(fv);
        std::vector<double> t(D + 2, 0.0);
        for (int v = 0; v < D + 2; ++v)
            for (long c = 0; c < fv.cells(); ++c) t[v] += u[v * fv.cells() + c];
        return t;
    };
    const std::vector<double> before = totals();
    for (int s = 0; s < 20; ++s) fv.step(fv.time_step());
    const std::vector<double> after = totals();
    double worst = 0;
    for (int v = 0; v < D + 2; ++v) worst = std::max(worst, std::abs(after[v] - before[v]) / std::max(std::abs(before[v]), before[0]));
    check.expect(worst < 1e-13, std::to_string(D) + "D periodic, 20 steps: largest change of a total " + sci(worst) + " (< 1e-13)");
}

// The CUDA kernel against the Kokkos kernel: the same bits and time steps without FMA contraction, rounding with it (CUDA builds only).
void kernel_forms(Check& check) {
#if defined(KOKKOS_ENABLE_CUDA)
    const std::array<int, 3> n = {32, 32, 64};
    const std::array<double, 3> length = {1.0, 0.75, 1.3};
    std::vector<double> state[2];
    double dt[2] = {0, 0};
    for (int f = 0; f < 2; ++f) {
        g::Options o;
        o.kernel = f == 0 ? g::Kernel::kokkos : g::Kernel::cuda;
        g::GridFiniteVolume<3> fv(Exec(), n, length, 1.4, o);
        set_state<3>(fv, [&](int i, int j, int k) { return g::to_conserved<3>(initial_primitives<3>(i, j, k, n[0], false), fv.params()); });
        for (int s = 0; s < 10; ++s) fv.step(fv.time_step());
        state[f] = host_state(fv);
        dt[f] = fv.time_step();
    }
    // With FMA contraction nvcc contracts the same source differently in the two kernels (measured: a quarter of the values differ
    // after 10 steps, by rounding), so the bits are asked for only without it.
    const Difference d = compare(state[1], state[0]);
    const bool ok = exact ? d.differ == 0 && dt[0] == dt[1] : d.relative < 1e-12 && std::abs(dt[0] / dt[1] - 1) < 1e-12;
    check.expect(ok, std::string("3D 32x32x64, 10 steps: CUDA kernel against Kokkos kernel (") + (exact ? "bits" : "rounding") + "), " +
                         std::to_string(d.differ) + " values differ (largest " + sci(d.relative) + "), next dt " + (dt[0] == dt[1] ? "equal" : "differs"));
#else
    (void)check;
#endif
}

// `key value` lines of a case's meta.txt. Used by: luna_cases.
std::map<std::string, std::string> read_meta(const std::string& path) {
    std::ifstream in(path);
    std::map<std::string, std::string> m;
    std::string line;
    while (std::getline(in, line)) {
        const std::size_t space = line.find(' ');
        if (space != std::string::npos) m[line.substr(0, space)] = line.substr(space + 1);
    }
    return m;
}

// One stored LUNA case: from LUNA's U_0 with LUNA's time steps; each step's dt and the saved states compared. Used by: luna_cases.
template <int D, class Riemann>
void luna_case(Check& check, const std::string& dir, const std::map<std::string, std::string>& meta, g::Kernel kernel) {
    std::array<int, D> n;
    std::array<double, D> length;
    std::istringstream shape(meta.at("shape")), lengths(meta.at("lengths"));
    for (int a = 0; a < D; ++a) shape >> n[a];
    for (int a = 0; a < D; ++a) lengths >> length[a];
    g::Options o;
    o.limiter = static_cast<g::Limiter>(std::stoi(meta.at("limiter")));
    o.boundary = static_cast<g::Boundary>(std::stoi(meta.at("boundary")));
    o.cfl = std::stod(meta.at("cfl"));
    o.floors = meta.at("floors") == "1";
    o.smallr = std::stod(meta.at("smallr"));
    o.smallc = std::stod(meta.at("smallc"));
    o.allow_first_order = meta.at("allow_first_order") == "1";
    o.kernel = kernel;
    g::GridFiniteVolume<D, Riemann> fv(Exec(), n, length, std::stod(meta.at("gamma")), o);
    const bool floors = o.floors;
    set_state<D>(fv, [&](int i, int j, int k) { return g::to_conserved<D>(initial_primitives<D>(i, j, k, n[0], floors), fv.params()); });
    Difference ic;
    if (std::filesystem::exists(dir + "/U_0.npy")) {  // start from LUNA's own U_0 (absent on the largest grids: the formula's state)
        const Npy U0 = read_npy(dir + "/U_0.npy");
        ic = compare(host_state(fv), U0.values);
        auto h = Kokkos::create_mirror_view(fv.state());
        for (std::size_t m = 0; m < U0.values.size(); ++m) h.data()[m] = U0.values[m];
        Kokkos::deep_copy(fv.state(), h);
        fv.refresh();
    }
    const Npy dts = read_npy(dir + "/dt.npy");
    std::istringstream saved(meta.at("steps"));
    int last = 0;
    std::vector<int> steps;
    for (int s; saved >> s;) steps.push_back(last = s);
    long dt_differ = 0;
    double dt_worst = 0;
    std::string states;
    bool ok = ic.differ == 0;
    for (int s = 1; s <= last; ++s) {
        const double ours = fv.time_step(), theirs = dts.values[s - 1];
        dt_differ += ours != theirs;
        dt_worst = std::max(dt_worst, std::abs(ours / theirs - 1));
        fv.step(theirs);
        for (int saved_step : steps) {
            if (saved_step != s) continue;
            const Difference d = compare(host_state(fv), read_npy(dir + "/U_" + std::to_string(s) + ".npy").values);
            states += " U_" + std::to_string(s) + ": " + std::to_string(d.differ) + " differ, largest " + sci(d.relative);
            ok = ok && (exact ? d.differ == 0 : d.relative < 1e-12);
        }
    }
    ok = ok && (exact ? dt_differ == 0 : dt_worst < 1e-12);
    const std::string form = kernel == g::Kernel::cuda ? "cuda" : "kokkos";
    check.expect(ok, "LUNA " + meta.at("case") + " (" + form + ", " + (exact ? "bits" : "rounding") + "): initial state " + std::to_string(ic.differ) +
                         " differ; dt " + std::to_string(dt_differ) + " of " + std::to_string(last) + " differ (largest " + sci(dt_worst) + ");" + states);
}

// Every case under LUNA_GRID_DATA, with each kernel this build has. Used by: main.
void luna_cases(Check& check, const std::string& root) {
    if (!std::filesystem::is_directory(root)) {
        check.expect(false, "LUNA cases: no directory " + root);
        return;
    }
    std::vector<std::string> dirs;
    for (const auto& e : std::filesystem::directory_iterator(root))
        if (e.is_directory()) dirs.push_back(e.path().string());
    std::sort(dirs.begin(), dirs.end());
    std::vector<g::Kernel> kernels = {g::Kernel::kokkos};
#if defined(KOKKOS_ENABLE_CUDA)
    kernels.push_back(g::Kernel::cuda);
#endif
    for (const std::string& dir : dirs) {
        const auto meta = read_meta(dir + "/meta.txt");
        const int dim = std::stoi(meta.at("dim"));
        const bool hll = meta.at("riemann") == "hll";
        for (g::Kernel k : kernels) {
            if (dim == 1) hll ? luna_case<1, hydro::HLL>(check, dir, meta, k) : luna_case<1, hydro::HLLC>(check, dir, meta, k);
            if (dim == 2) hll ? luna_case<2, hydro::HLL>(check, dir, meta, k) : luna_case<2, hydro::HLLC>(check, dir, meta, k);
            if (dim == 3) hll ? luna_case<3, hydro::HLL>(check, dir, meta, k) : luna_case<3, hydro::HLLC>(check, dir, meta, k);
        }
    }
    check.expect(!dirs.empty(), "LUNA cases: " + std::to_string(dirs.size()) + " in " + root);
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    Check check;
    cores<1, hydro::HLLC>(check, "HLLC");
    cores<2, hydro::HLLC>(check, "HLLC");
    cores<3, hydro::HLLC>(check, "HLLC");
    cores<3, hydro::HLL>(check, "HLL");
    uniform<1>(check);
    uniform<2>(check);
    uniform<3>(check);
    conservation<1>(check);
    conservation<2>(check);
    conservation<3>(check);
    kernel_forms(check);
    luna_cases(check, argc > 1 && argv[1][0] != '-' ? argv[1] : LUNA_GRID_DATA);
    return check.failed() != 0;
}
