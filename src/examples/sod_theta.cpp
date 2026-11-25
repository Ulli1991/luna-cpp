// The Sod tube's six parameters theta = (rho_L, P_L, u_L, rho_R, P_R, u_R) and the derivatives of a final-state objective with respect
// to them, by the hand-written tangents (FiniteVolume<1>::step_tangent, six of them) and adjoint (step_adjoint, one), at a fixed dt.
// The problem is astronomix's shock-tube sensitivity setup (arXiv:2607.18176 sec. 5.2); `usage` below has the numbers and objectives.
#include "finite_volume_1d.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// What --help prints.
static const char usage[] = R"(Sod tube on [0, 1], N = 200, gamma = 5/3, outflow walls, 250 steps of 8e-4 to t = 0.2.
Objectives: J = 1/2 int (rho^2 + v^2 + P^2) dx, and L = 1/2 int (rho - rho_obs)^2 dx, the misfit of the final density
against a target set by `target` (the inverse problem).

Usage: sod_theta [--n=200] [--steps=250] [--dt=8e-4] [--gamma=1.6666666666666667] [--limiter=minmod|mc|vanleer] [--order=1]
  serve: reads commands on stdin, one per line, and answers one line each (for scripts/sod_theta.py):
    forward t1..t6               -> "J <J> L <L>"
    adjoint J|L t1..t6           -> "<value> <g1..g6> <ms forward with the tape> <ms adjoint>"
    tangent J|L t1..t6           -> "<value> <g1..g6> <ms of the six tangent runs>"
    target t1..t6                -> "ok": rho_obs = the final density at theta
    dump <path> J|L t1..t6       -> "ok": raw doubles, N rows of (x, rho, u, P, qbar_rho, qbar_mom, qbar_E) at the end, qbar = dObj/dU0 / dx
    quit
  --cost=<N>: times of a forward step, a tangent step and an adjoint step at N cells (fixed dt, the Sod state), median of 5, and B/cell.
)";

namespace {

using Exec = Kokkos::DefaultExecutionSpace;
using Memory = Exec::memory_space;
using Mesh = voronoi::Mesh<1, Exec>;
using FV = hydro::FiniteVolume<1, Exec, hydro::HLLC>;
using Cells = FV::Cells;
using Host = Kokkos::View<double* [3], Kokkos::LayoutRight, Kokkos::HostSpace>;
using Tape = Kokkos::View<double** [3], Kokkos::LayoutRight, Memory>;
using Theta = std::array<double, 6>;

std::string flag(int argc, char** argv, const char* name, const char* fallback) {
    const std::size_t length = std::strlen(name);
    for (int a = 1; a < argc; ++a)
        if (std::strncmp(argv[a], name, length) == 0 && argv[a][length] == '=') return argv[a] + length + 1;
    return fallback;
}

double now_ms() {
    Kokkos::fence();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// The tube: mesh, solver, the tape of the states at the start of each step, and the target density.
struct Tube {
    int n, steps;
    double dt, gamma;
    Mesh mesh;
    std::vector<double> V, x, rho_obs;  // before fv: make() fills V and x
    double noise = 0;  // U0 + noise sin(2.3 i + v) (|U0| + 1): no cell exactly flat ("noise" command)
    FV fv;
    Tape tape;

    Tube(int n_, int steps_, double dt_, double gamma_, hydro::Limiter limiter, int order)
        : n(n_), steps(steps_), dt(dt_), gamma(gamma_), mesh(Exec{}, 1.0, voronoi::Ends::walls), fv(make(n_, gamma_, limiter, order)) {}

    FV make(int n_, double gamma_, hydro::Limiter limiter, int order) {
        Kokkos::View<double* [1], Memory> p("x", n_);
        auto h = Kokkos::create_mirror_view(p);
        for (int i = 0; i < n_; ++i) h(i, 0) = (i + 0.5) / n_;
        Kokkos::deep_copy(p, h);
        mesh.update(p);
        mesh.update_cells();
        auto hv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.volume());
        auto hx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.positions());
        for (int i = 0; i < n_; ++i) {
            V.push_back(hv(i));
            x.push_back(hx(i, 0));
        }
        return FV(Exec{}, mesh, hydro::Gas{gamma_}, order, limiter, hydro::Motion::static_mesh,
                  hydro::Walls{hydro::Wall::outflow, hydro::Wall::outflow});
    }

    // Q0(theta), extensive; the interface at x = 0.5.
    Host initial(const Theta& t) const {
        Host h("q0", n);
        for (int i = 0; i < n; ++i) {
            const bool left = x[i] < 0.5;
            const double r = left ? t[0] : t[3], P = left ? t[1] : t[4], u = left ? t[2] : t[5];
            h(i, 0) = r * V[i];
            h(i, 1) = r * u * V[i];
            h(i, 2) = (P / (gamma - 1) + 0.5 * r * u * u) * V[i];
            for (int v = 0; v < 3; ++v) h(i, v) += noise * std::sin(2.3 * i + v) * (std::abs(h(i, v) / V[i]) + 1.0) * V[i];
        }
        return h;
    }
    // theta bar from Q0 bar (the transpose of initial()'s derivative); also the tangent dQ0 of direction k.
    Theta theta_bar(const Theta& t, const Host& qbar) const {
        Theta g{};
        for (int i = 0; i < n; ++i) {
            const bool left = x[i] < 0.5;
            const int o = left ? 0 : 3;
            const double r = t[o], u = t[o + 2];
            const double b0 = qbar(i, 0) * V[i], b1 = qbar(i, 1) * V[i], b2 = qbar(i, 2) * V[i];
            g[o] += b0 + u * b1 + 0.5 * u * u * b2;
            g[o + 1] += b2 / (gamma - 1);
            g[o + 2] += r * b1 + r * u * b2;
        }
        return g;
    }
    Host direction(const Theta& t, int k) const {
        Host d("dq0", n);
        for (int i = 0; i < n; ++i) {
            const bool left = x[i] < 0.5;
            const int o = left ? 0 : 3;
            if (k < o || k >= o + 3) continue;
            const double r = t[o], u = t[o + 2];
            const int c = k - o;  // 0 rho, 1 P, 2 u
            if (c == 0) {
                d(i, 0) = V[i];
                d(i, 1) = u * V[i];
                d(i, 2) = 0.5 * u * u * V[i];
            } else if (c == 1) {
                d(i, 2) = V[i] / (gamma - 1);
            } else {
                d(i, 1) = r * V[i];
                d(i, 2) = r * u * V[i];
            }
        }
        return d;
    }

    // Objective of the final state Q (extensive) and its gradient dObj/dQ into qbar.
    double objective(char which, const Host& Q, Host* qbar) const {
        double value = 0;
        for (int i = 0; i < n; ++i) {
            const double v = V[i], m = Q(i, 0), p = Q(i, 1), E = Q(i, 2);
            const double r = m / v, u = p / m, P = (gamma - 1) * (E - 0.5 * p * p / m) / v;
            if (which == 'J') {
                value += 0.5 * (r * r + u * u + P * P) * v;
                if (qbar) {
                    (*qbar)(i, 0) = r + u * (-p / (m * m)) * v + P * (gamma - 1) * 0.5 * p * p / (m * m);
                    (*qbar)(i, 1) = u / m * v + P * (gamma - 1) * (-p / m);
                    (*qbar)(i, 2) = P * (gamma - 1);
                }
            } else {
                const double d = r - rho_obs[i];
                value += 0.5 * d * d * v;
                if (qbar) {
                    (*qbar)(i, 0) = d;
                    (*qbar)(i, 1) = 0;
                    (*qbar)(i, 2) = 0;
                }
            }
        }
        return value;
    }

    Host final_state() const {
        Host h("Q", n);
        Kokkos::deep_copy(h, Kokkos::subview(fv.state(), std::make_pair(0, n), Kokkos::ALL));
        return h;
    }
    void set(const Host& h) { Kokkos::deep_copy(Kokkos::subview(fv.state(), std::make_pair(0, n), Kokkos::ALL), h); }

    Host forward(const Theta& t, bool record) {
        if (record && tape.extent(0) != static_cast<std::size_t>(steps)) tape = Tape("tape", steps, n);
        set(initial(t));
        for (int s = 0; s < steps; ++s) {
            if (record) Kokkos::deep_copy(Kokkos::subview(tape, s, Kokkos::ALL, Kokkos::ALL), Kokkos::subview(fv.state(), std::make_pair(0, n), Kokkos::ALL));
            fv.step(dt);
        }
        return final_state();
    }

    // Value and dObj/dQ0 by one adjoint sweep over the tape.
    double adjoint(char which, const Theta& t, Host& q0bar, double& ms_forward, double& ms_adjoint) {
        double t0 = now_ms();
        const Host Q = forward(t, true);
        Host qbar("qbar", n);
        const double value = objective(which, Q, &qbar);
        ms_forward = now_ms() - t0;
        t0 = now_ms();
        Cells bar("bar", n);
        Kokkos::deep_copy(bar, qbar);
        for (int s = steps - 1; s >= 0; --s) {
            Kokkos::deep_copy(Kokkos::subview(fv.state(), std::make_pair(0, n), Kokkos::ALL), Kokkos::subview(tape, s, Kokkos::ALL, Kokkos::ALL));
            fv.step_adjoint(dt, bar);
        }
        Kokkos::deep_copy(q0bar, bar);
        ms_adjoint = now_ms() - t0;
        return value;
    }

    // Value and the six directional derivatives by six tangent runs.
    double tangents(char which, const Theta& t, Theta& g, double& ms) {
        const double t0 = now_ms();
        double value = 0;
        Cells dq("dq", n);
        for (int k = 0; k < 6; ++k) {
            set(initial(t));
            Kokkos::deep_copy(dq, direction(t, k));
            for (int s = 0; s < steps; ++s) fv.step_tangent(dt, dq);
            const Host Q = final_state();
            Host qbar("qbar", n), dQ("dQ", n);
            value = objective(which, Q, &qbar);
            Kokkos::deep_copy(dQ, dq);
            double d = 0;
            for (int i = 0; i < n; ++i)
                for (int v = 0; v < 3; ++v) d += qbar(i, v) * dQ(i, v);
            g[k] = d;
        }
        ms = now_ms() - t0;
        return value;
    }
};

Theta read_theta(std::istringstream& in) {
    Theta t{};
    for (double& v : t) in >> v;
    return t;
}

hydro::Limiter limiter_of(const std::string& s) {
    if (s == "minmod") return hydro::Limiter::minmod;
    if (s == "vanleer") return hydro::Limiter::van_leer;
    if (s == "mc") return hydro::Limiter::face_centroid;
    throw std::invalid_argument("limiter: minmod, mc or vanleer");
}

// Median of 5 of f, in ms.
template <class F>
double median_ms(F f) {
    std::vector<double> t;
    f();
    for (int r = 0; r < 5; ++r) {
        const double t0 = now_ms();
        f();
        t.push_back(now_ms() - t0);
    }
    std::sort(t.begin(), t.end());
    return t[2];
}

void cost(int n, double gamma, hydro::Limiter limiter, int order) {
    const double dt = 0.8 * 0.2 / 250 * 200.0 / n;  // the CFL of the N = 200 run, kept
    const int steps = 20;
    Tube tube(n, steps, dt, gamma, limiter, order);
    const Theta t{1, 1, 0, 0.125, 0.1, 0};
    const Host q0 = tube.initial(t);
    Cells dq("dq", n), bar("bar", n);
    Kokkos::deep_copy(dq, tube.direction(t, 0));
    Kokkos::deep_copy(bar, 1.0);
    const double fwd = median_ms([&] {
                           tube.set(q0);
                           for (int s = 0; s < steps; ++s) tube.fv.step(dt);
                       }) / steps;
    const double tan = median_ms([&] {
                           tube.set(q0);
                           for (int s = 0; s < steps; ++s) tube.fv.step_tangent(dt, dq);
                       }) / steps;
    tube.forward(t, true);
    const double adj = median_ms([&] {
                           for (int s = steps - 1; s >= 0; --s) {
                               Kokkos::deep_copy(Kokkos::subview(tube.fv.state(), std::make_pair(0, n), Kokkos::ALL),
                                                 Kokkos::subview(tube.tape, s, Kokkos::ALL, Kokkos::ALL));
                               tube.fv.step_adjoint(dt, bar);
                           }
                       }) / steps;
    // Bytes a cell beyond the forward's Q and Q_next: the tangent dq and its next (2 x 24), the adjoint qbar and its out (2 x 24);
    // the tape 24 B per step stored.
    std::printf("cost n %d: forward step %.4f ms (%.3f G cell updates/s), tangent step %.4f ms (%.2fx), adjoint step %.4f ms (%.2fx, "
                "the tape's copy in), tape 24 B/cell/step\n",
                n, fwd, n / fwd * 1e-6, tan, tan / fwd, adj, adj / fwd);
}

}  // namespace

int main(int argc, char** argv) {
    for (int a = 1; a < argc; ++a)
        if (std::string(argv[a]) == "--help") {
            std::cout << usage;
            return 0;
        }
    Kokkos::ScopeGuard kokkos(argc, argv);
    const double gamma = std::stod(flag(argc, argv, "--gamma", "1.6666666666666667"));
    const hydro::Limiter limiter = limiter_of(flag(argc, argv, "--limiter", "minmod"));
    const int order = std::stoi(flag(argc, argv, "--order", "1"));
    const std::string c = flag(argc, argv, "--cost", "");
    if (!c.empty()) {
        std::stringstream ss(c);
        std::string item;
        while (std::getline(ss, item, ',')) cost(std::stoi(item), gamma, limiter, order);
        return 0;
    }
    Tube tube(std::stoi(flag(argc, argv, "--n", "200")), std::stoi(flag(argc, argv, "--steps", "250")), std::stod(flag(argc, argv, "--dt", "8e-4")),
              gamma, limiter, order);
    std::string line;
    std::cout.precision(17);
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        std::string cmd;
        in >> cmd;
        if (cmd == "quit") break;
        if (cmd == "forward") {
            const Theta t = read_theta(in);
            const Host Q = tube.forward(t, false);
            std::cout << "J " << tube.objective('J', Q, nullptr) << " L " << (tube.rho_obs.empty() ? 0.0 : tube.objective('L', Q, nullptr)) << std::endl;
        } else if (cmd == "noise") {
            in >> tube.noise;
            std::cout << "ok" << std::endl;
        } else if (cmd == "fcell") {  // fcell i v h t1..t6: forward with U0(i, v) + h (per volume) -> "J <J> L <L>"
            int i = 0, v = 0;
            double h = 0;
            in >> i >> v >> h;
            const Theta t = read_theta(in);
            Host q0 = tube.initial(t);
            q0(i, v) += h * tube.V[i];
            tube.set(q0);
            for (int s = 0; s < tube.steps; ++s) tube.fv.step(tube.dt);
            const Host Q = tube.final_state();
            std::cout << "J " << tube.objective('J', Q, nullptr) << " L " << (tube.rho_obs.empty() ? 0.0 : tube.objective('L', Q, nullptr)) << std::endl;
        } else if (cmd == "target") {
            const Theta t = read_theta(in);
            const Host Q = tube.forward(t, false);
            tube.rho_obs.assign(tube.n, 0);
            for (int i = 0; i < tube.n; ++i) tube.rho_obs[i] = Q(i, 0) / tube.V[i];
            std::cout << "ok" << std::endl;
        } else if (cmd == "adjoint" || cmd == "tangent" || cmd == "dump") {
            std::string path, which;
            if (cmd == "dump") in >> path;
            in >> which;
            const Theta t = read_theta(in);
            Theta g{};
            double value = 0, ms1 = 0, ms2 = 0;
            if (cmd == "tangent") {
                value = tube.tangents(which[0], t, g, ms1);
                std::cout << value;
                for (double v : g) std::cout << ' ' << v;
                std::cout << ' ' << ms1 << std::endl;
                continue;
            }
            Host q0bar("q0bar", tube.n);
            value = tube.adjoint(which[0], t, q0bar, ms1, ms2);
            g = tube.theta_bar(t, q0bar);
            if (cmd == "adjoint") {
                std::cout << value;
                for (double v : g) std::cout << ' ' << v;
                std::cout << ' ' << ms1 << ' ' << ms2 << std::endl;
                continue;
            }
            const Host Q = tube.forward(t, false);
            std::ofstream out(path, std::ios::binary);
            for (int i = 0; i < tube.n; ++i) {
                const double v = tube.V[i], r = Q(i, 0) / v, u = Q(i, 1) / Q(i, 0), P = (gamma - 1) * (Q(i, 2) - 0.5 * Q(i, 1) * u) / v;
                const double row[7] = {tube.x[i], r, u, P, q0bar(i, 0), q0bar(i, 1), q0bar(i, 2)};
                out.write(reinterpret_cast<const char*>(row), sizeof row);
            }
            std::cout << "ok" << std::endl;
        } else {
            std::cout << "error unknown command" << std::endl;
        }
    }
    return 0;
}
