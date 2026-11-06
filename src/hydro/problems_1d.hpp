#pragma once

#include "finite_volume_1d.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>

// The 1D problems: the table every 1D test and bench reads, each with its own
// gamma. LUNA's setups (scripts/problems_1d.py has the file:line of each, and
// the same table for LUNA and for the exact solutions), plus two LUNA lacks:
// the exact isentropic simple wave and a pure shock at any Mach number.
namespace hydro::problems_1d {

// One problem: the line [origin, origin + L], its ends, t_end, gamma, and the
// conserved initial state of a cell [a, b] (in the problem's coordinates):
// the exact average where LUNA's setup has it, LUNA's own form where it has not.
struct Problem {
    double L, origin, t_end, gamma;
    Walls walls;
    std::function<State<1>(double a, double b)> average;
    voronoi::Ends ends = voronoi::Ends::walls;
    State<1> left{}, right{};  // a tube: the primitives either side of the jump at x0
    double x0 = 0;
};

constexpr double pi = 3.141592653589793;
constexpr double monatomic = 5.0 / 3.0;
constexpr double linear_amplitude = 1e-6;  // LUNA's EPS

inline State<1> conserved(const State<1>& W, double gamma) {
    return {W[0], W[0] * W[1], W[2] / (gamma - 1) + 0.5 * W[0] * W[1] * W[1]};
}
inline double fraction_below(double x, double a, double b) { return std::clamp((x - a) / (b - a), 0.0, 1.0); }

// A jump at x0: LUNA's exact cell averages (la14379 initial(), er.cell_average(..., t=1e-12)),
// the volume fractions.
inline Problem tube(double L, double origin, double t_end, double gamma, State<1> left, State<1> right, double x0, Walls walls) {
    const State<1> l = conserved(left, gamma), r = conserved(right, gamma);
    return {L, origin, t_end, gamma, walls, [=](double a, double b) {
                const double f = fraction_below(x0, a, b);
                return State<1>{f * l[0] + (1 - f) * r[0], f * l[1] + (1 - f) * r[1], f * l[2] + (1 - f) * r[2]};
            }, voronoi::Ends::walls, left, right, x0};
}

// A shock at x0 into 1 + amp sin(k x), LUNA's exact cell averages (shu_osher.py:26-46) on [a, b].
inline State<1> shock_into_wave(double a, double b, const State<1>& left, double x0, double amp, double k, double gamma) {
    const double h = b - a, f = fraction_below(x0, a, b), lo = std::max(a, x0);
    double rho = f * left[0] + (1 - f);
    if (b > lo) rho += amp * (std::cos(k * lo) - std::cos(k * b)) / k / h;
    return {rho, f * left[0] * left[1], f * (left[2] / (gamma - 1) + 0.5 * left[0] * left[1] * left[1]) + (1 - f) / (gamma - 1)};
}

// Woodward-Colella, LUNA's exact cell averages (woodward_colella.py:14-27).
inline State<1> woodward_colella(double a, double b) {
    const double f1 = fraction_below(0.1, a, b), f2 = fraction_below(0.9, a, b);
    const double p = f1 * 1000 + (f2 - f1) * 0.01 + (1 - f2) * 100;
    return {1.0, 0.0, p / (1.4 - 1)};
}

// Noh: cold inflow on a wall, brought to rest towards the far wall by a ramp,
// sampled at the cell's centre (test_hydro_hard_noh_planar.py, initial()).
inline State<1> noh(double a, double b) {
    constexpr double P0 = 1e-6, U0 = -1, L = 2, A = 1;
    const double x = 0.5 * (a + b), u = U0 * std::clamp((L - x) / (L - A), 0.0, 1.0);
    return {1.0, u, P0 / (monatomic - 1) + 0.5 * u * u};
}

// A linear eigenmode (test_hydro_accuracy_linear_waves.py, initial()) on the
// periodic unit line: rho0 = 1, c = 1, carried at v0, as exact cell averages.
inline State<1> eigenmode(double a, double b, double eps, double v0, double drho, double dv, double dp) {
    constexpr double k = 2 * pi, G = monatomic, P0 = 1 / G;
    const double h = b - a, half = 0.5 * k * h, s = std::sin(k * 0.5 * (a + b)) * std::sin(half) / half;
    return {1 + eps * drho * s, v0 + eps * (dv + v0 * drho) * s,
            P0 / (G - 1) + 0.5 * v0 * v0 + eps * (dp / (G - 1) + 0.5 * v0 * v0 * drho + v0 * dv) * s};
}

// The advected sound wave at the cell's centre (soundwave_advected.py).
inline State<1> advected(double a, double b, double eps) {
    const double s = eps * std::sin(2 * pi * 0.5 * (a + b));
    return conserved({1 + s, 1 + s, 1 + s}, monatomic);
}

// 4-point Gauss-Legendre average of a pointwise conserved state.
template <class F>
inline State<1> gauss_average(double a, double b, const F& f) {
    static constexpr double x[4] = {-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526};
    static constexpr double w[4] = {0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538};
    State<1> u{0, 0, 0};
    for (int q = 0; q < 4; ++q) {
        const State<1> at = f(0.5 * (a + b) + 0.5 * (b - a) * x[q]);
        for (int v = 0; v < 3; ++v) u[v] += 0.5 * w[q] * at[v];
    }
    return u;
}

// The exact isentropic simple wave on the periodic unit line, gamma = 1.4:
// c = 1 + amp sin(2 pi x), u = 5 (c - 1), rho = c^5, P = rho^gamma / gamma.
inline State<1> simple_wave(double a, double b, double amp) {
    return gauss_average(a, b, [=](double x) {
        constexpr double gamma = 1.4;
        const double c = 1 + amp * std::sin(2 * pi * x), rho = std::pow(c, 2 / (gamma - 1));
        return conserved({rho, 2 / (gamma - 1) * (c - 1), std::pow(rho, gamma) / gamma}, gamma);
    });
}

// A pure shock at x = 0.1 of Mach number `mach` into (1, 0, 1), running right,
// from Rankine-Hugoniot; t_end leaves it at x = 0.6.
inline Problem pure_shock(double mach, const Walls& outflow) {
    constexpr double g = 1.4;
    const double m2 = mach * mach, s = mach * std::sqrt(g);
    const double rho2 = (g + 1) * m2 / ((g - 1) * m2 + 2), p2 = (2 * g * m2 - (g - 1)) / (g + 1), v2 = s * (1 - 1 / rho2);
    return tube(1.0, 0.0, 0.5 / s, g, {rho2, v2, p2}, {1, 0, 1}, 0.1, outflow);
}

// amplitude: of the linear waves (signed: LUNA keeps the part odd in it) and
// of the simple wave (0: its default, 0.1).
inline Problem problem(const std::string& name, double amplitude = 0) {
    const Walls outflow{Wall::outflow, Wall::outflow}, closed{Wall::reflective, Wall::reflective};
    const auto toro = [&](State<1> l, State<1> r, double t) { return tube(1.0, 0.0, t, 1.4, l, r, 0.5, outflow); };
    // Toro (2009) Table 4.1 (exact_riemann.py:8-15); toro-1 is Sod.
    if (name == "sod" || name == "toro-1") return toro({1.0, 0.0, 1.0}, {0.125, 0.0, 0.1}, 0.25);
    if (name == "toro-2") return toro({1.0, -2.0, 0.4}, {1.0, 2.0, 0.4}, 0.15);
    if (name == "toro-3") return toro({1.0, 0.0, 1000.0}, {1.0, 0.0, 0.01}, 0.012);
    if (name == "toro-4") return toro({1.0, 0.0, 0.01}, {1.0, 0.0, 100.0}, 0.035);
    if (name == "toro-5") return toro({5.99924, 19.5975, 460.894}, {5.99242, -6.19633, 46.0950}, 0.035);
    // Einfeldt (einfeldt.py, Athena's 1-2-0-3, 1-1-0-5).
    if (name == "einfeldt-strong") return toro({1.0, -2.0, 0.4 * (3.0 - 0.5 * 4.0)}, {1.0, 2.0, 0.4 * (3.0 - 0.5 * 4.0)}, 0.15);
    if (name == "einfeldt-shear") return toro({1.0, -1.0, 0.4 * (5.0 - 0.5 * 1.0)}, {1.0, 1.0, 0.4 * (5.0 - 0.5 * 1.0)}, 0.15);
    // LA-14379 (test_hydro_hard_la14379.py, CASES, VACUUM).
    if (name == "la-peak")
        return tube(0.5, 0.1, 3.9e-3, 1.4, {0.1261192, 8.9047029, 782.92899}, {6.591493, 2.2654207, 3.1544874}, 0.5, outflow);
    if (name == "la-rcvcr") return toro({1.0, -4.0, 0.4}, {1.0, 4.0, 0.4}, 0.15);
    if (name == "la-stationary") return tube(1.0, 0.0, 0.012, 1.4, {1.0, -19.59745, 1e3}, {1.0, -19.59745, 1e-2}, 0.8, outflow);
    if (name == "la-slow") return toro({3.857143, -0.810631, 10.33333}, {1.0, -3.44, 1.0}, 1.0);
    if (name == "la-streams") return toro({1.0, 2.0, 0.1}, {1.0, -2.0, 0.1}, 0.8);
    if (name == "la-vacuum") return tube(6.0, -5.0, 0.75, 1.4, {0.0, 0.0, 0.0}, {1.0, 0.0, 1.0}, 0.0, outflow);
    if (name == "la-near-vacuum") return tube(6.0, -5.0, 0.75, 1.4, {1e-6, 0.0, 1e-6}, {1.0, 0.0, 1.0}, 0.0, outflow);
    // LeBlanc (leblanc.py).
    if (name == "leblanc")
        return tube(1.0, 0.0, 0.5, monatomic, {1.0, 0.0, (2.0 / 3.0) * 1e-1}, {1e-3, 0.0, (2.0 / 3.0) * 1e-10}, 0.3, outflow);
    if (name == "noh") return {2.0, 0.0, 0.6, monatomic, closed, noh};
    // Shock and wave: LUNA's Shu-Osher, Titarev-Toro.
    if (name == "shu-osher")
        return {10.0, -5.0, 1.8, 1.4, outflow, [](double a, double b) {
                    return shock_into_wave(a, b, {3.857143, 2.629369, 10.33333}, -4, 0.2, 5, 1.4);
                }};
    if (name == "titarev-toro")
        return {10.0, -5.0, 5.0, 1.4, outflow, [](double a, double b) {
                    return shock_into_wave(a, b, {1.515695, 0.523346, 1.805}, -4.5, 0.1, 20 * pi, 1.4);
                }};
    if (name == "woodward-colella") return {1.0, 0.0, 0.038, 1.4, closed, woodward_colella};
    // Linear waves, one period, periodic.
    const double eps = amplitude != 0 ? amplitude : linear_amplitude;
    const auto wave = [&](double t, double v0, double drho, double dv, double dp) {
        return Problem{1.0, 0.0, t, monatomic, {}, [=](double a, double b) { return eigenmode(a, b, eps, v0, drho, dv, dp); }, voronoi::Ends::periodic};
    };
    if (name == "sound-right") return wave(1.0, 0.0, 1.0, 1.0, 1.0);
    if (name == "sound-left") return wave(1.0, 0.0, 1.0, -1.0, 1.0);
    if (name == "entropy") return wave(2.0, 0.5, 1.0, 0.0, 0.0);
    if (name == "sound-advected")
        return {1.0, 0.0, 1.0, monatomic, {}, [=](double a, double b) { return advected(a, b, eps); }, voronoi::Ends::periodic};
    // Not in LUNA.
    if (name == "simple-wave") {
        const double amp = amplitude != 0 ? amplitude : 0.1;
        return {1.0, 0.0, 0.2, 1.4, {}, [=](double a, double b) { return simple_wave(a, b, amp); }, voronoi::Ends::periodic};
    }
    if (name.rfind("shock-", 0) == 0) return pure_shock(std::stod(name.substr(6)), outflow);
    throw std::invalid_argument("hydro_1d: unknown problem " + name);
}

}  // namespace hydro::problems_1d
