#pragma once

#include "finite_volume_1d.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>

// The 1D problems, each with its gamma: LUNA's setups (scripts/problems_1d.py has the same table, sources, exact solutions) and two it lacks.
// Used by: hydro_1d (problem(name) -> simulate), hydro1d_test, hydro1d_cases_test.
namespace hydro::problems_1d {

// One problem: the line [origin, origin + L], its ends, t_end, gamma, and the conserved initial state of a cell [a, b].
struct Problem {
    double L, origin, t_end, gamma;
    Walls walls;
    std::function<State<1>(double a, double b)> average;
    voronoi::Ends ends = voronoi::Ends::walls;
    State<1> left{}, right{};  // the primitives either side of a jump at x0
    double x0 = 0;
    Walls moving_walls{};  // LUNA's unless gas enters supersonically: there the wall holds the state, as the end cell's drifts as N^2
};

constexpr double pi = 3.141592653589793;
constexpr double gamma_gas = 1.4;  // the gamma of most of LUNA's 1D tests
constexpr double monatomic = 5.0 / 3.0;
constexpr double linear_amplitude = 1e-6;  // LUNA's EPS

inline State<1> conserved(const State<1>& W, double gamma) {
    return {W[0], W[0] * W[1], W[2] / (gamma - 1) + 0.5 * W[0] * W[1] * W[1]};
}
inline double fraction_below(double x, double a, double b) { return std::clamp((x - a) / (b - a), 0.0, 1.0); }

// A jump at x0: LUNA's exact cell averages, the volume fractions.
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
// Used by: problem ("woodward-colella"), as Problem::average.
inline State<1> woodward_colella(double a, double b) {
    const double f1 = fraction_below(0.1, a, b), f2 = fraction_below(0.9, a, b);
    const double p = f1 * 1000 + (f2 - f1) * 0.01 + (1 - f2) * 100;
    return {1.0, 0.0, p / (1.4 - 1)};
}

// Noh: cold inflow on a wall, brought to rest towards the far wall by a ramp, at the cell's centre (test_hydro_hard_noh_planar.py).
inline State<1> noh(double a, double b) {
    constexpr double P0 = 1e-6, U0 = -1, L = 2, A = 1;
    const double x = 0.5 * (a + b), u = U0 * std::clamp((L - x) / (L - A), 0.0, 1.0);
    return {1.0, u, P0 / (monatomic - 1) + 0.5 * u * u};
}

// A linear eigenmode (test_hydro_accuracy_linear_waves.py) on the periodic unit line: rho0 = 1, c = 1, carried at v0, exact averages.
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

// The exact isentropic simple wave on the periodic unit line, gamma = 1.4: c = 1 + amp sin(2 pi x), u = 5 (c - 1), rho = c^5.
inline State<1> simple_wave(double a, double b, double amp) {
    return gauss_average(a, b, [=](double x) {
        constexpr double gamma = 1.4;
        const double c = 1 + amp * std::sin(2 * pi * x), rho = std::pow(c, 2 / (gamma - 1));
        return conserved({rho, 2 / (gamma - 1) * (c - 1), std::pow(rho, gamma) / gamma}, gamma);
    });
}

// A pure shock at x = 0.1 of Mach number `mach` into (1, 0, 1), running right, to x = 0.6.
inline Problem pure_shock(double mach, const Walls& outflow) {
    constexpr double g = 1.4;
    const double m2 = mach * mach, s = mach * std::sqrt(g);
    const double rho2 = (g + 1) * m2 / ((g - 1) * m2 + 2), p2 = (2 * g * m2 - (g - 1)) / (g + 1), v2 = s * (1 - 1 / rho2);
    return tube(1.0, 0.0, 0.5 / s, g, {rho2, v2, p2}, {1, 0, 1}, 0.1, outflow);
}

// The super-hard tube (research_notes/super_tube_2026-11-10.md): from x on, the primitives (rho, v, P), up to the next x or 33.2.
struct Segment {
    double x, rho, v, P;
};
inline constexpr Segment super_tube_segments[] = {
    {0.0, 1, 0, 1},       {6.0, 10, 0, 1},         {6.05, 1, 0, 1},    {8.05, 1, 1, 1},     {8.80, 10, 1, 1},  {8.85, 1, 1, 1},
    {9.60, 1, 0, 1},      {11.60, 1, 0.75, 1},     {12.10, 0.125, 0, 0.1}, {12.60, 1, 0, 1}, {14.60, 1, 0, 0.1},
    {15.10, 1e-3, 0, 1e-10}, {15.60, 1, 0, 1},     {17.60, 3.857143, -0.810631, 10.33333}, {18.10, 1, -3.44, 1},
    {18.60, 1, 0, 1},     {22.60, 5.99924, 19.5975, 460.894}, {22.90, 5.99242, -6.19633, 46.0950}, {23.20, 1, 0, 1},
    {27.20, 1, -2, 0.4},  {27.70, 1, 2, 0.4},      {28.20, 1, 0, 1},   {32.20, 1, 0, 1000}, {32.30, 1, 0, 0.01}, {33.10, 1, 0, 100}};
inline constexpr double super_tube_length = 33.2, super_tube_wave = 1e-4, super_tube_wavelength = 0.25;

// Exact cell average on [a, b]: the segments' volume fractions; on [3, 4] a right sound wave and an entropy wave, linear in U.
inline State<1> super_tube(double a, double b) {
    constexpr double g = gamma_gas, k = 2 * pi / super_tube_wavelength;
    constexpr int count = sizeof super_tube_segments / sizeof super_tube_segments[0];
    State<1> u{0, 0, 0};
    for (int s = 0; s < count; ++s) {
        const Segment& w = super_tube_segments[s];
        const double end = s + 1 < count ? super_tube_segments[s + 1].x : super_tube_length;
        const double f = std::max(0.0, std::min(b, end) - std::max(a, w.x)) / (b - a);
        const State<1> U = conserved({w.rho, w.v, w.P}, g);
        for (int v = 0; v < 3; ++v) u[v] += f * U[v];
    }
    const double lo = std::max(a, 3.0), hi = std::min(b, 4.0);
    if (hi > lo) {  // c = sqrt(g) on the ambient: dU = s (1, c, c^2 / (g - 1)) + s (1, 0, 0)
        const double s = super_tube_wave * (std::cos(k * (lo - 3)) - std::cos(k * (hi - 3))) / k / (b - a);
        u[0] += 2 * s;
        u[1] += std::sqrt(g) * s;
        u[2] += g / (g - 1) * s;
    }
    return u;
}

// The super-hard tube v2 (research_notes/super_tube_v2_2026-11-18.md): v1's stages and five more, laid out by
// scripts/super_tube_layout.py v2 (exact Riemann fronts; no two stages meet before t = 0.15). Gamma 1.4; open left wall, reflecting right.
inline constexpr Segment super_tube_v2_segments[] = {
    {0.0, 1, -11.83215956619923, 1e-6},  // M: Mach 1e4 at rho = 1, P = 1e-6, running out through the open wall
    {2.0, 1, 0, 1}, {3.0, 1, 0, 1}, {4.0, 1, 0, 1}, {5.0, 1, 0, 1}, {6.0, 1, 0, 1},  // A quiet [3, 4], B waves [5, 6]
    {7.0, 10, 0, 1}, {7.05, 1, 0, 1},                                                  // C
    {8.05, 1, 1, 1}, {8.8, 10, 1, 1}, {8.85, 1, 1, 1}, {9.6, 1, 0, 1},               // C2
    {10.6, 1, 0.75, 1}, {11.1, 0.125, 0, 0.1}, {11.6, 1, 0, 1},                       // D
    {12.6, 1, 0, 0.1}, {13.1, 1e-3, 0, 1e-10}, {13.6, 1, 0, 1},                       // E
    {14.6, 1, 0, 0.1}, {15.1, 1e-6, 0, 1e-13}, {16.1, 1, 0, 1},                       // L: density ratio 1e6, pressure ratio 1e12
    {17.1, 3.857143, -0.810631, 10.33333}, {17.6, 1, -3.44, 1}, {18.1, 1, 0, 1},      // F
    {19.1, 3.857143, 2.629369, 10.33333}, {20.1, 1, 0, 1}, {23.9, 1, 0, 1},           // SO: Mach 3 shock at 20.1 into the sine
    {24.9, 1, 1, 1e-6}, {25.65, 1, -1, 1e-6}, {26.4, 1, 0, 1},                        // N: Noh's cold streams
    {28.4, 5.99924, 19.5975, 460.894}, {28.7, 5.99242, -6.19633, 46.0950}, {29.0, 1, 0, 1},  // S
    {31.75, 1, -2, 0.4}, {32.25, 1, 2, 0.4}, {32.75, 1, 0, 1},                        // G
    {33.75, 1, -4, 0.4}, {34.5, 1, 4, 0.4}, {35.25, 1, 0, 1},                         // V: true vacuum
    {39.25, 1, 0, 1000}, {39.35, 1, 0, 0.01}, {40.15, 1, 0, 100}};                    // H
inline constexpr double super_tube_v2_length = 40.25, super_tube_v2_b = 5.0;
inline constexpr double v2_mach_v = -11.83215956619923, v2_bump_x0 = 1.3, v2_bump_w = 0.5, v2_bump_amp = 0.5;
inline constexpr double v2_so_x = 20.1, v2_so_end = 20.1 + 6 * pi / 5, v2_so_k = 5, v2_so_amp = 0.2;

// Exact cell average on [a, b] of v2: the segments' volume fractions; v1's waves on [5, 6]; on M, rho += 0.5 sin^2(pi (x - 1.3) / 0.5)
// on [1.3, 1.8] at the stream's v and P; ahead of the Shu-Osher shock rho += 0.2 sin(5 (x - 20.1)) on [20.1, 20.1 + 6 pi / 5].
// scripts/problems_1d.py super2_initial() is the same. stream = false: stage M's stream and bump replaced by the ambient gas
// (super-tube-v2-m0, a diagnostic: the moving mesh stops at M, luna-cpp #14). Used by: table ("super-tube-v2[-m0]", "super-v2-<stage>").
inline State<1> super_tube_v2(double a, double b, bool stream = true) {
    constexpr double g = gamma_gas, k = 2 * pi / super_tube_wavelength, B = super_tube_v2_b;
    constexpr int count = sizeof super_tube_v2_segments / sizeof super_tube_v2_segments[0];
    constexpr Segment ambient{0.0, 1, 0, 1};
    State<1> u{0, 0, 0};
    for (int s = 0; s < count; ++s) {
        const Segment& w = s == 0 && !stream ? ambient : super_tube_v2_segments[s];
        const double end = s + 1 < count ? super_tube_v2_segments[s + 1].x : super_tube_v2_length;
        const double f = std::max(0.0, std::min(b, end) - std::max(a, w.x)) / (b - a);
        const State<1> U = conserved({w.rho, w.v, w.P}, g);
        for (int v = 0; v < 3; ++v) u[v] += f * U[v];
    }
    if (const double lo = std::max(a, B), hi = std::min(b, B + 1); hi > lo) {  // v1's stage B, moved
        const double s = super_tube_wave * (std::cos(k * (lo - B)) - std::cos(k * (hi - B))) / k / (b - a);
        u[0] += 2 * s;
        u[1] += std::sqrt(g) * s;
        u[2] += g / (g - 1) * s;
    }
    if (const double lo = std::clamp(a, v2_bump_x0, v2_bump_x0 + v2_bump_w), hi = std::clamp(b, v2_bump_x0, v2_bump_x0 + v2_bump_w); stream && hi > lo) {
        const auto integral = [](double s) { return 0.5 * s - v2_bump_w / (4 * pi) * std::sin(2 * pi * s / v2_bump_w); };
        const double d = v2_bump_amp * (integral(hi - v2_bump_x0) - integral(lo - v2_bump_x0)) / (b - a);
        u[0] += d;
        u[1] += d * v2_mach_v;
        u[2] += d * 0.5 * v2_mach_v * v2_mach_v;
    }
    if (const double lo = std::clamp(a, v2_so_x, v2_so_end), hi = std::clamp(b, v2_so_x, v2_so_end); hi > lo)
        u[0] += v2_so_amp * (std::cos(v2_so_k * (lo - v2_so_x)) - std::cos(v2_so_k * (hi - v2_so_x))) / v2_so_k / (b - a);
    return u;
}

// One stage of v2 alone, open walls: the cells of the tube at N = 4096 from `first` on, `cells` of them (problems_1d.py super2_window).
inline Problem super_v2_stage(int first, int cells) {
    constexpr double h = super_tube_v2_length / 4096;
    return {cells * h, first * h, 0.1, gamma_gas, {Wall::outflow, Wall::outflow}, [](double a, double b) { return super_tube_v2(a, b); }};
}

// amplitude: of the linear waves (signed: LUNA keeps the part odd in it) and of the simple wave (0: its default, 0.1).
inline Problem table(const std::string& name, double amplitude) {
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
                }, voronoi::Ends::walls, {3.857143, 2.629369, 10.33333}};
    if (name == "titarev-toro")
        return {10.0, -5.0, 5.0, 1.4, outflow, [](double a, double b) {
                    return shock_into_wave(a, b, {1.515695, 0.523346, 1.805}, -4.5, 0.1, 20 * pi, 1.4);
                }};
    if (name == "woodward-colella") return {1.0, 0.0, 0.038, 1.4, closed, woodward_colella};
    if (name == "super-tube") return {super_tube_length, 0.0, 2.0, gamma_gas, {Wall::outflow, Wall::reflective}, super_tube};
    // Stage E alone (luna-cpp #25), [13.60, 16.60]: on 370 k cells, the cells of the tube at N = 4096 k from 1678 k on.
    if (name == "super-tube-e") {
        constexpr double h = super_tube_length / 4096;
        return {370 * h, 1678 * h, 0.2, gamma_gas, outflow, super_tube};
    }
    if (name == "super-tube-v2" || name == "super-tube-v2-m0") {
        const bool stream = name == "super-tube-v2";
        return {super_tube_v2_length, 0.0, 4.0, gamma_gas, {Wall::outflow, Wall::reflective},
                [stream](double a, double b) { return super_tube_v2(a, b, stream); }};
    }
    // v2's new stages alone, t = 0.1 (run N = cells at the spacing of 4096, 8 cells for 32768).
    if (name == "super-v2-m") return super_v2_stage(0, 255);
    if (name == "super-v2-l") return super_v2_stage(1434, 256);
    if (name == "super-v2-so") return super_v2_stage(1892, 592);
    if (name == "super-v2-n") return super_v2_stage(2483, 306);
    if (name == "super-v2-v") return super_v2_stage(3383, 408);
    // Stage L of the tube v2 alone, a long-time exact test (luna-cpp #30): (1, 0, 0.1) next to (r, 0, 1e-7 r) at x = 0 on [-2, 9],
    // open walls, t = 4; r = 1e-3 (stage E's states), 1e-6 (stage L's), 1e-9. A single Riemann problem to t = 4: its waves stay
    // within [-1.50, 8.21] (scripts/stage_l.py check), so no wave reaches a wall. N = 1024: h = 0.0107, about the tube's at N = 4096.
    if (name == "stage-l-3") return tube(11.0, -2.0, 4.0, gamma_gas, {1.0, 0.0, 0.1}, {1e-3, 0.0, 1e-10}, 0.0, outflow);
    if (name == "stage-l-6") return tube(11.0, -2.0, 4.0, gamma_gas, {1.0, 0.0, 0.1}, {1e-6, 0.0, 1e-13}, 0.0, outflow);
    if (name == "stage-l-9") return tube(11.0, -2.0, 4.0, gamma_gas, {1.0, 0.0, 0.1}, {1e-9, 0.0, 1e-16}, 0.0, outflow);
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

inline Problem problem(const std::string& name, double amplitude = 0) {
    Problem p = table(name, amplitude);
    p.moving_walls = p.walls;
    const auto entering = [&](const State<1>& W, double side) {
        return W[0] > 0 && side * W[1] > std::sqrt(p.gamma * W[2] / W[0]);
    };
    if (p.walls.left == Wall::outflow && entering(p.left, 1)) {
        p.moving_walls.left = Wall::inflow;
        p.moving_walls.left_inflow = p.left;
    }
    if (p.walls.right == Wall::outflow && entering(p.right, -1)) {
        p.moving_walls.right = Wall::inflow;
        p.moving_walls.right_inflow = p.right;
    }
    return p;
}

}  // namespace hydro::problems_1d
