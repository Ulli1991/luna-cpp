#pragma once

#include "finite_volume_1d.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

// Stage 1 of hydro_1d (src/bench/hydro_1d.cpp, also hydro1d_test.cpp): problem(name) -> Problem -> simulate() seeds FiniteVolume<1>.
// LUNA's 1D tests: scripts/luna_1d_run.py has the file:line of each.
namespace hydro::problems_1d {

constexpr double gamma_gas = 1.4;

// One of LUNA's problems. L: length of the line; origin: x of the left end (cells are in [0, L]); t_end: end time;
// average(a, b): exact mean conserved state density on [a, b] in problem coordinates, the initial condition.
// Used by: problem (builds it), hydro_1d simulate (L, origin, t_end, average, walls or moving_walls for the FiniteVolume<1> constructor).
struct Problem {
    double L, origin, t_end;
    Walls walls;
    State<1> (*average)(double a, double b);
    // Used when Motion::moving; differs from `walls` only where gas enters supersonically (the end cell's state would drift as N^2).
    Walls moving_walls;
};

// The conserved density {rho, rho u, E} of the primitive state (rho, u, p) at gamma_gas.
// Used by: sod, shock_into_wave, woodward_colella.
inline State<1> conserved(double rho, double u, double p) {
    return {rho, rho * u, p / (gamma_gas - 1) + 0.5 * rho * u * u};
}
// Fraction of [a, b] that lies below x. Used by: sod, shock_into_wave, woodward_colella.
inline double fraction_below(double x, double a, double b) { return std::clamp((x - a) / (b - a), 0.0, 1.0); }

// Sod: exact cell averages of the jump at t -> 0 (LUNA: er.cell_average(..., t=1e-12)); at t = 0 the volume fraction.
// Used by: problem ("sod"), as Problem::average.
inline State<1> sod(double a, double b) {
    const double f = fraction_below(0.5, a, b);
    const State<1> l = conserved(1.0, 0.0, 1.0), r = conserved(0.125, 0.0, 0.1);
    return {f * l[0] + (1 - f) * r[0], f * l[1] + (1 - f) * r[1], f * l[2] + (1 - f) * r[2]};
}
// A shock at x0 into 1 + amp sin(k x), LUNA's exact cell averages on [a, b] (shu_osher.py:26-46).
// rhoL, uL, pL: the post-shock state left of x0; the gas to the right is at rest with p = 1.
// Used by: shu_osher, titarev_toro.
inline State<1> shock_into_wave(double a, double b, double rhoL, double uL, double pL, double x0, double amp, double k) {
    const double h = b - a, f = fraction_below(x0, a, b), lo = std::max(a, x0);
    double rho = f * rhoL + (1 - f);
    if (b > lo) rho += amp * (std::cos(k * lo) - std::cos(k * b)) / k / h;
    return {rho, f * rhoL * uL, f * (pL / (gamma_gas - 1) + 0.5 * rhoL * uL * uL) + (1 - f) / (gamma_gas - 1)};
}
// Used by: problem ("shu-osher"), as Problem::average.
inline State<1> shu_osher(double a, double b) { return shock_into_wave(a, b, 3.857143, 2.629369, 10.33333, -4, 0.2, 5); }
// Titarev-Toro 2004, titarev_toro.py: the high-frequency entropy wave.
// Used by: problem ("titarev-toro"), as Problem::average.
inline State<1> titarev_toro(double a, double b) {
    return shock_into_wave(a, b, 1.515695, 0.523346, 1.805, -4.5, 0.1, 20 * 3.141592653589793);
}
// Woodward-Colella, LUNA's exact cell averages (woodward_colella.py:14-27).
// Used by: problem ("woodward-colella"), as Problem::average.
inline State<1> woodward_colella(double a, double b) {
    const double f1 = fraction_below(0.1, a, b), f2 = fraction_below(0.9, a, b);
    const double p = f1 * 1000 + (f2 - f1) * 0.01 + (1 - f2) * 100;
    return {1.0, 0.0, p / (gamma_gas - 1)};
}

// The Problem called `name` (sod, shu-osher, titarev-toro, woodward-colella); throws on an unknown name.
// Used by: hydro_1d main (once, then passed to simulate), hydro1d_test.
inline Problem problem(const std::string& name) {
    const Walls outflow{Wall::outflow, Wall::outflow};
    if (name == "sod") return {1.0, 0.0, 0.25, outflow, sod, outflow};
    if (name == "shu-osher") {
        Walls inflow = outflow;
        inflow.left = Wall::inflow;
        inflow.left_inflow = {3.857143, 2.629369, 10.33333};
        return {10.0, -5.0, 1.8, outflow, shu_osher, inflow};
    }
    if (name == "titarev-toro") return {10.0, -5.0, 5.0, outflow, titarev_toro, outflow};  // subsonic: the wall must let waves out
    if (name == "woodward-colella")
        return {1.0, 0.0, 0.038, Walls{Wall::reflective, Wall::reflective}, woodward_colella, Walls{Wall::reflective, Wall::reflective}};
    throw std::invalid_argument("hydro_1d: unknown problem " + name);
}

}  // namespace hydro::problems_1d
