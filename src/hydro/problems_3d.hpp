#pragma once

#include "euler.hpp"

#include <array>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// 3D problems with an exact solution, gamma 1.4, shared by hydro_wave3d and the test `hydro_wave`: sound, simple, vortex, tube.
// Boxes, cell counts and end times of each: docs/design.md.
namespace hydro {

constexpr double wave_gamma = 1.4;

struct Problem3D {
    std::array<int, 3> cells;
    std::array<double, 3> side, origin;
    double t_end;
};

inline Problem3D problem_3d(const std::string& name, int nx) {
    if (name == "vortex") return {{nx, nx, nx / 4}, {10, 10, 2.5}, {-5, -5, 0}, 10};
    if (name == "tube") return {{2 * nx, nx / 4, nx / 4}, {2, 0.25, 0.25}, {0, 0, 0}, 0.25};
    if (name == "sound") return {{nx, nx / 4, nx / 4}, {1, 0.25, 0.25}, {0, 0, 0}, 1};
    if (name == "simple") return {{nx, nx / 4, nx / 4}, {1, 0.25, 0.25}, {0, 0, 0}, 0.2};
    throw std::invalid_argument("hydro::problem_3d: sound, simple, vortex or tube");
}

// The lattice of the problem, each point moved by up to `jitter` spacings along each axis, in the box [0, side).
inline std::vector<std::array<double, 3>> jittered_points(const Problem3D& p, double jitter) {
    std::vector<std::array<double, 3>> points;
    std::mt19937_64 rng(1);
    std::uniform_real_distribution<double> shift(-jitter, jitter);
    for (int i = 0; i < p.cells[0]; ++i)
        for (int j = 0; j < p.cells[1]; ++j)
            for (int k = 0; k < p.cells[2]; ++k) {
                const int index[3] = {i, j, k};
                std::array<double, 3> q;
                for (int a = 0; a < 3; ++a) {
                    const double h = p.side[a] / p.cells[a];
                    q[a] = std::fmod((index[a] + 0.5 + shift(rng)) * h + p.side[a], p.side[a]);
                }
                points.push_back(q);
            }
    return points;
}

// The primitive state at t = 0 at (x, y), box coordinates plus the origin.
inline State<3> wave_initial(const std::string& name, double x, double y, double amp) {
    constexpr double G = wave_gamma, two_pi = 6.283185307179586;
    if (name == "simple") {
        const double c = 1 + amp * std::sin(two_pi * x), rho = std::pow(c, 5.0);
        return {rho, 5 * (c - 1), 0, 0, std::pow(rho, G) / G};
    }
    if (name == "sound") {
        const double w = amp * std::sin(two_pi * x);
        return {1 + w, w, 0, 0, (1 + G * w) / G};
    }
    if (name == "vortex") {
        const double beta = 5, r2 = x * x + y * y, e = std::exp(1 - r2);
        const double T = 1 - (G - 1) * beta * beta / (8 * G * two_pi * two_pi / 4) * e;
        const double rho = std::pow(T, 1 / (G - 1)), w = beta / two_pi * std::exp(0.5 * (1 - r2));
        return {rho, -w * y, w * x, 0, rho * T};
    }
    const bool left = x >= 0.5 && x < 1.5;
    return left ? State<3>{1, 0, 0, 0, 1} : State<3>{0.125, 0, 0, 0, 0.1};
}

}  // namespace hydro
