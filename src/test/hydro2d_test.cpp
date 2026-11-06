// The finite-volume hydro on the 2D Voronoi mesh: uniform states, conservation, same bits, no NaN on ties, sound-wave order,
// Sod along an axis, Noh, a blast with a bulk velocity equals the blast at rest; moving meshes run both integrators.

#include "check.hpp"
#include "exact_riemann.hpp"
#include "finite_volume.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

using Exec = Kokkos::DefaultExecutionSpace;
using Memory = Exec::memory_space;
using Fv = hydro::FiniteVolume<2, Exec>;
using Mesh = Fv::Mesh;
using Box = exact_predicates::BoxN<2>;
using X = Kokkos::View<double* [2], Memory>;
using State = hydro::State<2>;

template <class V>
auto host(const V& v) {
    auto h = Kokkos::create_mirror_view(v);
    Kokkos::deep_copy(h, v);
    return h;
}

// nx x ny points on cell centres, moved by up to jitter / 2 spacings.
X lattice(int nx, int ny, const Box& box, double jitter) {
    X x("x", nx * ny);
    auto h = Kokkos::create_mirror_view(x);
    std::mt19937_64 rng(5);
    std::uniform_real_distribution<double> unit(-0.5, 0.5);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j) {
            const double px = (i + 0.5 + jitter * unit(rng)) * box.side[0] / nx;
            const double py = (j + 0.5 + jitter * unit(rng)) * box.side[1] / ny;
            h(i * ny + j, 0) = px - box.side[0] * std::floor(px / box.side[0]);
            h(i * ny + j, 1) = py - box.side[1] * std::floor(py / box.side[1]);
        }
    Kokkos::deep_copy(x, h);
    return x;
}

// Primitive state W(x, y) into the cells, from their centroids.
void fill(Fv& fv, const Mesh& mesh, const hydro::Gas& gas, const std::function<State(double, double)>& W) {
    auto q = Kokkos::create_mirror_view(fv.state());
    const auto volume = host(mesh.volume());
    const auto positions = host(mesh.positions());
    const auto centroid = host(mesh.centroid());
    for (int i = 0; i < mesh.particles(); ++i) {
        const State U = hydro::conserved(W(positions(i, 0) + centroid(i, 0), positions(i, 1) + centroid(i, 1)), gas);
        for (int v = 0; v < 4; ++v) q(i, v) = U[v] * volume(i);
    }
    Kokkos::deep_copy(fv.state(), q);
}

// The average of f(x, y) over each cell's actual polygon: triangles from the generator to each face, midpoint rule (exact for quadratics).
std::vector<double> cell_averages(const Mesh& mesh, const std::function<double(double, double)>& f) {
    const auto start = host(mesh.cell_face_start()), list = host(mesh.cell_face());
    const auto owner = host(mesh.owner()), neighbour = host(mesh.neighbour());
    const auto area = host(mesh.area()), centroid = host(mesh.face_centroid());
    const auto positions = host(mesh.positions());
    const Box box = mesh.sides();
    std::vector<double> averages(mesh.particles());
    for (int i = 0; i < mesh.particles(); ++i) {
        double sum = 0, total = 0;
        for (int k = start(i); k < start(i + 1); ++k) {
            const int e = list(k), o = owner(e);
            // The face's end points relative to generator i.
            double cx = centroid(e, 0), cy = centroid(e, 1);
            if (o != i)
                for (int a = 0; a < 2; ++a) {
                    const double d = positions(o, a) - positions(i, a);
                    (a ? cy : cx) += d - box.side[a] * std::nearbyint(d / box.side[a]);
                }
            const double hx = -0.5 * area(e, 1), hy = 0.5 * area(e, 0);
            const double ax = cx + hx, ay = cy + hy, bx = cx - hx, by = cy - hy;
            const double twice = std::abs(ax * by - ay * bx);
            const double px = positions(i, 0), py = positions(i, 1);
            sum += 0.5 * twice * (f(px + 0.5 * ax, py + 0.5 * ay) + f(px + 0.5 * bx, py + 0.5 * by) + f(px + cx, py + cy)) / 3;
            total += 0.5 * twice;
        }
        averages[i] = sum / total;
    }
    return averages;
}

struct Run {
    int steps = 0;
    double t = 0;
};
Run advance(Fv& fv, double t_end, double courant = 0.3, int most = 100000) {
    Run run;
    while (run.t < t_end * (1 - 1e-12) && run.steps < most) {
        const double dt = std::min(fv.time_step(courant), t_end - run.t);
        fv.step(dt);
        run.t += dt;
        ++run.steps;
    }
    return run;
}

std::uint64_t hash_of(const Fv& fv) {
    const auto q = host(fv.state());
    std::uint64_t hash = 14695981039346656037ull;
    for (std::size_t i = 0; i < q.extent(0); ++i)
        for (int v = 0; v < 4; ++v) {
            std::uint64_t u;
            std::memcpy(&u, &q(i, v), 8);
            hash = (hash ^ u) * 1099511628211ull;
        }
    return hash;
}

// The integrators of a run at order 1: the default first.
std::vector<hydro::Integrator> integrators(hydro::Motion motion) {
    if (motion == hydro::Motion::moving) return {hydro::Integrator::two_pass, hydro::Integrator::one_pass};
    return {hydro::Integrator::one_pass};
}

std::string label(hydro::Motion motion, hydro::Integrator integrator) {
    if (motion != hydro::Motion::moving) return "static";
    return integrator == hydro::Integrator::two_pass ? "moving, two passes" : "moving, one pass";
}

void uniform(Check& check) {
    const hydro::Gas gas{5.0 / 3.0};
    const Box box(1.0);
    for (double jitter : {0.0, 0.2, 0.4})
        for (auto motion : {hydro::Motion::static_mesh, hydro::Motion::moving})
            for (auto integrator : integrators(motion))
            for (double bulk : {0.0, 0.7}) {
                Mesh mesh(Exec(), box);
                mesh.update(lattice(32, 32, box, jitter));
                mesh.update_cells();
                Fv fv(Exec(), mesh, gas, 1, hydro::Update::per_face, hydro::Limiter::face_centroid, motion, integrator);
                fill(fv, mesh, gas, [&](double, double) { return State{1.5, bulk, -0.5 * bulk, 2.0}; });
                advance(fv, 1.0, 0.3, 30);
                const auto q = host(fv.state());
                const auto volume = host(mesh.volume());
                double worst = 0;
                for (int i = 0; i < mesh.particles(); ++i) {
                    worst = std::max(worst, std::abs(q(i, 0) / volume(i) / 1.5 - 1));
                    worst = std::max(worst, std::abs(q(i, 1) / volume(i) - 1.5 * bulk));
                    worst = std::max(worst, std::abs(q(i, 2) / q(i, 0) + 0.5 * bulk));
                }
                const std::string name = label(motion, integrator) + " jitter " + std::to_string(jitter) + " bulk " + std::to_string(bulk);
                // Moving: the grid step shows in the volumes (~3e-8). jitter 0.4: the update on the start-of-step mesh keeps a uniform state only to first order.
                const bool moving = motion == hydro::Motion::moving;
                const double bound = !moving ? 1e-11 : jitter > 0.3 ? 1e-3 : 1e-7;
                check.expect(worst < bound, name + ": uniform state stays, worst " + sci(worst));
            }
}

void conservation(Check& check) {
    const hydro::Gas gas{1.4};
    const Box box(2.0, 1.0);
    const auto blast = [](double x, double y) {
        const double r2 = (x - 1) * (x - 1) + (y - 0.5) * (y - 0.5);
        return State{1 + 0.2 * std::sin(6 * x), 0.3, 0.1 * std::cos(4 * y), r2 < 0.01 ? 50.0 : 1.0};
    };
    for (auto motion : {hydro::Motion::static_mesh, hydro::Motion::moving})
        for (auto integrator : integrators(motion))
        for (auto update : {hydro::Update::per_face, hydro::Update::per_cell}) {
            const std::string name = label(motion, integrator) + (update == hydro::Update::per_face ? " per-face" : " per-cell");
            std::uint64_t hash[2];
            std::array<long double, 4> before{}, after{};
            for (int r = 0; r < 2; ++r) {
                Mesh mesh(Exec(), box);
                mesh.update(lattice(64, 32, box, 0.3));
                mesh.update_cells();
                Fv fv(Exec(), mesh, gas, 1, update, hydro::Limiter::face_centroid, motion, integrator);
                fill(fv, mesh, gas, blast);
                auto sums = [&](std::array<long double, 4>& s) {
                    const auto q = host(fv.state());
                    s = {};
                    for (int i = 0; i < mesh.particles(); ++i)
                        for (int v = 0; v < 4; ++v) s[v] += q(i, v);
                };
                sums(before);
                advance(fv, 1.0, 0.3, 40);
                sums(after);
                hash[r] = hash_of(fv);
            }
            double worst = 0;
            for (int v = 0; v < 4; ++v) worst = std::max(worst, double(std::abs(after[v] - before[v]) / std::max(1.0L, std::abs(before[v]))));
            check.expect(worst < 1e-12, name + ": conserved, worst " + sci(worst));
            check.expect(hash[0] == hash[1], name + ": two runs, the same bits");
        }
}

// The L1 error of rho of a sound wave along x after one period: exact polygon averages at start and end (a point value would add a term of the order measured).
double wave_error(int n, double jitter, hydro::Motion motion, hydro::Integrator integrator) {
    const hydro::Gas gas{5.0 / 3.0};
    const Box box(1.0);
    const double amplitude = 1e-4, rho0 = 1, c = 1, p0 = rho0 * c * c / gas.gamma;
    const auto wave = [=](double x, double) {
        const double s = amplitude * std::sin(2 * M_PI * x);
        return State{rho0 * (1 + s), c * s, 0.0, p0 * (1 + gas.gamma * s)};
    };
    Mesh mesh(Exec(), box);
    mesh.update(lattice(n, n, box, jitter));
    mesh.update_cells();
    Fv fv(Exec(), mesh, gas, 1, hydro::Update::per_face, hydro::Limiter::face_centroid, motion, integrator);
    {
        auto q = Kokkos::create_mirror_view(fv.state());
        const auto volume = host(mesh.volume());
        std::vector<double> mean[4];
        for (int v = 0; v < 4; ++v)
            mean[v] = cell_averages(mesh, [&](double x, double y) { return hydro::conserved(wave(x, y), gas)[v]; });
        for (int i = 0; i < mesh.particles(); ++i)
            for (int v = 0; v < 4; ++v) q(i, v) = mean[v][i] * volume(i);
        Kokkos::deep_copy(fv.state(), q);
    }
    advance(fv, 1.0, 0.3);
    const auto q = host(fv.state());
    const auto volume = host(mesh.volume());
    const std::vector<double> exact = cell_averages(mesh, [&](double x, double y) { return wave(x, y)[0]; });
    double error = 0;
    for (int i = 0; i < mesh.particles(); ++i) error += std::abs(q(i, 0) / volume(i) - exact[i]) * volume(i);
    return error / amplitude;
}

void sound_wave(Check& check) {
    for (double jitter : {0.0, 0.4})
        for (auto motion : {hydro::Motion::static_mesh, hydro::Motion::moving})
            for (auto integrator : integrators(motion)) {
                const double e1 = wave_error(32, jitter, motion, integrator), e2 = wave_error(64, jitter, motion, integrator);
                const double order = std::log2(e1 / e2);
                const std::string name = label(motion, integrator) + " jitter " + std::to_string(jitter);
                const bool one_moving = motion == hydro::Motion::moving && integrator == hydro::Integrator::one_pass;
                check.expect(order > (one_moving && jitter > 0 ? 1.0 : 1.5), name + ": sound wave order " + sci(order) + " (L1 " + sci(e1) + ", " + sci(e2) + ")");
            }
}

void sod(Check& check) {
    const hydro::Gas gas{1.4};
    const exact_riemann::Primitive L{1, 0, 1}, R{0.125, 0, 0.1};
    const exact_riemann::Solver exact(L, R, 1.4);
    const Box box(2.0, 2.0 / 16);
    for (auto motion : {hydro::Motion::static_mesh, hydro::Motion::moving}) {
        const hydro::Integrator integrator = hydro::default_integrator(motion, 1);
        Mesh mesh(Exec(), box);
        mesh.update(lattice(256, 16, box, 0));
        mesh.update_cells();
        Fv fv(Exec(), mesh, gas, 1, hydro::Update::per_face, hydro::Limiter::face_centroid, motion, integrator);
        fill(fv, mesh, gas, [&](double x, double) {
            const bool left = x >= 0.5 && x < 1.5;
            return left ? State{L.rho, 0.0, 0.0, L.p} : State{R.rho, 0.0, 0.0, R.p};
        });
        const double t = 0.2;
        advance(fv, t, 0.3);
        const auto q = host(fv.state());
        const auto volume = host(mesh.volume());
        const auto centroid = host(mesh.centroid());
        const auto positions = host(mesh.positions());
        double error = 0, mass = 0;
        for (int i = 0; i < mesh.particles(); ++i) {
            const double x = positions(i, 0) + centroid(i, 0), V = volume(i);
            if (x < 1.0 || x >= 2.0) continue;
            error += std::abs(q(i, 0) / V - exact.sample((x - 1.5) / t).rho) * V;
            mass += V;
        }
        const std::string name = label(motion, integrator);
        check.expect(error / mass < 0.02, name + ": Sod along x, L1 rho " + sci(error / mass));
    }
}

// Noh's implosion in 2D, gamma 5/3: shock at r = t / 3, rho = 16 inside, 1 + t / r outside (Noh 1987); the box keeps the seam out of r < 0.38.
// The L1 bound is 1.1 times the moving mesh's measured 0.31, static 0.16: a regression bound, not a claim of accuracy.
[[maybe_unused]] void noh(Check& check) {
    const hydro::Gas gas{5.0 / 3.0};
    const Box box(2.0);
    const double t = 0.6;
    for (auto motion : {hydro::Motion::static_mesh, hydro::Motion::moving}) {
        const hydro::Integrator integrator = hydro::default_integrator(motion, 1);
        Mesh mesh(Exec(), box);
        mesh.update(lattice(256, 256, box, 0));
        mesh.update_cells();
        Fv fv(Exec(), mesh, gas, 1, hydro::Update::per_face, hydro::Limiter::face_centroid, motion, integrator);
        fill(fv, mesh, gas, [&](double x, double y) {
            const double dx = x - 1, dy = y - 1, r = std::hypot(dx, dy);
            return State{1.0, r > 0 ? -dx / r : 0.0, r > 0 ? -dy / r : 0.0, 1e-6};
        });
        const Run run = advance(fv, t, 0.3);
        const auto q = host(fv.state());
        const auto volume = host(mesh.volume());
        const auto centroid = host(mesh.centroid());
        const auto positions = host(mesh.positions());
        double inside = 0, inside_v = 0, outside = 0, outside_v = 0, swept = 0;
        for (int i = 0; i < mesh.particles(); ++i) {
            const double dx = positions(i, 0) + centroid(i, 0) - 1, dy = positions(i, 1) + centroid(i, 1) - 1, r = std::hypot(dx, dy);
            const double V = volume(i), rho = q(i, 0) / V;
            if (r > 0.06 && r < 0.17) {
                inside += rho * V;
                inside_v += V;
            }
            if (r > 0.25 && r < 0.38) {
                outside += std::abs(rho - (1 + t / r)) * V;
                outside_v += V;
            }
            if (rho > 8) swept += V;
        }
        const double shock = std::sqrt(swept / M_PI);
        const std::string name = label(motion, integrator);
        check.expect(std::abs(inside / inside_v / 16 - 1) < 0.15 && outside / outside_v < 0.35 && std::abs(shock - t / 3) < 0.02,
                     name + ": Noh: rho inside " + sci(inside / inside_v) + " (16), L1 outside " + sci(outside / outside_v) +
                         ", shock at " + sci(shock) + " (0.2), " + std::to_string(run.steps) + " steps");
    }
}

void galilean(Check& check) {
    const hydro::Gas gas{5.0 / 3.0};
    const Box box(1.0);
    const double u = 3.0, t = 0.1;
    auto rho_at = [&](hydro::Motion motion, hydro::Integrator integrator, double bulk) {
        Mesh mesh(Exec(), box);
        mesh.update(lattice(96, 96, box, 0.2));
        mesh.update_cells();
        Fv fv(Exec(), mesh, gas, 1, hydro::Update::per_face, hydro::Limiter::face_centroid, motion, integrator);
        fill(fv, mesh, gas, [&](double x, double y) {
            const double r2 = (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5);
            return State{1.0, bulk, 0.0, 1.0 + 9.0 * std::exp(-r2 / 0.002)};
        });
        advance(fv, t, 0.3);
        const auto q = host(fv.state());
        const auto volume = host(mesh.volume());
        std::vector<double> kept(mesh.particles());
        for (int i = 0; i < mesh.particles(); ++i) kept[i] = q(i, 0) / volume(i);
        return kept;
    };
    auto distance = [&](const std::vector<double>& a, const std::vector<double>& b) {
        double s = 0;
        for (std::size_t i = 0; i < a.size(); ++i) s += std::abs(a[i] - b[i]);
        return s / a.size();
    };
    for (auto integrator : integrators(hydro::Motion::moving)) {
        const double moving = distance(rho_at(hydro::Motion::moving, integrator, 0.0), rho_at(hydro::Motion::moving, integrator, u));
        check.expect(moving < 2e-3, label(hydro::Motion::moving, integrator) + ": a blast carried at Mach ~2.3 equals the blast at rest, cell by cell, L1 " + sci(moving));
    }
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::ScopeGuard guard(argc, argv);
    std::cout << std::unitbuf;
    Check check;
    uniform(check);
    conservation(check);
    sound_wave(check);
    sod(check);
#ifndef __SANITIZE_ADDRESS__
    noh(check);  // the long run (65536 cells, 1900 steps), about 11x slower under sanitizers
#endif
    galilean(check);
    return check.failed() != 0;
}
