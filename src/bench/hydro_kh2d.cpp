// Smooth Kelvin-Helmholtz after McNally, Lyra & Passy 2012 on the 2D Voronoi
// mesh, static or moving, timed.
//
//   hydro_kh2d <N> [--moving] [--order=0|1] [--dt=luna|arepo|fixed] [--courant=<c>] [--fixed-dt=<dt>]
//              [--riemann=hllc|hll] [--limiter=face|cell|none] [--update=face|cell] [--jitter=<spacings>]
//              [--integrator=one-pass|two-pass] [--bulk=<u>] [--t-end=<t>] [--steps=<max>] [--repeats=<r>] [--phases] [--memory] [--out=<prefix>]
//
// The setup is LUNA's KH-McNally-2D (ic_generation/kh_mcnally.py): the unit
// box, rho and V_x in two exponential shear layers (eqs. 1, 3; L = 0.025,
// rho 1 / 2, V_x 0.5 / -0.5), P = 2.5, gamma 5/3, V_y = 0.01 sin 4 pi x, point
// values at the generators: N^2 points on the cell centres, moved by up to
// --jitter / 2 spacings. --bulk adds a velocity along x. Particles are numbered
// along a Morton curve.
//
// M(t) (McNally eqs. 6-9) at t = 0.02 k: s, c and d summed over the cells
// at their centroids, weighted by volume (on a square lattice LUNA's sums,
// test_hydro_hard_mcnally_kh.py:52-61); K = max rho V_y^2 / 2. Observing is
// not timed. --dt: LUNA's rule, default Courant 0.8; AREPO's, default 0.3; or a
// constant. --steps stops after that many steps (throughput; no observations).
//
// --integrator: hydro::Integrator, default two-pass on a moving mesh at order 1,
// else one-pass; two-pass exists only there.
//
// Every run is printed, the fastest gives the summary, in M cell updates/s =
// N^2 * steps / stepping time; the final state hash must agree between runs.
// --phases fences the phases and prints their rates. --out writes
// <prefix>.txt (t M K) and <prefix>.bin (doubles per cell: generator x y, centroid
// x y, V, rho, rho_vy).

#include "device_memory.hpp"
#include "finite_volume.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

using Exec = Kokkos::DefaultExecutionSpace;
using Memory = Exec::memory_space;
using Box = exact_predicates::BoxN<2>;
constexpr double gas_gamma = 5.0 / 3.0, layer = 0.025, pressure = 2.5, seed_amplitude = 0.01;

std::string flag(int argc, char** argv, const char* name, const char* fallback) {
    const std::size_t length = std::strlen(name);
    for (int a = 2; a < argc; ++a)
        if (std::strncmp(argv[a], name, length) == 0 && argv[a][length] == '=') return argv[a] + length + 1;
    return fallback;
}
bool has(int argc, char** argv, const char* name) {
    for (int a = 2; a < argc; ++a)
        if (std::strcmp(argv[a], name) == 0) return true;
    return false;
}

struct Options {
    int n = 0, order = 1, steps = 0, repeats = 1;
    double jitter = 0, bulk = 0, courant = 0.8, fixed_dt = 0, t_end = 1.5;
    bool moving = false, phases = false, memory = false, fixed = false;
    hydro::StepRule rule = hydro::StepRule::luna;
    hydro::Integrator integrator = hydro::Integrator::one_pass;
    hydro::Limiter limiter = hydro::Limiter::face_centroid;
    hydro::Update update = hydro::Update::per_face;
    std::string out, riemann = "hllc", dt = "luna";
};

// McNally eq. 1 for y in [0, 1), mirrored about 1/2.
double profile(double y, double f1, double f2) {
    const double h = y < 0.5 ? y : 1.0 - y, fm = 0.5 * (f1 - f2);
    return h < 0.25 ? f1 - fm * std::exp((h - 0.25) / layer) : f2 + fm * std::exp((0.25 - h) / layer);
}

std::uint64_t morton(std::uint32_t x, std::uint32_t y) {
    auto spread = [](std::uint64_t v) {
        v = (v | v << 16) & 0x0000ffff0000ffffull;
        v = (v | v << 8) & 0x00ff00ff00ff00ffull;
        v = (v | v << 4) & 0x0f0f0f0f0f0f0f0full;
        v = (v | v << 2) & 0x3333333333333333ull;
        return (v | v << 1) & 0x5555555555555555ull;
    };
    return spread(x) << 1 | spread(y);
}

struct Mode {
    double M, K;
};

template <class Fv, class Mesh>
Mode mode_amplitude(const Exec& exec, const Fv& fv, const Mesh& mesh, bool moving) {
    const auto q = fv.state();
    const auto x = moving ? fv.positions() : typename Fv::Points();
    const auto generators = mesh.positions();
    const auto centroid = mesh.centroid();
    const auto volume = mesh.volume();
    double s = 0, c = 0, d = 0, K = 0;
    Kokkos::parallel_reduce(
        "kh::mode", Kokkos::RangePolicy<Exec>(exec, 0, mesh.particles()),
        KOKKOS_LAMBDA(int i, double& ss, double& cc, double& dd, double& kk) {
            const double px = (moving ? x(i, 0) : generators(i, 0)) + centroid(i, 0);
            const double raw = (moving ? x(i, 1) : generators(i, 1)) + centroid(i, 1);
            const double y = raw - Kokkos::floor(raw);
            const double vy = q(i, 2) / q(i, 0), V = volume(i);
            const double h = y < 0.5 ? y : 1.0 - y, w = Kokkos::exp(-4 * M_PI * Kokkos::abs(h - 0.25));
            ss += vy * Kokkos::sin(4 * M_PI * px) * w * V;
            cc += vy * Kokkos::cos(4 * M_PI * px) * w * V;
            dd += w * V;
            kk = Kokkos::max(kk, 0.5 * q(i, 2) * q(i, 2) / q(i, 0) / V);
        },
        s, c, d, Kokkos::Max<double>(K));
    return {2 * std::hypot(s / d, c / d), K};
}

template <class Riemann>
int run(const Options& o) {
    using Fv = hydro::FiniteVolume<2, Exec, Riemann>;
    using Mesh = typename Fv::Mesh;
    using X = Kokkos::View<double* [2], Memory>;
    Exec exec;
    const int n = o.n * o.n;
    const Box box(1.0);
    const double spacing = 1.0 / o.n;
    // Points on the cell centres, moved, in Morton order.
    std::vector<int> order(n);
    {
        std::vector<std::uint64_t> keys(n);
        for (int i = 0; i < n; ++i) keys[i] = morton(i / o.n, i % o.n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return keys[a] < keys[b]; });
    }
    X x0("x0", n);
    auto x0h = Kokkos::create_mirror_view(x0);
    std::mt19937_64 rng(20261105);
    std::uniform_real_distribution<double> unit(-0.5, 0.5);
    for (int m = 0; m < n; ++m) {
        const int i = order[m] / o.n, j = order[m] % o.n;
        const double px = (i + 0.5 + o.jitter * unit(rng)) * spacing, py = (j + 0.5 + o.jitter * unit(rng)) * spacing;
        x0h(m, 0) = px - std::floor(px);
        x0h(m, 1) = py - std::floor(py);
    }
    Kokkos::deep_copy(x0, x0h);

    double best = 1e300;
    std::uint64_t first_hash = 0;
    bool same = true;
    long long steps_best = 0;
    double build_ms = 0;
    hydro::Timings best_phases;
    hydro::MeshStats stats_best;
    std::vector<std::array<double, 3>> series;
    for (int r = 0; r < o.repeats; ++r) {
        Kokkos::Timer build_timer;
        Mesh mesh(exec, box);
        mesh.update(x0);
        mesh.update_cells();
        exec.fence();
        build_ms = build_timer.seconds() * 1e3;
        Fv fv(exec, mesh, hydro::Gas{gas_gamma}, o.order, o.update, o.limiter,
              o.moving ? hydro::Motion::moving : hydro::Motion::static_mesh, o.integrator);
        if (o.rule == hydro::StepRule::luna) fv.set_step_rule(hydro::StepRule::luna);
        {
            auto q = Kokkos::create_mirror_view(fv.state());
            auto volume = Kokkos::create_mirror_view(mesh.volume());
            Kokkos::deep_copy(volume, mesh.volume());
            auto positions = Kokkos::create_mirror_view(mesh.positions());
            Kokkos::deep_copy(positions, mesh.positions());
            auto centroids = Kokkos::create_mirror_view(mesh.centroid());
            Kokkos::deep_copy(centroids, mesh.centroid());
            for (int i = 0; i < n; ++i) {
                const double wrapped = positions(i, 1) + centroids(i, 1);
                const double px = positions(i, 0) + centroids(i, 0), py = wrapped - std::floor(wrapped);
                const double rho = profile(py, 1.0, 2.0), vx = profile(py, 0.5, -0.5) + o.bulk;
                const double vy = seed_amplitude * std::sin(4 * M_PI * px);
                const double E = pressure / (gas_gamma - 1) + 0.5 * rho * (vx * vx + vy * vy), V = volume(i);
                q(i, 0) = rho * V;
                q(i, 1) = rho * vx * V;
                q(i, 2) = rho * vy * V;
                q(i, 3) = E * V;
            }
            Kokkos::deep_copy(fv.state(), q);
        }
        series.clear();
        const Mode start = mode_amplitude(exec, fv, mesh, o.moving);
        series.push_back({0.0, start.M, start.K});
        double t = 0, wall = 0;
        long long steps = 0;
        hydro::Timings phases;
        hydro::Timings* timing = o.phases ? &phases : nullptr;
        const int outputs = o.steps > 0 ? 1 : static_cast<int>(std::lround(o.t_end / 0.02));
        Kokkos::Timer timer;
        for (int k = 1; k <= outputs; ++k) {
            const double target = o.steps > 0 ? 1e300 : k * 0.02;
            timer.reset();
            while (t < target * (1 - 1e-13) && (o.steps == 0 || steps < o.steps)) {
                double dt = o.fixed ? o.fixed_dt : 0;
                const double step = fv.time_step(o.courant, timing);
                if (!o.fixed) dt = step;
                dt = std::min(dt, target - t);
                fv.step(dt, timing);
                t = dt == target - t ? target : t + dt;
                ++steps;
                if (!(dt > 0)) {
                    std::cerr << "kh: dt " << dt << " at t = " << t << "\n";
                    return 1;
                }
            }
            exec.fence();
            wall += timer.seconds();
            if (o.steps == 0) {
                const Mode m = mode_amplitude(exec, fv, mesh, o.moving);
                series.push_back({t, m.M, m.K});
            }
        }
        auto q = Kokkos::create_mirror_view(fv.state());
        Kokkos::deep_copy(q, fv.state());
        std::uint64_t hash = 14695981039346656037ull;
        for (int i = 0; i < n; ++i)
            for (int v = 0; v < 4; ++v) {
                std::uint64_t u;
                std::memcpy(&u, &q(i, v), 8);
                hash = (hash ^ u) * 1099511628211ull;
            }
        const double rate = n * double(steps) / wall * 1e-6;
        std::printf("run: wall_s=%.4f steps=%lld M_cell_updates_per_s=%.1f hash=%016llx\n", wall, steps, rate,
                    static_cast<unsigned long long>(hash));
        same = same && (r == 0 || hash == first_hash);
        if (r == 0) first_hash = hash;
        if (wall < best) {
            best = wall;
            steps_best = steps;
            best_phases = phases;
            stats_best = fv.mesh_stats();
        }
        if (r + 1 == o.repeats && !o.out.empty()) {
            std::ofstream text(o.out + ".txt");
            text.precision(17);
            for (const auto& row : series) text << row[0] << " " << row[1] << " " << row[2] << "\n";
            auto volume = Kokkos::create_mirror_view(mesh.volume());
            Kokkos::deep_copy(volume, mesh.volume());
            auto centroid = Kokkos::create_mirror_view(mesh.centroid());
            Kokkos::deep_copy(centroid, mesh.centroid());
            auto positions = Kokkos::create_mirror_view(mesh.positions());
            Kokkos::deep_copy(positions, mesh.positions());
            std::ofstream binary(o.out + ".bin", std::ios::binary);
            for (int i = 0; i < n; ++i) {
                const double row[7] = {positions(i, 0), positions(i, 1), positions(i, 0) + centroid(i, 0), positions(i, 1) + centroid(i, 1),
                                       volume(i), q(i, 0) / volume(i), q(i, 2) / volume(i)};
                binary.write(reinterpret_cast<const char*>(row), sizeof(row));
            }
        }
    }
    const std::string mesh_kind = o.moving ? "moving" : "static";
    std::printf("ours: N=%d mesh=%s integrator=%s order=%d solver=%s limiter=%s dt=%s courant=%g jitter=%g bulk=%g update=%s steps=%lld "
                "wall_s=%.4f build_ms=%.2f M_cell_updates_per_s=%.1f same_hash=%d mesh_rebuilds=%lld stuck=%lld host_patches=%lld "
                "device_flips=%lld M_final=%.8e\n",
                o.n, mesh_kind.c_str(), o.integrator == hydro::Integrator::two_pass ? "two-pass" : "one-pass", o.order, o.riemann.c_str(),
                o.limiter == hydro::Limiter::face_centroid ? "face" : o.limiter == hydro::Limiter::cell_centroid ? "cell" : "none",
                o.dt.c_str(), o.courant, o.jitter, o.bulk, o.update == hydro::Update::per_face ? "face" : "cell", steps_best, best,
                build_ms, n * double(steps_best) / best * 1e-6, same ? 1 : 0, stats_best.rebuilds, stats_best.stuck,
                stats_best.host_flips, stats_best.device_flips, series.back()[1]);
    if (o.phases) {
        const hydro::Timings& p = best_phases;
        const double per = 1e-3 / double(steps_best);
        const auto rate = [&](double ms) { return n / (ms * per) * 1e-6; };
        const double hydro_ms = p.primitives + p.sweep + p.fluxes + p.update, mesh_ms = p.drift + p.mesh + p.cells;
        std::printf("phase: M cells/s hydro %.1f (primitives %.0f sweep %.0f fluxes %.0f update %.0f) mesh %.1f (drift %.0f update %.0f cells %.0f) "
                    "whole %.1f\n",
                    rate(hydro_ms), rate(p.primitives), rate(p.sweep), rate(p.fluxes), rate(p.update), mesh_ms > 0 ? rate(mesh_ms) : 0.0,
                    p.drift > 0 ? rate(p.drift) : 0.0, p.mesh > 0 ? rate(p.mesh) : 0.0, p.cells > 0 ? rate(p.cells) : 0.0,
                    rate(hydro_ms + mesh_ms));
    }
    if (o.memory) std::printf("memory: peak %.1f B/cell, %.3f GB\n", double(DeviceMemory<Memory>::peak) / n, double(DeviceMemory<Memory>::peak) * 1e-9);
    return same ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::ScopeGuard guard(argc, argv);
    if (argc < 2) {
        std::puts("usage: hydro_kh2d <N> [--moving] [--dt=luna|arepo|fixed] [--courant=c] ... (see the top of the source)");
        return 2;
    }
    Options o;
    o.n = std::atoi(argv[1]);
    o.moving = has(argc, argv, "--moving");
    o.phases = has(argc, argv, "--phases");
    o.memory = has(argc, argv, "--memory");
    o.order = std::stoi(flag(argc, argv, "--order", "1"));
    const std::string integrator = flag(argc, argv, "--integrator", "");
    if (!integrator.empty() && integrator != "one-pass" && integrator != "two-pass") throw std::invalid_argument("hydro_kh2d: --integrator=one-pass|two-pass");
    o.integrator = integrator.empty() ? hydro::default_integrator(o.moving ? hydro::Motion::moving : hydro::Motion::static_mesh, o.order)
                   : integrator == "two-pass" ? hydro::Integrator::two_pass
                                              : hydro::Integrator::one_pass;
    o.dt = flag(argc, argv, "--dt", "luna");
    o.courant = std::stod(flag(argc, argv, "--courant", o.dt == "arepo" ? "0.3" : "0.8"));
    o.rule = o.dt == "arepo" ? hydro::StepRule::arepo : hydro::StepRule::luna;
    o.fixed = o.dt == "fixed";
    o.fixed_dt = std::stod(flag(argc, argv, "--fixed-dt", "0"));
    o.riemann = flag(argc, argv, "--riemann", "hllc");
    const std::string limiter = flag(argc, argv, "--limiter", "face");
    o.limiter = limiter == "cell" ? hydro::Limiter::cell_centroid : limiter == "none" ? hydro::Limiter::none : hydro::Limiter::face_centroid;
    o.update = flag(argc, argv, "--update", "face") == "cell" ? hydro::Update::per_cell : hydro::Update::per_face;
    o.jitter = std::stod(flag(argc, argv, "--jitter", "0"));
    o.bulk = std::stod(flag(argc, argv, "--bulk", "0"));
    o.t_end = std::stod(flag(argc, argv, "--t-end", "1.5"));
    o.steps = std::stoi(flag(argc, argv, "--steps", "0"));
    o.repeats = std::stoi(flag(argc, argv, "--repeats", "1"));
    o.out = flag(argc, argv, "--out", "");
    if (o.memory) DeviceMemory<Memory>::start();
    return o.riemann == "hll" ? run<hydro::HLL>(o) : run<hydro::HLLC>(o);
}
