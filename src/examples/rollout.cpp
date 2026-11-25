// A rollout of the 2D or 3D finite-volume step at a fixed dt and its reverse-mode derivative, for optimisers in Python
// (scripts/examples/rollout.py): the KH twin experiment, the field-level inference and the emergence examples. The mesh is the
// Voronoi mesh of a regular n^D lattice in the periodic unit box (the Cartesian grid), static or moving.
#include "finite_volume.hpp"
#include "mesh_derivative.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// What --help prints.
static const char usage[] = R"(Usage: rollout --dim=2|3 --n=<cells a side> --steps=<k> --dt=<dt> [--gamma=5/3] [--moving] [--courant=0.3]
  serve: one command a line on stdin, one answer a line (paths are raw float64 files, C order):
    forward <U0 in> <final out>     U0: N x V intensive (rho, rho v, E) at the lattice cells; runs the steps, keeping the tape;
                                    final: N x (V + D + 1) rows of (Q extensive, x, volume). Answers "ok <ms> <cfl violations>".
    adjoint <bars in> <U0bar out>   bars: N x (V + D + 1) rows of (Q bar, x bar, volume bar) at the end of the last forward;
                                    U0bar = dL/dU0 (intensive, N x V). Answers "ok <ms>".
    tangent <U0 in> <dU0 in> <dQ out>  the forward with the tangent dU0 (intensive) beside it, the positions' tangent 0;
                                    dQ: N x V (extensive) at the end. Answers "ok <ms>".
    cfl <U0 in>                     "ok <time_step(courant) at U0>".
    geometry <out>                  N x (D + 1) rows of (x, volume) of the initial lattice. Answers "ok".
    memory                          "ok <peak device bytes> <bytes now>" (Kokkos Tools allocation callbacks).
    quit
The fixed dt is checked against time_step(courant) of every step; the steps where it is larger are counted.
)";

namespace {

using Exec = Kokkos::DefaultExecutionSpace;
using Memory = Exec::memory_space;

std::string flag(int argc, char** argv, const char* name, const char* fallback) {
    const std::size_t length = std::strlen(name);
    for (int a = 1; a < argc; ++a)
        if (std::strncmp(argv[a], name, length) == 0 && argv[a][length] == '=') return argv[a] + length + 1;
    return fallback;
}
bool has(int argc, char** argv, const char* name) {
    for (int a = 1; a < argc; ++a)
        if (std::strcmp(argv[a], name) == 0) return true;
    return false;
}
double now_ms() {
    Kokkos::fence();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Peak of the device allocations. Used by: main ("memory").
struct DeviceMemory {
    static inline std::map<const void*, std::uint64_t> live;
    static inline std::uint64_t current = 0, peak = 0;
    static bool device(const Kokkos_Profiling_SpaceHandle& h) { return std::strcmp(h.name, Memory::name()) == 0; }
    static void allocate(Kokkos_Profiling_SpaceHandle h, const char*, const void* ptr, std::uint64_t size) {
        if (!device(h)) return;
        live[ptr] = size;
        current += size;
        peak = std::max(peak, current);
    }
    static void deallocate(Kokkos_Profiling_SpaceHandle h, const char*, const void* ptr, std::uint64_t) {
        if (!device(h)) return;
        const auto it = live.find(ptr);
        if (it == live.end()) return;
        current -= it->second;
        live.erase(it);
    }
    static void start() {
        Kokkos::Tools::Experimental::set_allocate_data_callback(allocate);
        Kokkos::Tools::Experimental::set_deallocate_data_callback(deallocate);
    }
};

std::vector<double> read_file(const std::string& path, std::size_t count) {
    std::vector<double> v(count);
    std::ifstream in(path, std::ios::binary);
    in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(count * sizeof(double)));
    if (!in) throw std::runtime_error("rollout: short read of " + path);
    return v;
}
void write_file(const std::string& path, const std::vector<double>& v) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(double)));
}

template <int D>
struct Rollout {
    using FV = hydro::FiniteVolume<D, Exec, hydro::HLLC, hydro::Precision<>>;
    using Mesh = typename FV::Mesh;
    using Cells = typename FV::Cells;
    using Points = typename FV::Points;
    using Scalars = Kokkos::View<double*, Memory>;
    using HostCells = Kokkos::View<double* [D + 2], Kokkos::LayoutRight, Kokkos::HostSpace>;
    using HostPoints = Kokkos::View<double* [D], Kokkos::LayoutRight, Kokkos::HostSpace>;
    using Tape = Kokkos::View<double** [D + 2], Kokkos::LayoutRight, Memory>;
    using PointTape = Kokkos::View<double** [D], Kokkos::LayoutRight, Memory>;
    static constexpr int V = D + 2;

    int n, side, steps;
    double dt, courant;
    bool moving;
    exact_predicates::BoxN<D> box;
    Points x0;
    Mesh mesh;
    FV fv;
    Tape tape;
    PointTape xtape;
    std::vector<double> V0;

    static Points lattice(int side, const exact_predicates::BoxN<D>& box) {
        int n = 1;
        for (int a = 0; a < D; ++a) n *= side;
        Points x("x0", n);
        auto h = Kokkos::create_mirror_view(x);
        for (int i = 0; i < n; ++i) {
            int r = i;
            for (int a = D - 1; a >= 0; --a) {  // C order: the last axis fastest
                const int k = r % side;
                r /= side;
                h(i, a) = std::floor((k + 0.5) / side / box.step) * box.step;
            }
        }
        Kokkos::deep_copy(x, h);
        return x;
    }
    static Mesh& built(Mesh& m, const Points& x) {
        m.update(x);
        m.update_cells();
        return m;
    }

    Rollout(int side_, int steps_, double dt_, double gamma, bool moving_, double courant_)
        : n(static_cast<int>(std::pow(side_, D))), side(side_), steps(steps_), dt(dt_), courant(courant_), moving(moving_), box(1.0),
          x0(lattice(side_, box)), mesh(Exec{}, box),
          fv(Exec{}, built(mesh, x0), hydro::Gas{gamma}, 1, hydro::Update::per_face, hydro::Limiter::face_centroid,
             moving_ ? hydro::Motion::moving : hydro::Motion::static_mesh) {
        auto hv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.volume());
        for (int i = 0; i < n; ++i) V0.push_back(hv(i));
        tape = Tape("tape", steps, n);
        if (moving) xtape = PointTape("xtape", steps, n);
    }

    // Puts (q, x) into fv and the mesh (moving).
    void load_positions(const Points& x) {
        Kokkos::deep_copy(fv.positions(), x);
        mesh.update(fv.positions());
        mesh.update_cells();
        fv.mesh_moved();
    }

    // Returns the steps whose time_step(courant) was below dt.
    int forward(const std::vector<double>& U0) {
        if (moving) load_positions(x0);
        HostCells h("q0", n);
        for (int i = 0; i < n; ++i)
            for (int v = 0; v < V; ++v) h(i, v) = U0[i * V + v] * V0[i];
        Kokkos::deep_copy(Kokkos::subview(fv.state(), std::make_pair(0, n), Kokkos::ALL), h);
        int violations = 0;
        for (int s = 0; s < steps; ++s) {
            Kokkos::deep_copy(Kokkos::subview(tape, s, Kokkos::ALL, Kokkos::ALL), Kokkos::subview(fv.state(), std::make_pair(0, n), Kokkos::ALL));
            if (moving) Kokkos::deep_copy(Kokkos::subview(xtape, s, Kokkos::ALL, Kokkos::ALL), Kokkos::subview(fv.positions(), std::make_pair(0, n), Kokkos::ALL));
            const double cfl_dt = fv.time_step(courant);
            violations += cfl_dt < dt;
            fv.step(dt);
        }
        return violations;
    }
    std::vector<double> final_rows() {
        const int w = V + D + 1;
        std::vector<double> rows(static_cast<std::size_t>(n) * w);
        HostCells q("q", n);
        Kokkos::deep_copy(q, Kokkos::subview(fv.state(), std::make_pair(0, n), Kokkos::ALL));
        auto x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, moving ? fv.positions() : x0);
        auto vol = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.volume());
        for (int i = 0; i < n; ++i) {
            for (int v = 0; v < V; ++v) rows[i * w + v] = q(i, v);
            for (int a = 0; a < D; ++a) rows[i * w + V + a] = x(i, a);
            rows[i * w + V + D] = vol(i);
        }
        return rows;
    }
    // The forward again with the tangent dU0 (intensive; the positions' tangent 0) beside it; returns dQ at the end (extensive).
    std::vector<double> tangent(const std::vector<double>& U0, const std::vector<double>& dU0) {
        if (moving) load_positions(x0);
        HostCells h("q0", n), d("dq0", n);
        for (int i = 0; i < n; ++i)
            for (int v = 0; v < V; ++v) {
                h(i, v) = U0[i * V + v] * V0[i];
                d(i, v) = dU0[i * V + v] * V0[i];
            }
        Kokkos::deep_copy(Kokkos::subview(fv.state(), std::make_pair(0, n), Kokkos::ALL), h);
        Cells dq("dq", n);
        Kokkos::deep_copy(dq, d);
        Points dx("dx", n);
        for (int s = 0; s < steps; ++s) {
            if (moving) {
                fv.time_step(courant);
                const hydro::MeshDerivative<Mesh> geometry(mesh);
                fv.step_moving_tangent(dt, dq, dx, geometry);
            } else {
                fv.time_step_tangent(courant, dq);
                fv.step_tangent(dt, dq);
            }
        }
        Kokkos::deep_copy(d, dq);
        std::vector<double> out(static_cast<std::size_t>(n) * V);
        for (int i = 0; i < n; ++i)
            for (int v = 0; v < V; ++v) out[i * V + v] = d(i, v);
        return out;
    }
    double cfl_dt(const std::vector<double>& U0) {
        if (moving) load_positions(x0);
        HostCells h("q0", n);
        for (int i = 0; i < n; ++i)
            for (int v = 0; v < V; ++v) h(i, v) = U0[i * V + v] * V0[i];
        Kokkos::deep_copy(Kokkos::subview(fv.state(), std::make_pair(0, n), Kokkos::ALL), h);
        return fv.time_step(courant);
    }

    // bars at the end of the last forward (the mesh is the final one) -> dL/dU0 (intensive).
    std::vector<double> adjoint(const std::vector<double>& bars) {
        const int w = V + D + 1;
        HostCells hq("qbar", n);
        HostPoints hx("xbar", n);
        Kokkos::View<double*, Kokkos::HostSpace> hV("Vbar", n);
        for (int i = 0; i < n; ++i) {
            for (int v = 0; v < V; ++v) hq(i, v) = bars[i * w + v];
            for (int a = 0; a < D; ++a) hx(i, a) = bars[i * w + V + a];
            hV(i) = bars[i * w + V + D];
        }
        Cells qbar("qbar", n);
        Kokkos::deep_copy(qbar, hq);
        if (!moving) {
            for (int s = steps - 1; s >= 0; --s) {
                Kokkos::deep_copy(Kokkos::subview(fv.state(), std::make_pair(0, n), Kokkos::ALL), Kokkos::subview(tape, s, Kokkos::ALL, Kokkos::ALL));
                fv.step_adjoint(dt, qbar);
            }
        } else {
            Points xbar("xbar", n), zero_c("cbar", n);
            Kokkos::deep_copy(xbar, hx);
            Scalars Vbar("Vbar", n);
            Kokkos::deep_copy(Vbar, hV);
            const int faces = static_cast<int>(mesh.faces());
            Points Abar("Abar", faces), xfbar("xfbar", faces);
            {
                const hydro::MeshDerivative<Mesh> geometry(mesh);  // the final mesh: the volumes' bars onto the positions
                geometry.adjoint(Vbar, zero_c, Abar, xfbar, xbar);
            }
            for (int s = steps - 1; s >= 0; --s) {
                Points xs("xs", n);
                Kokkos::deep_copy(xs, Kokkos::subview(xtape, s, Kokkos::ALL, Kokkos::ALL));
                load_positions(xs);
                Kokkos::deep_copy(Kokkos::subview(fv.state(), std::make_pair(0, n), Kokkos::ALL), Kokkos::subview(tape, s, Kokkos::ALL, Kokkos::ALL));
                const hydro::MeshDerivative<Mesh> geometry(mesh);
                fv.step_moving_adjoint(dt, qbar, xbar, geometry);
            }
        }
        Kokkos::deep_copy(hq, qbar);
        std::vector<double> out(static_cast<std::size_t>(n) * V);
        for (int i = 0; i < n; ++i)
            for (int v = 0; v < V; ++v) out[i * V + v] = hq(i, v) * V0[i];
        return out;
    }
};

template <int D>
int serve(int argc, char** argv) {
    const bool moving = has(argc, argv, "--moving");
    Rollout<D> r(std::stoi(flag(argc, argv, "--n", "64")), std::stoi(flag(argc, argv, "--steps", "100")), std::stod(flag(argc, argv, "--dt", "1e-3")),
                 std::stod(flag(argc, argv, "--gamma", "1.6666666666666667")), moving, std::stod(flag(argc, argv, "--courant", "0.3")));
    const int V = Rollout<D>::V;
    std::string line;
    std::cout.precision(17);
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        std::string cmd, a, b;
        in >> cmd >> a >> b;
        if (cmd == "quit") break;
        if (cmd == "forward") {
            const auto U0 = read_file(a, static_cast<std::size_t>(r.n) * V);
            const double t0 = now_ms();
            const int violations = r.forward(U0);
            const double ms = now_ms() - t0;
            write_file(b, r.final_rows());
            std::cout << "ok " << ms << ' ' << violations << std::endl;
        } else if (cmd == "adjoint") {
            const auto bars = read_file(a, static_cast<std::size_t>(r.n) * (V + D + 1));
            const double t0 = now_ms();
            const auto out = r.adjoint(bars);
            const double ms = now_ms() - t0;
            write_file(b, out);
            std::cout << "ok " << ms << std::endl;
        } else if (cmd == "tangent") {
            std::string c;
            in >> c;
            const auto U0 = read_file(a, static_cast<std::size_t>(r.n) * V), dU0 = read_file(b, static_cast<std::size_t>(r.n) * V);
            const double t0 = now_ms();
            const auto out = r.tangent(U0, dU0);
            const double ms = now_ms() - t0;
            write_file(c, out);
            std::cout << "ok " << ms << std::endl;
        } else if (cmd == "cfl") {
            const auto U0 = read_file(a, static_cast<std::size_t>(r.n) * V);
            std::cout << "ok " << r.cfl_dt(U0) << std::endl;
        } else if (cmd == "geometry") {
            auto x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, r.x0);
            std::vector<double> rows;
            for (int i = 0; i < r.n; ++i) {
                for (int k = 0; k < D; ++k) rows.push_back(x(i, k));
                rows.push_back(r.V0[i]);
            }
            write_file(a, rows);
            std::cout << "ok" << std::endl;
        } else if (cmd == "memory") {
            std::cout << "ok " << DeviceMemory::peak << ' ' << DeviceMemory::current << std::endl;
        } else {
            std::cout << "error unknown command" << std::endl;
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    for (int a = 1; a < argc; ++a)
        if (std::string(argv[a]) == "--help") {
            std::cout << usage;
            return 0;
        }
    Kokkos::ScopeGuard kokkos(argc, argv);
    DeviceMemory::start();
    const int dim = std::stoi(flag(argc, argv, "--dim", "2"));
    return dim == 3 ? serve<3>(argc, argv) : serve<2>(argc, argv);
}
