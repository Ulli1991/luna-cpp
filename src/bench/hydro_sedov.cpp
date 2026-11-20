// Sedov-Taylor blast (LUNA's sedov3d_run.py). Usage: hydro_sedov <side> <jittered|uniform> [flags] (docs/design.md).
// Drives: main -> run_precision -> run: Mesh::update/update_cells -> FiniteVolume (set Q) -> loop time_step, step -> write_snapshot.


#include "pin_notice.hpp"
#include "finite_volume.hpp"
#include "grid_finite_volume.hpp"
#include "snapshot.hpp"
#include "support.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace {

using namespace support;

// Device memory through the Kokkos Tools allocation callbacks: now and at the phase peak, with the live allocations there.
// Used by: main (start, with --memory), run (phase("mesh"), phase("hydro_setup"), ...).
struct DeviceMemory {
    static inline bool on = false;
    static inline std::unordered_map<const void*, std::pair<std::string, std::uint64_t>> live;
    static inline std::uint64_t current = 0, peak = 0;
    static inline std::map<std::string, std::uint64_t> at_peak;

    static bool device(const Kokkos_Profiling_SpaceHandle& h) { return std::strcmp(h.name, Memory::name()) == 0; }
    static void allocate(Kokkos_Profiling_SpaceHandle h, const char* label, const void* ptr, std::uint64_t size) {
        if (!device(h)) return;
        live[ptr] = {label, size};
        current += size;
        if (current <= peak) return;
        peak = current;
        at_peak.clear();
        for (const auto& [p, entry] : live) at_peak[entry.first] += entry.second;
    }
    static void deallocate(Kokkos_Profiling_SpaceHandle h, const char*, const void* ptr, std::uint64_t) {
        if (!device(h)) return;
        const auto it = live.find(ptr);
        if (it == live.end()) return;
        current -= it->second.second;
        live.erase(it);
    }
    static void start() {
        on = true;
        Kokkos::Tools::Experimental::set_allocate_data_callback(allocate);
        Kokkos::Tools::Experimental::set_deallocate_data_callback(deallocate);
    }
    static void phase(const std::string& name, long long n) {
        if (!on) return;
        Kokkos::fence();
        std::cout << "memory: phase=" << name << " current_MB=" << current / 1e6 << " peak_MB=" << peak / 1e6
                  << " peak_bytes_per_cell=" << double(peak) / n << " current_bytes_per_cell=" << double(current) / n;
#ifdef KOKKOS_ENABLE_CUDA
        std::size_t free = 0, total = 0;
        cudaMemGetInfo(&free, &total);
        std::cout << " gpu_used_MB=" << (total - free) / 1e6 << " gpu_total_MB=" << total / 1e6;
#endif
        std::cout << "\n";
        for (const auto& [label, bytes] : at_peak)
            std::cout << "  " << label << " " << bytes / 1e6 << " MB " << double(bytes) / n << " B/cell\n";
        peak = current;
        at_peak.clear();
        for (const auto& [p, entry] : live) at_peak[entry.first] += entry.second;
    }
};

// The value of `--name=value` ("1" for a bare `--name`), else `fallback`. Used by: main.
std::string flag(int argc, char** argv, const std::string& name, const std::string& fallback) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.rfind(name + "=", 0) == 0) return arg.substr(name.size() + 1);
        if (arg == name) return "1";
    }
    return fallback;
}

// The points in Morton order (21 bits per coordinate), so neighbouring cells sit close in memory.
// box: the periodic box that scales the coordinates to [0, 1). Used by: run (before Mesh::update, unless --input-order).
Points morton_order(Points points, const Box& box) {
    auto spread = [](std::uint64_t v) {  // the 21 bits of v to every third bit
        v &= 0x1fffff;
        v = (v | v << 32) & 0x1f00000000ffffULL;
        v = (v | v << 16) & 0x1f0000ff0000ffULL;
        v = (v | v << 8) & 0x100f00f00f00f00fULL;
        v = (v | v << 4) & 0x10c30c30c30c30c3ULL;
        v = (v | v << 2) & 0x1249249249249249ULL;
        return v;
    };
    std::vector<std::pair<std::uint64_t, std::size_t>> keys(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        std::uint64_t key = 0;
        for (int a = 0; a < 3; ++a) key |= spread(static_cast<std::uint64_t>(points[i][a] * box.inverse[a] * (1 << 21))) << a;
        keys[i] = {key, i};
    }
    std::sort(keys.begin(), keys.end());
    Points sorted(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) sorted[i] = points[keys[i].second];
    return sorted;
}

// Median of the per-step phase times. Used by: run (the phases_ms_median and throughput lines).
double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// The Sedov set-up of LUNA (sedov3d_run.py, sedov3d.py): units, box edge (kpc), gas and the blast/background state in code units.
// Used by: run (initial state, t_end, snapshot units), main (box edge).
struct Sedov {
    static constexpr double year_s = 3.15576e7, kpc_cm = 3.08568e21, msun_g = 1.98892e33;
    static constexpr double proton_g = 1.6726219e-24, boltzmann = 1.380649e-16;
    hydro::Units units{kpc_cm, 1e10 * msun_g, 1e5};
    double box = 50.0e-3;  // kpc
    double gamma = 5.0 / 3.0;
    double time_s() const { return units.length_cm / units.velocity_cm_s; }
    double density_cgs() const { return units.mass_g / std::pow(units.length_cm, 3); }
    double pressure_cgs() const { return density_cgs() * units.velocity_cm_s * units.velocity_cm_s; }
    double energy_cgs() const { return units.mass_g * units.velocity_cm_s * units.velocity_cm_s; }
    double t_end() const { return 1.0e4 * year_s / time_s(); }
    double rho() const { return 1.0 * 1.0 * proton_g / density_cgs(); }
    double pressure() const { return 1.0 * boltzmann * 1.0e4 / pressure_cgs(); }
    double energy() const { return 1.0e51 / energy_cgs(); }
};

// Sums of the conserved Q over all cells. Used by: totals, run (before/after the loop, for the conservation report).
struct Totals {
    double mass = 0, px = 0, py = 0, pz = 0, energy = 0;
};

// Sums mass, momenta and energy of fv.state() on the device. Used by: run.
template <class FV>
Totals totals(const FV& fv) {
    const auto q = fv.state();
    Totals t;
    Kokkos::parallel_reduce(
        "totals", Kokkos::RangePolicy<Exec>(0, static_cast<int>(q.extent(0))),
        KOKKOS_LAMBDA(int i, double& m, double& x, double& y, double& z, double& e) {
            m += q(i, 0);
            x += q(i, 1);
            y += q(i, 2);
            z += q(i, 3);
            e += q(i, 4);
        },
        t.mass, t.px, t.py, t.pz, t.energy);
    return t;
}

// FNV-1a over the bits of every value of a host view (2D, rows x columns); h chains several views into one hash.
// Used by: run (the final checksum line of Q, positions and volume).
template <class View>
std::uint64_t bits_hash(const View& v, std::uint64_t h = 1469598103934665603ULL) {
    for (std::size_t i = 0; i < v.extent(0); ++i)
        for (std::size_t a = 0; a < v.extent(1); ++a) {
            const double x = v(i, a);
            std::uint64_t u = 0;
            std::memcpy(&u, &x, sizeof(u));
            for (int b = 0; b < 8; ++b) {
                h ^= (u >> (8 * b)) & 0xff;
                h *= 1099511628211ULL;
            }
        }
    return h;
}

// Parsed command line. Filled by: main; Used by: run_precision, run.
struct Options {
    int side = 0, order = 1, snapshots = 10;
    std::string kind, loop, limit, riemann, out, precision;
    hydro::Update update = hydro::Update::per_face;
    hydro::Limiter limiter = hydro::Limiter::face_centroid;
    bool moving = false, input_order = false, blast = true, gradients = false;
    double cold = 1;
    double bulk = 0;
    long long max_steps = 0;
    Box box{1.0};
    bool phases = false;       // --phases: fence and time each phase of a step (off: one wall time a step)
    bool plain_faces = false;  // --plain-faces: the mesh layout as the flags set it (the hydro's default: owner bit, and faces by owner when static)
    bool keep_mesh = false;    // --keep-mesh: a static run keeps the whole mesh (by default the hydro lets go of it, FiniteVolume::release_mesh)
    voronoi::Layout layout;  // the mesh's memory layout (--curve, --every, --far, --owner-faces, --owner-bit)
};

// The whole run with the hydro's precision policy P (the mesh carries float face rows if P asks for them) and Riemann solver R.
// points: the generators in the periodic box o.box. Used by: run_precision.
template <class P, class R>
void run(const Options& o, Points points) {
    using FV = hydro::FiniteVolume<3, Exec, R, P>;
    using Mesh = typename FV::Mesh;
    // CourantFac of AREPO next generation's Sedov examples.
    const double courant = 0.3;
    const Sedov sedov;
    const Box& box = o.box;
    const long long n = static_cast<long long>(points.size());
    const hydro::Gas gas{sedov.gamma};
    const Exec exec;

    Kokkos::Timer timer;
    auto owned_mesh = std::make_unique<Mesh>(exec, box);  // let go of after the hydro's set-up on a static mesh (release_mesh)
    Mesh& mesh = *owned_mesh;
    mesh.set_layout(o.plain_faces ? o.layout : FV::fast_layout(o.moving ? hydro::Motion::moving : hydro::Motion::static_mesh, o.layout));
    {
        if (!o.input_order) points = morton_order(points, box);
        const Positions x = to_device(points);
        mesh.update(x);
        mesh.update_cells();
        exec.fence();
    }
    std::cout << "mesh: n=" << n << " kind=" << o.kind << (o.input_order ? " order=input" : " order=morton") << " faces=" << mesh.faces()
              << " faces_per_cell=" << double(mesh.faces()) / n << " build_s=" << timer.seconds() << "\n";
    DeviceMemory::phase("mesh", n);

    const auto positions = to_host(mesh.positions());
    const auto volume = to_host(mesh.volume());
    // LUNA's 8 central cells: 8 mean cell volumes.
    const double target = 8 * box.volume() / static_cast<double>(n);
    const double centre[3] = {0.5 * box.side[0], 0.5 * box.side[1], 0.5 * box.side[2]};
    auto radius = [&](long long i) {
        double r2 = 0;
        for (int a = 0; a < 3; ++a) {
            const double d = tet_mesh::wrap(positions(i, a) - centre[a], box.side[a], box.inverse[a]);
            r2 += d * d;
        }
        return std::sqrt(r2);
    };
    std::vector<int> by_distance(n);
    std::iota(by_distance.begin(), by_distance.end(), 0);
    std::vector<double> r(n);
    for (long long i = 0; i < n; ++i) r[i] = radius(i);
    std::partial_sort(by_distance.begin(), by_distance.begin() + std::min<long long>(n, 1000), by_distance.end(),
                      [&](int a, int b) { return r[a] < r[b] || (r[a] == r[b] && a < b); });
    int deposit = 0;
    double deposited = 0;
    for (int k = 0; k < std::min<long long>(n, 1000); ++k) {
        const double next = deposited + volume(by_distance[k]);
        if (std::abs(next - target) >= std::abs(deposited - target)) break;
        deposited = next;
        ++deposit;
    }
    if (!o.blast) deposit = 0;
    const double p_inject = (gas.gamma - 1.0) * sedov.energy() / deposited;
    std::cout << "sedov: rho=" << sedov.rho() << " P=" << sedov.pressure() << " E=" << sedov.energy() << " t_end=" << sedov.t_end()
              << " deposit_cells=" << deposit << " deposit_volume=" << deposited << " luna_volume=" << target
              << " deposit_radius_max=" << (deposit > 0 ? r[by_distance[deposit - 1]] : 0.0) << " p_inject=" << p_inject << "\n";

    FV fv(exec, mesh, gas, o.order, o.update, o.limiter, o.moving ? hydro::Motion::moving : hydro::Motion::static_mesh,
          !o.plain_faces);
    {  // Q = U V: the background moving with the bulk velocity, the blast's pressure on top
        auto Q = Kokkos::create_mirror_view(fv.state());
        for (long long i = 0; i < n; ++i) {
            const hydro::State<3> W{sedov.rho(), o.bulk, 0, 0, o.cold * sedov.pressure()};
            const hydro::State<3> u = hydro::conserved(W, gas);
            for (int v = 0; v < 5; ++v) Q(i, v) = u[v] * volume(i);
        }
        for (int k = 0; k < deposit; ++k) Q(by_distance[k], 4) += p_inject / (gas.gamma - 1.0) * volume(by_distance[k]);
        Kokkos::deep_copy(fv.state(), Q);
    }
    DeviceMemory::phase("hydro_setup", n);
    // A static run lets go of the mesh: the hydro holds the arrays it reads (--keep-mesh: keep the whole mesh, as before).
    if (!o.moving && !o.keep_mesh) {
        fv.release_mesh();
        owned_mesh.reset();
        DeviceMemory::phase("released", n);
    }

    hydro::SnapshotData snap;
    for (int a = 0; a < 3; ++a) snap.sides[a] = box.side[a];
    snap.order = o.order;
    snap.gas = gas;
    snap.units = sedov.units;
    // Writes snapshot `index` at time t (copies state to the host, prints the disturbed radius: the farthest cell whose rho or P
    // differs from the background by > 1e-6; --no-blast: the largest deviation) and calls write_snapshot if --out is set.
    auto write = [&](int index, double t) {
        const auto Q = to_host(fv.state());
        const auto x = to_host(fv.geometry().positions);
        const auto V = to_host(fv.geometry().volume);
        snap.index = index;
        snap.time = t;
        for (auto* c : {&snap.x, &snap.y, &snap.z, &snap.volume}) c->resize(n);
        for (int v = 0; v < 5; ++v) snap.U[v].resize(n);
        const double cx = centre[0] + o.bulk * t;
        double disturbed = 0, drho = 0, dP = 0, dv = 0;
        const double speed = std::max(std::abs(o.bulk), 1e-300);
        for (long long i = 0; i < n; ++i) {
            snap.x[i] = x(i, 0);
            snap.y[i] = x(i, 1);
            snap.z[i] = x(i, 2);
            snap.volume[i] = V(i);
            for (int v = 0; v < 5; ++v) snap.U[v][i] = Q(i, v) / V(i);
            const hydro::State<3> W =
                hydro::primitive(hydro::State<3>{snap.U[0][i], snap.U[1][i], snap.U[2][i], snap.U[3][i], snap.U[4][i]}, gas);
            drho = std::max(drho, std::abs(W[0] / sedov.rho() - 1));
            dP = std::max(dP, std::abs(W[4] / sedov.pressure() - 1));
            dv = std::max({dv, std::abs(W[1] - o.bulk) / speed, std::abs(W[2]) / speed, std::abs(W[3]) / speed});
            if (std::abs(W[0] / sedov.rho() - 1) <= 1e-6 && std::abs(W[4] / sedov.pressure() - 1) <= 1e-6) continue;
            double r2 = 0;
            for (int a = 0; a < 3; ++a) {
                const double d = tet_mesh::wrap(x(i, a) - (a == 0 ? cx : centre[a]), box.side[a], box.inverse[a]);
                r2 += d * d;
            }
            disturbed = std::max(disturbed, std::sqrt(r2));
        }
        std::cout << "snapshot: index=" << index << " time=" << t << " disturbed_radius=" << disturbed
                  << " shortest_half_box=" << 0.5 * box.shortest();
        if (!o.blast) std::cout << " uniform_deviation: rho=" << drho << " P=" << dP << " v=" << dv;
        if (!o.out.empty()) std::cout << " file=" << hydro::write_snapshot(o.out, snap);
        if (o.gradients && index == o.snapshots) {  // the run is over: the gradients of the end state
            fv.time_step(courant);
            const auto gradient = fv.density_gradients();
            const auto c = to_host(fv.geometry().centroid);
            std::ofstream rows(o.out + "/gradients.bin", std::ios::binary);
            for (long long i = 0; i < n; ++i) {
                const double row[6] = {double(c(i, 0)), double(c(i, 1)), double(c(i, 2)), gradient(i, 0), gradient(i, 1), gradient(i, 2)};
                rows.write(reinterpret_cast<const char*>(row), sizeof(row));
            }
        }
        std::cout << "\n";
    };

    const Totals before = totals(fv);
    if (!o.out.empty()) write(0, 0.0);
    std::vector<double> prim_ms, sweep_ms, flux_ms, update_ms, drift_ms, mesh_ms, cells_ms, wall_ms;
    double t = 0, gpu_ms = 0;
    long long steps = 0, fallbacks = 0, most_fallbacks = 0, steps_with_fallbacks = 0;
    int next = 1;
    Kokkos::Timer wall;
    while (t < sedov.t_end() && (o.max_steps == 0 || steps < o.max_steps)) {
        // --phases: each phase fenced and timed (Timings); otherwise one wall time a step, ended by the fence of fallbacks().
        Kokkos::Timer step_timer;
        hydro::Timings timings;
        hydro::Timings* phases = o.phases ? &timings : nullptr;
        double dt = fv.time_step(courant, phases);
        const double t_next = sedov.t_end() * next / o.snapshots;
        bool snapshot = false;
        if (t + dt >= t_next) {
            dt = t_next - t;
            snapshot = true;
        }
        fv.step(dt, phases);
        const long long step_fallbacks = fv.fallbacks();  // fences: the step is done
        wall_ms.push_back(step_timer.seconds() * 1e3);
        prim_ms.push_back(timings.primitives);
        sweep_ms.push_back(timings.sweep);
        flux_ms.push_back(timings.fluxes);
        update_ms.push_back(timings.update);
        drift_ms.push_back(timings.drift);
        mesh_ms.push_back(timings.mesh);
        cells_ms.push_back(timings.cells);
        gpu_ms += o.phases ? timings.primitives + timings.sweep + timings.fluxes + timings.update + timings.drift + timings.mesh + timings.cells
                           : wall_ms.back();
        fallbacks += step_fallbacks;
        most_fallbacks = std::max(most_fallbacks, step_fallbacks);
        steps_with_fallbacks += step_fallbacks > 0;
        t = snapshot ? t_next : t + dt;
        ++steps;
        if (snapshot) {
            if (!o.out.empty() || next == o.snapshots) write(next, t);
            ++next;
        }
    }
    exec.fence();
    const double wall_s = wall.seconds();
    if (t < sedov.t_end()) write(next, t);
    DeviceMemory::phase("steps", n);
    const Totals after = totals(fv);
    const double hydro_ms = median(prim_ms) + median(sweep_ms) + median(flux_ms) + median(update_ms);
    const double step_ms = hydro_ms + median(drift_ms) + median(mesh_ms) + median(cells_ms);
    std::cout << "run: precision=" << o.precision
              << " order=" << o.order << " update=" << (o.update == hydro::Update::per_cell ? "cell" : "face")
              << " steps=" << steps << " limiter=" << o.limit << " riemann=" << o.riemann << " mesh=" << (o.moving ? "moving" : "static")
              << " bulk=" << o.bulk << " blast=" << o.blast << " t=" << t << " wall_s=" << wall_s << " s_per_step=" << wall_s / steps
              << " gpu_s=" << gpu_ms * 1e-3 << " step_wall_ms_median=" << median(wall_ms)
              << " G_cell_updates_per_s=" << n / (median(wall_ms) * 1e-3) / 1e9 << " phases=" << o.phases << "\n";
    std::cout << "fallbacks: total=" << fallbacks << " steps_with=" << steps_with_fallbacks << " most_in_a_step=" << most_fallbacks
              << " face_sides_per_step=" << 2 * static_cast<long long>(fv.faces()) << "\n";
    const hydro::MeshStats& ms = fv.mesh_stats();
    std::cout << "mesh_updates: updates=" << ms.updates << " rebuilds=" << ms.rebuilds << " resorts=" << ms.resorts << " stuck=" << ms.stuck
              << " stuck_steps=" << ms.stuck_steps << " local_repairs=" << ms.local_repairs << " local_cells=" << ms.local_cells
              << " host_flips=" << ms.host_flips << " device_flips=" << ms.device_flips << "\n";
    auto rate = [&](double ms) { return ms > 0 ? n / (ms * 1e-3) / 1e6 : 0.0; };
    if (o.phases) std::cout << "phases_ms_median: primitives=" << median(prim_ms) << " sweep=" << median(sweep_ms) << " fluxes=" << median(flux_ms)
              << " update=" << median(update_ms) << " drift=" << median(drift_ms) << " mesh=" << median(mesh_ms)
              << " cells=" << median(cells_ms) << " hydro=" << hydro_ms << " step=" << step_ms
              << " cell_updates_per_s=" << n / (step_ms * 1e-3) << "\n";
    if (o.phases) std::cout << "throughput_M_per_s: primitives=" << rate(median(prim_ms)) << " sweep=" << rate(median(sweep_ms))
              << " fluxes=" << rate(median(flux_ms)) << " update=" << rate(median(update_ms)) << " mesh=" << rate(median(mesh_ms))
              << " cells=" << rate(median(cells_ms)) << " hydro=" << rate(hydro_ms) << " step=" << rate(step_ms)
              << " hydro_G_face_updates_per_s=" << fv.faces() / (hydro_ms * 1e-3) / 1e9 << "\n";
    std::cout << "conservation: mass=" << (after.mass - before.mass) / before.mass
              << " energy=" << (after.energy - before.energy) / before.energy << " momentum=(" << after.px - before.px << ","
              << after.py - before.py << "," << after.pz - before.pz << ") momentum_scale=" << before.mass * std::sqrt(2 * before.energy / before.mass)
              << "\n";
    const auto Qh = to_host(fv.state());
    const auto xh = to_host(fv.geometry().positions);
    const auto Vh = to_host(Kokkos::View<const double* [1], Kokkos::LayoutLeft, Memory>(fv.geometry().volume.data(), n));
    std::cout << "checksum: steps=" << steps << " Q=" << std::hex << bits_hash(Qh) << " x=" << bits_hash(xh) << " volume=" << bits_hash(Vh)
              << std::dec << "\n";
}

// Writes the grid's state (5, n, n, n) as a float64 .npy file. Used by: run_grid (--out=file.npy).
void write_state_npy(const std::string& path, const std::vector<double>& U, int n) {
    char header[128];
    int length = std::snprintf(header, sizeof header, "{'descr': '<f8', 'fortran_order': False, 'shape': (5, %d, %d, %d), }", n, n, n);
    while ((10 + length + 1) % 64 != 0) header[length++] = ' ';
    header[length++] = '\n';
    std::ofstream file(path, std::ios::binary);
    file.write("\x93NUMPY\x01\x00", 8);
    const unsigned short size = static_cast<unsigned short>(length);
    file.write(reinterpret_cast<const char*>(&size), 2);
    file.write(header, length);
    file.write(reinterpret_cast<const char*>(U.data()), static_cast<std::streamsize>(U.size() * sizeof(double)));
}

// --grid: LUNA's fixed-grid Sedov (sedov3d_run.py: units on, 1e51 erg in the central 2 x 2 x 2 cells, BOUNDARY 1, CFL 0.8, MUSCL-Hancock,
// --limiter=minmod|mc|vanleer, minmod by default as LUNA's run) with hydro::grid::GridFiniteVolume (research_notes/fixed_grid_2026-11-19.md).
// --kernel=kokkos|cuda, --tile=x,y,z, --cfl. Each step's wall time includes the time step (one fence a step). Used by: main.
template <class R>
void run_grid(const Options& o, int argc, char** argv) {
    namespace g = hydro::grid;
    const Sedov sedov;
    const int n = o.side;
    const Exec exec;
    g::Options go;
    const std::string limiter = flag(argc, argv, "--limiter", "minmod");
    go.limiter = limiter == "mc" ? g::Limiter::mc : limiter == "vanleer" ? g::Limiter::van_leer : g::Limiter::minmod;
    go.boundary = g::Boundary::outflow;
    go.cfl = std::stod(flag(argc, argv, "--cfl", "0.8"));
    const std::string kernel = flag(argc, argv, "--kernel", "cuda");
    go.kernel = kernel == "kokkos" ? g::Kernel::kokkos : g::Kernel::cuda;
    if (const std::string t = flag(argc, argv, "--tile", ""); !t.empty()) std::sscanf(t.c_str(), "%d,%d,%d", &go.tile[0], &go.tile[1], &go.tile[2]);
    Kokkos::Timer timer;
    g::GridFiniteVolume<3, R> fv(exec, {n, n, n}, {o.box.side[0], o.box.side[1], o.box.side[2]}, sedov.gamma, go);
    const g::Params& p = fv.params();
    {  // sedov3d.sedov_blast_3d with units: background at rest, p_inject = (gamma - 1) E / (8 dx dy dz) on top in the central cells
        const double V = 8 * p.dx[0] * p.dx[1] * p.dx[2], p_inject = (sedov.gamma - 1.0) * (sedov.energy() / V);
        auto U = Kokkos::create_mirror_view(fv.state());
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                for (int k = 0; k < n; ++k) {
                    const bool inject = i >= n / 2 - 1 && i <= n / 2 && j >= n / 2 - 1 && j <= n / 2 && k >= n / 2 - 1 && k <= n / 2;
                    const hydro::State<3> W{sedov.rho(), 0, 0, 0, inject ? p_inject + sedov.pressure() : sedov.pressure()};
                    const hydro::State<3> u = g::to_conserved<3>(W, p);
                    for (int v = 0; v < 5; ++v) U(v, (static_cast<long>(i) * n + j) * n + k) = u[v];
                }
        Kokkos::deep_copy(fv.state(), U);
        fv.refresh();
    }
    const double dV = p.dx[0] * p.dx[1] * p.dx[2];
    auto sums = [&] {
        const auto U = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fv.state());
        std::array<double, 5> s{};
        for (int v = 0; v < 5; ++v)
            for (long c = 0; c < fv.cells(); ++c) s[v] += U(v, c) * dV;
        return s;
    };
    const std::array<double, 5> before = sums();
    std::cout << "grid: n=" << n << "^3 kernel=" << kernel << " tile=" << fv.tile()[0] << "," << fv.tile()[1] << "," << fv.tile()[2]
              << " limiter=" << limiter << " riemann=" << o.riemann << " cfl=" << go.cfl << " t_end=" << sedov.t_end()
              << " setup_s=" << timer.seconds() << "\n";
    std::vector<double> wall_ms;
    double t = 0;
    long long steps = 0;
    Kokkos::Timer wall;
    double dt = fv.time_step();
    while (t < sedov.t_end() && (o.max_steps == 0 || steps < o.max_steps)) {
        Kokkos::Timer step_timer;
        if (t + dt > sedov.t_end()) dt = sedov.t_end() - t;
        fv.step(dt);
        t += dt;
        dt = fv.time_step();  // fences: the step is done
        wall_ms.push_back(step_timer.seconds() * 1e3);
        ++steps;
        if (!(dt > 0)) {
            std::cout << "grid: stopped at step " << steps << ", t=" << t << ": dt=" << dt << "\n";
            break;
        }
    }
    const double wall_s = wall.seconds();
    const std::array<double, 5> after = sums();
    // The median after 3 warm-up steps.
    const std::vector<double> warm(wall_ms.begin() + static_cast<long>(std::min<std::size_t>(wall_ms.size() - 1, 3)), wall_ms.end());
    const double ms = median(warm);
    const double cells = static_cast<double>(fv.cells());
    std::cout << "run: grid=1 kernel=" << kernel << " steps=" << steps << " t=" << t << " wall_s=" << wall_s << " step_wall_ms_median=" << ms
              << " G_cell_updates_per_s=" << cells / (ms * 1e-3) / 1e9 << " ns_per_cell=" << ms * 1e6 / cells << "\n";
    std::cout << "conservation: mass=" << (after[0] - before[0]) / before[0] << " energy=" << (after[4] - before[4]) / before[4]
              << " momentum=(" << after[1] - before[1] << "," << after[2] - before[2] << "," << after[3] - before[3] << ")\n";
    const auto Uh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fv.state());
    const std::vector<double> U(Uh.data(), Uh.data() + Uh.size());
    std::uint64_t h = 1469598103934665603ULL;
    for (double x : U) {
        std::uint64_t u = 0;
        std::memcpy(&u, &x, sizeof(u));
        for (int b = 0; b < 8; ++b) {
            h ^= (u >> (8 * b)) & 0xff;
            h *= 1099511628211ULL;
        }
    }
    std::cout << "checksum: steps=" << steps << " U=" << std::hex << h << std::dec << "\n";
    if (!o.out.empty()) {
        write_state_npy(o.out, U, n);
        std::cout << "grid: wrote " << o.out << "\n";
    }
}

// The precision policies of --precision, as <name>: geometry / gradient weights / gradients / face fluxes / Riemann.
// default: the hydro's default; double: double throughout, what older logs call `default`. R: the Riemann solver from --riemann.
// Used by: main (picks R); calls run<P, R>.
template <class R>
void run_precision(const Options& o, const Points& points) {
    using hydro::Half;
    using hydro::Precision;
    const std::string& p = o.precision;
    if (p == "default") return run<hydro::DefaultPrecision<3>, R>(o, points);
    if (p == "double") return run<Precision<>, R>(o, points);
    if constexpr (std::is_same_v<R, hydro::HLLC>) {
        if (p == "grad32") return run<hydro::Grad32, R>(o, points);
        if (p == "grad16") return run<Precision<double, Half>, R>(o, points);
        if (p == "riemann32") return run<Precision<double, float, double, float>, R>(o, points);
        if (p == "single") return run<Precision<double, float, float, float>, R>(o, points);
        if (p == "mixed") return run<Precision<float, float, float>, R>(o, points);
        if (p == "low") return run<Precision<float, Half, float, float>, R>(o, points);
    }
    throw std::invalid_argument("hydro_sedov: unknown --precision " + p + " (HLL: default or double only)");
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    luna::pin_notice();
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]).rfind("--", 0) != 0) args.push_back(argv[i]);
    if (args.size() < 2) {
        std::cerr << "usage: hydro_sedov <side> <jittered|uniform> [--order=0|1] [--update=cell|face] [--limiter=face|none]"
                     " [--riemann=hllc|hll] [--precision=default|double|grad32|grad16|riemann32|single|mixed|low]"
                     " [--moving] [--bulk=km/s] [--no-blast] [--box=fx,fy,fz] [--out=dir]"
                     " [--snapshots=10] [--steps=max] [--input-order] [--memory] [--gradients] [--cold=f]"
                     " [--curve=morton|hilbert|moore|gray|row|rcm] [--every=k] [--far=share] [--owner-faces] [--owner-bit] [--plain-faces] [--phases] [--keep-mesh]\n"
                     "       hydro_sedov <side> grid --grid [--kernel=cuda|kokkos] [--limiter=minmod|mc|vanleer] [--riemann=hllc|hll] [--cfl=0.8]"
                     " [--tile=x,y,z] [--steps=max] [--out=file.npy]\n";
        return 2;
    }
    if (flag(argc, argv, "--memory", "") == "1") DeviceMemory::start();
    Options o;
    o.side = std::stoi(args[0]);
    o.kind = args[1];
    o.order = std::stoi(flag(argc, argv, "--order", "1"));
    // Default loop: the faster one measured on the A100.
    o.loop = flag(argc, argv, "--update", o.order == 1 ? "face" : "cell");
    o.update = o.loop == "face" ? hydro::Update::per_face : hydro::Update::per_cell;
    o.limit = flag(argc, argv, "--limiter", "face");
    o.limiter = o.limit == "none" ? hydro::Limiter::none : hydro::Limiter::face_centroid;
    o.riemann = flag(argc, argv, "--riemann", "hllc");
    o.precision = flag(argc, argv, "--precision", "default");
    o.moving = flag(argc, argv, "--moving", "") == "1";
    o.blast = flag(argc, argv, "--no-blast", "") != "1";
    o.cold = std::stod(flag(argc, argv, "--cold", "1"));
    o.bulk = std::stod(flag(argc, argv, "--bulk", "0"));  // km/s along x
    o.out = flag(argc, argv, "--out", "");
    o.snapshots = std::stoi(flag(argc, argv, "--snapshots", "10"));
    o.gradients = flag(argc, argv, "--gradients", "") == "1";
    o.max_steps = std::stoll(flag(argc, argv, "--steps", "0"));
    o.input_order = flag(argc, argv, "--input-order", "") == "1";
    const std::map<std::string, voronoi::Curve> curves = {{"morton", voronoi::Curve::morton}, {"hilbert", voronoi::Curve::hilbert},
                                                          {"moore", voronoi::Curve::moore},   {"gray", voronoi::Curve::gray},
                                                          {"row", voronoi::Curve::row_major}, {"rcm", voronoi::Curve::rcm}};
    if (const std::string curve = flag(argc, argv, "--curve", ""); !curve.empty()) o.layout.curve = curves.at(curve);
    // Re-sort every k updates; 0 (off) unless asked, explicitly: voronoi-cpp's own default is 25 (FiniteVolume::no_resort).
    o.layout.every = std::stoi(flag(argc, argv, "--every", "0"));
    o.layout.far = std::stod(flag(argc, argv, "--far", "0"));
    o.layout.owner_faces = flag(argc, argv, "--owner-faces", "") == "1";
    o.layout.owner_bit = flag(argc, argv, "--owner-bit", "") == "1";
    o.plain_faces = flag(argc, argv, "--plain-faces", "") == "1";
    o.phases = flag(argc, argv, "--phases", "") == "1";
    o.keep_mesh = flag(argc, argv, "--keep-mesh", "") == "1";
    // The periodic box: --box=fx,fy,fz times 50 pc (a cube by default); the lattice keeps about equal spacings per axis.
    const Sedov sedov;
    const std::string shape_arg = flag(argc, argv, "--box", "1,1,1");
    double f[3] = {1, 1, 1};
    std::sscanf(shape_arg.c_str(), "%lf,%lf,%lf", &f[0], &f[1], &f[2]);
    o.box = Box(sedov.box * f[0], sedov.box * f[1], sedov.box * f[2]);
    if (flag(argc, argv, "--grid", "") == "1") {  // the fixed grid: no mesh, no points (`kind` is ignored)
        if (o.riemann == "hll")
            run_grid<hydro::HLL>(o, argc, argv);
        else
            run_grid<hydro::HLLC>(o, argc, argv);
        return 0;
    }
    const Points points = scaled(make_points_off_grid(o.kind, o.side * o.side * o.side, shape(o.box)), o.box);
    if (o.riemann == "hll")
        run_precision<hydro::HLL>(o, points);
    else
        run_precision<hydro::HLLC>(o, points);
    return 0;
}
