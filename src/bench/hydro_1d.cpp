// LUNA's 1D tests, timed. Usage: hydro_1d <problem of problems_1d.hpp> <N> [flags] (docs/design.md).
// Drives: main -> problem() -> simulate: Mesh::update -> FiniteVolume<1> (set state) -> loop time_step / step[_and_time_step] -> --out npy.

#include "pin_notice.hpp"
#include "finite_volume_1d.hpp"
#include "problems_1d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace {

using Exec = Kokkos::DefaultExecutionSpace;
using Memory = Exec::memory_space;
using Mesh = voronoi::Mesh<1, Exec>;
using State = hydro::State<1>;
using hydro::problems_1d::Problem;
using hydro::problems_1d::problem;

// Peak of the device allocations, through the Kokkos Tools callbacks. Used by: main (start, with --memory; peak in the report).
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

// The value of `--name=value` among argv[3..], else `fallback`. Used by: main.
std::string flag(int argc, char** argv, const char* name, const char* fallback) {
    const std::size_t length = std::strlen(name);
    for (int a = 3; a < argc; ++a)
        if (std::strncmp(argv[a], name, length) == 0 && argv[a][length] == '=') return argv[a] + length + 1;
    return fallback;
}
// Whether the bare flag `name` is among argv[3..]. Used by: main.
bool has(int argc, char** argv, const char* name) {
    for (int a = 3; a < argc; ++a)
        if (std::strcmp(argv[a], name) == 0) return true;
    return false;
}

// x in %.4e, for the .json. Used by: main.
std::string sci(double x) {
    char b[32];
    std::snprintf(b, sizeof b, "%.4e", x);
    return b;
}

// Parsed flags. Filled by: main; Used by: simulate.
struct Options {
    int n = 0, order = 1;
    double jitter = 0, courant = 0.8, fixed_dt = 0, inflow_spacing = 1, max_wall = 0;
    // Refinement and derefinement of a moving line (hydro::Adapt). --amr=mass (default) | length | off. Mass: --refine-mass=2
    // --derefine-mass=0.5 times each cell's target mass, within --split-min / --merge-max (h), and --derefine=f (default 0.03 h) as the
    // safeguard; --refine=F (F L / N) also splits wider cells.
    // Given without --amr, --derefine / --refine choose length. --split=linear (default) | constant.
    hydro::Adapt amr = hydro::Adapt::mass;
    double derefine = -1, refine = -1;  // -1: not given (mass keeps its safeguard, length has none)
    double split_min = hydro::split_min_default, merge_max = hydro::merge_max_default;  // --split-min=, --merge-max= (h); mass only
    double refine_mass = hydro::refine_mass_default, derefine_mass = hydro::derefine_mass_default;
    bool refine_linear = true;
    hydro::Limiter limiter = hydro::Limiter::face_centroid;
    hydro::Motion motion = hydro::Motion::static_mesh;
    bool fixed = false, phases = false, regularise = true, fused = true, smallest = false;
    hydro::TimeStep rule = hydro::TimeStep::luna;
    long long max_steps = 0;
    std::vector<double> times;  // --times=t1,t2,...: snapshots, increasing
};

// What one simulate() run reports (steps, sorts, final cell count, wall time, state hash). Used by: main.
struct Result {
    long long steps = 0, sorts = 0, cell_updates = 0;
    int cells = 0;
    long long removed = 0, refined = 0;  // cells removed by --derefine, added by --refine
    bool reached = false;
    double wall_s = 0, t = 0, smallest_h = 0;  // smallest_h: with --smallest, in units of L / N
    std::uint64_t hash = 0;
    hydro::Timings timings;
};

// FNV-1a over the bits of the state (and the positions of a moving run): `h` accumulates, host(i, c) for rows x columns.
// Used by: simulate (the final hash, compared across --repeats in main).
template <class V>
void hash_into(std::uint64_t& h, const V& host, int rows, int columns) {
    for (int i = 0; i < rows; ++i)
        for (int c = 0; c < columns; ++c) {
            std::uint64_t bits;
            const double value = host(i, c);
            std::memcpy(&bits, &value, sizeof bits);
            h = (h ^ bits) * 1099511628211ull;
        }
}

// Writes rows of (x, x + centroid, volume, rho, u, p) as a .npy file. Used by: simulate (--out).
void write_npy(const std::string& path, const std::vector<std::array<double, 6>>& rows) {
    char header[128];
    int length = std::snprintf(header, sizeof header, "{'descr': '<f8', 'fortran_order': False, 'shape': (%zu, 6), }", rows.size());
    while ((10 + length + 1) % 64 != 0) header[length++] = ' ';
    header[length++] = '\n';
    std::ofstream file(path, std::ios::binary);
    file.write("\x93NUMPY\x01\x00", 8);
    const unsigned short size = static_cast<unsigned short>(length);
    file.write(reinterpret_cast<const char*>(&size), 2);
    file.write(header, length);
    file.write(reinterpret_cast<const char*>(rows.data()), static_cast<std::streamsize>(rows.size() * sizeof rows[0]));
}

// One run: pb is the problem from problem(), o the flags, `out` the .npy path (empty: none). Riemann: HLLC or HLL.
// Used by: main (once per --repeats).
template <class Riemann>
Result simulate(const Problem& pb, const Options& o, const std::string& out) {
    const Exec exec;
    const int n = o.n;
    Mesh mesh(exec, pb.L, pb.ends);
    {
        Kokkos::View<double* [1], Memory> x("x", n);
        auto h = Kokkos::create_mirror_view(x);
        std::mt19937_64 rng(1);
        std::uniform_real_distribution<double> unit(-0.5, 0.5);
        for (int i = 0; i < n; ++i) h(i, 0) = pb.L * (i + 0.5 + o.jitter * unit(rng)) / n;
        Kokkos::deep_copy(x, h);
        mesh.update(x);
    }
    hydro::FiniteVolume<1, Exec, Riemann> fv(exec, mesh, hydro::Gas{pb.gamma}, o.order, o.limiter, o.motion,
                                             o.motion == hydro::Motion::moving ? pb.moving_walls : pb.walls, o.regularise);
    fv.set_inflow_spacing(o.inflow_spacing * pb.L / n);
    fv.set_adapt(o.amr);
    fv.set_adapt_mass(o.refine_mass, o.derefine_mass);
    fv.set_mass_bounds(o.split_min * pb.L / n, o.merge_max * pb.L / n, hydro::derefine_length_default * pb.L / n);
    if (o.derefine >= 0) fv.set_derefine(o.derefine * pb.L / n);
    if (o.refine >= 0) fv.set_refine(o.refine * pb.L / n);
    fv.set_split(o.refine_linear);
    {
        const auto x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.positions());
        const auto c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.centroid());
        const auto V = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.volume());
        auto q = Kokkos::create_mirror_view(fv.state());
        for (int i = 0; i < n; ++i) {
            const double a = x(i, 0) + c(i, 0) - 0.5 * V(i), b = a + V(i);
            const State u = pb.average(a + pb.origin, b + pb.origin);
            for (int v = 0; v < 3; ++v) q(i, v) = u[v] * V(i);
        }
        Kokkos::deep_copy(fv.state(), q);
    }
    exec.fence();
    Result r;
    hydro::Timings* timings = o.phases ? &r.timings : nullptr;
    Kokkos::Timer timer;
    double t = 0;
    const bool fused = o.fused && !o.fixed && !o.phases && o.rule == hydro::TimeStep::luna;
    double next = fused ? fv.time_step(o.courant, o.rule) : 0;
    double smallest = 1e300;
    // --times: at each of these times (the step shortened to land on it) the generators and conserved totals, rows of (x, Q0, Q1, Q2)
    // in <out stem>_t<time>.npy, and a line "t steps wall_s cells" in <out stem>_times.txt (begun anew by each run). Reads only: the run is the same without it.
    std::size_t snap = 0;
    const auto snapshot = [&](double at) {
        const auto q = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, fv.state());
        std::vector<double> x(static_cast<std::size_t>(fv.cells()));
        if (o.motion == hydro::Motion::moving) {
            const auto p = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, fv.positions());
            for (int i = 0; i < fv.cells(); ++i) x[i] = p(i, 0);
        } else {
            const auto p = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.positions());
            for (int i = 0; i < fv.cells(); ++i) x[i] = p(i, 0);
        }
        std::vector<std::array<double, 6>> rows(static_cast<std::size_t>(fv.cells()));
        for (int i = 0; i < fv.cells(); ++i) rows[i] = std::array<double, 6>{x[i] + pb.origin, q(i, 0), q(i, 1), q(i, 2), 0.0, 0.0};
        std::sort(rows.begin(), rows.end());
        const std::string stem = out.substr(0, out.rfind(".npy"));
        char name[64];
        std::snprintf(name, sizeof name, "_t%g.npy", o.times[snap]);
        write_npy(stem + name, rows);
        std::ofstream(stem + "_times.txt", snap == 0 ? std::ios::trunc : std::ios::app) << at << " " << r.steps << " " << timer.seconds() << " " << fv.cells() << "\n";
    };
    while (t < pb.t_end * (1 - 1e-12) && (o.max_steps == 0 || r.steps < o.max_steps)) {
        double dt = fused ? next : o.fixed ? o.fixed_dt : fv.time_step(o.courant, o.rule, timings);
        if (!(dt > 1e-15 * pb.t_end) || (o.max_wall > 0 && timer.seconds() > o.max_wall)) break;
        dt = std::min(dt, pb.t_end - t);
        if (snap < o.times.size()) dt = std::min(dt, o.times[snap] - t);
        r.cell_updates += fv.cells();
        if (fused)
            next = fv.step_and_time_step(dt, o.courant);
        else
            fv.step(dt, timings);
        t += dt;
        ++r.steps;
        while (snap < o.times.size() && t >= o.times[snap] * (1 - 1e-12) && !out.empty()) snapshot(t), ++snap;
        if (o.smallest && o.motion == hydro::Motion::moving) {  // --smallest: the smallest inner cell over all steps, as hydro1d_cases
            const auto p = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, fv.positions());
            std::vector<double> at(p.data(), p.data() + fv.cells());
            std::sort(at.begin(), at.end());
            for (std::size_t i = 1; i + 1 < at.size(); ++i) smallest = std::min(smallest, 0.5 * (at[i + 1] - at[i - 1]));
        }
    }
    r.smallest_h = smallest * n / pb.L;
    exec.fence();
    r.wall_s = timer.seconds();
    r.t = t;
    r.reached = t >= pb.t_end * (1 - 1e-12);
    r.sorts = mesh.sorts();
    r.cells = fv.cells();
    r.removed = fv.cells_removed();
    r.refined = fv.cells_refined();
    const auto q = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, fv.state());
    r.hash = 14695981039346656037ull;
    hash_into(r.hash, q, r.cells, 3);
    if (o.motion == hydro::Motion::moving)
        hash_into(r.hash, Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, fv.positions()), r.cells, 1);
    if (!out.empty()) {
        mesh.update_cells();
        const auto x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.positions());
        const auto c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.centroid());
        const auto V = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.volume());
        std::vector<std::array<double, 6>> rows;
        rows.reserve(static_cast<std::size_t>(r.cells));
        for (int i = 0; i < r.cells; ++i) {
            const State W = hydro::primitive(State{q(i, 0) / V(i), q(i, 1) / V(i), q(i, 2) / V(i)}, fv.gas());
            rows.push_back({x(i, 0) + pb.origin, x(i, 0) + c(i, 0) + pb.origin, V(i), W[0], W[1], W[2]});
        }
        std::sort(rows.begin(), rows.end());
        write_npy(out, rows);
    }
    return r;
}

}  // namespace

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    luna::pin_notice();
    if (argc < 3) {
        std::cerr << "usage: hydro_1d <problem> <N> [--moving] [--riemann=hllc|hll] [--order=0|1]"
                     " [--dt=luna|arepo|fixed] [--courant=c] [--fixed-dt=dt] [--jitter=s] [--limiter=face|none|minmod|vanleer]"
                     " [--amp=a] [--pad=length] [--steps=max] [--max-wall=s] [--repeats=r] [--phases] [--memory] [--no-regularise] [--unfused] [--smallest]"
                     " [--inflow-spacing=s] [--t-end=t] [--out=file.npy] [--series=name] [--times=t1,t2,...]"
                     " [--amr=mass|length|off] [--refine-mass=2] [--derefine-mass=0.5] [--split-min=0.12] [--merge-max=2] [--derefine=f]"
                     " [--refine=F] [--split=linear|constant]\n"
                     "  refinement of a moving line: --amr=mass (default) splits cells above refine-mass times their initial mass if wider than"
                     " split-min h,\n  removes cells below derefine-mass times it if narrower than merge-max h, and any cell narrower than"
                     " --derefine=f h (default 0.03);\n  --derefine / --refine without --amr: --amr=length (cells narrower than f h go, wider"
                     " than F h split); --amr=off: neither\n";
        return 1;
    }
    const std::string name = argv[1];
    Options o;
    o.n = std::stoi(argv[2]);
    const bool moving = has(argc, argv, "--moving");
    o.motion = moving ? hydro::Motion::moving : hydro::Motion::static_mesh;
    const std::string riemann = flag(argc, argv, "--riemann", "hllc"), rule_name = flag(argc, argv, "--dt", moving ? "arepo" : "luna");
    const std::string limit = flag(argc, argv, "--limiter", "face");
    o.order = std::stoi(flag(argc, argv, "--order", "1"));
    o.rule = rule_name == "arepo" ? hydro::TimeStep::arepo : hydro::TimeStep::luna;
    o.fixed = rule_name == "fixed";
    o.fixed_dt = std::stod(flag(argc, argv, "--fixed-dt", "0"));
    if (o.fixed && !(o.fixed_dt > 0)) throw std::invalid_argument("hydro_1d: --dt=fixed needs --fixed-dt > 0");
    const std::string courant_default = std::to_string(o.rule == hydro::TimeStep::arepo ? hydro::courant_moving_1d : hydro::courant_static_1d);
    o.courant = std::stod(flag(argc, argv, "--courant", courant_default.c_str()));
    o.jitter = std::stod(flag(argc, argv, "--jitter", "0"));
    o.inflow_spacing = std::stod(flag(argc, argv, "--inflow-spacing", "1"));
    o.max_steps = std::stoll(flag(argc, argv, "--steps", "0"));
    o.max_wall = std::stod(flag(argc, argv, "--max-wall", "0"));
    o.phases = has(argc, argv, "--phases");
    o.regularise = !has(argc, argv, "--no-regularise");
    o.fused = !has(argc, argv, "--unfused");
    o.smallest = has(argc, argv, "--smallest");
    o.derefine = std::stod(flag(argc, argv, "--derefine", "-1"));
    o.refine = std::stod(flag(argc, argv, "--refine", "-1"));
    o.split_min = std::stod(flag(argc, argv, "--split-min", std::to_string(hydro::split_min_default).c_str()));
    o.merge_max = std::stod(flag(argc, argv, "--merge-max", std::to_string(hydro::merge_max_default).c_str()));
    o.refine_mass = std::stod(flag(argc, argv, "--refine-mass", std::to_string(hydro::refine_mass_default).c_str()));
    o.derefine_mass = std::stod(flag(argc, argv, "--derefine-mass", std::to_string(hydro::derefine_mass_default).c_str()));
    o.refine_linear = std::string(flag(argc, argv, "--split", "linear")) == "linear";
    {
        // An explicitly given criterion replaces the default: lengths alone mean --amr=length.
        const std::string amr = flag(argc, argv, "--amr", (o.derefine >= 0 || o.refine >= 0) ? "length" : "mass");
        if (amr != "mass" && amr != "length" && amr != "off") throw std::invalid_argument("hydro_1d: --amr=mass|length|off");
        o.amr = amr == "mass" ? hydro::Adapt::mass : amr == "length" ? hydro::Adapt::length : hydro::Adapt::off;
    }
    for (std::string list = flag(argc, argv, "--times", ""); !list.empty();) {
        const std::size_t comma = list.find(',');
        o.times.push_back(std::stod(list.substr(0, comma)));
        list = comma == std::string::npos ? "" : list.substr(comma + 1);
    }
    const int repeats = std::stoi(flag(argc, argv, "--repeats", "2"));
    const std::string out = flag(argc, argv, "--out", "");
    o.limiter = limit == "none"    ? hydro::Limiter::none
                : limit == "minmod"  ? hydro::Limiter::minmod
                : limit == "vanleer" ? hydro::Limiter::van_leer
                                     : hydro::Limiter::face_centroid;
    if (has(argc, argv, "--memory")) DeviceMemory::start();
    Problem pb = problem(name, std::stod(flag(argc, argv, "--amp", "0")));
    if (const std::string t_end = flag(argc, argv, "--t-end", ""); !t_end.empty()) pb.t_end = std::stod(t_end);
    const int box_cells = o.n, pad_cells = static_cast<int>(std::ceil(std::stod(flag(argc, argv, "--pad", "0")) * o.n / pb.L));
    pb.origin -= pad_cells * pb.L / o.n;
    pb.L += 2 * pad_cells * pb.L / o.n;
    o.n += 2 * pad_cells;
    if (pad_cells > 0) pb.ends = voronoi::Ends::periodic;
    Result best;
    bool same = true;
    for (int run = 0; run < repeats; ++run) {
        const std::string file = run == repeats - 1 ? out : std::string();
        const Result r = riemann == "hll" ? simulate<hydro::HLL>(pb, o, file) : simulate<hydro::HLLC>(pb, o, file);
        std::cout << "run: " << run << " wall_s=" << r.wall_s << " M_cell_updates_per_s=" << double(r.cell_updates) / r.wall_s / 1e6
                  << " hash=" << r.hash << "\n";
        same = same && (run == 0 || r.hash == best.hash);
        if (run == 0 || r.wall_s < best.wall_s) best = r;
    }
    const double rate = double(best.cell_updates) / best.wall_s;
    std::cout << "ours: problem=" << name << " N=" << o.n << " mesh=" << (moving ? "moving" : "static") << " riemann=" << riemann
              << " order=" << o.order << " dt=" << rule_name << " courant=" << o.courant << " jitter=" << o.jitter
              << " steps=" << best.steps << " t=" << best.t << " reached=" << (best.reached ? 1 : 0) << " sorts=" << best.sorts << " wall_s=" << best.wall_s
              << " final_cells=" << best.cells << " cell_updates=" << double(best.cell_updates) << " M_cell_updates_per_s=" << rate / 1e6 << " hash=" << best.hash
              << " same_bits=" << (same ? 1 : 0) << " backend=" << Exec::name();
    if (o.smallest) std::cout << " smallest_h=" << best.smallest_h;
    if (moving) {
        const char* amr = o.amr == hydro::Adapt::mass ? "mass" : o.amr == hydro::Adapt::length ? "length" : "off";
        std::cout << " amr=" << amr;
        if (o.amr == hydro::Adapt::mass)
            std::cout << " refine_mass=" << o.refine_mass << " derefine_mass=" << o.derefine_mass << " split_min=" << o.split_min
                      << " merge_max=" << o.merge_max << " derefine=" << (o.derefine >= 0 ? o.derefine : hydro::derefine_length_default);
        if (o.amr != hydro::Adapt::off) {
            if (o.amr == hydro::Adapt::length && o.derefine >= 0) std::cout << " derefine=" << o.derefine;
            if (o.refine >= 0) std::cout << " refine=" << o.refine;
            std::cout << " split=" << (o.refine_linear ? "linear" : "constant") << " removed=" << best.removed << " refined=" << best.refined;
        }
    }
    std::cout << "\n";
    if (o.phases) {
        const hydro::Timings& t = best.timings;
        auto phase = [&](const char* label, double ms) {
            if (ms > 0) std::cout << "phase: " << label << " M_cell_updates_per_s=" << double(best.cell_updates) / (ms * 1e-3) / 1e6 << "\n";
        };
        phase("time_step", t.sweep);
        phase("hydro", t.fluxes);
        phase("mesh", t.mesh);
        phase("whole", t.sweep + t.fluxes + t.mesh);
    }
    const std::string series = flag(argc, argv, "--series", "");
    if (!series.empty() && !out.empty()) {
        std::ofstream json(out + ".json");
        json << "{\"series\": \"" << series << "\", \"problem\": \"" << name << "\", \"N\": " << box_cells << ", \"steps\": " << best.steps
             << ", \"wall_s\": " << best.wall_s << ", \"peak_MB\": " << DeviceMemory::peak / 1e6 << ", \"sorts\": " << best.sorts
             << ", \"t\": " << best.t << ", \"reached\": " << (best.reached ? 1 : 0) << (o.smallest ? ", \"smallest_h\": " + sci(best.smallest_h) : std::string())
             << ", \"final_cells\": " << best.cells << ", \"cell_updates\": " << best.cell_updates << ", \"file\": \"" << out.substr(out.find_last_of('/') + 1) << "\"}\n";
    }
    if (DeviceMemory::peak > 0)
        std::cout << "memory: peak_MB=" << DeviceMemory::peak / 1e6 << " peak_bytes_per_cell=" << double(DeviceMemory::peak) / o.n << "\n";
    return 0;
}
