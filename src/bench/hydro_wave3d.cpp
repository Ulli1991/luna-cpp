// 3D problems with an exact solution; prints steps, wall seconds and cell updates/s. Usage: hydro_wave3d <sound|simple|vortex|tube> <nx> [flags].
// Drives: main -> run<P>: problem_3d -> jittered_points -> Mesh::update/update_cells -> FiniteVolume<3> (wave_initial) -> time_step, step -> --out npy.

#include "pin_notice.hpp"
#include "finite_volume.hpp"
#include "problems_3d.hpp"
#include "support.hpp"

#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace support;
using State = hydro::State<3>;

// The value of `--name=value` among argv[3..], else `fallback`. Used by: run, main.
std::string flag(int argc, char** argv, const char* name, const char* fallback) {
    const std::size_t length = std::strlen(name);
    for (int a = 3; a < argc; ++a)
        if (std::strncmp(argv[a], name, length) == 0 && argv[a][length] == '=') return argv[a] + length + 1;
    return fallback;
}
// Whether the bare flag `name` is among argv[3..]. Used by: run.
bool has(int argc, char** argv, const char* name) {
    for (int a = 3; a < argc; ++a)
        if (std::strcmp(argv[a], name) == 0) return true;
    return false;
}

// One run with the precision policy P (--precision, as in hydro_sedov); argv[1] problem, argv[2] nx.
// Used by: main.
template <class P>
int run(int argc, char** argv) {
    using FV = hydro::FiniteVolume<3, Exec, hydro::HLLC, P>;
    const std::string problem = argv[1];
    const int nx = std::stoi(argv[2]);
    const double courant = std::stod(flag(argc, argv, "--courant", "0.3")), jitter = std::stod(flag(argc, argv, "--jitter", "0.1"));
    const double amp = std::stod(flag(argc, argv, "--amp", "0.1"));
    const long long max_steps = std::stoll(flag(argc, argv, "--max-steps", "0"));
    const bool moving = has(argc, argv, "--moving");
    const std::string out = flag(argc, argv, "--out", "");
    const hydro::Problem3D p = hydro::problem_3d(problem, nx);
    const double t_end = std::stod(flag(argc, argv, "--t-end", std::to_string(p.t_end).c_str()));
    const Box box(p.side[0], p.side[1], p.side[2]);
    const hydro::Gas gas{hydro::wave_gamma};
    Points points;
    for (const auto& q : hydro::jittered_points(p, jitter)) points.push_back(Point{q[0], q[1], q[2]});
    const long long n = static_cast<long long>(points.size());
    const Exec exec;
    typename FV::Mesh mesh(exec, box);
    mesh.update(to_device(points));
    mesh.update_cells();
    const std::string integrator = flag(argc, argv, "--integrator", moving ? "two-pass" : "one-pass");
    if (integrator != "one-pass" && integrator != "two-pass") throw std::invalid_argument("hydro_wave3d: --integrator=one-pass|two-pass");
    FV fv(exec, mesh, gas, 1, hydro::Update::per_face, has(argc, argv, "--limiter=none") ? hydro::Limiter::none : hydro::Limiter::face_centroid,
          moving ? hydro::Motion::moving : hydro::Motion::static_mesh,
          integrator == "two-pass" ? hydro::Integrator::two_pass : hydro::Integrator::one_pass);
    {
        const auto x = to_host(mesh.positions());
        const auto c = to_host(mesh.centroid());
        const auto V = to_host(mesh.volume());
        auto Q = Kokkos::create_mirror_view(fv.state());
        for (long long i = 0; i < n; ++i) {
            double at[2];
            for (int a = 0; a < 2; ++a) at[a] = std::fmod(x(i, a) + c(i, a) + 2 * p.side[a], p.side[a]) + p.origin[a];
            const State u = hydro::conserved(hydro::wave_initial(problem, at[0], at[1], amp), gas);
            for (int v = 0; v < 5; ++v) Q(i, v) = u[v] * V(i);
        }
        Kokkos::deep_copy(fv.state(), Q);
    }
    Kokkos::Timer timer;
    double t = 0;
    long long steps = 0, fallbacks = 0;
    while (t < t_end * (1 - 1e-12) && (max_steps == 0 || steps < max_steps)) {
        const double dt = std::min(fv.time_step(courant), t_end - t);
        fv.step(dt);
        fallbacks += fv.fallbacks();
        t += dt;
        ++steps;
    }
    exec.fence();
    const double wall = timer.seconds();
    const hydro::MeshStats& ms = fv.mesh_stats();
    std::cout << "ours3d: problem=" << problem << " nx=" << nx << " n=" << n << " moving=" << moving << " integrator=" << integrator << " courant=" << courant << " jitter=" << jitter
              << " steps=" << steps << " t=" << t << " rebuilds=" << ms.rebuilds << " fallbacks=" << fallbacks << " precision=" << flag(argc, argv, "--precision", "default") << " wall_s=" << wall << " M_cell_updates_per_s=" << n * steps / wall / 1e6 << "\n";
    if (!out.empty()) {
        mesh.update_cells();
        const auto x = to_host(mesh.positions());
        const auto c = to_host(mesh.centroid());
        const auto V = to_host(mesh.volume());
        const auto Q = to_host(fv.state());
        std::vector<std::array<double, 12>> rows(static_cast<std::size_t>(n));
        for (long long i = 0; i < n; ++i) {
            const State W = hydro::primitive(State{Q(i, 0) / V(i), Q(i, 1) / V(i), Q(i, 2) / V(i), Q(i, 3) / V(i), Q(i, 4) / V(i)}, gas);
            auto& r = rows[static_cast<std::size_t>(i)];
            for (int a = 0; a < 3; ++a) {
                r[a] = x(i, a) + p.origin[a];
                r[3 + a] = x(i, a) + c(i, a) + p.origin[a];
            }
            r[6] = V(i);
            for (int v = 0; v < 5; ++v) r[7 + v] = W[v];
        }
        char header[128];
        int length = std::snprintf(header, sizeof header, "{'descr': '<f8', 'fortran_order': False, 'shape': (%zu, 12), }", rows.size());
        while ((10 + length + 1) % 64 != 0) header[length++] = ' ';
        header[length++] = '\n';
        std::ofstream file(out, std::ios::binary);
        file.write("\x93NUMPY\x01\x00", 8);
        const unsigned short size = static_cast<unsigned short>(length);
        file.write(reinterpret_cast<const char*>(&size), 2);
        file.write(header, length);
        file.write(reinterpret_cast<const char*>(rows.data()), static_cast<std::streamsize>(rows.size() * sizeof rows[0]));
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    luna::pin_notice();
    const std::string precision = flag(argc, argv, "--precision", "default");
    if (precision == "default") return run<hydro::DefaultPrecision<3>>(argc, argv);
    if (precision == "double") return run<hydro::Precision<>>(argc, argv);
    if (precision == "grad32") return run<hydro::Grad32>(argc, argv);
    if (precision == "grad16") return run<hydro::Precision<double, void, hydro::Half>>(argc, argv);
    throw std::invalid_argument("hydro_wave3d: --precision=default|double|grad32|grad16");
}
