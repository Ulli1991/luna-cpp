// CTest `hydro_wave`: L1 accuracy of the moving 3D step on smooth flow with a known end state (vortex, sound wave), both integrators.
// Bounds are measured values (same to 3 digits on CPU and A100) plus 10 %; the rates sit just below measured, never a nominal 2.
// Two passes: rho at rate 2.2 and 1.7, one pass 1.1 and 0.

#include "check.hpp"
#include "finite_volume.hpp"
#include "problems_3d.hpp"
#include "support.hpp"

#include <array>
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>

namespace {

using namespace support;
using FV = hydro::FiniteVolume<3, Exec>;
using State = hydro::State<3>;

// L1 of rho, |v|, P against `exact` at the centroids.
template <class Exact>
std::array<double, 3> run(const std::string& problem, int nx, hydro::Integrator integrator, double amp, double jitter, double t_end, Exact exact) {
    const hydro::Problem3D p = hydro::problem_3d(problem, nx);
    const hydro::Gas gas{hydro::wave_gamma};
    Points points;
    for (const auto& q : hydro::jittered_points(p, jitter)) points.push_back(Point{q[0], q[1], q[2]});
    const long long n = static_cast<long long>(points.size());
    FV::Mesh mesh(Exec{}, Box(p.side[0], p.side[1], p.side[2]));
    mesh.update(to_device(points));
    mesh.update_cells();
    FV fv(Exec{}, mesh, gas, 1, hydro::Update::per_face, hydro::Limiter::face_centroid, hydro::Motion::moving, integrator);
    auto at = [&](const auto& x, const auto& c, long long i, int a) { return std::fmod(x(i, a) + c(i, a) + 2 * p.side[a], p.side[a]) + p.origin[a]; };
    {
        const auto x = to_host(mesh.positions());
        const auto c = to_host(mesh.centroid());
        const auto V = to_host(mesh.volume());
        auto Q = Kokkos::create_mirror_view(fv.state());
        for (long long i = 0; i < n; ++i) {
            const State u = hydro::conserved(hydro::wave_initial(problem, at(x, c, i, 0), at(x, c, i, 1), amp), gas);
            for (int v = 0; v < 5; ++v) Q(i, v) = u[v] * V(i);
        }
        Kokkos::deep_copy(fv.state(), Q);
    }
    double t = 0;
    while (t < t_end * (1 - 1e-12)) {
        const double dt = std::min(fv.time_step(0.3), t_end - t);
        fv.step(dt);
        t += dt;
    }
    Exec{}.fence();
    mesh.update_cells();
    const auto x = to_host(mesh.positions());
    const auto c = to_host(mesh.centroid());
    const auto V = to_host(mesh.volume());
    const auto Q = to_host(fv.state());
    std::array<double, 3> l1{};
    double total = 0;
    for (long long i = 0; i < n; ++i) {
        const State W = hydro::primitive(State{Q(i, 0) / V(i), Q(i, 1) / V(i), Q(i, 2) / V(i), Q(i, 3) / V(i), Q(i, 4) / V(i)}, gas);
        const State E = exact(at(x, c, i, 0), at(x, c, i, 1));
        l1[0] += V(i) * std::abs(W[0] - E[0]);
        l1[1] += V(i) * std::sqrt((W[1] - E[1]) * (W[1] - E[1]) + (W[2] - E[2]) * (W[2] - E[2]) + (W[3] - E[3]) * (W[3] - E[3]));
        l1[2] += V(i) * std::abs(W[4] - E[4]);
        total += V(i);
    }
    for (double& e : l1) e /= total;
    return l1;
}

// The largest allowed L1 of rho, v, P at both resolutions, and the smallest rate of rho between them.
struct Gate {
    const char* name;
    hydro::Integrator integrator;
    std::array<double, 3> coarse, fine;
    double rate;
};

void gate(Check& check, const std::string& problem, const Gate& g, int nx, double amp, double jitter, double t_end) {
    auto exact = [&](double x, double y) { return hydro::wave_initial(problem, x, y, amp); };
    const auto a = run(problem, nx, g.integrator, amp, jitter, t_end, exact);
    const auto b = run(problem, 2 * nx, g.integrator, amp, jitter, t_end, exact);
    const double rate = std::log2(a[0] / b[0]);
    const std::string head = problem + " " + g.name;
    for (int k = 0; k < 3; ++k) {
        const char* what = k == 0 ? "rho" : k == 1 ? "v" : "P";
        check.expect(a[k] < g.coarse[k], head + " nx=" + std::to_string(nx) + " L1 " + what + " " + sci(a[k]) + " < " + sci(g.coarse[k]));
        check.expect(b[k] < g.fine[k], head + " nx=" + std::to_string(2 * nx) + " L1 " + what + " " + sci(b[k]) + " < " + sci(g.fine[k]));
    }
    check.expect(rate > g.rate, head + " rate of rho " + std::to_string(rate) + " > " + std::to_string(g.rate));
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    Check check;
    constexpr auto one = hydro::Integrator::one_pass;
    constexpr auto two = hydro::Integrator::two_pass;
    gate(check, "vortex", {"one-pass", one, {1.9e-3, 6.8e-3, 3.2e-3}, {8.9e-4, 1.3e-3, 7.8e-4}, 1.0}, 24, 0, 0.1, 2);
    gate(check, "vortex", {"two-pass", two, {2.6e-3, 7.2e-3, 3.6e-3}, {5.7e-4, 1.3e-3, 7.6e-4}, 2.0}, 24, 0, 0.1, 2);
    // At jitter 0.25 rho of one pass sits on a floor of the regularised mesh motion, the rate near 0.
    gate(check, "sound", {"one-pass", one, {7.2e-5, 2.2e-5, 1.9e-5}, {7.3e-5, 5.5e-6, 5.9e-6}, -0.15}, 24, 1e-3, 0.25, 1);
    gate(check, "sound", {"two-pass", two, {2.7e-5, 2.7e-5, 2.7e-5}, {8.2e-6, 7.4e-6, 7.3e-6}, 1.6}, 24, 1e-3, 0.25, 1);
    return check.failed() != 0;
}
