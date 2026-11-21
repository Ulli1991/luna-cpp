// Checks the hand-written tangents and adjoints of the 3D hydro (euler.hpp, finite_volume.hpp): per piece on random states in a kernel,
// and whole static steps on a small mesh (FiniteVolume::step_tangent, step_adjoint). Tangent against central differences away from
// switches, adjoint against tangent (<w, J v> = <J^T w, v>). `--cost N`: times of step, step_tangent and step_adjoint on N cells.

#include "check.hpp"
#include "finite_volume.hpp"
#include "finite_volume_1d.hpp"
#include "luna_derivative_values.hpp"
#include "problems_1d.hpp"
#include "support.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace support;
constexpr int D = 3, V = D + 2;
using State = hydro::State<D>;
using Vec = hydro::Vec<D>;
using Grad = hydro::Gradient<D>;
using FV = hydro::FiniteVolume<D, Exec, hydro::HLLC, hydro::Precision<>>;
using FVHLL = hydro::FiniteVolume<D, Exec, hydro::HLL, hydro::Precision<>>;
using FV16 = hydro::FiniteVolume<D, Exec, hydro::HLLC, hydro::Grad16>;
using FV32 = hydro::FiniteVolume<D, Exec, hydro::HLLC, hydro::Grad32>;

// splitmix64: a reproducible stream per sample, on host and device. Used by: every piece's make().
KOKKOS_INLINE_FUNCTION double uniform(std::uint64_t& s, double lo, double hi) {
    std::uint64_t z = (s += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    z ^= z >> 31;
    return lo + (hi - lo) * static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0);
}
KOKKOS_INLINE_FUNCTION Vec unit_vector(std::uint64_t& s) {
    Vec n{};
    double r = 0;
    while (r < 1e-2) {
        for (int a = 0; a < D; ++a) n[a] = uniform(s, -1, 1);
        r = dot(n, n);
    }
    return n * (1.0 / Kokkos::sqrt(r));
}
// A physical W: rho in [0.1, 2], v in [-2, 2], P in [0.1, 3].
KOKKOS_INLINE_FUNCTION State random_W(std::uint64_t& s) {
    return State{uniform(s, 0.1, 2), uniform(s, -2, 2), uniform(s, -2, 2), uniform(s, -2, 2), uniform(s, 0.1, 3)};
}

// What a piece's samples gave: the largest relative differences, how many samples were kept or skipped (a branch switch within
// +- h), how many exceeded 1e-7 and 1e-6, and the kept samples per branch category (the piece's category()).
struct Stats {
    double fd = 0, dot = 0;
    long long kept = 0, skipped = 0, above7 = 0, above6 = 0;
    long long category[4] = {0, 0, 0, 0};
};

// Runs a piece on `samples` random inputs: Piece::make(seed) -> sample with x; f(sample, x, y) -> branch pattern; tangent, adjoint.
template <class Piece>
struct Sampler {
    using value_type = Stats;
    Piece piece;
    double h;
    double corrupt = 0;  // teeth: the tangent's first output and the adjoint's first input scaled by 1 + corrupt
    KOKKOS_INLINE_FUNCTION void init(Stats& s) const { s = Stats{}; }
    KOKKOS_INLINE_FUNCTION void join(Stats& a, const Stats& b) const {
        a.fd = Kokkos::fmax(a.fd, b.fd);
        a.dot = Kokkos::fmax(a.dot, b.dot);
        a.kept += b.kept;
        a.skipped += b.skipped;
        a.above7 += b.above7;
        a.above6 += b.above6;
        for (int c = 0; c < 4; ++c) a.category[c] += b.category[c];
    }
    KOKKOS_INLINE_FUNCTION void operator()(int index, Stats& stats) const {
        constexpr int NI = Piece::inputs, NO = Piece::outputs;
        std::uint64_t seed = 0x5eed0000ull + static_cast<std::uint64_t>(index) * 7919ull;
        typename Piece::Sample s = piece.make(seed);
        const int ni = piece.used_inputs(s);
        double dx[NI], xp[NI], xm[NI], y[NO], yp[NO], ym[NO], dy[NO], ybar[NO], xbar[NI];
        for (int i = 0; i < NI; ++i) {
            dx[i] = i < ni ? uniform(seed, -1, 1) * Kokkos::fmax(Kokkos::fabs(s.x[i]), 0.1) : 0.0;
            xp[i] = s.x[i] + h * dx[i];
            xm[i] = s.x[i] - h * dx[i];
        }
        const long long pattern = piece.f(s, s.x, y);
        if (piece.f(s, xp, yp) != pattern || piece.f(s, xm, ym) != pattern) {
            ++stats.skipped;
            return;
        }
        piece.tangent(s, s.x, dx, dy);
        dy[0] *= 1 + corrupt;
        double diff = 0, scale = 0;
        for (int o = 0; o < NO; ++o) {
            const double fd = (yp[o] - ym[o]) / (2 * h);
            diff = Kokkos::fmax(diff, Kokkos::fabs(dy[o] - fd));
            scale = Kokkos::fmax(scale, Kokkos::fmax(Kokkos::fabs(fd), Kokkos::fabs(dy[o])));
        }
        const double fd = scale > 0 ? diff / scale : 0.0;
        for (int o = 0; o < NO; ++o) ybar[o] = uniform(seed, -1, 1);
        for (int i = 0; i < NI; ++i) xbar[i] = 0;
        piece.adjoint(s, s.x, ybar, xbar);
        xbar[0] *= 1 + corrupt;
        double lhs = 0, rhs = 0, size = 0;
        for (int o = 0; o < NO; ++o) {
            lhs += ybar[o] * dy[o];
            size += Kokkos::fabs(ybar[o] * dy[o]);
        }
        for (int i = 0; i < NI; ++i) {
            rhs += xbar[i] * dx[i];
            size += Kokkos::fabs(xbar[i] * dx[i]);
        }
        ++stats.kept;
        stats.fd = Kokkos::fmax(stats.fd, fd);
        stats.dot = Kokkos::fmax(stats.dot, size > 0 ? Kokkos::fabs(lhs - rhs) / size : 0.0);
        stats.above7 += fd > 1e-7;
        stats.above6 += fd > 1e-6;
        ++stats.category[piece.category(pattern)];
    }
};

KOKKOS_INLINE_FUNCTION State state_at(const double* x) { return State{x[0], x[1], x[2], x[3], x[4]}; }
KOKKOS_INLINE_FUNCTION void put(const State& W, double* y) {
    for (int v = 0; v < V; ++v) y[v] = W[v];
}
KOKKOS_INLINE_FUNCTION void add(const State& W, double* y) {
    for (int v = 0; v < V; ++v) y[v] += W[v];
}

// primitive(): U -> W, away from the floors (they are branches of their own, derivative 0 in what they replace).
struct PrimitivePiece {
    static constexpr int inputs = V, outputs = V;
    struct Sample {
        double x[V];
    };
    hydro::Gas gas;
    KOKKOS_INLINE_FUNCTION Sample make(std::uint64_t& seed) const {
        Sample s;
        put(hydro::conserved(random_W(seed), gas), s.x);
        return s;
    }
    KOKKOS_INLINE_FUNCTION int used_inputs(const Sample&) const { return inputs; }
    KOKKOS_INLINE_FUNCTION long long f(const Sample&, const double* x, double* y) const {
        put(hydro::primitive(state_at(x), gas), y);
        return hydro::primitive_branches(state_at(x), gas);
    }
    KOKKOS_INLINE_FUNCTION void tangent(const Sample&, const double* x, const double* dx, double* dy) const {
        put(hydro::primitive_tangent(state_at(x), state_at(dx), gas), dy);
    }
    KOKKOS_INLINE_FUNCTION void adjoint(const Sample&, const double* x, const double* ybar, double* xbar) const {
        add(hydro::primitive_adjoint(state_at(x), state_at(ybar), gas), xbar);
    }
    KOKKOS_INLINE_FUNCTION int category(long long pattern) const { return static_cast<int>(pattern & 3); }
};

// FaceFlux::side_value(): (W, g) -> the face state (half step plus the offset r), r and dt fixed; category 1: the side fell back.
struct SidePiece {
    static constexpr int inputs = V + V * D, outputs = V;
    struct Sample {
        double x[V + V * D];
        Vec r;
        double dt;
    };
    hydro::Gas gas;
    KOKKOS_INLINE_FUNCTION Sample make(std::uint64_t& seed) const {
        Sample s;
        put(random_W(seed), s.x);
        for (int v = 0; v < V; ++v)
            for (int a = 0; a < D; ++a) s.x[V + D * v + a] = uniform(seed, -1, 1) * Kokkos::fabs(s.x[v]) * 2;
        s.r = unit_vector(seed) * uniform(seed, 0.05, 0.5);
        s.dt = uniform(seed, 0.01, 0.1);
        return s;
    }
    KOKKOS_INLINE_FUNCTION static Grad gradient(const double* x) {
        Grad g;
        for (int v = 0; v < V; ++v) g[v] = hydro::make_vec<D>([&](int a) { return x[V + D * v + a]; });
        return g;
    }
    KOKKOS_INLINE_FUNCTION int used_inputs(const Sample&) const { return inputs; }
    KOKKOS_INLINE_FUNCTION long long f(const Sample& s, const double* x, double* y) const {
        bool fallback;
        put(FV::FaceFlux<1, false>::side_value(state_at(x), gradient(x), s.r, s.dt, gas, fallback), y);
        return fallback;
    }
    KOKKOS_INLINE_FUNCTION void tangent(const Sample& s, const double* x, const double* dx, double* dy) const {
        put(FV::FaceFlux<1, false>::side_tangent(state_at(x), state_at(dx), gradient(x), gradient(dx), s.r, s.dt, gas), dy);
    }
    KOKKOS_INLINE_FUNCTION void adjoint(const Sample& s, const double* x, const double* ybar, double* xbar) const {
        State Wbar{};
        Grad gbar;
        for (int v = 0; v < V; ++v) gbar[v] = Vec{};
        FV::FaceFlux<1, false>::side_adjoint(state_at(x), gradient(x), s.r, s.dt, gas, state_at(ybar), Wbar, gbar);
        add(Wbar, xbar);
        for (int v = 0; v < V; ++v)
            for (int a = 0; a < D; ++a) xbar[V + D * v + a] += gbar[v][a];
    }
    KOKKOS_INLINE_FUNCTION int category(long long pattern) const { return static_cast<int>(pattern); }
};

// Riemann::flux(): (L, R) -> the lab-frame flux through a face of unit normal n moving with w (both fixed).
// Category: HLLC F_L, F*_L, F*_R, F_R; HLL star, F_L, F_R.
template <class Riemann>
struct FluxPiece {
    static constexpr int inputs = 2 * V, outputs = V;
    struct Sample {
        double x[2 * V];
        Vec n, w;
    };
    hydro::Gas gas;
    KOKKOS_INLINE_FUNCTION Sample make(std::uint64_t& seed) const {
        Sample s;
        put(random_W(seed), s.x);
        put(random_W(seed), s.x + V);
        s.n = unit_vector(seed);
        s.w = hydro::make_vec<D>([&](int) { return uniform(seed, -1, 1); });
        return s;
    }
    KOKKOS_INLINE_FUNCTION int used_inputs(const Sample&) const { return inputs; }
    KOKKOS_INLINE_FUNCTION long long f(const Sample& s, const double* x, double* y) const {
        put(Riemann::flux(state_at(x), state_at(x + V), s.n, s.w, gas), y);
        return Riemann::branches(state_at(x), state_at(x + V), s.n, s.w, gas);
    }
    KOKKOS_INLINE_FUNCTION void tangent(const Sample& s, const double* x, const double* dx, double* dy) const {
        put(Riemann::flux_tangent(state_at(x), state_at(x + V), state_at(dx), state_at(dx + V), s.n, s.w, gas), dy);
    }
    KOKKOS_INLINE_FUNCTION void adjoint(const Sample& s, const double* x, const double* ybar, double* xbar) const {
        State Lbar{}, Rbar{};
        Riemann::flux_adjoint(state_at(x), state_at(x + V), s.n, s.w, gas, state_at(ybar), Lbar, Rbar);
        add(Lbar, xbar);
        add(Rbar, xbar + V);
    }
    KOKKOS_INLINE_FUNCTION int category(long long pattern) const {
        if constexpr (std::is_same_v<Riemann, hydro::HLLC>) {
            const bool left = pattern & 4, star = pattern & 8;
            return left ? (star ? 1 : 0) : (star ? 2 : 3);
        } else {
            return static_cast<int>((pattern >> 2) & 3);
        }
    }
};

// The limited gradient of one cell with up to 16 neighbours: (W_i, W_j) -> g, the stencil's geometry fixed.
// Category: bit 0 some variable clipped, bit 1 some variable clipped to zero.
constexpr int most = 16;
struct GradientPiece {
    static constexpr int inputs = V + most * V, outputs = V * D;
    struct Sample {
        double x[V + most * V];
        Vec between[most], offset[most];
        double weight[most];
        int count;
    };
    // The Entries of FiniteVolume::limited_gradient*() over a sample (x: W_i then the W_j; dx the same for the tangent).
    struct Entries {
        const Sample* s;
        const double* x;
        const double* dx;
        KOKKOS_INLINE_FUNCTION int count() const { return s->count; }
        KOKKOS_INLINE_FUNCTION State W(int k) const { return state_at(x + V + V * k); }
        KOKKOS_INLINE_FUNCTION State dW(int k) const { return state_at(dx + V + V * k); }
        KOKKOS_INLINE_FUNCTION Vec between(int k) const { return s->between[k]; }
        KOKKOS_INLINE_FUNCTION double weight(int k) const { return s->weight[k]; }
        KOKKOS_INLINE_FUNCTION Vec offset(int k) const { return s->offset[k]; }
        template <bool Tangent>
        KOKKOS_INLINE_FUNCTION FV::Entry load(int k) const {
            FV::Entry e;
            e.W = W(k);
            if constexpr (Tangent) e.dW = dW(k);
            e.between = between(k);
            e.weight = weight(k);
            return e;
        }
    };
    KOKKOS_INLINE_FUNCTION Sample make(std::uint64_t& seed) const {
        Sample s;
        s.count = 8 + static_cast<int>(uniform(seed, 0, most - 8 + 1 - 1e-9));
        const State Wi = random_W(seed);
        put(Wi, s.x);
        Grad slope;
        for (int v = 0; v < V; ++v) slope[v] = hydro::make_vec<D>([&](int) { return uniform(seed, -1, 1) * Wi[v]; });
        for (int k = 0; k < most; ++k) {
            s.between[k] = unit_vector(seed) * uniform(seed, 0.5, 1.5);
            s.offset[k] = s.between[k] * 0.5 + unit_vector(seed) * uniform(seed, 0, 0.15);
            s.weight[k] = uniform(seed, 0.1, 1);
            for (int v = 0; v < V; ++v)
                s.x[V + V * k + v] = k < s.count ? Wi[v] + dot(slope[v], s.between[k]) + uniform(seed, -0.3, 0.3) * Wi[v] : 0.0;
        }
        return s;
    }
    KOKKOS_INLINE_FUNCTION int used_inputs(const Sample& s) const { return V + V * s.count; }
    KOKKOS_INLINE_FUNCTION static void put_gradient(const Grad& g, double* y) {
        for (int v = 0; v < V; ++v)
            for (int a = 0; a < D; ++a) y[D * v + a] = g[v][a];
    }
    KOKKOS_INLINE_FUNCTION long long f(const Sample& s, const double* x, double* y) const {
        FV::LimitPattern p;
        Grad g0;
        put_gradient(FV::limited_gradient(state_at(x), Entries{&s, x, nullptr}, hydro::Limiter::face_centroid, p, g0), y);
        long long code = 0;
        for (int v = 0; v < V; ++v) code = code * 1031 + (p.last[v] + 1) * 37 + (p.extreme[v] + 1) * 2 + p.upper[v] + 4 * p.zero[v] * 1000;
        long long clipped = 0, zero = 0;
        for (int v = 0; v < V; ++v) clipped |= p.last[v] >= 0, zero |= p.zero[v];
        return code * 4 + clipped + 2 * zero;
    }
    KOKKOS_INLINE_FUNCTION void tangent(const Sample& s, const double* x, const double* dx, double* dy) const {
        Grad g;
        put_gradient(FV::limited_gradient_tangent(state_at(x), state_at(dx), Entries{&s, x, dx}, hydro::Limiter::face_centroid, g), dy);
    }
    KOKKOS_INLINE_FUNCTION void adjoint(const Sample& s, const double* x, const double* ybar, double* xbar) const {
        Grad gbar;
        for (int v = 0; v < V; ++v) gbar[v] = hydro::make_vec<D>([&](int a) { return ybar[D * v + a]; });
        const Entries e{&s, x, nullptr};
        const FV::GradientAdjoint adjoint = FV::limited_gradient_adjoint(state_at(x), e, hydro::Limiter::face_centroid, gbar);
        add(adjoint.self, xbar);
        for (int k = 0; k < s.count; ++k) add(FV::neighbour_bar(e, k, adjoint), xbar + V + V * k);
    }
    KOKKOS_INLINE_FUNCTION int category(long long pattern) const { return static_cast<int>(pattern & 3); }
};

// One piece over `samples` random inputs on the default execution space; prints its table row.
template <class Piece>
void piece_test(Check& check, const std::string& name, const Piece& piece, int samples, const char* categories) {
    Stats stats;
    Kokkos::parallel_reduce("derivative_samples", Kokkos::RangePolicy<Exec>(0, samples), Sampler<Piece>{piece, 1e-6}, stats);
    std::printf("piece %-16s kept %7lld skipped %5lld  tangent-FD max %.2e (>1e-7: %lld, >1e-6: %lld)  dot max %.2e  %s %lld %lld %lld %lld\n",
                name.c_str(), stats.kept, stats.skipped, stats.fd, stats.above7, stats.above6, stats.dot, categories, stats.category[0],
                stats.category[1], stats.category[2], stats.category[3]);
    check.expect(stats.kept > samples / 2, name + ": most samples away from switches (" + std::to_string(stats.kept) + " of " +
                                              std::to_string(samples) + ")");
    check.expect(stats.above6 * 1000 <= stats.kept && stats.fd < 1e-4,
                 name + ": tangent equals central differences (h = 1e-6) to 1e-6 in 999 of 1000 samples (" + std::to_string(stats.above6) +
                     " above, max " + sci(stats.fd) + ")");
    check.expect(stats.dot < 1e-13, name + ": <w, J v> = <J^T w, v> (" + sci(stats.dot) + ")");
    // Teeth: a derivative off by 1 % in one entry must fail both comparisons.
    Stats teeth;
    Kokkos::parallel_reduce("derivative_teeth", Kokkos::RangePolicy<Exec>(0, samples / 10), Sampler<Piece>{piece, 1e-6, 0.01}, teeth);
    check.expect(teeth.fd > 1e-4 && teeth.dot > 1e-5,
                 name + ": a tangent and an adjoint off by 1 % in one entry are caught (" + sci(teeth.fd) + ", " + sci(teeth.dot) + ")");
}

using HostCells = Kokkos::View<double* [V], Kokkos::LayoutRight, Kokkos::HostSpace>;

template <class F>
HostCells read(const F& fv) {
    HostCells h("read", fv.state().extent(0));
    Kokkos::deep_copy(h, fv.state());
    return h;
}
template <class F>
void write(F& fv, const HostCells& h) {
    Kokkos::deep_copy(fv.state(), h);
}
HostCells combine(const HostCells& a, double s, const HostCells& b) {
    HostCells c("combine", a.extent(0));
    for (std::size_t i = 0; i < a.extent(0); ++i)
        for (int v = 0; v < V; ++v) c(i, v) = a(i, v) + s * b(i, v);
    return c;
}
double inner(const HostCells& a, const HostCells& b, double* size = nullptr) {
    double sum = 0, abs = 0;
    for (std::size_t i = 0; i < a.extent(0); ++i)
        for (int v = 0; v < V; ++v) {
            sum += a(i, v) * b(i, v);
            abs += std::abs(a(i, v) * b(i, v));
        }
    if (size) *size = abs;
    return sum;
}

// A smooth state with noise of relative size `noise` (0.01: some limiters clip, most do not; 0.3: many clip, some sides fall back), extensive.
HostCells smooth_state(const Points& points, const Kokkos::View<const double*, Kokkos::HostSpace>& volume, const hydro::Gas& gas, double amplitude = 0.01) {
    std::mt19937_64 rng(5);
    std::uniform_real_distribution<double> noise(-amplitude, amplitude);
    HostCells h("smooth", points.size());
    const double k = 2 * M_PI;
    for (std::size_t i = 0; i < points.size(); ++i) {
        const auto& x = points[i];
        const State W{1 + 0.2 * std::sin(k * x[0]) + noise(rng), 0.3 * std::sin(k * x[1]) + noise(rng), 0.2 * std::cos(k * x[2]) + noise(rng),
                      0.1 + noise(rng), 1 + 0.2 * std::cos(k * x[0] + k * x[1]) + noise(rng)};
        const State U = hydro::conserved(W, gas);
        for (int v = 0; v < V; ++v) h(i, v) = U[v] * volume(i);
    }
    return h;
}

// A whole static step (and `steps` steps): forward bits, tangent against central differences, adjoint against tangent. A reduced
// gradient policy (grad32, grad16): central differences of its forward are rounding below h = 1e-6 (reported, not checked); its
// tangent against `reference`, the double policy's on the same mesh and state. Returns the tangent.
template <class F>
HostCells step_test(Check& check, const std::string& name, int n, int steps, double noise = 0.01, double h = 1e-6,
                    const HostCells* reference = nullptr, bool fused = true) {
    const hydro::Gas gas{5.0 / 3.0};
    const Points points = make_points_off_grid("jittered", n);
    typename F::Mesh mesh(Exec{}, 1.0);
    mesh.update(to_device(points));
    mesh.update_cells();
    F fv(Exec{}, mesh, gas, 1, hydro::Update::per_face);
    const auto volume = to_host(mesh.volume());
    const HostCells q0 = smooth_state(points, volume, gas, noise);
    std::mt19937_64 rng(17);
    std::uniform_real_distribution<double> unit(-1, 1);
    HostCells dq0("dq0", n), wbar("wbar", n);
    for (int i = 0; i < n; ++i)
        for (int v = 0; v < V; ++v) {
            dq0(i, v) = unit(rng) * std::abs(q0(i, v)) + 0.1 * unit(rng) * volume(i);
            wbar(i, v) = unit(rng);
        }
    // Forward: the time steps and the checkpoints Q^0 .. Q^steps.
    write(fv, q0);
    std::vector<double> dts;
    std::vector<HostCells> checkpoints{q0};
    for (int s = 0; s < steps; ++s) {
        dts.push_back(fv.time_step(0.3));
        fv.step(dts.back());
        checkpoints.push_back(read(fv));
    }
    auto run = [&](const HostCells& start) {
        write(fv, start);
        for (int s = 0; s < steps; ++s) fv.step(dts[s]);
        return read(fv);
    };
    // Tangent: the same forward bits, and J dq.
    write(fv, q0);
    typename F::Cells dqd("dq", n);
    Kokkos::deep_copy(dqd, dq0);
    // The tangent run's own trajectory: the states its derivatives linearise at, and the adjoint's checkpoints.
    std::vector<HostCells> path;
    bool same_dt = true;
    for (int s = 0; s < steps; ++s) {
        path.push_back(read(fv));
        if (fused) {  // time_step_tangent: the sweep and its tangent in one kernel, and the same dt as time_step()
            const double dt = fv.time_step_tangent(0.3, dqd);
            if (s == 0) same_dt = dt == dts[0];  // later steps start from the tangent run's own Q (equal to rounding)
        }
        fv.step_tangent(dts[s], dqd);
    }
    if (fused) check.expect(same_dt, name + ": time_step_tangent gives time_step()'s dt");
    const HostCells forward = read(fv);
    HostCells tangent("tangent", n);
    Kokkos::deep_copy(tangent, dqd);
    double bits = 0;
    for (int i = 0; i < n; ++i)
        for (int v = 0; v < V; ++v)
            bits = std::max(bits, std::abs(forward(i, v) - checkpoints.back()(i, v)) / std::max(std::abs(checkpoints.back()(i, v)), 1e-300));
    const bool same = std::memcmp(forward.data(), checkpoints.back().data(), sizeof(double) * V * n) == 0;
    // Double: equal to rounding (the flux from sides computed once, 1e-14 measured). grad32 and grad16: the float face states of
    // FaceFlux contract differently inside the fused tangent kernel than in step()'s (A100: 1e-8 to 1.5e-6 over 3 steps).
    check.expect(bits < (std::is_same_v<typename F::G, double> ? 1e-11 : 1e-5), name + ": step_tangent's Q equals step()'s over " + std::to_string(steps) + " steps (" + sci(bits) +
                                   (same ? ", same bits)" : ")"));
    // Central differences with step h (relative to each entry of dq0).
    const HostCells plus = run(combine(q0, h, dq0)), minus = run(combine(q0, -h, dq0));
    // Central for every cell; where it differs (a switch between x - h dq and x + h dq), the tangent must equal one of the two
    // one-sided differences: the side without the switch (the one-sided derivative the tangent takes, or the other side's).
    const HostCells& at = checkpoints.back();
    double worst = 0, scale = 0, worst_one_sided = 0;
    int cells_off = 0, worst_cell = 0;
    for (int i = 0; i < n; ++i)
        for (int v = 0; v < V; ++v) scale = std::max(scale, std::abs(plus(i, v) - minus(i, v)) / (2 * h));
    for (int i = 0; i < n; ++i) {
        double cell_diff = 0, cell_scale = 1e-6 * scale, plus_diff = 0, minus_diff = 0;
        for (int v = 0; v < V; ++v) {
            const double fd = (plus(i, v) - minus(i, v)) / (2 * h);
            cell_diff = std::max(cell_diff, std::abs(tangent(i, v) - fd));
            cell_scale = std::max(cell_scale, std::abs(fd));
            plus_diff = std::max(plus_diff, std::abs(tangent(i, v) - (plus(i, v) - at(i, v)) / h));
            minus_diff = std::max(minus_diff, std::abs(tangent(i, v) - (at(i, v) - minus(i, v)) / h));
        }
        if (cell_diff > worst) worst_cell = i;
        worst = std::max(worst, cell_diff);
        if (cell_diff > 1e-6 * cell_scale) {
            ++cells_off;
            worst_one_sided = std::max(worst_one_sided, std::min(plus_diff, minus_diff) / cell_scale);
        }
    }
    if (cells_off > 0)
        for (int v = 0; v < V; ++v)
            std::printf("  worst cell %d v %d: tangent %.10e  central %.10e  one-sided +h %.10e  -h %.10e\n", worst_cell, v, tangent(worst_cell, v),
                        (plus(worst_cell, v) - minus(worst_cell, v)) / (2 * h), (plus(worst_cell, v) - at(worst_cell, v)) / h,
                        (at(worst_cell, v) - minus(worst_cell, v)) / h);
    // Smooth: no switch within +- h, the central difference everywhere. Rough: switches within +- h (a clip, a wave case, a
    // fallback); there one side must match (its own O(h) error: 1e-4 of the cell's largest entry).
    const bool rough = noise > 0.05;
    constexpr bool exact = std::is_same_v<typename F::G, double>;
    if (exact)
        check.expect(rough ? worst_one_sided < 1e-4 : worst < 1e-6 * scale,
                     name + ": J dq equals central differences (h = " + sci(h) + ") over " + std::to_string(steps) + " steps (" +
                         sci(worst / scale) + " of the largest; " + std::to_string(cells_off) + " of " + std::to_string(n) +
                         " cells at a switch, where it equals a one-sided difference to " + sci(worst_one_sided) + ")");
    // A reduced policy: its tangent is the double scheme's derivative at its own rows, so it differs from the double policy's by
    // the rows' rounding (FP16: 2^-11 of a group).
    double to_reference = 0;
    if (reference) {
        double largest = 0;
        for (int i = 0; i < n; ++i)
            for (int v = 0; v < V; ++v) {
                to_reference = std::max(to_reference, std::abs(tangent(i, v) - (*reference)(i, v)));
                largest = std::max(largest, std::abs((*reference)(i, v)));
            }
        to_reference /= largest;
        check.expect(to_reference < 1e-2, name + ": J dq against the double policy's over " + std::to_string(steps) + " steps (" +
                                              sci(to_reference) + " of the largest)");
    }
    // Adjoint, backwards from the checkpoints.
    typename F::Cells bar("bar", n);
    Kokkos::deep_copy(bar, wbar);
    for (int s = steps - 1; s >= 0; --s) {
        write(fv, path[s]);
        fv.step_adjoint(dts[s], bar);
    }
    HostCells adjoint("adjoint", n);
    Kokkos::deep_copy(adjoint, bar);
    double size_l = 0, size_r = 0;
    const double lhs = inner(wbar, tangent, &size_l), rhs = inner(adjoint, dq0, &size_r);
    const double dot = std::abs(lhs - rhs) / std::max(size_l, size_r);
    check.expect(dot < 1e-13, name + ": <w, J dq> = <J^T w, dq> over " + std::to_string(steps) + " steps (" + sci(dot) + ")");
    const HostCells after = read(fv);
    check.expect(std::memcmp(after.data(), path[0].data(), sizeof(double) * V * n) == 0, name + ": step_adjoint leaves Q as it was");
    // No atomics: a second reverse pass gives the same bits.
    Kokkos::deep_copy(bar, wbar);
    for (int s = steps - 1; s >= 0; --s) {
        write(fv, path[s]);
        fv.step_adjoint(dts[s], bar);
    }
    HostCells again("again", n);
    Kokkos::deep_copy(again, bar);
    check.expect(std::memcmp(again.data(), adjoint.data(), sizeof(double) * V * n) == 0, name + ": two reverse passes give the same bits");
    std::printf("step %-6s n %d steps %d: forward max diff %.2e%s, tangent-FD %.2e (cells at a switch %d, one-sided there %.2e), dot %.2e, "
                "against double %.2e\n",
                name.c_str(), n, steps, bits, same ? " (same bits)" : "", worst / scale, cells_off, worst_one_sided, dot, to_reference);
    return tangent;
}

// Times of the forward step, step_tangent and step_adjoint of policy F on n jittered cells; median of 5.
template <class F>
void cost(int n, const char* policy) {
    const hydro::Gas gas{5.0 / 3.0};
    const Points points = make_points_off_grid("jittered", n);
    typename F::Mesh mesh(Exec{}, 1.0);
    mesh.update(to_device(points));
    mesh.update_cells();
    F fv(Exec{}, mesh, gas, 1, hydro::Update::per_face);
    const auto volume = to_host(mesh.volume());
    const HostCells q0 = smooth_state(points, volume, gas);
    typename F::Cells dq("dq", n), bar("bar", n);
    Kokkos::deep_copy(dq, 1.0);
    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    auto time = [&](auto&& f) {
        std::vector<double> t;
        for (int r = 0; r < 6; ++r) {
            write(fv, q0);
            Kokkos::fence();
            Kokkos::Timer timer;
            f();
            Kokkos::fence();
            if (r > 0) t.push_back(timer.seconds() * 1e3);
        }
        return median(t);
    };
    write(fv, q0);
    const double dt = fv.time_step(0.3);
    fv.step(dt);
    const double forward = time([&] { fv.step(fv.time_step(0.3)); });
    const double step_only = time([&] {
        fv.time_step(0.3);
        Kokkos::fence();
    });
    // As a run would: time_step() (sweep and dt), then the step with its tangent.
    const double tangent = time([&] { fv.step_tangent(fv.time_step_tangent(0.3, dq), dq); });
    const double unfused = time([&] { fv.step_tangent(fv.time_step(0.3), dq); });  // the tangent's sweep in a kernel of its own
    const double adjoint = time([&] {
        Kokkos::deep_copy(bar, 1.0);
        fv.step_adjoint(dt, bar);
    });
    hydro::Timings tt, ta;
    write(fv, q0);
    fv.step_tangent(fv.time_step_tangent(0.3, dq, &tt), dq, &tt);
    write(fv, q0);
    fv.step_adjoint(dt, bar, &ta);
    std::printf("cost n %d faces %d (%s): time_step+step %.3f ms (time_step alone %.3f), step_tangent %.3f (%.2fx; unfused sweep %.3f, %.2fx), step_adjoint %.3f (%.2fx)\n",
                n, fv.faces(), policy, forward, step_only, tangent, tangent / forward, unfused, unfused / forward, adjoint, adjoint / forward);
    std::printf("  tangent phases: primitives %.3f sweep %.3f fluxes %.3f update %.3f ms\n", tt.primitives, tt.sweep, tt.fluxes, tt.update);
    std::printf("  adjoint phases: forward recompute %.3f fluxes %.3f sweep %.3f gather %.3f ms\n", ta.primitives, ta.fluxes, ta.sweep, ta.update);
    hydro::Timings tf;
    write(fv, q0);
    fv.step(fv.time_step(0.3, &tf), &tf);
    std::printf("  forward phases: primitives %.3f sweep+dt %.3f fluxes %.3f update %.3f ms\n", tf.primitives, tf.sweep, tf.fluxes, tf.update);
}

// ---- 1D (finite_volume_1d.hpp): the gradient of each limiter as a piece, and whole static steps with every kind of wall.
using FV1 = hydro::FiniteVolume<1, Exec, hydro::HLLC>;
using FV1HLL = hydro::FiniteVolume<1, Exec, hydro::HLL>;
using State1 = hydro::State<1>;
using HostCells1 = Kokkos::View<double* [3], Kokkos::LayoutRight, Kokkos::HostSpace>;

// Stencil::gradient(): (W of the left neighbour, the cell, the right neighbour) -> the limited slope of each variable, gaps and centroids
// fixed. Category: bit 0 some variable clipped (face centroid) or with a one-signed slope pair (minmod, van Leer), bit 1 some to zero.
struct GradientPiece1D {
    static constexpr int inputs = 9, outputs = 3;
    struct Sample {
        double x[9];
        double gapL, gapR, c[3];
    };
    hydro::Limiter limiter;
    using Cell1 = FV1::Stencil::Cell;
    KOKKOS_INLINE_FUNCTION Sample make(std::uint64_t& seed) const {
        Sample s;
        const double Wi[3] = {uniform(seed, 0.2, 2), uniform(seed, -1, 1), uniform(seed, 0.2, 2)};
        for (int k = 0; k < 3; ++k)
            for (int v = 0; v < 3; ++v) s.x[3 * k + v] = k == 1 ? Wi[v] : Wi[v] + uniform(seed, -0.5, 0.5) * (v == 1 ? 1.0 : Wi[v]);
        s.gapL = uniform(seed, 0.5, 1.5);
        s.gapR = uniform(seed, 0.5, 1.5);
        for (int k = 0; k < 3; ++k) s.c[k] = uniform(seed, -0.15, 0.15);
        return s;
    }
    KOKKOS_INLINE_FUNCTION void cells(const Sample& s, const double* x, Cell1 (&cell)[3]) const {
        for (int k = 0; k < 3; ++k) {
            cell[k].W = State1{x[3 * k], x[3 * k + 1], x[3 * k + 2]};
            cell[k].c = s.c[k];
        }
    }
    KOKKOS_INLINE_FUNCTION FV1::Stencil stencil() const {
        FV1::Stencil st{};
        st.limiter = limiter;
        return st;
    }
    KOKKOS_INLINE_FUNCTION int used_inputs(const Sample&) const { return inputs; }
    KOKKOS_INLINE_FUNCTION long long f(const Sample& s, const double* x, double* y) const {
        Cell1 cell[3];
        cells(s, x, cell);
        const FV1::Stencil st = stencil();
        long long code = 0;
        for (int v = 0; v < 3; ++v) {
            y[v] = st.gradient(cell + 1, s.gapL, s.gapR, v);
            const double dL = -s.gapL + cell[0].c - cell[1].c, dR = s.gapR + cell[2].c - cell[1].c;
            const double sL = (cell[0].W[v] - cell[1].W[v]) / dL, sR = (cell[2].W[v] - cell[1].W[v]) / dR;
            int last = -1;
            bool upper = false, zero = false;
            if (limiter == hydro::Limiter::face_centroid) st.clips(cell + 1, s.gapL, s.gapR, v, last, upper, zero);
            const int c = (sL * sR > 0) | (Kokkos::fabs(sL) < Kokkos::fabs(sR)) << 1 | (last + 1) << 2 | upper << 4 | zero << 5 |
                          (cell[2].W[v] > cell[0].W[v]) << 6;
            code = code * 128 + c;
        }
        return code;
    }
    KOKKOS_INLINE_FUNCTION void tangent(const Sample& s, const double* x, const double* dx, double* dy) const {
        Cell1 cell[3];
        cells(s, x, cell);
        const FV1::Stencil st = stencil();
        for (int v = 0; v < 3; ++v) dy[v] = st.gradient_tangent(cell + 1, s.gapL, s.gapR, v, dx[v], dx[3 + v], dx[6 + v]);
    }
    KOKKOS_INLINE_FUNCTION void adjoint(const Sample& s, const double* x, const double* ybar, double* xbar) const {
        Cell1 cell[3];
        cells(s, x, cell);
        const FV1::Stencil st = stencil();
        for (int v = 0; v < 3; ++v) st.gradient_adjoint(cell + 1, s.gapL, s.gapR, v, ybar[v], xbar[v], xbar[3 + v], xbar[6 + v]);
    }
    KOKKOS_INLINE_FUNCTION int category(long long pattern) const {
        int clipped = 0, zero = 0;
        for (int v = 0; v < 3; ++v, pattern /= 128) {
            const int c = static_cast<int>(pattern % 128);
            clipped |= limiter == hydro::Limiter::face_centroid ? (c >> 2 & 3) != 0 : c & 1;
            zero |= limiter == hydro::Limiter::face_centroid ? (c >> 5 & 1) : !(c & 1);
        }
        return clipped + 2 * zero;
    }
};

// A static 1D line of n cells on [0, 1] (periodic, or walls), its state smooth with noise of relative size `noise`.
struct Line1D {
    voronoi::Mesh<1, Exec> mesh;
    Line1D(int n, bool periodic) : mesh(Exec{}, 1.0, periodic ? voronoi::Ends::periodic : voronoi::Ends::walls) {
        Kokkos::View<double* [1], Exec::memory_space> x("x", n);
        auto h = Kokkos::create_mirror_view(x);
        for (int i = 0; i < n; ++i) h(i, 0) = (i + 0.5) / n;
        Kokkos::deep_copy(x, h);
        mesh.update(x);
        mesh.update_cells();
    }
};
HostCells1 state_1d(voronoi::Mesh<1, Exec>& mesh, const hydro::Gas& gas, double noise) {
    std::mt19937_64 rng(23);
    std::uniform_real_distribution<double> unit(-noise, noise);
    const auto x = to_host(mesh.positions());
    const auto c = to_host(mesh.centroid());
    const auto V = to_host(mesh.volume());
    const int n = static_cast<int>(V.extent(0));
    HostCells1 h("state", n);
    const double k = 2 * M_PI;
    for (int i = 0; i < n; ++i) {
        const double p = x(i, 0) + c(i, 0);
        const State1 W{1 + 0.2 * std::sin(k * p) + unit(rng), 0.3 * std::sin(k * p + 0.5) + unit(rng), 1 + 0.2 * std::cos(k * p) + unit(rng)};
        const State1 U = hydro::conserved(W, gas);
        for (int v = 0; v < 3; ++v) h(i, v) = U[v] * V(i);
    }
    return h;
}

// Whole static 1D steps: forward bits against step(), J dq against central differences (one-sided at a switch, `rough`), the dot test.
template <class F>
void step_test_1d(Check& check, const std::string& name, int n, int steps, bool periodic, hydro::Walls walls, hydro::Limiter limiter, int order,
                  double noise = 0.01) {
    const hydro::Gas gas{5.0 / 3.0};
    Line1D line(n, periodic);
    F fv(Exec{}, line.mesh, gas, order, limiter, hydro::Motion::static_mesh, walls);
    const HostCells1 q0 = state_1d(line.mesh, gas, noise);
    auto write1 = [&](const HostCells1& h) { Kokkos::deep_copy(fv.state(), h); };
    auto read1 = [&]() {
        HostCells1 h("read", n);
        Kokkos::deep_copy(h, fv.state());
        return h;
    };
    std::mt19937_64 rng(31);
    std::uniform_real_distribution<double> unit(-1, 1);
    HostCells1 dq0("dq0", n), wbar("wbar", n);
    for (int i = 0; i < n; ++i)
        for (int v = 0; v < 3; ++v) {
            dq0(i, v) = unit(rng) * std::abs(q0(i, v)) + 0.1 * unit(rng) / n;
            wbar(i, v) = unit(rng);
        }
    write1(q0);
    std::vector<double> dts;
    std::vector<HostCells1> checkpoints{q0};
    for (int s = 0; s < steps; ++s) {
        dts.push_back(fv.time_step(0.8, hydro::TimeStep::luna));
        fv.step(dts.back());
        checkpoints.push_back(read1());
    }
    auto run = [&](const HostCells1& start) {
        write1(start);
        for (int s = 0; s < steps; ++s) fv.step(dts[s]);
        return read1();
    };
    auto combine1 = [&](const HostCells1& a, double s, const HostCells1& b) {
        HostCells1 c("c", n);
        for (int i = 0; i < n; ++i)
            for (int v = 0; v < 3; ++v) c(i, v) = a(i, v) + s * b(i, v);
        return c;
    };
    write1(q0);
    typename F::Cells dq("dq", n);
    Kokkos::deep_copy(dq, dq0);
    std::vector<HostCells1> path;
    for (int s = 0; s < steps; ++s) {
        path.push_back(read1());
        fv.step_tangent(dts[s], dq);
    }
    const HostCells1 forward = read1();
    HostCells1 tangent("tangent", n);
    Kokkos::deep_copy(tangent, dq);
    double bits = 0;
    for (int i = 0; i < n; ++i)
        for (int v = 0; v < 3; ++v)
            bits = std::max(bits, std::abs(forward(i, v) - checkpoints.back()(i, v)) / std::max(std::abs(checkpoints.back()(i, v)), 1e-300));
    check.expect(bits < 1e-11, name + ": step_tangent's Q equals step()'s over " + std::to_string(steps) + " steps (" + sci(bits) + ")");
    const double h = 1e-6;
    const HostCells1 plus = run(combine1(q0, h, dq0)), minus = run(combine1(q0, -h, dq0));
    const HostCells1& at = checkpoints.back();
    double worst = 0, scale = 0, worst_one_sided = 0;
    int cells_off = 0;
    for (int i = 0; i < n; ++i)
        for (int v = 0; v < 3; ++v) scale = std::max(scale, std::abs(plus(i, v) - minus(i, v)) / (2 * h));
    for (int i = 0; i < n; ++i) {
        double cell_diff = 0, cell_scale = 1e-6 * scale, plus_diff = 0, minus_diff = 0;
        for (int v = 0; v < 3; ++v) {
            const double fd = (plus(i, v) - minus(i, v)) / (2 * h);
            cell_diff = std::max(cell_diff, std::abs(tangent(i, v) - fd));
            cell_scale = std::max(cell_scale, std::abs(fd));
            plus_diff = std::max(plus_diff, std::abs(tangent(i, v) - (plus(i, v) - at(i, v)) / h));
            minus_diff = std::max(minus_diff, std::abs(tangent(i, v) - (at(i, v) - minus(i, v)) / h));
        }
        worst = std::max(worst, cell_diff);
        if (cell_diff > 1e-6 * cell_scale) {
            ++cells_off;
            worst_one_sided = std::max(worst_one_sided, std::min(plus_diff, minus_diff) / cell_scale);
        }
    }
    const bool rough = noise > 0.05;
    check.expect(rough ? worst_one_sided < 1e-4 : worst < 1e-6 * scale,
                 name + ": J dq equals central differences (h = 1e-6) over " + std::to_string(steps) + " steps (" + sci(worst / scale) + "; " +
                     std::to_string(cells_off) + " of " + std::to_string(n) + " cells at a switch, one-sided there " + sci(worst_one_sided) + ")");
    typename F::Cells bar("bar", n);
    Kokkos::deep_copy(bar, wbar);
    for (int s = steps - 1; s >= 0; --s) {
        write1(path[s]);
        fv.step_adjoint(dts[s], bar);
    }
    HostCells1 adjoint("adjoint", n);
    Kokkos::deep_copy(adjoint, bar);
    double lhs = 0, rhs = 0, size = 0;
    for (int i = 0; i < n; ++i)
        for (int v = 0; v < 3; ++v) {
            lhs += wbar(i, v) * tangent(i, v);
            rhs += adjoint(i, v) * dq0(i, v);
            size += std::abs(wbar(i, v) * tangent(i, v)) + std::abs(adjoint(i, v) * dq0(i, v));
        }
    const double dot = std::abs(lhs - rhs) / size;
    check.expect(dot < 1e-13, name + ": <w, J dq> = <J^T w, dq> over " + std::to_string(steps) + " steps (" + sci(dot) + ")");
    std::printf("step1d %-34s n %d steps %d: forward %.2e, tangent-FD %.2e (cells at a switch %d, one-sided %.2e), dot %.2e\n", name.c_str(), n, steps,
                bits, worst / scale, cells_off, worst_one_sided, dot);
}

// LUNA's JAX derivatives (luna_derivative_values.hpp) against ours: the same problem, N, limiter (LUNA's 0 minmod, 1 MC = our face
// centroid, 2 van Leer), HLLC, dt and steps; U per volume, so J dU = (J dq) / V and J^T w = (J^T (w / V)) V on our extensive Q.
// Prints the largest differences relative to LUNA's largest entry, and the cells where they exceed 1e-8 (a branch either code takes
// differently: a flat state, a tie). Used by: tests_1d().
void luna_case(Check& check, const char* problem, int n, int limiter, int steps, double dt, const double (*U0)[3], const double (*dU)[3],
               const double (*w)[3], const double (*U)[3], const double (*JdU)[3], const double (*JTw)[3], bool gate = true) {
    const auto pb = hydro::problems_1d::problem(problem);
    const hydro::Gas gas{pb.gamma};
    voronoi::Mesh<1, Exec> mesh(Exec{}, pb.L, pb.ends);
    Kokkos::View<double* [1], Exec::memory_space> x("x", n);
    auto hx = Kokkos::create_mirror_view(x);
    for (int i = 0; i < n; ++i) hx(i, 0) = pb.L * (i + 0.5) / n;
    Kokkos::deep_copy(x, hx);
    mesh.update(x);
    mesh.update_cells();
    const hydro::Limiter lim = limiter == 0 ? hydro::Limiter::minmod : limiter == 2 ? hydro::Limiter::van_leer : hydro::Limiter::face_centroid;
    FV1 fv(Exec{}, mesh, gas, 1, lim, hydro::Motion::static_mesh, pb.walls);
    const auto V = to_host(mesh.volume());
    HostCells1 q0("q0", n), dq0("dq0", n), bar0("bar0", n);
    for (int i = 0; i < n; ++i)
        for (int v = 0; v < 3; ++v) {
            q0(i, v) = U0[i][v] * V(i);
            dq0(i, v) = dU[i][v] * V(i);
            bar0(i, v) = w[i][v] / V(i);
        }
    Kokkos::deep_copy(fv.state(), q0);
    typename FV1::Cells dq("dq", n), bar("bar", n);
    Kokkos::deep_copy(dq, dq0);
    std::vector<HostCells1> path;
    for (int s = 0; s < steps; ++s) {
        HostCells1 h("path", n);
        Kokkos::deep_copy(h, fv.state());
        path.push_back(h);
        fv.step_tangent(dt, dq);
    }
    HostCells1 Q("Q", n), J("J", n), JT("JT", n);
    Kokkos::deep_copy(Q, fv.state());
    Kokkos::deep_copy(J, dq);
    Kokkos::deep_copy(bar, bar0);
    for (int s = steps - 1; s >= 0; --s) {
        Kokkos::deep_copy(fv.state(), path[s]);
        fv.step_adjoint(dt, bar);
    }
    Kokkos::deep_copy(JT, bar);
    double scale[3] = {0, 0, 0}, diff[3] = {0, 0, 0};
    int cells[3] = {0, 0, 0};
    for (int k = 0; k < 3; ++k) {
        for (int i = 0; i < n; ++i)
            for (int v = 0; v < 3; ++v) scale[k] = std::max(scale[k], std::abs(k == 0 ? U[i][v] : k == 1 ? JdU[i][v] : JTw[i][v]));
        for (int i = 0; i < n; ++i) {
            double cell = 0;
            for (int v = 0; v < 3; ++v) {
                const double ours = k == 0 ? Q(i, v) / V(i) : k == 1 ? J(i, v) / V(i) : JT(i, v) * V(i);
                const double theirs = k == 0 ? U[i][v] : k == 1 ? JdU[i][v] : JTw[i][v];
                cell = std::max(cell, std::abs(ours - theirs) / scale[k]);
            }
            diff[k] = std::max(diff[k], cell);
            cells[k] += cell > 1e-8;
        }
    }
    const std::string name = std::string(problem) + " N = " + std::to_string(n) + " LIMITER = " + std::to_string(limiter);
    std::printf("luna %-32s forward %.2e, J dU %.2e (%d cells above 1e-8), J^T w %.2e (%d cells)\n", name.c_str(), diff[0], diff[1], cells[1],
                diff[2], cells[2]);
    check.expect(diff[0] < 1e-12, name + ": our forward equals LUNA's after " + std::to_string(steps) + " steps (" + sci(diff[0]) + ")");
    // 1e-8: van Leer's derivative divides by (sL + sR)^2, and on the 1e-6 sound wave the two codes' arrangements round apart (7e-10).
    if (gate) check.expect(diff[1] < 1e-8 && diff[2] < 1e-8, name + ": J dU and J^T w equal LUNA's jax.jvp and jax.vjp (" + sci(diff[1]) + ", " +
                                                         sci(diff[2]) + ")");
}

// The 1D tests. Used by: main.
void tests_1d(Check& check, int samples) {
    using L = hydro::Limiter;
    using W = hydro::Wall;
    piece_test(check, "1D gradient MC", GradientPiece1D{L::face_centroid}, samples, "unclipped/clipped/zero/both:");
    piece_test(check, "1D gradient minmod", GradientPiece1D{L::minmod}, samples, "-/one-signed/zero/-:");
    piece_test(check, "1D gradient van Leer", GradientPiece1D{L::van_leer}, samples, "-/one-signed/zero/-:");
    const hydro::Walls periodic{}, outflow{W::outflow, W::outflow}, reflective{W::reflective, W::reflective};
    for (const auto& [limiter, label] : {std::pair{L::face_centroid, "MC"}, std::pair{L::minmod, "minmod"}, std::pair{L::van_leer, "van Leer"}}) {
        step_test_1d<FV1>(check, std::string("HLLC ") + label + " periodic", 64, 5, true, periodic, limiter, 1);
        step_test_1d<FV1>(check, std::string("HLLC ") + label + " outflow", 64, 5, false, outflow, limiter, 1);
        step_test_1d<FV1>(check, std::string("HLLC ") + label + " reflective", 64, 5, false, reflective, limiter, 1);
    }
    step_test_1d<FV1HLL>(check, "HLL MC outflow", 64, 5, false, outflow, L::face_centroid, 1);
    step_test_1d<FV1>(check, "HLLC order 0 reflective", 64, 5, false, reflective, L::face_centroid, 0);
    step_test_1d<FV1>(check, "HLLC MC rough reflective", 64, 5, false, reflective, L::face_centroid, 1, 0.3);
    // Against LUNA's jax.jvp and jax.vjp of the same five steps (luna_derivative_values.hpp, scripts/luna_derivatives.py).
#define LUNA_CASE(problem, ns, ...) luna_case(check, problem, ns::n, ns::limiter, ns::steps, ns::dt, ns::U0, ns::dU, ns::w, ns::U, ns::JdU, ns::JTw __VA_OPT__(, ) __VA_ARGS__)
    // Sod with MC starts exactly flat either side of the jump, where MC has no derivative: LUNA's jvp takes the zero branch of its
    // where(a b > 0, ...), ours the unlimited central slope (as LUNA's own limiter_tangent_eps gives for minmod). Reported, not checked;
    // the same problem with no cell exactly flat (noisy) is checked.
    LUNA_CASE("sod", luna_d_sod_64_l1, false);
    LUNA_CASE("sod", luna_d_sod_64_l1_noisy);
    LUNA_CASE("sod", luna_d_sod_64_l2_noisy);
    LUNA_CASE("sound-right", luna_d_sound_right_64_l2);
    LUNA_CASE("sod", luna_d_sod_64_l0);
    LUNA_CASE("sound-right", luna_d_sound_right_64_l1);
    LUNA_CASE("sound-right", luna_d_sound_right_64_l0);
#undef LUNA_CASE
}

// ---- 2D (FiniteVolume<2>: the same templates as 3D): whole static steps on a jittered 2D mesh, double and grad32 (the 2D default).
using FV2 = hydro::FiniteVolume<2, Exec, hydro::HLLC, hydro::Precision<>>;
using FV2HLL = hydro::FiniteVolume<2, Exec, hydro::HLL, hydro::Precision<>>;
using FV2_32 = hydro::FiniteVolume<2, Exec, hydro::HLLC, hydro::Grad32>;
using State2 = hydro::State<2>;
using HostCells2 = Kokkos::View<double* [4], Kokkos::LayoutRight, Kokkos::HostSpace>;

// The checks of step_test() on an nx x nx jittered (0.3) periodic 2D mesh. Returns the tangent; reference: the double policy's.
template <class F>
HostCells2 step_test_2d(Check& check, const std::string& name, int nx, int steps, double noise = 0.01, const HostCells2* reference = nullptr) {
    const hydro::Gas gas{5.0 / 3.0};
    const exact_predicates::BoxN<2> box(1.0);
    const int n = nx * nx;
    Kokkos::View<double* [2], Exec::memory_space> x("x", n);
    auto hx = Kokkos::create_mirror_view(x);
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> jit(-0.15, 0.15), unit(-1, 1), small(-noise, noise);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < nx; ++j) {
            hx(i * nx + j, 0) = (i + 0.5 + jit(rng)) / nx;
            hx(i * nx + j, 1) = (j + 0.5 + jit(rng)) / nx;
        }
    Kokkos::deep_copy(x, hx);
    typename F::Mesh mesh(Exec{}, box);
    mesh.update(x);
    mesh.update_cells();
    F fv(Exec{}, mesh, gas, 1, hydro::Update::per_face);
    const auto V = to_host(mesh.volume());
    const auto pos = to_host(mesh.positions());
    const auto cen = to_host(mesh.centroid());
    HostCells2 q0("q0", n), dq0("dq0", n), wbar("wbar", n);
    const double k = 2 * M_PI;
    for (int i = 0; i < n; ++i) {
        const double px = pos(i, 0) + cen(i, 0), py = pos(i, 1) + cen(i, 1);
        const State2 W{1 + 0.2 * std::sin(k * px) + small(rng), 0.3 * std::sin(k * py) + small(rng), 0.2 * std::cos(k * px) + small(rng),
                       1 + 0.2 * std::cos(k * (px + py)) + small(rng)};
        const State2 U = hydro::conserved(W, gas);
        for (int v = 0; v < 4; ++v) q0(i, v) = U[v] * V(i);
    }
    for (int i = 0; i < n; ++i)
        for (int v = 0; v < 4; ++v) {
            dq0(i, v) = unit(rng) * std::abs(q0(i, v)) + 0.1 * unit(rng) * V(i);
            wbar(i, v) = unit(rng);
        }
    auto write2 = [&](const HostCells2& h) { Kokkos::deep_copy(fv.state(), h); };
    auto read2 = [&]() {
        HostCells2 h("read", n);
        Kokkos::deep_copy(h, fv.state());
        return h;
    };
    write2(q0);
    std::vector<double> dts;
    std::vector<HostCells2> checkpoints{q0};
    for (int s = 0; s < steps; ++s) {
        dts.push_back(fv.time_step(0.3));
        fv.step(dts.back());
        checkpoints.push_back(read2());
    }
    auto run = [&](const HostCells2& start) {
        write2(start);
        for (int s = 0; s < steps; ++s) fv.step(dts[s]);
        return read2();
    };
    auto combine2 = [&](const HostCells2& a, double s, const HostCells2& b) {
        HostCells2 c("c", n);
        for (int i = 0; i < n; ++i)
            for (int v = 0; v < 4; ++v) c(i, v) = a(i, v) + s * b(i, v);
        return c;
    };
    write2(q0);
    typename F::Cells dq("dq", n);
    Kokkos::deep_copy(dq, dq0);
    std::vector<HostCells2> path;
    for (int s = 0; s < steps; ++s) {
        path.push_back(read2());
        fv.time_step_tangent(0.3, dq);
        fv.step_tangent(dts[s], dq);
    }
    const HostCells2 forward = read2();
    HostCells2 tangent("tangent", n);
    Kokkos::deep_copy(tangent, dq);
    double bits = 0;
    for (int i = 0; i < n; ++i)
        for (int v = 0; v < 4; ++v)
            bits = std::max(bits, std::abs(forward(i, v) - checkpoints.back()(i, v)) / std::max(std::abs(checkpoints.back()(i, v)), 1e-300));
    constexpr bool exact = std::is_same_v<typename F::G, double>;
    check.expect(bits < (exact ? 1e-11 : 1e-5), name + ": step_tangent's Q equals step()'s over " + std::to_string(steps) + " steps (" + sci(bits) + ")");
    const double h = 1e-7;  // the jittered mesh puts clips within 1e-6 of a switch; their spread over 3 steps reaches ~60 cells
    const HostCells2 plus = run(combine2(q0, h, dq0)), minus = run(combine2(q0, -h, dq0));
    // As step_test(): central for every cell; at a cell where it differs (a switch between x - h dq and x + h dq), one of the
    // one-sided differences must match.
    const HostCells2& at = checkpoints.back();
    double worst = 0, scale = 0, worst_one_sided = 0;
    int cells_off = 0, worst_cell = 0;
    for (int i = 0; i < n; ++i)
        for (int v = 0; v < 4; ++v) scale = std::max(scale, std::abs(plus(i, v) - minus(i, v)) / (2 * h));
    for (int i = 0; i < n; ++i) {
        double cell_diff = 0, cell_scale = 1e-6 * scale, plus_diff = 0, minus_diff = 0;
        for (int v = 0; v < 4; ++v) {
            const double fd = (plus(i, v) - minus(i, v)) / (2 * h);
            cell_diff = std::max(cell_diff, std::abs(tangent(i, v) - fd));
            cell_scale = std::max(cell_scale, std::abs(fd));
            plus_diff = std::max(plus_diff, std::abs(tangent(i, v) - (plus(i, v) - at(i, v)) / h));
            minus_diff = std::max(minus_diff, std::abs(tangent(i, v) - (at(i, v) - minus(i, v)) / h));
        }
        if (cell_diff > worst) worst_cell = i;
        worst = std::max(worst, cell_diff);
        if (cell_diff > 1e-6 * cell_scale) {
            ++cells_off;
            worst_one_sided = std::max(worst_one_sided, std::min(plus_diff, minus_diff) / cell_scale);
        }
    }
    if (cells_off > 0)
        for (int v = 0; v < 4; ++v)
            std::printf("  worst cell %d v %d: tangent %.10e  central %.10e  one-sided +h %.10e  -h %.10e\n", worst_cell, v, tangent(worst_cell, v),
                        (plus(worst_cell, v) - minus(worst_cell, v)) / (2 * h), (plus(worst_cell, v) - at(worst_cell, v)) / h,
                        (at(worst_cell, v) - minus(worst_cell, v)) / h);
    if (exact)
        check.expect(worst_one_sided < 1e-4,
                     name + ": J dq equals central differences (h = 1e-7) over " + std::to_string(steps) + " steps (" + sci(worst / scale) +
                         " of the largest; " + std::to_string(cells_off) + " of " + std::to_string(n) +
                         " cells at a switch, where it equals a one-sided difference to " + sci(worst_one_sided) + ")");
    double to_reference = 0;
    if (reference) {
        double largest = 0;
        for (int i = 0; i < n; ++i)
            for (int v = 0; v < 4; ++v) {
                to_reference = std::max(to_reference, std::abs(tangent(i, v) - (*reference)(i, v)));
                largest = std::max(largest, std::abs((*reference)(i, v)));
            }
        to_reference /= largest;
        check.expect(to_reference < 1e-4, name + ": J dq against the double policy's (" + sci(to_reference) + ")");
    }
    typename F::Cells bar("bar", n);
    Kokkos::deep_copy(bar, wbar);
    for (int s = steps - 1; s >= 0; --s) {
        write2(path[s]);
        fv.step_adjoint(dts[s], bar);
    }
    HostCells2 adjoint("adjoint", n);
    Kokkos::deep_copy(adjoint, bar);
    double lhs = 0, rhs = 0, size = 0;
    for (int i = 0; i < n; ++i)
        for (int v = 0; v < 4; ++v) {
            lhs += wbar(i, v) * tangent(i, v);
            rhs += adjoint(i, v) * dq0(i, v);
            size += std::abs(wbar(i, v) * tangent(i, v)) + std::abs(adjoint(i, v) * dq0(i, v));
        }
    const double dot = std::abs(lhs - rhs) / size;
    check.expect(dot < 1e-13, name + ": <w, J dq> = <J^T w, dq> over " + std::to_string(steps) + " steps (" + sci(dot) + ")");
    std::printf("step2d %-14s n %d steps %d: forward %.2e, tangent-FD %.2e, dot %.2e, against double %.2e\n", name.c_str(), n, steps, bits,
                worst / scale, dot, to_reference);
    return tangent;
}

// The 2D tests. Used by: main.
void tests_2d(Check& check) {
    const HostCells2 reference = step_test_2d<FV2>(check, "2D HLLC", 32, 3);
    step_test_2d<FV2HLL>(check, "2D HLL", 32, 3);
    step_test_2d<FV2_32>(check, "2D HLLC grad32", 32, 3, 0.01, &reference);
}

// One warm-up of each, then time_step + step, time_step + step_tangent, step_adjoint once each: the kernels ncu measures.
template <class F>
void profile(int n) {
    const hydro::Gas gas{5.0 / 3.0};
    const Points points = make_points_off_grid("jittered", n);
    typename F::Mesh mesh(Exec{}, 1.0);
    mesh.update(to_device(points));
    mesh.update_cells();
    F fv(Exec{}, mesh, gas, 1, hydro::Update::per_face);
    const HostCells q0 = smooth_state(points, to_host(mesh.volume()), gas);
    typename F::Cells dq("dq", n), bar("bar", n);
    Kokkos::deep_copy(dq, 1.0);
    for (int r = 0; r < 2; ++r) {
        write(fv, q0);
        const double dt = fv.time_step(0.3);
        fv.step(dt);
        write(fv, q0);
        fv.step_tangent(fv.time_step_tangent(0.3, dq), dq);
        write(fv, q0);
        Kokkos::deep_copy(bar, 1.0);
        fv.step_adjoint(dt, bar);
    }
    Kokkos::fence();
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    for (int a = 1; a + 1 < argc; ++a)
        if (std::string(argv[a]) == "--cost") {
            cost<FV>(std::stoi(argv[a + 1]), "double");
            cost<FV16>(std::stoi(argv[a + 1]), "grad16");
            return 0;
        } else if (std::string(argv[a]) == "--profile") {  // for ncu: one warm-up, then one forward, tangent and adjoint step each
            profile<FV>(std::stoi(argv[a + 1]));
            profile<FV16>(std::stoi(argv[a + 1]));
            return 0;
        }
    Check check;
    // `--scan`: the rough step's cells off against h; where differences cross switches they fall with h, a wrong derivative does not.
    for (int a = 1; a < argc; ++a)
        if (std::string(argv[a]) == "--scan") {
            for (const double h : {1e-5, 1e-6, 1e-7, 1e-8, 1e-9}) {
                step_test<FV>(check, "HLLC rough h " + sci(h), 1000, 1, 0.3, h);
                step_test<FVHLL>(check, "HLL rough h " + sci(h), 1000, 1, 0.3, h);
            }
            return 0;
        }
    const hydro::Gas gas{5.0 / 3.0};
    const int samples = 20000;
    piece_test(check, "primitive", PrimitivePiece{gas}, samples, "floors none/rho/e/both:");
    piece_test(check, "face side", SidePiece{gas}, samples, "extrapolated/fallback:");
    piece_test(check, "HLL", FluxPiece<hydro::HLL>{gas}, samples, "star/F_L/F_R/-:");
    piece_test(check, "HLLC", FluxPiece<hydro::HLLC>{gas}, samples, "F_L/F*_L/F*_R/F_R:");
    piece_test(check, "limited gradient", GradientPiece{}, samples, "unclipped/clipped/zero/both:");
    step_test<FV>(check, "HLLC", 1000, 1);
    const HostCells reference = step_test<FV>(check, "HLLC", 1000, 3);
    step_test<FVHLL>(check, "HLL", 1000, 1);
    step_test<FV>(check, "HLLC rough", 1000, 3, 0.3);
    step_test<FV>(check, "HLLC unfused", 1000, 3, 0.01, 1e-6, nullptr, false);  // time_step, then step_tangent's own tangent sweep
    // The policies in use: grad16 (the 3D default) and grad32, rows as stored, the derivative in double.
    step_test<FV16>(check, "HLLC grad16", 1000, 3, 0.01, 1e-6, &reference);
    step_test<FV32>(check, "HLLC grad32", 1000, 3, 0.01, 1e-6, &reference);
    tests_1d(check, samples);
    tests_2d(check);
    return check.failed() != 0;
}
