// Checks the hand-written tangents and adjoints of the 3D hydro (euler.hpp, finite_volume.hpp): per piece on random states in a kernel,
// and whole static steps on a small mesh (FiniteVolume::step_tangent, step_adjoint). Tangent against central differences away from
// switches, adjoint against tangent (<w, J v> = <J^T w, v>). `--cost N`: times of step, step_tangent and step_adjoint on N cells.

#include "check.hpp"
#include "finite_volume.hpp"
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

// A whole static step (and `steps` steps): forward bits, tangent against central differences, adjoint against tangent.
template <class F>
void step_test(Check& check, const std::string& name, int n, int steps, double noise = 0.01, double h = 1e-6) {
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
    for (int s = 0; s < steps; ++s) fv.step_tangent(dts[s], dqd);
    const HostCells forward = read(fv);
    HostCells tangent("tangent", n);
    Kokkos::deep_copy(tangent, dqd);
    double bits = 0;
    for (int i = 0; i < n; ++i)
        for (int v = 0; v < V; ++v)
            bits = std::max(bits, std::abs(forward(i, v) - checkpoints.back()(i, v)) / std::max(std::abs(checkpoints.back()(i, v)), 1e-300));
    const bool same = std::memcmp(forward.data(), checkpoints.back().data(), sizeof(double) * V * n) == 0;
    // The same bits on the host; on the A100 the tangent's sweep contracts multiply-adds differently from Sweep (measured 8e-14
    // smooth, 5e-13 rough over 3 steps), as the two update loops do in `hydro`.
    check.expect(bits < 1e-11, name + ": step_tangent's Q equals step()'s over " + std::to_string(steps) + " steps (" + sci(bits) +
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
    check.expect(rough ? worst_one_sided < 1e-4 : worst < 1e-6 * scale,
                 name + ": J dq equals central differences (h = " + sci(h) + ") over " + std::to_string(steps) + " steps (" + sci(worst / scale) +
                     " of the largest; " + std::to_string(cells_off) + " of " + std::to_string(n) +
                     " cells at a switch, where it equals a one-sided difference to " + sci(worst_one_sided) + ")");
    // Adjoint, backwards from the checkpoints.
    typename F::Cells bar("bar", n);
    Kokkos::deep_copy(bar, wbar);
    for (int s = steps - 1; s >= 0; --s) {
        write(fv, checkpoints[s]);
        fv.step_adjoint(dts[s], bar);
    }
    HostCells adjoint("adjoint", n);
    Kokkos::deep_copy(adjoint, bar);
    double size_l = 0, size_r = 0;
    const double lhs = inner(wbar, tangent, &size_l), rhs = inner(adjoint, dq0, &size_r);
    const double dot = std::abs(lhs - rhs) / std::max(size_l, size_r);
    check.expect(dot < 1e-13, name + ": <w, J dq> = <J^T w, dq> over " + std::to_string(steps) + " steps (" + sci(dot) + ")");
    const HostCells after = read(fv);
    check.expect(std::memcmp(after.data(), checkpoints[0].data(), sizeof(double) * V * n) == 0, name + ": step_adjoint leaves Q as it was");
    // No atomics: a second reverse pass gives the same bits.
    Kokkos::deep_copy(bar, wbar);
    for (int s = steps - 1; s >= 0; --s) {
        write(fv, checkpoints[s]);
        fv.step_adjoint(dts[s], bar);
    }
    HostCells again("again", n);
    Kokkos::deep_copy(again, bar);
    check.expect(std::memcmp(again.data(), adjoint.data(), sizeof(double) * V * n) == 0, name + ": two reverse passes give the same bits");
    std::printf("step %-6s n %d steps %d: forward max diff %.2e%s, tangent-FD %.2e (cells at a switch %d, one-sided there %.2e), dot %.2e\n",
                name.c_str(), n, steps, bits, same ? " (same bits)" : "", worst / scale, cells_off, worst_one_sided, dot);
}

// Times of the forward step (double and the default policy), step_tangent and step_adjoint on n jittered cells; median of 5.
void cost(int n) {
    const hydro::Gas gas{5.0 / 3.0};
    const Points points = make_points_off_grid("jittered", n);
    FV::Mesh mesh(Exec{}, 1.0);
    mesh.update(to_device(points));
    mesh.update_cells();
    FV fv(Exec{}, mesh, gas, 1, hydro::Update::per_face);
    const auto volume = to_host(mesh.volume());
    const HostCells q0 = smooth_state(points, volume, gas);
    FV::Cells dq("dq", n), bar("bar", n);
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
    const double tangent = time([&] { fv.step_tangent(dt, dq); });
    const double adjoint = time([&] {
        Kokkos::deep_copy(bar, 1.0);
        fv.step_adjoint(dt, bar);
    });
    hydro::Timings tt, ta;
    write(fv, q0);
    fv.step_tangent(dt, dq, &tt);
    write(fv, q0);
    fv.step_adjoint(dt, bar, &ta);
    std::printf("cost n %d faces %d (double): time_step+step %.3f ms (time_step alone %.3f), step_tangent %.3f (%.2fx), step_adjoint %.3f (%.2fx)\n",
                n, fv.faces(), forward, step_only, tangent, tangent / forward, adjoint, adjoint / forward);
    std::printf("  tangent phases: primitives %.3f sweep %.3f fluxes %.3f update %.3f ms\n", tt.primitives, tt.sweep, tt.fluxes, tt.update);
    std::printf("  adjoint phases: forward recompute %.3f fluxes %.3f sweep %.3f gather %.3f ms\n", ta.primitives, ta.fluxes, ta.sweep, ta.update);
    hydro::Timings tf;
    write(fv, q0);
    fv.step(fv.time_step(0.3, &tf), &tf);
    std::printf("  forward phases: primitives %.3f sweep+dt %.3f fluxes %.3f update %.3f ms\n", tf.primitives, tf.sweep, tf.fluxes, tf.update);
    // The default policy (grad16) on the same points, for scale.
    using Default = hydro::FiniteVolume<D, Exec>;
    Default::Mesh dmesh(Exec{}, 1.0);
    dmesh.update(to_device(points));
    dmesh.update_cells();
    Default dfv(Exec{}, dmesh, gas, 1, hydro::Update::per_face);
    std::vector<double> t;
    for (int r = 0; r < 6; ++r) {
        Kokkos::deep_copy(dfv.state(), q0);
        Kokkos::fence();
        Kokkos::Timer timer;
        dfv.step(dfv.time_step(0.3));
        Kokkos::fence();
        if (r > 0) t.push_back(timer.seconds() * 1e3);
    }
    std::printf("  default policy (grad16) time_step+step %.3f ms\n", median(t));
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    for (int a = 1; a + 1 < argc; ++a)
        if (std::string(argv[a]) == "--cost") {
            cost(std::stoi(argv[a + 1]));
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
    step_test<FV>(check, "HLLC", 1000, 3);
    step_test<FVHLL>(check, "HLL", 1000, 1);
    step_test<FV>(check, "HLLC rough", 1000, 3, 0.3);
    return check.failed() != 0;
}
