// Checks the hand-written tangents and adjoints of the 3D hydro (euler.hpp, finite_volume.hpp): per piece on random states in a kernel,
// and whole static steps on a small mesh (FiniteVolume::step_tangent, step_adjoint). Tangent against central differences away from
// switches, adjoint against tangent (<w, J v> = <J^T w, v>). `--cost N`: times of step, step_tangent and step_adjoint on N cells.

#include "build_info.hpp"
#include "check.hpp"
#include "finite_volume.hpp"
#include "finite_volume_1d.hpp"
#include "luna_derivative_values.hpp"
#include "problems_1d.hpp"
#include "reference_geometry_2d.hpp"
#include "support.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
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

// MovingFace: (L, R, A, w) -> |A| flux(L, R, A / |A|, w), every input free (the moving step's face). Category as FluxPiece.
template <class Riemann>
struct MovingFacePiece {
    static constexpr int inputs = 2 * V + 2 * D, outputs = V;
    using Face = hydro::MovingFace<Riemann, D>;
    struct Sample {
        double x[2 * V + 2 * D];
    };
    hydro::Gas gas;
    KOKKOS_INLINE_FUNCTION static Vec vec_at(const double* x) { return hydro::make_vec<D>([&](int a) { return x[a]; }); }
    KOKKOS_INLINE_FUNCTION Sample make(std::uint64_t& seed) const {
        Sample s;
        put(random_W(seed), s.x);
        put(random_W(seed), s.x + V);
        const Vec A = unit_vector(seed) * uniform(seed, 0.2, 2);
        for (int a = 0; a < D; ++a) {
            s.x[2 * V + a] = A[a];
            s.x[2 * V + D + a] = uniform(seed, -1, 1);
        }
        return s;
    }
    KOKKOS_INLINE_FUNCTION int used_inputs(const Sample&) const { return inputs; }
    KOKKOS_INLINE_FUNCTION long long f(const Sample&, const double* x, double* y) const {
        const Vec A = vec_at(x + 2 * V), w = vec_at(x + 2 * V + D);
        const double s = Kokkos::sqrt(dot(A, A));
        const Vec n = A * (1.0 / s);
        State F = Riemann::flux(state_at(x), state_at(x + V), n, w, gas);
        for (int v = 0; v < V; ++v) F[v] *= s;
        put(F, y);
        return Riemann::branches(state_at(x), state_at(x + V), n, w, gas);
    }
    KOKKOS_INLINE_FUNCTION void tangent(const Sample&, const double* x, const double* dx, double* dy) const {
        put(Face::tangent(state_at(x), state_at(x + V), vec_at(x + 2 * V), vec_at(x + 2 * V + D), state_at(dx), state_at(dx + V),
                          vec_at(dx + 2 * V), vec_at(dx + 2 * V + D), gas),
            dy);
    }
    KOKKOS_INLINE_FUNCTION void adjoint(const Sample&, const double* x, const double* ybar, double* xbar) const {
        State Lbar{}, Rbar{};
        Vec Abar{}, wbar{};
        Face::adjoint(state_at(x), state_at(x + V), vec_at(x + 2 * V), vec_at(x + 2 * V + D), gas, state_at(ybar), Lbar, Rbar, Abar, wbar);
        add(Lbar, xbar);
        add(Rbar, xbar + V);
        for (int a = 0; a < D; ++a) {
            xbar[2 * V + a] += Abar[a];
            xbar[2 * V + D + a] += wbar[a];
        }
    }
    KOKKOS_INLINE_FUNCTION int category(long long pattern) const { return FluxPiece<Riemann>{gas}.category(pattern); }
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
// The 2D reference geometry derivative (reference_geometry_2d.hpp) on an nx x nx jittered periodic mesh with the points on the
// mesh's grid: its forward values against the mesh's, the tangent against central differences of the mesh's own geometry (the
// points moved by whole grid steps, so the differences are exact up to rounding), the identities, the dot test.
void geometry_test_2d(Check& check, int nx) {
    using Mesh = FV2::Mesh;
    using GD = hydro::GeometryDerivative<2, Exec::memory_space>;
    const exact_predicates::BoxN<2> box(1.0);
    const int n = nx * nx;
    Kokkos::View<double* [2], Exec::memory_space> x("x", n);
    auto hx = Kokkos::create_mirror_view(x);
    std::mt19937_64 rng(11);
    std::uniform_real_distribution<double> jit(-0.15, 0.15), unit(-1, 1);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < nx; ++j) {
            hx(i * nx + j, 0) = std::floor((i + 0.5 + jit(rng)) / nx / box.step) * box.step;
            hx(i * nx + j, 1) = std::floor((j + 0.5 + jit(rng)) / nx / box.step) * box.step;
        }
    Kokkos::deep_copy(x, hx);
    Mesh mesh(Exec{}, box);
    mesh.update(x);
    mesh.update_cells();
    const reference::Geometry2D<Mesh, Exec::memory_space> ref(mesh);
    const int faces = mesh.faces();
    struct Values {
        Kokkos::View<double*, Kokkos::HostSpace> V;
        Kokkos::View<double* [2], Kokkos::LayoutRight, Kokkos::HostSpace> c, A, xf;
        std::vector<std::pair<int, int>> sides;
    };
    auto read = [&]() {
        Values v;
        // Copies (on a host build the mesh's views are already host views, and const).
        v.V = decltype(v.V)("V", n);
        v.c = decltype(v.c)("c", n);
        v.A = decltype(v.A)("A", mesh.faces());
        v.xf = decltype(v.xf)("xf", mesh.faces());
        Kokkos::deep_copy(v.V, mesh.volume());
        Kokkos::deep_copy(v.c, mesh.centroid());
        Kokkos::deep_copy(v.A, mesh.area());
        Kokkos::deep_copy(v.xf, mesh.face_centroid());
        const auto o = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.owner());
        const auto j = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.neighbour());
        for (int f = 0; f < mesh.faces(); ++f) v.sides.emplace_back(o(f), j(f));
        return v;
    };
    const Values base = read();
    // Forward: the reference's values against the mesh's (relative to a cell's size).
    double forward = 0;
    for (int i = 0; i < n; ++i) {
        forward = std::max(forward, std::abs(ref.volume(i) - base.V(i)) * n);
        for (int a = 0; a < 2; ++a) forward = std::max(forward, std::abs(ref.centroid(i)[a] - base.c(i, a)) * nx);
    }
    for (int f = 0; f < faces; ++f)
        for (int a = 0; a < 2; ++a)
            forward = std::max({forward, std::abs(ref.area(f)[a] - base.A(f, a)) * nx, std::abs(ref.face_centroid(f)[a] - base.xf(f, a)) * nx});
    check.expect(forward < 1e-10, "2D geometry: the reference's V, c, A, x_f equal the mesh's (" + sci(forward) + " of a cell)");
    // A direction in whole grid steps (up to 1e4: 1e-5 of the box, 3e-4 of a cell).
    typename GD::Points dx("dx", n), dc("dc", n), dA("dA", faces), dxf("dxf", faces);
    typename GD::Scalars dV("dV", n);
    auto hdx = Kokkos::create_mirror_view(dx);
    for (int i = 0; i < n; ++i)
        for (int a = 0; a < 2; ++a) hdx(i, a) = std::round(1e3 * unit(rng)) * box.step;
    Kokkos::deep_copy(dx, hdx);
    ref.tangent(dx, dV, dc, dA, dxf);
    const auto tV = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dV);
    const auto tc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dc);
    const auto tA = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dA);
    const auto txf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dxf);
    auto moved = [&](double s) {
        Kokkos::View<double* [2], Exec::memory_space> y("y", n);
        auto h = Kokkos::create_mirror_view(y);
        for (int i = 0; i < n; ++i)
            for (int a = 0; a < 2; ++a) {
                const double v = hx(i, a) + s * hdx(i, a);
                h(i, a) = v - std::floor(v);
            }
        Kokkos::deep_copy(y, h);
        mesh.update(y);
        mesh.update_cells();
        return read();
    };
    const Values plus = moved(1), minus = moved(-1);
    mesh.update(x);
    mesh.update_cells();
    // The same faces (by their two cells; an update may number them differently): each moved face's number in base's order.
    auto renumbered = [&](const Values& v, std::vector<int>& to) {
        if (v.sides.size() != base.sides.size()) return false;
        std::map<std::pair<int, int>, int> at;
        for (std::size_t f = 0; f < v.sides.size(); ++f) at[v.sides[f]] = static_cast<int>(f);
        to.assign(base.sides.size(), -1);
        for (std::size_t f = 0; f < base.sides.size(); ++f) {
            const auto it = at.find(base.sides[f]);
            if (it == at.end()) return false;
            to[f] = it->second;
        }
        return true;
    };
    std::vector<int> fp, fm;
    const bool same = renumbered(plus, fp) && renumbered(minus, fm);
    check.expect(same, "2D geometry: x +- dx keeps the topology (the same pairs of cells share a face)");
    std::printf("geometry2d: faces renumbered by the updates: %s\n", plus.sides == base.sides && minus.sides == base.sides ? "no" : "yes");
    if (same) {
        double worst[4] = {}, scale[4] = {};
        auto compare = [&](int k, double t, double p, double m) {
            const double fd = (p - m) / 2;
            worst[k] = std::max(worst[k], std::abs(t - fd));
            scale[k] = std::max(scale[k], std::abs(fd));
        };
        for (int i = 0; i < n; ++i) {
            compare(0, tV(i), plus.V(i), minus.V(i));
            for (int a = 0; a < 2; ++a) compare(1, tc(i, a), plus.c(i, a), minus.c(i, a));
        }
        for (int f = 0; f < faces; ++f)
            for (int a = 0; a < 2; ++a) {
                compare(2, tA(f, a), plus.A(fp[f], a), minus.A(fm[f], a));
                compare(3, txf(f, a), plus.xf(fp[f], a), minus.xf(fm[f], a));
            }
        const char* names[4] = {"V", "c", "A", "x_f"};
        for (int k = 0; k < 4; ++k)
            check.expect(worst[k] < 1e-5 * scale[k], std::string("2D geometry: d") + names[k] + " equals central differences (" +
                                                         sci(worst[k] / scale[k]) + " of the largest)");
    }
    // Identities: sum dV = 0 (the box stays filled); a translation changes nothing (all relative to the generators).
    double sum = 0, size = 0;
    for (int i = 0; i < n; ++i) {
        sum += tV(i);
        size += std::abs(tV(i));
    }
    check.expect(std::abs(sum) < 1e-12 * size, "2D geometry: sum dV = 0 (" + sci(std::abs(sum) / size) + ")");
    {
        auto h = Kokkos::create_mirror_view(dx);
        for (int i = 0; i < n; ++i) {
            h(i, 0) = 0.3;
            h(i, 1) = -0.7;
        }
        Kokkos::deep_copy(dx, h);
        ref.tangent(dx, dV, dc, dA, dxf);
        const auto c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dc);
        const auto A = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dA);
        const auto xf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dxf);
        const auto Vh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dV);
        double moved_any = 0;
        for (int i = 0; i < n; ++i) {
            moved_any = std::max(moved_any, std::abs(Vh(i)) * n);
            for (int a = 0; a < 2; ++a) moved_any = std::max(moved_any, std::abs(c(i, a)) * nx);
        }
        for (int f = 0; f < faces; ++f)
            for (int a = 0; a < 2; ++a) moved_any = std::max({moved_any, std::abs(A(f, a)) * nx, std::abs(xf(f, a)) * nx});
        check.expect(moved_any < 1e-10, "2D geometry: a translation leaves V, c, A, x_f as they are (" + sci(moved_any) + ")");
        Kokkos::deep_copy(dx, hdx);
    }
    // The dot test: <bars, J dx> = <J^T bars, dx>.
    typename GD::Points cbar("cbar", n), Abar("Abar", faces), xfbar("xfbar", faces), xbar("xbar", n);
    typename GD::Scalars Vbar("Vbar", n);
    auto hVb = Kokkos::create_mirror_view(Vbar);
    auto hcb = Kokkos::create_mirror_view(cbar);
    auto hAb = Kokkos::create_mirror_view(Abar);
    auto hfb = Kokkos::create_mirror_view(xfbar);
    double lhs = 0, magnitude = 0;
    for (int i = 0; i < n; ++i) {
        hVb(i) = unit(rng);
        lhs += hVb(i) * tV(i);
        magnitude += std::abs(hVb(i) * tV(i));
        for (int a = 0; a < 2; ++a) {
            hcb(i, a) = unit(rng);
            lhs += hcb(i, a) * tc(i, a);
            magnitude += std::abs(hcb(i, a) * tc(i, a));
        }
    }
    for (int f = 0; f < faces; ++f)
        for (int a = 0; a < 2; ++a) {
            hAb(f, a) = unit(rng);
            hfb(f, a) = unit(rng);
            lhs += hAb(f, a) * tA(f, a) + hfb(f, a) * txf(f, a);
            magnitude += std::abs(hAb(f, a) * tA(f, a)) + std::abs(hfb(f, a) * txf(f, a));
        }
    Kokkos::deep_copy(Vbar, hVb);
    Kokkos::deep_copy(cbar, hcb);
    Kokkos::deep_copy(Abar, hAb);
    Kokkos::deep_copy(xfbar, hfb);
    ref.adjoint(Vbar, cbar, Abar, xfbar, xbar);
    const auto hxb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, xbar);
    double rhs = 0;
    for (int i = 0; i < n; ++i)
        for (int a = 0; a < 2; ++a) {
            rhs += hxb(i, a) * hdx(i, a);
            magnitude += std::abs(hxb(i, a) * hdx(i, a));
        }
    const double dot = std::abs(lhs - rhs) / magnitude;
    check.expect(dot < 1e-13, "2D geometry: <bars, J dx> = <J^T bars, dx> (" + sci(dot) + ")");
    // The library's (voronoi::GeometryDerivative, voronoi-cpp #63) against this reference: the same tangent and transpose.
    {
        const hydro::MeshDerivative<Mesh> library(mesh);
        typename GD::Points lc("lc", n), lA("lA", faces), lxf("lxf", faces), lxb("lxb", n);
        typename GD::Scalars lV("lV", n);
        library.tangent(dx, lV, lc, lA, lxf);
        library.adjoint(Vbar, cbar, Abar, xfbar, lxb);
        const auto hV = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, lV);
        const auto hc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, lc);
        const auto hA = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, lA);
        const auto hf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, lxf);
        const auto hl = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, lxb);
        double t = 0, tscale = 0, a = 0, ascale = 0;
        for (int i = 0; i < n; ++i) {
            t = std::max(t, std::abs(hV(i) - tV(i)));
            tscale = std::max(tscale, std::abs(tV(i)));
            for (int c = 0; c < 2; ++c) {
                t = std::max(t, std::abs(hc(i, c) - tc(i, c)));
                tscale = std::max(tscale, std::abs(tc(i, c)));
                a = std::max(a, std::abs(hl(i, c) - hxb(i, c)));
                ascale = std::max(ascale, std::abs(hxb(i, c)));
            }
        }
        for (int f = 0; f < faces; ++f)
            for (int c = 0; c < 2; ++c) {
                t = std::max({t, std::abs(hA(f, c) - tA(f, c)), std::abs(hf(f, c) - txf(f, c))});
                tscale = std::max({tscale, std::abs(tA(f, c)), std::abs(txf(f, c))});
            }
        check.expect(t < 1e-12 * tscale && a < 1e-12 * ascale, "2D geometry: the library's tangent and adjoint equal the reference's (" +
                                                                    sci(t / tscale) + ", " + sci(a / ascale) + ")");
    }
    std::printf("geometry2d n %d faces %d: forward %.2e, dot %.2e\n", n, faces, forward, dot);
}

// Whole moving steps at order 0 on an nx x nx jittered periodic 2D mesh, the state (Q, x): step_moving_tangent against central
// differences (x moved by whole grid steps, so the first step's geometry is exact; later steps read x' quantised to the grid,
// one grid step in 1e4), one-sided at a switch; step_moving_adjoint against the tangent (the dot test). The geometry's derivative:
// the tests' reference (reference_geometry_2d.hpp).
template <class F>
void moving_test_2d(Check& check, const std::string& name, int nx, int steps, double gate, double scale = 1, int order = 0, bool steps_fixed = false,
                    double noise = 0.01) {
    using Mesh = typename F::Mesh;
    using Points2 = typename F::Points;
    using HostPoints = Kokkos::View<double* [2], Kokkos::LayoutRight, Kokkos::HostSpace>;
    const hydro::Gas gas{5.0 / 3.0};
    const exact_predicates::BoxN<2> box(1.0);
    const int n = nx * nx;
    HostPoints x0("x0", n), dx0("dx0", n);
    std::mt19937_64 rng(5);
    std::uniform_real_distribution<double> jit(-0.15, 0.15), unit(-1, 1), small(-noise, noise);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < nx; ++j) {
            x0(i * nx + j, 0) = std::floor((i + 0.5 + jit(rng)) / nx / box.step) * box.step;
            x0(i * nx + j, 1) = std::floor((j + 0.5 + jit(rng)) / nx / box.step) * box.step;
        }
    Points2 xd("x", n);
    Kokkos::deep_copy(xd, x0);
    Mesh mesh(Exec{}, box);
    mesh.update(xd);
    mesh.update_cells();
    F fv(Exec{}, mesh, gas, order, hydro::Update::per_face, hydro::Limiter::face_centroid, hydro::Motion::moving);
    const auto V = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.volume());
    const auto cen = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.centroid());
    HostCells2 q0("q0", n), dq0("dq0", n), wbar("wbar", n);
    HostPoints xbar0("xbar0", n);
    const double k = 2 * M_PI;
    for (int i = 0; i < n; ++i) {
        const double px = x0(i, 0) + cen(i, 0), py = x0(i, 1) + cen(i, 1);
        const State2 W{1 + 0.2 * std::sin(k * px) + small(rng), 0.3 * std::sin(k * py) + small(rng), 0.2 * std::cos(k * px) + small(rng),
                       1 + 0.2 * std::cos(k * (px + py)) + small(rng)};
        const State2 U = hydro::conserved(W, gas);
        for (int v = 0; v < 4; ++v) q0(i, v) = U[v] * V(i);
    }
    // The direction: Q by 1e-5 of each entry, x by up to 1e4 whole grid steps (1e-5 of the box).
    for (int i = 0; i < n; ++i) {
        for (int v = 0; v < 4; ++v) {
            dq0(i, v) = scale * 1e-5 * (unit(rng) * std::abs(q0(i, v)) + 0.1 * unit(rng) * V(i));
            wbar(i, v) = unit(rng);
        }
        for (int a = 0; a < 2; ++a) {
            dx0(i, a) = std::round(scale * 1e4 * unit(rng)) * box.step;
            xbar0(i, a) = unit(rng) * 1e3;
        }
    }
    // Puts (q, x) into fv and the mesh.
    auto load = [&](const HostCells2& q, const HostPoints& x) {
        Kokkos::deep_copy(fv.positions(), x);
        mesh.update(fv.positions());
        mesh.update_cells();
        fv.mesh_moved();
        Kokkos::deep_copy(fv.state(), q);
    };
    auto read_q = [&]() {
        HostCells2 h("q", n);
        Kokkos::deep_copy(h, fv.state());
        return h;
    };
    auto read_x = [&]() {
        HostPoints h("x", n);
        Kokkos::deep_copy(h, fv.positions());
        return h;
    };
    load(q0, x0);
    std::vector<double> dts;
    std::vector<HostCells2> qs{q0};
    std::vector<HostPoints> xs{x0};
    for (int s = 0; s < steps; ++s) {
        dts.push_back(fv.time_step(0.3));
        fv.step(dts.back());
        qs.push_back(read_q());
        xs.push_back(read_x());
    }
    auto run = [&](double sign, HostCells2& q_out, HostPoints& x_out) {
        HostCells2 q("q", n);
        HostPoints x("x", n);
        for (int i = 0; i < n; ++i) {
            for (int v = 0; v < 4; ++v) q(i, v) = q0(i, v) + sign * dq0(i, v);
            for (int a = 0; a < 2; ++a) {
                const double y = x0(i, a) + sign * dx0(i, a);
                x(i, a) = y - std::floor(y);
            }
        }
        load(q, x);
        for (int s = 0; s < steps; ++s) fv.step(dts[s]);
        q_out = read_q();
        x_out = read_x();
    };
    // The tangent along the forward path.
    load(q0, x0);
    typename F::Cells dq("dq", n);
    Points2 dx("dx", n);
    Kokkos::deep_copy(dq, dq0);
    Kokkos::deep_copy(dx, dx0);
    for (int s = 0; s < steps; ++s) {
        const hydro::MeshDerivative<Mesh> ref(mesh);  // the library's (voronoi-cpp #63); the reference agrees (geometry_test_2d)
        fv.step_moving_tangent(dts[s], dq, dx, ref, steps_fixed);
    }
    const HostCells2 q_end = read_q();
    const HostPoints x_end = read_x();
    double bits = 0;
    for (int i = 0; i < n; ++i)
        for (int v = 0; v < 4; ++v) bits = std::max(bits, std::abs(q_end(i, v) - qs.back()(i, v)));
    check.expect(bits == 0, name + ": the forward of step_moving_tangent is step()'s (" + sci(bits) + ")");
    HostCells2 tq("tq", n);
    HostPoints tx("tx", n);
    Kokkos::deep_copy(tq, dq);
    Kokkos::deep_copy(tx, dx);
    HostCells2 qp, qm;
    HostPoints xp, xm;
    run(1, qp, xp);
    run(-1, qm, xm);
    auto wrapped = [](double d) { return d - std::nearbyint(d); };
    // steps_fixed (order 1, one step): the cells whose whole grid steps x +- the direction changed (the tangent holds them).
    int steps_changed = 0;
    if (steps_fixed && steps == 1)
        for (int i = 0; i < n; ++i)
            for (int a = 0; a < 2; ++a) {
                const double base = std::nearbyint(wrapped(xs[1](i, a) - x0(i, a)) / box.step);
                const double p = std::nearbyint(wrapped(xp(i, a) - x0(i, a) - dx0(i, a)) / box.step);
                const double m = std::nearbyint(wrapped(xm(i, a) - x0(i, a) + dx0(i, a)) / box.step);
                if (p != base || m != base) {
                    ++steps_changed;
                    break;
                }
            }
    if (steps_fixed) std::printf("moving2d %-12s cells whose grid steps changed: %d\n", name.c_str(), steps_changed);
    // Per cell: central difference of (Q, x), or where it differs (a switch within +- the direction), a one-sided one.
    double worst = 0, scale_q = 0, scale_x = 0, worst_one_sided = 0;
    int cells_off = 0;
    for (int i = 0; i < n; ++i) {
        for (int v = 0; v < 4; ++v) scale_q = std::max(scale_q, std::abs(qp(i, v) - qm(i, v)) / 2);
        for (int a = 0; a < 2; ++a) scale_x = std::max(scale_x, std::abs(wrapped(xp(i, a) - xm(i, a))) / 2);
    }
    for (int i = 0; i < n; ++i) {
        double central = 0, plus = 0, minus = 0;
        for (int v = 0; v < 4; ++v) {
            central = std::max(central, std::abs(tq(i, v) - (qp(i, v) - qm(i, v)) / 2) / scale_q);
            plus = std::max(plus, std::abs(tq(i, v) - (qp(i, v) - q_end(i, v))) / scale_q);
            minus = std::max(minus, std::abs(tq(i, v) - (q_end(i, v) - qm(i, v))) / scale_q);
        }
        for (int a = 0; a < 2; ++a) {
            central = std::max(central, std::abs(tx(i, a) - wrapped(xp(i, a) - xm(i, a)) / 2) / scale_x);
            plus = std::max(plus, std::abs(tx(i, a) - wrapped(xp(i, a) - x_end(i, a))) / scale_x);
            minus = std::max(minus, std::abs(tx(i, a) - wrapped(x_end(i, a) - xm(i, a))) / scale_x);
        }
        worst = std::max(worst, central);
        if (central > 1e-5) {
            ++cells_off;
            worst_one_sided = std::max(worst_one_sided, std::min(plus, minus));
        }
    }
    // gate < 0: reported only (several HLLC steps: the contact switch at S* = 0, where a mesh moving with the flow sits, is crossed
    // within the direction, and the positions after a step are quantised to the grid, 1 step in 1e4).
    if (gate > 0 && steps_fixed) {
        // A cell whose whole grid steps changed jumps, and so do its faces' cells: there one side must match.
        check.expect(worst_one_sided < gate && cells_off <= 8 * steps_changed,
                     name + ": J (dq, dx) equals central differences away from the " + std::to_string(steps_changed) +
                         " cells whose grid steps changed, a one-sided one at the " + std::to_string(cells_off) + " cells next to them (" +
                         sci(worst_one_sided) + ")");
    } else if (gate > 0) {
        check.expect(worst < gate, name + ": J (dq, dx) equals central differences over " + std::to_string(steps) + " steps (" + sci(worst) +
                                       " of the largest; direction x" + sci(scale) + ")");
    }
    // The adjoint, backwards along the forward path.
    typename F::Cells qbar("qbar", n);
    Points2 xbar("xbar", n);
    Kokkos::deep_copy(qbar, wbar);
    Kokkos::deep_copy(xbar, xbar0);
    for (int s = steps - 1; s >= 0; --s) {
        load(qs[s], xs[s]);
        const hydro::MeshDerivative<Mesh> ref(mesh);
        fv.step_moving_adjoint(dts[s], qbar, xbar, ref, steps_fixed);
    }
    HostCells2 aq("aq", n);
    HostPoints ax("ax", n);
    Kokkos::deep_copy(aq, qbar);
    Kokkos::deep_copy(ax, xbar);
    double lhs = 0, rhs = 0, size = 0;
    for (int i = 0; i < n; ++i) {
        for (int v = 0; v < 4; ++v) {
            lhs += wbar(i, v) * tq(i, v);
            rhs += aq(i, v) * dq0(i, v);
            size += std::abs(wbar(i, v) * tq(i, v)) + std::abs(aq(i, v) * dq0(i, v));
        }
        for (int a = 0; a < 2; ++a) {
            lhs += xbar0(i, a) * tx(i, a);
            rhs += ax(i, a) * dx0(i, a);
            size += std::abs(xbar0(i, a) * tx(i, a)) + std::abs(ax(i, a) * dx0(i, a));
        }
    }
    const double dot = std::abs(lhs - rhs) / size;
    check.expect(dot < 1e-13, name + ": <w, J v> = <J^T w, v> in (Q, x) over " + std::to_string(steps) + " steps (" + sci(dot) + ")");
    std::printf("moving2d %-12s n %d steps %d: forward %.2e, tangent-FD %.2e (cells off %d, one-sided %.2e), dot %.2e\n", name.c_str(), n,
                steps, bits, worst, cells_off, worst_one_sided, dot);
}

void tests_2d(Check& check) {
    geometry_test_2d(check, 32);
    moving_test_2d<FV2>(check, "2D moving HLLC", 32, 1, 1e-8, 0.01);  // x0.01: no switch crossed (the scan: --scan)
    moving_test_2d<FV2HLL>(check, "2D moving HLL", 32, 1, 1e-8, 0.01);
    moving_test_2d<FV2HLL>(check, "2D moving HLL", 32, 3, 1e-3);  // quantised positions after the first step: 1 grid step in 1e4
    moving_test_2d<FV2>(check, "2D moving HLLC", 32, 3, -1);
    // Order 1 (step_moving): the grid steps held, a direction of up to 1 grid step and 1e-9 of Q (few steps change); the dot test also
    // with the steps' derivative (dw-hat = dw) over 3 steps.
    moving_test_2d<FV2>(check, "2D moving order 1 HLLC", 32, 1, 1e-5, 1e-4, 1, true);
    moving_test_2d<FV2HLL>(check, "2D moving order 1 HLL", 32, 1, 1e-5, 1e-4, 1, true);
    moving_test_2d<FV2>(check, "2D moving order 1 HLLC", 32, 3, -1, 1.0, 1);
    const HostCells2 reference = step_test_2d<FV2>(check, "2D HLLC", 32, 3);
    step_test_2d<FV2HLL>(check, "2D HLL", 32, 3);
    step_test_2d<FV2_32>(check, "2D HLLC grad32", 32, 3, 0.01, &reference);
}

// Per label while on: kernel and copy time (fenced at both ends), and the device allocations' count and bytes.
// Used by: cost_moving_2d (its profile), DeviceBytes.
struct KernelClock {
    struct Row {
        long long calls = 0, bytes = 0;
        double ms = 0;
    };
    static inline bool on = false;
    static inline std::map<std::string, Row> kernels, copies, allocations;
    static inline std::vector<std::string> open;
    static inline Kokkos::Timer timer;
    static void begin(const char* name, const uint32_t, uint64_t* id) {
        if (!on) return;
        Kokkos::fence();
        open.emplace_back(name);
        *id = open.size() - 1;
        timer.reset();
    }
    static void end(uint64_t) {
        if (!on || open.empty()) return;
        Kokkos::fence();
        Row& r = kernels[open.back()];
        r.ms += timer.seconds() * 1e3;
        ++r.calls;
        open.pop_back();
    }
    static void copy_begin(Kokkos_Profiling_SpaceHandle dst, const char* dst_name, const void*, Kokkos_Profiling_SpaceHandle src, const char*,
                           const void*, uint64_t size) {
        if (!on) return;
        Kokkos::fence();
        open.push_back(std::string("deep_copy ") + src.name + "->" + dst.name + " [" + dst_name + "]");
        copies[open.back()].bytes += static_cast<long long>(size);
        timer.reset();
    }
    static void copy_end() {
        if (!on || open.empty()) return;
        Kokkos::fence();
        Row& r = copies[open.back()];
        r.ms += timer.seconds() * 1e3;
        ++r.calls;
        open.pop_back();
    }
    static void allocated(const char* label, uint64_t size) {
        if (!on) return;
        Row& r = allocations[label];
        ++r.calls;
        r.bytes += static_cast<long long>(size);
    }
    static void install() {
        Kokkos::Tools::Experimental::set_begin_parallel_for_callback(begin);
        Kokkos::Tools::Experimental::set_end_parallel_for_callback(end);
        Kokkos::Tools::Experimental::set_begin_parallel_reduce_callback(begin);
        Kokkos::Tools::Experimental::set_end_parallel_reduce_callback(end);
        Kokkos::Tools::Experimental::set_begin_parallel_scan_callback(begin);
        Kokkos::Tools::Experimental::set_end_parallel_scan_callback(end);
        Kokkos::Tools::Experimental::set_begin_deep_copy_callback(copy_begin);
        Kokkos::Tools::Experimental::set_end_deep_copy_callback(copy_end);
    }
    static void reset() {
        kernels.clear();
        copies.clear();
        allocations.clear();
    }
    // The table: kernels, then copies, by time; allocations by bytes; ns and B per cell; the rest of `total_ms` as "outside".
    static void print(const char* what, double total_ms, double reference_ms, int n) {
        auto sorted = [](const std::map<std::string, Row>& m, bool by_bytes) {
            std::vector<std::pair<std::string, Row>> v(m.begin(), m.end());
            std::sort(v.begin(), v.end(), [&](const auto& a, const auto& b) { return by_bytes ? a.second.bytes > b.second.bytes : a.second.ms > b.second.ms; });
            return v;
        };
        double inside = 0;
        std::printf("profile %s: %.3f ms in all, %.3f ms the reference geometry (host), n %d\n", what, total_ms, reference_ms, n);
        for (const auto& [name, r] : sorted(kernels, false)) {
            inside += r.ms;
            std::printf("  kernel %8.3f ms %7.2f ns/cell %4lld x  %s\n", r.ms, r.ms * 1e6 / n, r.calls, name.c_str());
        }
        for (const auto& [name, r] : sorted(copies, false)) {
            inside += r.ms;
            std::printf("  copy   %8.3f ms %7.2f ns/cell %4lld x %7.1f B/cell  %s\n", r.ms, r.ms * 1e6 / n, r.calls, double(r.bytes) / n, name.c_str());
        }
        long long bytes = 0, count = 0;
        for (const auto& [name, r] : sorted(allocations, true)) {
            bytes += r.bytes;
            count += r.calls;
            std::printf("  alloc  %7.1f B/cell %4lld x  %s\n", double(r.bytes) / n, r.calls, name.c_str());
        }
        std::printf("  kernels and copies %.3f ms; outside them (host work, allocation, launch gaps) %.3f ms without the reference; "
                    "%lld device allocations, %.1f B/cell\n",
                    inside, total_ms - inside - reference_ms, count, double(bytes) / n);
    }
};
// Device bytes in use and their peak, from Kokkos's allocation callbacks (every space but the host's). Used by: cost_moving_2d.
struct DeviceBytes {
    static inline long long now = 0, peak = 0;
    static void allocate(const Kokkos_Profiling_SpaceHandle h, const char* label, const void*, const uint64_t size) {
        if (std::string(h.name) == "Host") return;
        now += static_cast<long long>(size);
        peak = std::max(peak, now);
        KernelClock::allocated(label, size);
    }
    static void deallocate(const Kokkos_Profiling_SpaceHandle h, const char*, const void*, const uint64_t size) {
        if (std::string(h.name) != "Host") now -= static_cast<long long>(size);
    }
};
// A geometry derivative of zero on the device: for timing the hydro alone (the library's version replaces the tests' host reference).
struct NullGeometry : hydro::GeometryDerivative<2, Exec::memory_space> {
    void tangent(const Points&, const Scalars& dV, const Points& dc, const Points& dA, const Points& dxf) const override {
        Kokkos::deep_copy(dV, 0.0);
        Kokkos::deep_copy(dc, 0.0);
        Kokkos::deep_copy(dA, 0.0);
        Kokkos::deep_copy(dxf, 0.0);
    }
    void adjoint(const Scalars&, const Points&, const Points&, const Points&, const Points&) const override {}
};
// A GeometryDerivative that times the one it wraps (the tests' host reference: its time is not the hydro's).
struct TimedGeometry : hydro::GeometryDerivative<2, Exec::memory_space> {
    const hydro::GeometryDerivative<2, Exec::memory_space>& inner;
    mutable double ms = 0;
    explicit TimedGeometry(const hydro::GeometryDerivative<2, Exec::memory_space>& g) : inner(g) {}
    void tangent(const Points& dx, const Scalars& dV, const Points& dc, const Points& dA, const Points& dxf) const override {
        Kokkos::fence();
        Kokkos::Timer t;
        const bool clock = KernelClock::on;  // the reference's own kernels and copies are not the hydro's
        KernelClock::on = false;
        inner.tangent(dx, dV, dc, dA, dxf);
        KernelClock::on = clock;
        Kokkos::fence();
        ms += t.seconds() * 1e3;
    }
    void adjoint(const Scalars& Vbar, const Points& cbar, const Points& Abar, const Points& xfbar, const Points& xbar) const override {
        Kokkos::fence();
        Kokkos::Timer t;
        const bool clock = KernelClock::on;
        KernelClock::on = false;
        inner.adjoint(Vbar, cbar, Abar, xfbar, xbar);
        KernelClock::on = clock;
        Kokkos::fence();
        ms += t.seconds() * 1e3;
    }
};

// The moving step's derivatives on an nx x nx jittered periodic 2D mesh, order 1: time_step + step against step_moving_tangent and
// step_moving_adjoint (medians of 5 after a warm-up; each from the same state, the reload not timed). The geometry's derivative is a
// null one on the device (zeros), standing in for the library's: the host reference would dominate and blur the hydro's time.
template <class F>
void cost_moving_2d(int nx, bool profile = false) {
    using Mesh = typename F::Mesh;
    const hydro::Gas gas{5.0 / 3.0};
    const exact_predicates::BoxN<2> box(1.0);
    const int n = nx * nx;
    typename F::Points x0("x0", n);
    auto hx = Kokkos::create_mirror_view(x0);
    std::mt19937_64 rng(5);
    std::uniform_real_distribution<double> jit(-0.15, 0.15), small(-0.01, 0.01);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < nx; ++j) {
            hx(i * nx + j, 0) = std::floor((i + 0.5 + jit(rng)) / nx / box.step) * box.step;
            hx(i * nx + j, 1) = std::floor((j + 0.5 + jit(rng)) / nx / box.step) * box.step;
        }
    Kokkos::deep_copy(x0, hx);
    Mesh mesh(Exec{}, box);
    mesh.update(x0);
    mesh.update_cells();
    F fv(Exec{}, mesh, gas, 1, hydro::Update::per_face, hydro::Limiter::face_centroid, hydro::Motion::moving);
    const auto V = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.volume());
    const auto cen = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.centroid());
    HostCells2 q0("q0", n);
    const double k = 2 * M_PI;
    for (int i = 0; i < n; ++i) {
        const double px = hx(i, 0) + cen(i, 0), py = hx(i, 1) + cen(i, 1);
        const State2 W{1 + 0.2 * std::sin(k * px) + small(rng), 0.3 * std::sin(k * py) + small(rng), 0.2 * std::cos(k * px) + small(rng),
                       1 + 0.2 * std::cos(k * (px + py)) + small(rng)};
        const State2 U = hydro::conserved(W, gas);
        for (int v = 0; v < 4; ++v) q0(i, v) = U[v] * V(i);
    }
    auto load = [&]() {
        Kokkos::deep_copy(fv.positions(), x0);
        mesh.update(fv.positions());
        mesh.update_cells();
        fv.mesh_moved();
        Kokkos::deep_copy(fv.state(), q0);
        Kokkos::fence();
    };
    typename F::Cells dq("dq", n), qbar("qbar", n);
    typename F::Points dx("dx", n), xbar("xbar", n);
    load();
    const double dt = fv.time_step(0.3);
    struct Result {
        double ms = 0, geometry_ms = 0;
        long long peak = 0;
    };
    auto time = [&](auto&& f) {
        std::vector<double> t, g;
        long long peak = 0;
        for (int r = 0; r < 6; ++r) {
            load();
            Kokkos::deep_copy(dq, 1e-6);
            Kokkos::deep_copy(dx, 0.0);
            Kokkos::deep_copy(qbar, 1.0);
            Kokkos::deep_copy(xbar, 1.0);
            Kokkos::fence();
            const long long before = DeviceBytes::now;
            DeviceBytes::peak = DeviceBytes::now;
            Kokkos::Timer timer;
            const double geometry_ms = f();
            Kokkos::fence();
            if (r > 0) {
                t.push_back(timer.seconds() * 1e3);
                g.push_back(geometry_ms);
            }
            peak = std::max(peak, DeviceBytes::peak - before);
        }
        std::sort(t.begin(), t.end());
        std::sort(g.begin(), g.end());
        return Result{t[t.size() / 2], g[g.size() / 2], peak};
    };
    const NullGeometry ref;
    const Result forward = time([&] {
        fv.step(fv.time_step(0.3));
        return 0.0;
    });
    const Result tangent = time([&] {
        const TimedGeometry timed(ref);
        fv.step_moving_tangent(dt, dq, dx, timed);
        return timed.ms;
    });
    const Result adjoint = time([&] {
        const TimedGeometry timed(ref);
        fv.step_moving_adjoint(dt, qbar, xbar, timed);
        return timed.ms;
    });
    long long resident = DeviceBytes::now;
    auto line = [&](const char* what, const Result& r) {
        const double hydro_ms = r.ms - r.geometry_ms;
        std::printf("  %-8s %9.3f ms (%5.2fx), without the geometry derivative %9.3f ms (%5.2fx, %.4f G cell steps/s); peak +%.0f B/cell\n", what,
                    r.ms, r.ms / forward.ms, hydro_ms, hydro_ms / forward.ms, n / (hydro_ms * 1e-3) * 1e-9, double(r.peak) / n);
    };
    std::printf("cost_moving_2d n %d faces %d, order 1, resident %.0f B/cell:\n", n, mesh.faces(), double(resident) / n);
    line("forward", forward);
    line("tangent", tangent);
    line("adjoint", adjoint);
    if (!profile) return;
    // The speed-of-light table: each kernel and copy fenced and timed by label, each allocation counted (one call each).
    KernelClock::install();
    auto table = [&](const char* what, auto&& f) {
        load();
        Kokkos::deep_copy(dq, 1e-6);
        Kokkos::deep_copy(dx, 0.0);
        Kokkos::deep_copy(qbar, 1.0);
        Kokkos::deep_copy(xbar, 1.0);
        Kokkos::fence();
        KernelClock::reset();
        KernelClock::on = true;
        Kokkos::Timer timer;
        const double reference_ms = f();
        Kokkos::fence();
        const double total = timer.seconds() * 1e3;
        KernelClock::on = false;
        KernelClock::print(what, total, reference_ms, n);
    };
    table("forward", [&] {
        fv.step(fv.time_step(0.3));
        return 0.0;
    });
    table("tangent", [&] {
        const TimedGeometry timed(ref);
        fv.step_moving_tangent(dt, dq, dx, timed);
        return timed.ms;
    });
    table("adjoint", [&] {
        const TimedGeometry timed(ref);
        fv.step_moving_adjoint(dt, qbar, xbar, timed);
        return timed.ms;
    });
    // The allocations alone, as the adjoint makes them (two face-row arrays of 40, the list entries' 9, twenty cell arrays of 2), each
    // freed at the end, with and without the zero fill: what the arrays cost outside the kernels.
    for (const bool fill : {true, false}) {
        Kokkos::fence();
        Kokkos::Timer timer;
        {
            using Rows = Kokkos::View<double**, Kokkos::LayoutRight, Exec::memory_space>;
            auto make = [&](const char* label, int rows, int width) {
                return fill ? Rows(label, rows, width) : Rows(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(label)), rows, width);
            };
            std::vector<Rows> keep;
            keep.push_back(make("probe_rows", mesh.faces(), 40));
            keep.push_back(make("probe_rows2", mesh.faces(), 40));
            keep.push_back(make("probe_entries", 2 * mesh.faces(), 9));
            for (int a = 0; a < 20; ++a) keep.push_back(make("probe_cells", n, 2));
            Kokkos::fence();
        }
        Kokkos::fence();
        std::printf("probe: 23 allocations and frees (%.0f B/cell)%s: %.3f ms\n", (2 * 40 * 8.0 * mesh.faces() + 2 * 9 * 8.0 * mesh.faces()) / n + 20 * 16,
                    fill ? ", zero-filled" : ", not filled", timer.seconds() * 1e3);
    }
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
    luna::print_build_info();
    for (int a = 1; a + 1 < argc; ++a)
        if (std::string(argv[a]) == "--cost-moving") {  // the 2D moving step's derivatives, nx x nx cells
            Kokkos::Tools::Experimental::set_allocate_data_callback(DeviceBytes::allocate);
            Kokkos::Tools::Experimental::set_deallocate_data_callback(DeviceBytes::deallocate);
            cost_moving_2d<FV2>(std::stoi(argv[a + 1]), a + 2 < argc && std::string(argv[a + 2]) == "profile");
            return 0;
        } else if (std::string(argv[a]) == "--cost") {
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
            for (const double s : {1.0, 0.1, 0.01}) {
                moving_test_2d<FV2>(check, "2D moving HLLC x" + sci(s), 32, 1, -1, s);
                moving_test_2d<FV2HLL>(check, "2D moving HLL x" + sci(s), 32, 1, -1, s);
            }
            for (const double s : {1e-4, 1e-3, 1e-2, 1.0}) {  // order 1 with the grid steps held
                moving_test_2d<FV2>(check, "2D moving order 1 HLLC fixed x" + sci(s), 32, 1, -1, s, 1, true);
                moving_test_2d<FV2HLL>(check, "2D moving order 1 HLL fixed x" + sci(s), 32, 1, -1, s, 1, true);
            }
            moving_test_2d<FV2HLL>(check, "2D moving order 1 HLL", 32, 1, -1, 1.0, 1);
            return 0;
        }
    const hydro::Gas gas{5.0 / 3.0};
    const int samples = 20000;
    piece_test(check, "primitive", PrimitivePiece{gas}, samples, "floors none/rho/e/both:");
    piece_test(check, "face side", SidePiece{gas}, samples, "extrapolated/fallback:");
    piece_test(check, "HLL", FluxPiece<hydro::HLL>{gas}, samples, "star/F_L/F_R/-:");
    piece_test(check, "HLLC", FluxPiece<hydro::HLLC>{gas}, samples, "F_L/F*_L/F*_R/F_R:");
    piece_test(check, "limited gradient", GradientPiece{}, samples, "unclipped/clipped/zero/both:");
    piece_test(check, "moving face HLL", MovingFacePiece<hydro::HLL>{gas}, samples, "star/F_L/F_R/-:");
    piece_test(check, "moving face HLLC", MovingFacePiece<hydro::HLLC>{gas}, samples, "F_L/F*_L/F*_R/F_R:");
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
