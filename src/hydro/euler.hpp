#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

// Physics kernels of the solver (D = 1, 2, 3), called per cell/face by FiniteVolume<D> (finite_volume.hpp, finite_volume_1d.hpp).
// Per face: primitive() -> half_step() -> Riemann::flux() (HLL/HLLC). W = (rho, v_1 .. v_D, P), U = (rho, rho v_1 .. rho v_D, E) per volume.
namespace hydro {

// Used by: everything in the solver. A state of D + 2 entries: W or U.
template <int D, class T = double>
using StateT = Kokkos::Array<T, D + 2>;
template <int D>
using State = StateT<D, double>;
// The dimension of a state of N entries (State<D> does not deduce D).
template <std::size_t N>
constexpr int dimension_of = static_cast<int>(N) - 2;

// Used by: dot(), velocity(), Gradient, the Riemann solvers (positions, normals, velocities).
template <int D, class T = double>
struct Vec {
    T v[D];
    KOKKOS_INLINE_FUNCTION T& operator[](int a) { return v[a]; }
    KOKKOS_INLINE_FUNCTION const T& operator[](int a) const { return v[a]; }
};
using Vec3 = Vec<3>;

// T{f(0), f(1), .., f(N - 1)}: a brace initializer for any N. Used by: braced().
template <class T, class F, std::size_t... I>
KOKKOS_INLINE_FUNCTION T braced_from(const F& f, std::index_sequence<I...>) {
    return T{f(static_cast<int>(I))...};
}
// Used by: make_vec(), make_state().
template <class T, std::size_t N, class F>
KOKKOS_INLINE_FUNCTION T braced(const F& f) {
    return braced_from<T>(f, std::make_index_sequence<N>{});
}
template <int D, class T = double, class F>
KOKKOS_INLINE_FUNCTION Vec<D, T> make_vec(const F& f) {
    return braced<Vec<D, T>, D>(f);
}
template <int D, class T = double, class F>
KOKKOS_INLINE_FUNCTION StateT<D, T> make_state(const F& f) {
    return braced<StateT<D, T>, D + 2>(f);
}

template <int D, class T>
KOKKOS_INLINE_FUNCTION Vec<D, T> operator+(Vec<D, T> a, Vec<D, T> b) {
    return make_vec<D, T>([&](int k) { return a[k] + b[k]; });
}
template <int D, class T>
KOKKOS_INLINE_FUNCTION Vec<D, T> operator-(Vec<D, T> a, Vec<D, T> b) {
    return make_vec<D, T>([&](int k) { return a[k] - b[k]; });
}
template <int D, class T>
KOKKOS_INLINE_FUNCTION Vec<D, T> operator*(Vec<D, T> a, T s) {
    return make_vec<D, T>([&](int k) { return a[k] * s; });
}
// Element-wise conversion to the real type T. Used by: FiniteVolume<D> reduced-precision paths.
template <class T, int D, class S>
KOKKOS_INLINE_FUNCTION Vec<D, T> cast(Vec<D, S> a) {
    return make_vec<D, T>([&](int k) { return static_cast<T>(a[k]); });
}
// Left to right, written out per D (a loop could be contracted otherwise).
template <int D, class T>
KOKKOS_INLINE_FUNCTION T dot(Vec<D, T> a, Vec<D, T> b) {
    if constexpr (D == 1) return a[0] * b[0];
    if constexpr (D == 2) return a[0] * b[0] + a[1] * b[1];
    if constexpr (D == 3) return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
// The velocity (or momentum) entries 1 .. D of a state. Used by: primitive(), conserved(), half_step_rate(), the solvers.
template <class T, std::size_t N, int D = dimension_of<N>>
KOKKOS_INLINE_FUNCTION Vec<D, T> velocity(const Kokkos::Array<T, N>& W) {
    return make_vec<D, T>([&](int k) { return W[1 + k]; });
}

// Ideal gas with LUNA's RAMSES floors (density >= smallr, internal energy >= smallc^2 / (gamma (gamma - 1))), on primitives only.
// Used by: every function below taking `gas`.
struct Gas {
    double gamma = 5.0 / 3.0;
    double smallr = 1e-10, smallc = 1e-10;
    // 1 / (gamma - 1), once per gas instead of once per face side; in double and in float, each rounded as computed in
    // that type (the float Riemann solve). Used by: detail::side() through inverse_gamma_minus_one().
    double inverse_gm1 = 1.0 / (5.0 / 3.0 - 1.0);
    float inverse_gm1_float = 1.0f / (static_cast<float>(5.0 / 3.0) - 1.0f);

    Gas() = default;
    KOKKOS_INLINE_FUNCTION Gas(double gamma_, double smallr_ = 1e-10, double smallc_ = 1e-10)
        : gamma(gamma_), smallr(smallr_), smallc(smallc_), inverse_gm1(1.0 / (gamma_ - 1.0)),
          inverse_gm1_float(1.0f / (static_cast<float>(gamma_) - 1.0f)) {}
    template <class T>
    KOKKOS_INLINE_FUNCTION T inverse_gamma_minus_one() const {
        if constexpr (std::is_same_v<T, float>)
            return inverse_gm1_float;
        else
            return static_cast<T>(inverse_gm1);
    }
};

// U -> W with the floors (LUNA's cons_to_prim_floored). Used by: FiniteVolume<D>::primitives(), side(), 1D primitive_of(), snapshot.
template <std::size_t N, int D = dimension_of<N>>
KOKKOS_INLINE_FUNCTION State<D> primitive(const Kokkos::Array<double, N>& U, const Gas& gas) {
    const double rho = Kokkos::fmax(U[0], gas.smallr);
    const double inv_rho = 1.0 / rho;
    const Vec<D> v = velocity(U) * inv_rho;
    const double smalle = gas.smallc * gas.smallc / gas.gamma / (gas.gamma - 1.0);
    const double kinetic = 0.5 * rho * dot(v, v);
    const double e = Kokkos::fmax(U[D + 1] - kinetic, smalle * rho);
    return make_state<D>([&](int k) { return k == 0 ? rho : k == D + 1 ? (gas.gamma - 1.0) * e : v[k - 1]; });
}

// Derivatives (tangent-linear and adjoint) sit beside each forward function: f_tangent(x, dx) = J dx, f_adjoint(x, ybar) = J^T ybar,
// in double for every precision policy (the forward state is linearised as stored). At a switch (a floor, a wave case, a guard)
// each takes the derivative of the branch the forward took. Checked by: hydro_derivatives_test.

// Which floors primitive() applied: bit 0 density, bit 1 internal energy. Used by: the derivative functions, the tests (branch patterns).
template <std::size_t N, int D = dimension_of<N>>
KOKKOS_INLINE_FUNCTION int primitive_branches(const Kokkos::Array<double, N>& U, const Gas& gas) {
    const double rho = Kokkos::fmax(U[0], gas.smallr);
    const Vec<D> v = velocity(U) * (1.0 / rho);
    const double smalle = gas.smallc * gas.smallc / gas.gamma / (gas.gamma - 1.0);
    return (U[0] >= gas.smallr ? 0 : 1) | (U[D + 1] - 0.5 * rho * dot(v, v) >= smalle * rho ? 0 : 2);
}

// dW = J dU of primitive(); a floored density or energy has derivative 0 in what the floor replaced.
// Used by: FiniteVolume::step_tangent (primitives), the derivative tests.
template <std::size_t N, int D = dimension_of<N>>
KOKKOS_INLINE_FUNCTION State<D> primitive_tangent(const Kokkos::Array<double, N>& U, const Kokkos::Array<double, N>& dU, const Gas& gas) {
    const int floors = primitive_branches(U, gas);
    const double rho = Kokkos::fmax(U[0], gas.smallr), inv_rho = 1.0 / rho, drho = floors & 1 ? 0.0 : dU[0];
    const Vec<D> v = velocity(U) * inv_rho;
    const Vec<D> dv = (velocity(dU) - v * drho) * inv_rho;
    const double smalle = gas.smallc * gas.smallc / gas.gamma / (gas.gamma - 1.0);
    const double dkinetic = 0.5 * drho * dot(v, v) + rho * dot(v, dv);
    const double de = floors & 2 ? smalle * drho : dU[D + 1] - dkinetic;
    return make_state<D>([&](int k) { return k == 0 ? drho : k == D + 1 ? (gas.gamma - 1.0) * de : dv[k - 1]; });
}

// Ubar = J^T Wbar of primitive(). Used by: FiniteVolume::step_adjoint (gather), the derivative tests.
template <std::size_t N, int D = dimension_of<N>>
KOKKOS_INLINE_FUNCTION State<D> primitive_adjoint(const Kokkos::Array<double, N>& U, const Kokkos::Array<double, N>& Wbar, const Gas& gas) {
    const int floors = primitive_branches(U, gas);
    const double rho = Kokkos::fmax(U[0], gas.smallr), inv_rho = 1.0 / rho;
    const Vec<D> v = velocity(U) * inv_rho;
    const double smalle = gas.smallc * gas.smallc / gas.gamma / (gas.gamma - 1.0);
    const double ebar = (gas.gamma - 1.0) * Wbar[D + 1];
    double rhobar = Wbar[0], energybar = 0;
    Vec<D> vbar = velocity(Wbar);
    if (floors & 2) {
        rhobar += smalle * ebar;
    } else {  // e = U_P - 0.5 rho v.v
        energybar = ebar;
        rhobar -= 0.5 * dot(v, v) * ebar;
        vbar = vbar - v * (rho * ebar);
    }
    const Vec<D> mbar = vbar * inv_rho;  // v = m / rho
    rhobar -= dot(v, vbar) * inv_rho;
    return make_state<D>([&](int k) { return k == 0 ? (floors & 1 ? 0.0 : rhobar) : k == D + 1 ? energybar : mbar[k - 1]; });
}

// W -> U, the inverse of primitive() without floors. Used by: initial conditions of the benches and tests.
template <std::size_t N, int D = dimension_of<N>>
KOKKOS_INLINE_FUNCTION State<D> conserved(const Kokkos::Array<double, N>& W, const Gas& gas) {
    const double rho = W[0];
    const Vec<D> v = velocity(W);
    const double kinetic = 0.5 * rho * dot(v, v);
    return make_state<D>(
        [&](int k) { return k == 0 ? rho : k == D + 1 ? W[D + 1] / (gas.gamma - 1.0) + kinetic : rho * W[k]; });
}

// Used by: time-step rules in FiniteVolume<D>, the 1D drift. c = sqrt(gamma P / rho).
template <std::size_t N, int D = dimension_of<N>>
KOKKOS_INLINE_FUNCTION double sound_speed(const Kokkos::Array<double, N>& W, const Gas& gas) {
    return Kokkos::sqrt(gas.gamma * W[D + 1] / W[0]);
}

// Gradients of the primitives: g[v] = grad W[v]. Written by the sweep, used by: half_step(), half_step_rate().
template <int D, class T = double>
using Gradient = Kokkos::Array<Vec<D, T>, D + 2>;

// The rate of the MUSCL-Hancock half step, sum_d A_d(W) dW/dx_d, with the primitive Jacobians summed over the axes in order.
// Used by: half_step(), FaceFlux::side() (reduced precision).
template <class T, std::size_t N, int D = dimension_of<N>>
KOKKOS_INLINE_FUNCTION StateT<D, T> half_step_rate(const Kokkos::Array<T, N>& W, const Gradient<D, T>& g,
                                                   const Gas& gas) {
    constexpr int P = D + 1;
    const T rho = W[0], inv_rho = T(1) / W[0], rho_c2 = T(gas.gamma) * W[P];
    StateT<D, T> A;
    for (int v = 0; v < D + 2; ++v) A[v] = 0;
    for (int d = 0; d < D; ++d) {
        const T vn = W[1 + d];
        A[0] += vn * g[0][d] + rho * g[1 + d][d];
        for (int k = 0; k < D; ++k) {
            T row = vn * g[1 + k][d];
            if (k == d) row = row + inv_rho * g[P][d];
            A[1 + k] += row;
        }
        A[P] += rho_c2 * g[1 + d][d] + vn * g[P][d];
    }
    return A;
}

// MUSCL-Hancock half step: W - dt/2 sum_d A_d(W) dW/dx_d. Used by: FaceFlux::side(), 1D Kernel::face_state(); feeds Riemann::flux().
template <std::size_t N, int D = dimension_of<N>>
KOKKOS_INLINE_FUNCTION State<D> half_step(const Kokkos::Array<double, N>& W, const Gradient<D>& g, double dt,
                                          const Gas& gas) {
    const State<D> A = half_step_rate(W, g, gas);
    State<D> half;
    for (int v = 0; v < D + 2; ++v) half[v] = W[v] - 0.5 * dt * A[v];
    return half;
}

// d(half_step) for tangents dW, dg (dt fixed). Used by: FiniteVolume::FaceFlux::side_tangent, the derivative tests.
template <std::size_t N, int D = dimension_of<N>>
KOKKOS_INLINE_FUNCTION State<D> half_step_tangent(const Kokkos::Array<double, N>& W, const Kokkos::Array<double, N>& dW,
                                                  const Gradient<D>& g, const Gradient<D>& dg, double dt, const Gas& gas) {
    constexpr int P = D + 1;
    const double rho = W[0], inv_rho = 1.0 / W[0], rho_c2 = gas.gamma * W[P];
    const double drho = dW[0], dinv_rho = -inv_rho * inv_rho * dW[0], drho_c2 = gas.gamma * dW[P];
    State<D> dA{};
    for (int d = 0; d < D; ++d) {
        const double vn = W[1 + d], dvn = dW[1 + d];
        dA[0] += dvn * g[0][d] + vn * dg[0][d] + drho * g[1 + d][d] + rho * dg[1 + d][d];
        for (int k = 0; k < D; ++k) {
            double row = dvn * g[1 + k][d] + vn * dg[1 + k][d];
            if (k == d) row += dinv_rho * g[P][d] + inv_rho * dg[P][d];
            dA[1 + k] += row;
        }
        dA[P] += drho_c2 * g[1 + d][d] + rho_c2 * dg[1 + d][d] + dvn * g[P][d] + vn * dg[P][d];
    }
    State<D> dhalf;
    for (int v = 0; v < D + 2; ++v) dhalf[v] = dW[v] - 0.5 * dt * dA[v];
    return dhalf;
}

// The adjoint of half_step(): adds J^T halfbar to Wbar and gbar. Used by: FiniteVolume::FaceFlux::side_adjoint, the derivative tests.
template <std::size_t N, int D = dimension_of<N>>
KOKKOS_INLINE_FUNCTION void half_step_adjoint(const Kokkos::Array<double, N>& W, const Gradient<D>& g, double dt, const Gas& gas,
                                              const Kokkos::Array<double, N>& halfbar, State<D>& Wbar, Gradient<D>& gbar) {
    constexpr int P = D + 1;
    const double rho = W[0], inv_rho = 1.0 / W[0], rho_c2 = gas.gamma * W[P];
    State<D> Abar;
    for (int v = 0; v < D + 2; ++v) {
        Wbar[v] += halfbar[v];
        Abar[v] = -0.5 * dt * halfbar[v];
    }
    double inv_rhobar = 0, rho_c2bar = 0;
    for (int d = 0; d < D; ++d) {
        const double vn = W[1 + d];
        Wbar[1 + d] += Abar[0] * g[0][d];
        gbar[0][d] += Abar[0] * vn;
        Wbar[0] += Abar[0] * g[1 + d][d];
        gbar[1 + d][d] += Abar[0] * rho;
        for (int k = 0; k < D; ++k) {
            Wbar[1 + d] += Abar[1 + k] * g[1 + k][d];
            gbar[1 + k][d] += Abar[1 + k] * vn;
            if (k == d) {
                inv_rhobar += Abar[1 + k] * g[P][d];
                gbar[P][d] += Abar[1 + k] * inv_rho;
            }
        }
        rho_c2bar += Abar[P] * g[1 + d][d];
        gbar[1 + d][d] += Abar[P] * rho_c2;
        Wbar[1 + d] += Abar[P] * g[P][d];
        gbar[P][d] += Abar[P] * vn;
    }
    Wbar[0] -= inv_rho * inv_rho * inv_rhobar;
    Wbar[P] += gas.gamma * rho_c2bar;
}

// Riemann solvers (Riemann parameter of FiniteVolume<D>): flux(L, R, n, w, gas) along unit normal n of a face moving with w.
// States are taken relative to w, the lab-frame flux is per area, L to R. Vectors stay in box axes, not LUNA's (normal, tangents) rotation; equal up to rounding (docs/design.md).
namespace detail {

// One side of a face in the face frame. Used by: HLL::flux(), HLLC::flux().
template <int D, class T>
struct Side {
    Vec<D, T> v;
    T vn, c, E;         // normal velocity, sound speed (0 if c^2 <= 0), energy
    StateT<D, T> F, U;  // flux along n and conserved state
};

// One side with its normal velocity vn = (v - w) . n given, without the sound speed (s.c = 0). n, w: face normal and
// velocity. Used by: side(), HLLC::flux() (which has vn already and never reads c).
template <int D, class T>
KOKKOS_INLINE_FUNCTION Side<D, T> side_with(const StateT<D, T>& W, Vec<D, T> n, Vec<D, T> w, const Gas& gas, T vn) {
    constexpr int P = D + 1;
    Side<D, T> s;
    s.v = velocity(W) - w;
    s.vn = vn;
    s.c = T(0);
    s.E = W[P] * gas.inverse_gamma_minus_one<T>() + W[0] * (T(0.5) * dot(s.v, s.v));
    const T m = W[0] * s.vn;
    s.F = make_state<D, T>([&](int k) { return k == 0 ? m : k == P ? (s.E + W[P]) * s.vn : m * s.v[k - 1] + W[P] * n[k - 1]; });
    s.U = make_state<D, T>([&](int k) { return k == 0 ? W[0] : k == P ? s.E : W[0] * s.v[k - 1]; });
    return s;
}

// The whole side, sound speed included. Used by: HLL::flux().
template <int D, class T>
KOKKOS_INLINE_FUNCTION Side<D, T> side(const StateT<D, T>& W, Vec<D, T> n, Vec<D, T> w, const Gas& gas) {
    constexpr int P = D + 1;
    Side<D, T> s = side_with(W, n, w, gas, dot(velocity(W) - w, n));
    const T c2 = T(gas.gamma) * W[P] / W[0];
    s.c = c2 > 0 ? Kokkos::sqrt(c2) : T(0);
    return s;
}

// From the face frame to the lab frame (AREPO next generation's flux_convert_to_lab_frame). Used by: HLL::flux(), HLLC::flux(), FaceFlux::solve().
template <int D, class T>
KOKKOS_INLINE_FUNCTION StateT<D, T> to_lab(StateT<D, T> F, Vec<D, T> w) {
    if constexpr (D == 1) F[2] += w[0] * F[1] + T(0.5) * dot(w, w) * F[0];
    if constexpr (D == 2) F[3] += w[0] * F[1] + w[1] * F[2] + T(0.5) * dot(w, w) * F[0];
    if constexpr (D == 3) F[4] += w[0] * F[1] + w[1] * F[2] + w[2] * F[3] + T(0.5) * dot(w, w) * F[0];
    for (int k = 0; k < D; ++k) F[1 + k] += w[k] * F[0];
    return F;
}

// The derivatives of a side, in double. side_with(): dS = J dW with vn's tangent dvn given; s is the forward side.
// Used by: HLLC::flux_tangent, side_tangent().
template <int D>
KOKKOS_INLINE_FUNCTION Side<D, double> side_with_tangent(const State<D>& W, const State<D>& dW, Vec<D> n, const Gas& gas,
                                                         const Side<D, double>& s, double dvn) {
    constexpr int P = D + 1;
    Side<D, double> d;
    d.v = velocity(dW);
    d.vn = dvn;
    d.c = 0;
    d.E = dW[P] * gas.inverse_gamma_minus_one<double>() + dW[0] * (0.5 * dot(s.v, s.v)) + W[0] * dot(s.v, d.v);
    const double m = W[0] * s.vn, dm = dW[0] * s.vn + W[0] * dvn;
    d.F = make_state<D>([&](int k) {
        return k == 0 ? dm : k == P ? (d.E + dW[P]) * s.vn + (s.E + W[P]) * dvn : dm * s.v[k - 1] + m * d.v[k - 1] + dW[P] * n[k - 1];
    });
    d.U = make_state<D>([&](int k) { return k == 0 ? dW[0] : k == P ? d.E : dW[0] * s.v[k - 1] + W[0] * d.v[k - 1]; });
    return d;
}

// The tangent of side(), sound speed included (0 where c^2 <= 0, as the forward). Used by: HLL::flux_tangent.
template <int D>
KOKKOS_INLINE_FUNCTION Side<D, double> side_tangent(const State<D>& W, const State<D>& dW, Vec<D> n, const Gas& gas, const Side<D, double>& s) {
    constexpr int P = D + 1;
    Side<D, double> d = side_with_tangent(W, dW, n, gas, s, dot(velocity(dW), n));
    const double c2 = gas.gamma * W[P] / W[0];
    d.c = c2 > 0 ? (gas.gamma * dW[P] - c2 * dW[0]) / W[0] / (2 * s.c) : 0.0;
    return d;
}

// The adjoint of side_with(): from the bars of F, U, and of E and v used elsewhere, adds J^T to Wbar; returns vn's bar.
// Used by: HLLC::flux_adjoint, side_adjoint().
template <int D>
KOKKOS_INLINE_FUNCTION double side_with_adjoint(const State<D>& W, Vec<D> n, const Gas& gas, const Side<D, double>& s, const State<D>& Fbar,
                                                const State<D>& Ubar, double Ebar, Vec<D> vbar, State<D>& Wbar) {
    constexpr int P = D + 1;
    const double m = W[0] * s.vn;
    Ebar += Fbar[P] * s.vn + Ubar[P];
    double Pbar = Fbar[P] * s.vn, vnbar = Fbar[P] * (s.E + W[P]), mbar = Fbar[0], rhobar = Ubar[0];
    for (int k = 0; k < D; ++k) {
        Pbar += Fbar[1 + k] * n[k];
        mbar += Fbar[1 + k] * s.v[k];
        vbar[k] += Fbar[1 + k] * m + Ubar[1 + k] * W[0];
        rhobar += Ubar[1 + k] * s.v[k];
    }
    rhobar += mbar * s.vn;
    vnbar += mbar * W[0];
    Pbar += Ebar * gas.inverse_gamma_minus_one<double>();
    rhobar += Ebar * (0.5 * dot(s.v, s.v));
    vbar = vbar + s.v * (Ebar * W[0]);
    Wbar[0] += rhobar;
    for (int k = 0; k < D; ++k) Wbar[1 + k] += vbar[k];
    Wbar[P] += Pbar;
    return vnbar;
}

// The adjoint of side(): from the bars of F, U, vn and c, adds J^T to Wbar. Used by: HLL::flux_adjoint.
template <int D>
KOKKOS_INLINE_FUNCTION void side_adjoint(const State<D>& W, Vec<D> n, const Gas& gas, const Side<D, double>& s, const State<D>& Fbar,
                                         const State<D>& Ubar, double vnbar, double cbar, State<D>& Wbar) {
    constexpr int P = D + 1;
    vnbar += side_with_adjoint(W, n, gas, s, Fbar, Ubar, 0.0, Vec<D>{}, Wbar);
    for (int k = 0; k < D; ++k) Wbar[1 + k] += vnbar * n[k];
    const double c2 = gas.gamma * W[P] / W[0];
    if (c2 > 0) {
        const double c2bar = cbar / (2 * s.c);
        Wbar[P] += c2bar * gas.gamma / W[0];
        Wbar[0] -= c2bar * c2 / W[0];
    }
}

// The adjoint of to_lab() (linear in F, w fixed); its tangent is to_lab() itself. Used by: HLL/HLLC::flux_adjoint.
template <int D>
KOKKOS_INLINE_FUNCTION State<D> to_lab_adjoint(const State<D>& Fbar, Vec<D> w) {
    constexpr int P = D + 1;
    State<D> G = Fbar;
    G[0] += 0.5 * dot(w, w) * Fbar[P];
    for (int k = 0; k < D; ++k) {
        G[0] += w[k] * Fbar[1 + k];
        G[1 + k] += w[k] * Fbar[P];
    }
    return G;
}

}  // namespace detail

// HLL, LUNA's _hll_core. Used by: FaceFlux::solve() / 1D Kernel as the Riemann parameter.
struct HLL {
    template <int D, class T>
    KOKKOS_INLINE_FUNCTION static StateT<D, T> flux(const StateT<D, T>& L, const StateT<D, T>& R, Vec<D, T> n,
                                                    Vec<D, T> w, const Gas& gas) {
        const detail::Side<D, T> l = detail::side(L, n, w, gas), r = detail::side(R, n, w, gas);
        const T SL = Kokkos::fmin(l.vn - l.c, r.vn - r.c);
        const T SR = Kokkos::fmax(l.vn + l.c, r.vn + r.c);
        StateT<D, T> F;
        if (SL >= T(0)) {
            F = l.F;
        } else if (SR <= T(0)) {
            F = r.F;
        } else {
            const T inv_dS = T(1) / (SR - SL);
            for (int v = 0; v < D + 2; ++v) F[v] = (SR * l.F[v] - SL * r.F[v] + SL * SR * (r.U[v] - l.U[v])) * inv_dS;
        }
        return detail::to_lab(F, w);
    }

    // The branch pattern of flux(): bit 0 S_L from L, bit 1 S_R from L, bits 2-3 the case (0 star, 1 F_L, 2 F_R), bits 4-5 c^2 > 0 on L, R.
    // n, w fixed. Used by: flux_tangent(), flux_adjoint(), the derivative tests (a sample is kept only if x +- h dx has the same pattern).
    template <int D>
    KOKKOS_INLINE_FUNCTION static int branches(const State<D>& L, const State<D>& R, Vec<D> n, Vec<D> w, const Gas& gas) {
        const detail::Side<D, double> l = detail::side(L, n, w, gas), r = detail::side(R, n, w, gas);
        const double a = l.vn - l.c, b = r.vn - r.c, A = l.vn + l.c, B = r.vn + r.c;
        const double SL = Kokkos::fmin(a, b), SR = Kokkos::fmax(A, B);
        const int wave = SL >= 0 ? 1 : SR <= 0 ? 2 : 0;
        const int cL = gas.gamma * L[D + 1] / L[0] > 0, cR = gas.gamma * R[D + 1] / R[0] > 0;
        return (b < a ? 0 : 1) | (B > A ? 0 : 2) | wave << 2 | cL << 4 | cR << 5;
    }
    // dF = J (dL, dR) of flux(), n and w fixed; S_L and S_R take the derivative of the side min/max picked. Used by: FiniteVolume::step_tangent, tests.
    template <int D>
    KOKKOS_INLINE_FUNCTION static State<D> flux_tangent(const State<D>& L, const State<D>& R, const State<D>& dL, const State<D>& dR, Vec<D> n,
                                                        Vec<D> w, const Gas& gas) {
        const detail::Side<D, double> l = detail::side(L, n, w, gas), r = detail::side(R, n, w, gas);
        const detail::Side<D, double> dl = detail::side_tangent(L, dL, n, gas, l), dr = detail::side_tangent(R, dR, n, gas, r);
        const int pattern = branches(L, R, n, w, gas);
        const double SL = pattern & 1 ? l.vn - l.c : r.vn - r.c, dSL = pattern & 1 ? dl.vn - dl.c : dr.vn - dr.c;
        const double SR = pattern & 2 ? l.vn + l.c : r.vn + r.c, dSR = pattern & 2 ? dl.vn + dl.c : dr.vn + dr.c;
        const int wave = (pattern >> 2) & 3;
        State<D> dF;
        if (wave == 1) {
            dF = dl.F;
        } else if (wave == 2) {
            dF = dr.F;
        } else {
            const double inv_dS = 1.0 / (SR - SL);
            for (int v = 0; v < D + 2; ++v) {
                const double dU = r.U[v] - l.U[v], F = (SR * l.F[v] - SL * r.F[v] + SL * SR * dU) * inv_dS;
                dF[v] = (dSR * l.F[v] + SR * dl.F[v] - dSL * r.F[v] - SL * dr.F[v] + (dSL * SR + SL * dSR) * dU + SL * SR * (dr.U[v] - dl.U[v]) -
                         F * (dSR - dSL)) *
                        inv_dS;
            }
        }
        return detail::to_lab(dF, w);
    }
    // (Lbar, Rbar) = J^T Fbar of flux(): adds to Lbar and Rbar. Used by: FiniteVolume::step_adjoint, the derivative tests.
    template <int D>
    KOKKOS_INLINE_FUNCTION static void flux_adjoint(const State<D>& L, const State<D>& R, Vec<D> n, Vec<D> w, const Gas& gas,
                                                    const State<D>& Fbar_lab, State<D>& Lbar, State<D>& Rbar) {
        const detail::Side<D, double> l = detail::side(L, n, w, gas), r = detail::side(R, n, w, gas);
        const int pattern = branches(L, R, n, w, gas);
        const double SL = pattern & 1 ? l.vn - l.c : r.vn - r.c, SR = pattern & 2 ? l.vn + l.c : r.vn + r.c;
        const int wave = (pattern >> 2) & 3;
        const State<D> Fbar = detail::to_lab_adjoint(Fbar_lab, w);
        State<D> lF{}, rF{}, lU{}, rU{};
        double SLbar = 0, SRbar = 0;
        if (wave == 1) {
            lF = Fbar;
        } else if (wave == 2) {
            rF = Fbar;
        } else {
            const double inv_dS = 1.0 / (SR - SL);
            for (int v = 0; v < D + 2; ++v) {
                const double dU = r.U[v] - l.U[v], F = (SR * l.F[v] - SL * r.F[v] + SL * SR * dU) * inv_dS;
                const double N = Fbar[v] * inv_dS;  // the numerator's bar
                SRbar += -Fbar[v] * F * inv_dS + N * (l.F[v] + SL * dU);
                SLbar += Fbar[v] * F * inv_dS + N * (-r.F[v] + SR * dU);
                lF[v] = SR * N;
                rF[v] = -SL * N;
                rU[v] = SL * SR * N;
                lU[v] = -SL * SR * N;
            }
        }
        double lvn = 0, lc = 0, rvn = 0, rc = 0;
        (pattern & 1 ? lvn : rvn) += SLbar;
        (pattern & 1 ? lc : rc) -= SLbar;
        (pattern & 2 ? lvn : rvn) += SRbar;
        (pattern & 2 ? lc : rc) += SRbar;
        detail::side_adjoint(L, n, gas, l, lF, lU, lvn, lc, Lbar);
        detail::side_adjoint(R, n, gas, r, rF, rU, rvn, rc, Rbar);
    }
};

// HLLC (the default Riemann), LUNA's _hllc_core with its guards: c^2 < 0 on either side takes F_R, divisions are kept off zero by 1e-20.
struct HLLC {
    // Used by: flux().
    template <class T>
    KOKKOS_INLINE_FUNCTION static T nonzero(T d) {
        constexpr T eps = T(1e-20);
        return Kokkos::fabs(d) < eps ? (d < 0 ? -eps : eps) : d;
    }
    template <int D, class T>
    KOKKOS_INLINE_FUNCTION static StateT<D, T> flux(const StateT<D, T>& L, const StateT<D, T>& R, Vec<D, T> n,
                                                    Vec<D, T> w, const Gas& gas) {
        constexpr int P = D + 1;
        // Wave speeds and the contact speed from the normal velocities and sound speeds alone; the flux is then built for one side only.
        const T vnL = dot(velocity(L) - w, n), vnR = dot(velocity(R) - w, n);
        // 1 / rho once per side, for c^2 and E* (products, not divisions: rounding differs from a division).
        const T inv_rhoL = T(1) / L[0], inv_rhoR = T(1) / R[0];
        const T c2L = T(gas.gamma) * L[P] * inv_rhoL, c2R = T(gas.gamma) * R[P] * inv_rhoR;
        const bool ok = c2L >= 0 && c2R >= 0;
        const T cL = c2L > 0 ? Kokkos::sqrt(c2L) : T(0), cR = c2R > 0 ? Kokkos::sqrt(c2R) : T(0);
        const T SL = Kokkos::fmin(vnL - cL, vnR - cR);
        const T SR = Kokkos::fmax(vnL + cL, vnR + cR);
        const T denomL = L[0] * (SL - vnL), denomR = R[0] * (SR - vnR);
        const T SM = (R[P] - L[P] + denomL * vnL - denomR * vnR) / nonzero(denomL - denomR);
        // LUNA's branch order: F_L, F*_L, F*_R, F_R.
        const bool left = ok && (SL >= 0 || SM >= 0);
        const bool star = ok && SL < 0 && (SM >= 0 || SR > 0);
        // The upwind state copied entry by entry: a reference to one of the two arrays put both in local memory.
        const StateT<D, T> W = make_state<D, T>([&](int v) { return left ? L[v] : R[v]; });
        const detail::Side<D, T> s = detail::side_with(W, n, w, gas, left ? vnL : vnR);
        if (!star) return detail::to_lab(s.F, w);
        // The star state: rho*, rho* (v_t + S_M n), rho* E*.
        const T S = left ? SL : SR, denom = left ? denomL : denomR;
        const T rho_star = denom / nonzero(S - SM);
        const T E_star = s.E * (left ? inv_rhoL : inv_rhoR) + (SM - s.vn) * (SM + W[P] * (T(1) / nonzero(denom)));
        const Vec<D, T> v_star = s.v + n * (SM - s.vn);
        const StateT<D, T> U_star = make_state<D, T>(
            [&](int k) { return k == 0 ? rho_star : k == P ? rho_star * E_star : rho_star * v_star[k - 1]; });
        StateT<D, T> F;
        for (int v = 0; v < D + 2; ++v) F[v] = s.F[v] + S * (U_star[v] - s.U[v]);
        return detail::to_lab(F, w);
    }

    // flux()'s wave speeds in double, as flux() computes them, and its branch pattern: bit 0 S_L from L, bit 1 S_R from L, bit 2 left,
    // bit 3 star, bit 4 ok (c^2 >= 0 both sides), bits 5-6 c^2 > 0 on L, R, bits 7-9 the 1e-20 guard on S_M's, rho*'s and E*'s divisor.
    // Used by: branches(), flux_tangent(), flux_adjoint().
    template <int D>
    struct Waves {
        double vnL, vnR, inv_rhoL, inv_rhoR, c2L, c2R, cL, cR, SL, SR, denomL, denomR, den, SM;
        int pattern;
    };
    template <int D>
    KOKKOS_INLINE_FUNCTION static Waves<D> waves(const State<D>& L, const State<D>& R, Vec<D> n, Vec<D> w, const Gas& gas) {
        constexpr int P = D + 1;
        constexpr double eps = 1e-20;
        Waves<D> a;
        a.vnL = dot(velocity(L) - w, n);
        a.vnR = dot(velocity(R) - w, n);
        a.inv_rhoL = 1.0 / L[0];
        a.inv_rhoR = 1.0 / R[0];
        a.c2L = gas.gamma * L[P] * a.inv_rhoL;
        a.c2R = gas.gamma * R[P] * a.inv_rhoR;
        const bool ok = a.c2L >= 0 && a.c2R >= 0;
        a.cL = a.c2L > 0 ? Kokkos::sqrt(a.c2L) : 0.0;
        a.cR = a.c2R > 0 ? Kokkos::sqrt(a.c2R) : 0.0;
        const double sl = a.vnL - a.cL, sr = a.vnR - a.cR, tl = a.vnL + a.cL, tr = a.vnR + a.cR;
        a.SL = Kokkos::fmin(sl, sr);
        a.SR = Kokkos::fmax(tl, tr);
        a.denomL = L[0] * (a.SL - a.vnL);
        a.denomR = R[0] * (a.SR - a.vnR);
        a.den = a.denomL - a.denomR;
        a.SM = (R[P] - L[P] + a.denomL * a.vnL - a.denomR * a.vnR) / nonzero(a.den);
        const bool left = ok && (a.SL >= 0 || a.SM >= 0);
        const bool star = ok && a.SL < 0 && (a.SM >= 0 || a.SR > 0);
        const double S = left ? a.SL : a.SR, denom = left ? a.denomL : a.denomR;
        a.pattern = (sr < sl ? 0 : 1) | (tr > tl ? 0 : 2) | left << 2 | star << 3 | ok << 4 | (a.c2L > 0) << 5 | (a.c2R > 0) << 6 |
                    (Kokkos::fabs(a.den) < eps) << 7 | (star && Kokkos::fabs(S - a.SM) < eps) << 8 | (star && Kokkos::fabs(denom) < eps) << 9;
        return a;
    }
    // The branch pattern of flux() (Waves::pattern). Used by: the derivative tests (a sample is kept only if x +- h dx has the same pattern).
    template <int D>
    KOKKOS_INLINE_FUNCTION static int branches(const State<D>& L, const State<D>& R, Vec<D> n, Vec<D> w, const Gas& gas) {
        return waves(L, R, n, w, gas).pattern;
    }
    // dF = J (dL, dR) of flux(), n and w fixed: the derivative of the case flux() took; a guarded divisor has derivative 0.
    // Used by: FiniteVolume::step_tangent, the derivative tests.
    template <int D>
    KOKKOS_INLINE_FUNCTION static State<D> flux_tangent(const State<D>& L, const State<D>& R, const State<D>& dL, const State<D>& dR, Vec<D> n,
                                                        Vec<D> w, const Gas& gas) {
        constexpr int P = D + 1;
        const Waves<D> a = waves(L, R, n, w, gas);
        const bool left = a.pattern & 4, star = a.pattern & 8;
        const double dvnL = dot(velocity(dL), n), dvnR = dot(velocity(dR), n);
        const double dinvL = -a.inv_rhoL * a.inv_rhoL * dL[0], dinvR = -a.inv_rhoR * a.inv_rhoR * dR[0];
        const double dcL = a.c2L > 0 ? gas.gamma * (dL[P] * a.inv_rhoL + L[P] * dinvL) / (2 * a.cL) : 0.0;
        const double dcR = a.c2R > 0 ? gas.gamma * (dR[P] * a.inv_rhoR + R[P] * dinvR) / (2 * a.cR) : 0.0;
        const double dSL = a.pattern & 1 ? dvnL - dcL : dvnR - dcR, dSR = a.pattern & 2 ? dvnL + dcL : dvnR + dcR;
        const double ddenomL = dL[0] * (a.SL - a.vnL) + L[0] * (dSL - dvnL), ddenomR = dR[0] * (a.SR - a.vnR) + R[0] * (dSR - dvnR);
        const double dnum = dR[P] - dL[P] + ddenomL * a.vnL + a.denomL * dvnL - ddenomR * a.vnR - a.denomR * dvnR;
        const double dden = a.pattern & 128 ? 0.0 : ddenomL - ddenomR, den = nonzero(a.den);
        const double dSM = (dnum - a.SM * dden) / den;
        // The upwind state copied entry by entry, as flux() does (a reference to one of two arrays puts both in local memory).
        const State<D> W = make_state<D>([&](int v) { return left ? L[v] : R[v]; });
        const State<D> dW = make_state<D>([&](int v) { return left ? dL[v] : dR[v]; });
        const double vn = left ? a.vnL : a.vnR, dvn = left ? dvnL : dvnR;
        const detail::Side<D, double> s = detail::side_with(W, n, w, gas, vn);
        const detail::Side<D, double> ds = detail::side_with_tangent(W, dW, n, gas, s, dvn);
        if (!star) return detail::to_lab(ds.F, w);
        const double S = left ? a.SL : a.SR, dS = left ? dSL : dSR, denom = left ? a.denomL : a.denomR, ddenom = left ? ddenomL : ddenomR;
        const double inv_side = left ? a.inv_rhoL : a.inv_rhoR, dinv_side = left ? dinvL : dinvR;
        const double q = nonzero(S - a.SM), dq = a.pattern & 256 ? 0.0 : dS - dSM;
        const double rho_star = denom / q, drho_star = (ddenom - rho_star * dq) / q;
        const double inv_denom = 1.0 / nonzero(denom), dinv_denom = a.pattern & 512 ? 0.0 : -inv_denom * inv_denom * ddenom;
        const double a2 = a.SM - s.vn, da2 = dSM - dvn;
        const double a3 = a.SM + W[P] * inv_denom, da3 = dSM + dW[P] * inv_denom + W[P] * dinv_denom;
        const double E_star = s.E * inv_side + a2 * a3;
        const double dE_star = ds.E * inv_side + s.E * dinv_side + da2 * a3 + a2 * da3;
        const Vec<D> v_star = s.v + n * a2, dv_star = ds.v + n * da2;
        State<D> dF;
        for (int v = 0; v < D + 2; ++v) {
            const double U_star = v == 0 ? rho_star : v == P ? rho_star * E_star : rho_star * v_star[v - 1];
            const double dU_star = v == 0   ? drho_star
                                   : v == P ? drho_star * E_star + rho_star * dE_star
                                            : drho_star * v_star[v - 1] + rho_star * dv_star[v - 1];
            dF[v] = ds.F[v] + dS * (U_star - s.U[v]) + S * (dU_star - ds.U[v]);
        }
        return detail::to_lab(dF, w);
    }
    // (Lbar, Rbar) = J^T Fbar of flux(): adds to Lbar and Rbar; only the case flux() took is differentiated, so an unselected
    // branch cannot put a NaN into the bars (LUNA needed its guards for that). Used by: FiniteVolume::step_adjoint, the derivative tests.
    template <int D>
    KOKKOS_INLINE_FUNCTION static void flux_adjoint(const State<D>& L, const State<D>& R, Vec<D> n, Vec<D> w, const Gas& gas,
                                                    const State<D>& Fbar_lab, State<D>& Lbar, State<D>& Rbar) {
        constexpr int P = D + 1;
        const Waves<D> a = waves(L, R, n, w, gas);
        const bool left = a.pattern & 4, star = a.pattern & 8;
        const State<D> Fbar = detail::to_lab_adjoint(Fbar_lab, w);
        const State<D> W = make_state<D>([&](int v) { return left ? L[v] : R[v]; });
        const double vn = left ? a.vnL : a.vnR;
        const detail::Side<D, double> s = detail::side_with(W, n, w, gas, vn);
        State<D> Wbar{};
        double vnLbar = 0, vnRbar = 0;
        if (!star) {
            (left ? vnLbar : vnRbar) += detail::side_with_adjoint(W, n, gas, s, Fbar, State<D>{}, 0.0, Vec<D>{}, Wbar);
        } else {
            const double S = left ? a.SL : a.SR, denom = left ? a.denomL : a.denomR, inv_side = left ? a.inv_rhoL : a.inv_rhoR;
            const double q = nonzero(S - a.SM), rho_star = denom / q, inv_denom = 1.0 / nonzero(denom);
            const double a2 = a.SM - s.vn, a3 = a.SM + W[P] * inv_denom, E_star = s.E * inv_side + a2 * a3;
            const Vec<D> v_star = s.v + n * a2;
            // F = s.F + S (U* - s.U)
            double Sbar = 0, rho_starbar = 0, E_starbar = 0;
            State<D> sUbar;
            Vec<D> v_starbar;
            for (int v = 0; v < D + 2; ++v) {
                const double U_star = v == 0 ? rho_star : v == P ? rho_star * E_star : rho_star * v_star[v - 1];
                Sbar += Fbar[v] * (U_star - s.U[v]);
                sUbar[v] = -S * Fbar[v];
                const double U_starbar = S * Fbar[v];
                if (v == 0) {
                    rho_starbar += U_starbar;
                } else if (v == P) {
                    rho_starbar += U_starbar * E_star;
                    E_starbar = U_starbar * rho_star;
                } else {
                    rho_starbar += U_starbar * v_star[v - 1];
                    v_starbar[v - 1] = U_starbar * rho_star;
                }
            }
            // v* = s.v + n a2, E* = s.E inv_side + a2 a3, a2 = S_M - vn, a3 = S_M + W_P / denom, rho* = denom / (S - S_M)
            const double a2bar = dot(v_starbar, n) + E_starbar * a3, a3bar = E_starbar * a2;
            double SMbar = a3bar + a2bar, vnbar = -a2bar;
            const double inv_denombar = a3bar * W[P];
            double denombar = (a.pattern & 512 ? 0.0 : -inv_denom * inv_denom * inv_denombar) + rho_starbar / q;
            const double qbar = a.pattern & 256 ? 0.0 : -rho_starbar * rho_star / q;
            Sbar += qbar;
            SMbar -= qbar;
            const double sEbar = E_starbar * inv_side, inv_sidebar = E_starbar * s.E;
            vnbar += detail::side_with_adjoint(W, n, gas, s, Fbar, sUbar, sEbar, v_starbar, Wbar);
            Wbar[P] += a3bar * inv_denom;
            // S_M = num / den, num = R_P - L_P + denomL vnL - denomR vnR, den = denomL - denomR
            const double den = nonzero(a.den), numbar = SMbar / den, denbar = a.pattern & 128 ? 0.0 : -SMbar * a.SM / den;
            double denomLbar = denbar + numbar * a.vnL, denomRbar = -denbar - numbar * a.vnR;
            (left ? denomLbar : denomRbar) += denombar;
            vnLbar += numbar * a.denomL;
            vnRbar -= numbar * a.denomR;
            Rbar[P] += numbar;
            Lbar[P] -= numbar;
            (left ? vnLbar : vnRbar) += vnbar;
            double invLbar = 0, invRbar = 0;
            (left ? invLbar : invRbar) += inv_sidebar;
            // denomL = L_0 (S_L - vnL), denomR = R_0 (S_R - vnR)
            double SLbar = denomLbar * L[0], SRbar = denomRbar * R[0];
            (left ? SLbar : SRbar) += Sbar;
            Lbar[0] += denomLbar * (a.SL - a.vnL);
            vnLbar -= denomLbar * L[0];
            Rbar[0] += denomRbar * (a.SR - a.vnR);
            vnRbar -= denomRbar * R[0];
            // S_L = min(vnL - cL, vnR - cR), S_R = max(vnL + cL, vnR + cR)
            double cLbar = 0, cRbar = 0;
            (a.pattern & 1 ? vnLbar : vnRbar) += SLbar;
            (a.pattern & 1 ? cLbar : cRbar) -= SLbar;
            (a.pattern & 2 ? vnLbar : vnRbar) += SRbar;
            (a.pattern & 2 ? cLbar : cRbar) += SRbar;
            // c = sqrt(gamma P / rho), 1 / rho per side
            if (a.c2L > 0) {
                const double c2bar = cLbar / (2 * a.cL);
                Lbar[P] += c2bar * gas.gamma * a.inv_rhoL;
                invLbar += c2bar * gas.gamma * L[P];
            }
            if (a.c2R > 0) {
                const double c2bar = cRbar / (2 * a.cR);
                Rbar[P] += c2bar * gas.gamma * a.inv_rhoR;
                invRbar += c2bar * gas.gamma * R[P];
            }
            Lbar[0] -= a.inv_rhoL * a.inv_rhoL * invLbar;
            Rbar[0] -= a.inv_rhoR * a.inv_rhoR * invRbar;
        }
        for (int v = 0; v < D + 2; ++v) {  // entry by entry, not a reference to one of the two arrays
            Lbar[v] += left ? Wbar[v] : 0.0;
            Rbar[v] += left ? 0.0 : Wbar[v];
        }
        for (int k = 0; k < D; ++k) {
            Lbar[1 + k] += vnLbar * n[k];
            Rbar[1 + k] += vnRbar * n[k];
        }
    }
};

}  // namespace hydro
