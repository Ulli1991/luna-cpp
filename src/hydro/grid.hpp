#pragma once

#include "euler.hpp"

#include <Kokkos_Core.hpp>

#include <cstdint>

// Unrolls the loop below it on the device, where an array indexed by a loop counter would otherwise live in local memory
// (-DLUNA_GRID_NO_UNROLL: not, as before 2026-11-20, for comparisons).
#if defined(__CUDA_ARCH__) && !defined(LUNA_GRID_NO_UNROLL)
#define LUNA_GRID_UNROLL _Pragma("unroll")
#else
#define LUNA_GRID_UNROLL
#endif

// The fixed-grid MUSCL-Hancock step of LUNA (research_notes/fixed_grid_2026-11-19.md), per cell: LUNA's numerics operation for
// operation, so a build without FMA contraction gives LUNA's bits. The kernels that call cell_step() are in grid_finite_volume.hpp
// (Kokkos) and grid_cuda.hpp (CUDA). Names follow LUNA's: slopes.py, jacobian.py, fluxes.py, conversions.py, integrators.py.
namespace hydro::grid {

// LUNA's LIMITER 0, 1, 2 (slopes.slope_limiter). Used by: Params, limit().
enum class Limiter { minmod = 0, mc = 1, van_leer = 2 };
// LUNA's BOUNDARY codes with two ghost cells (utils/halo_local.pad_halo_local; the wide path). Used by: Field.
enum class Boundary { periodic = 0, outflow = 1, reflective = 2, diode = 4 };

// What a step needs besides the fields: gas, limiter, 1 / dx per axis, LUNA's face rule (slopes.face_rule) and floors
// (preserve_face_states). The floor constants are computed as LUNA's Python computes them. Built by: make_params(). Used by: cell_step().
struct Params {
    double gamma = 1.4, gm1 = 0.4, gamma_gm1 = 0.56;  // gamma, gamma - 1, gamma (gamma - 1)
    double inv[3] = {1, 1, 1}, dx[3] = {1, 1, 1};
    double cfl = 0.8;
    Limiter limiter = Limiter::mc;
    bool floors = false, allow_first_order = false;
    double smallr = 1e-10;
    double smallr_c2 = 0, c2_over_gamma = 0, smalle = 0;  // smallr smallc^2, smallc^2 / gamma, smallc^2 / gamma / (gamma - 1)
};

// Params of a grid with spacings dx (LUNA: dx = L / N, inv = 1 / dx). floors: preserve_face_states with (smallr, smallc).
// Used by: GridFiniteVolume, the tests.
inline Params make_params(double gamma, const double (&dx)[3], double cfl, Limiter limiter, bool floors = false, double smallr = 1e-10,
                          double smallc = 1e-10, bool allow_first_order = false) {
    Params p;
    p.gamma = gamma;
    p.gm1 = gamma - 1.0;
    p.gamma_gm1 = gamma * (gamma - 1.0);
    for (int a = 0; a < 3; ++a) {
        p.dx[a] = dx[a];
        p.inv[a] = 1.0 / dx[a];
    }
    p.cfl = cfl;
    p.limiter = limiter;
    p.floors = floors;
    p.allow_first_order = allow_first_order;
    p.smallr = smallr;
    const double c2 = smallc * smallc;
    p.smallr_c2 = smallr * c2;
    p.c2_over_gamma = c2 / gamma;
    p.smalle = c2 / gamma / (gamma - 1.0);
    return p;
}

// slopes.minmod: the smaller magnitude when a and b share a sign, else 0. Used by: limit().
template <class T>
KOKKOS_INLINE_FUNCTION T minmod(T a, T b) {
    if (!(a * b > T(0))) return T(0);
    const T m = nan_min(Kokkos::fabs(a), Kokkos::fabs(b));
    return a > T(0) ? m : -m;
}
// slopes.minmod3. Used by: limit() (MC).
template <class T>
KOKKOS_INLINE_FUNCTION T minmod3(T a, T b, T c) {
    if (!(a * b > T(0) && a * c > T(0))) return T(0);
    const T m = nan_min(Kokkos::fabs(a), nan_min(Kokkos::fabs(b), Kokkos::fabs(c)));
    return a > T(0) ? m : -m;
}
// slopes.slope_limiter on the left and right differences: minmod, MC (moncen) or van Leer (with LUNA's normal-float threshold).
// Used by: cell_step().
template <class T>
KOKKOS_INLINE_FUNCTION T limit(T dl, T dr, Limiter limiter) {
    if (limiter == Limiter::minmod) return minmod(dl, dr);
    if (limiter == Limiter::mc) return minmod3(T(2) * dl, T(2) * dr, T(0.5) * (dl + dr));
    const bool same_sign = dl * dr > Kokkos::Experimental::norm_min_v<T>;
    const T denom = same_sign ? dl + dr : T(1);
    return same_sign ? T(2) * dl * dr / denom : T(0);
}

// conversions.cons_to_prim, or cons_to_prim_floored when p.floors. Used by: cell_step(), primitives(), the drivers.
template <int D>
KOKKOS_INLINE_FUNCTION State<D> to_primitive(const State<D>& U, const Params& p) {
    constexpr int P = D + 1;
    State<D> W;
    const double rho = p.floors ? nan_max(U[0], p.smallr) : U[0];
    const double inv_rho = 1.0 / rho;
    double sum = 0;
    LUNA_GRID_UNROLL
    for (int a = 0; a < D; ++a) {
        W[1 + a] = U[1 + a] * inv_rho;
        sum = a == 0 ? W[1] * W[1] : sum + W[1 + a] * W[1 + a];
    }
    const double ke = 0.5 * rho * sum;
    W[0] = rho;
    W[P] = p.floors ? p.gm1 * nan_max(U[P] - ke, p.smalle * rho) : p.gm1 * (U[P] - ke);
    return W;
}

// conversions.prim_to_cons as XLA runs it: a division by the constant gamma - 1 becomes a product with its reciprocal (XLA rewrites x / c).
// Used by: the drivers' initial states.
template <int D>
KOKKOS_INLINE_FUNCTION State<D> to_conserved(const State<D>& W, const Params& p) {
    constexpr int P = D + 1;
    State<D> U;
    double sum = 0;
    LUNA_GRID_UNROLL
    for (int a = 0; a < D; ++a) {
        U[1 + a] = W[0] * W[1 + a];
        sum = a == 0 ? W[1] * W[1] : sum + W[1 + a] * W[1 + a];
    }
    U[0] = W[0];
    U[P] = W[P] * (1.0 / p.gm1) + 0.5 * W[0] * sum;
    return U;
}

// integrators.cell_dt of the conserved U, CFL / sum_d (|v_d| + c) / dx_d with c^2 = gamma P / rho, as XLA compiles LUNA's dt function
// (its HLO): P = (gamma - 1) e is not formed, the constants fold into c^2 = e (gamma (gamma - 1)) / rho, and / dx_d becomes (1 / dx_d).
// e = E - ke, floored as to_primitive(). NaN as -inf (so the minimum shows it). Used by: cell_step(), GridFiniteVolume::refresh().
template <int D>
KOKKOS_INLINE_FUNCTION double cell_dt(const State<D>& U, const Params& p) {
    constexpr int P = D + 1;
    const double rho = p.floors ? nan_max(U[0], p.smallr) : U[0];
    const double inv_rho = 1.0 / rho;
    double v[3] = {0, 0, 0}, sq = 0;
    LUNA_GRID_UNROLL
    for (int a = 0; a < D; ++a) {
        v[a] = U[1 + a] * inv_rho;
        sq = a == 0 ? v[0] * v[0] : sq + v[a] * v[a];
    }
    const double ke = 0.5 * rho * sq;
    const double e = p.floors ? nan_max(U[P] - ke, p.smalle * rho) : U[P] - ke;
    const double c = Kokkos::sqrt(e * p.gamma_gm1 / rho);
    double sum = 0;
    LUNA_GRID_UNROLL
    for (int a = 0; a < D; ++a) {
        const double term = (Kokkos::fabs(v[a]) + c) * p.inv[a];
        sum = a == 0 ? term : sum + term;
    }
    const double dt = p.cfl / sum;
    return dt != dt ? -Kokkos::Experimental::infinity_v<double> : dt;
}

// A^d dW summed over the axes (jacobian.primitive_jacobian_rows_1d, _2d, _3d_axis, each as LUNA sums it); g[d]: the slope along d
// times 1 / dx_d. Used by: predict().
template <int D>
KOKKOS_INLINE_FUNCTION State<D> rate(const State<D>& W, const State<D> (&g)[D], double gamma) {
    State<D> A;
    if constexpr (D == 1) {
        const double rho = W[0], vx = W[1], P = W[2], c2 = gamma * P / rho;
        A[0] = vx * g[0][0] + rho * g[0][1];
        A[1] = vx * g[0][1] + (1.0 / rho) * g[0][2];
        A[2] = rho * c2 * g[0][1] + vx * g[0][2];
    } else if constexpr (D == 2) {
        const double rho = W[0], vx = W[1], vy = W[2], P = W[3], c2 = gamma * P / rho, inv_rho = 1.0 / rho;
        const State<2>&gx = g[0], &gy = g[1];
        A[0] = (vx * gx[0] + rho * gx[1]) + (vy * gy[0] + rho * gy[2]);
        A[1] = (vx * gx[1] + inv_rho * gx[3]) + vy * gy[1];
        A[2] = vx * gx[2] + (vy * gy[2] + inv_rho * gy[3]);
        A[3] = (rho * c2 * gx[1] + vx * gx[3]) + (rho * c2 * gy[2] + vy * gy[3]);
    } else {
        const double rho = W[0], P = W[4], rho_c2 = gamma * P, inv_rho = 1.0 / rho;
        LUNA_GRID_UNROLL
        for (int d = 0; d < 3; ++d) {
            const double vn = W[1 + d];
            State<3> r;
            r[0] = vn * g[d][0] + rho * g[d][1 + d];
            LUNA_GRID_UNROLL
            for (int k = 0; k < 3; ++k) r[1 + k] = k == d ? vn * g[d][1 + k] + inv_rho * g[d][4] : vn * g[d][1 + k];
            r[4] = rho_c2 * g[d][1 + d] + vn * g[d][4];
            LUNA_GRID_UNROLL
            for (int v = 0; v < 5; ++v) A[v] = d == 0 ? r[v] : A[v] + r[v];
        }
    }
    return A;
}

// The half-step predictor W - (dt / 2) A dW (Pallas _predict). h = 0.5 dt. Used by: cell_step().
template <int D>
KOKKOS_INLINE_FUNCTION State<D> predict(const State<D>& W, const State<D> (&slope)[D], const Params& p, double h) {
    State<D> g[D];
    LUNA_GRID_UNROLL
    for (int d = 0; d < D; ++d)
        LUNA_GRID_UNROLL
        for (int v = 0; v < D + 2; ++v) g[d][v] = slope[d][v] * p.inv[d];
    const State<D> A = rate<D>(W, g, p.gamma);
    State<D> half;
    LUNA_GRID_UNROLL
    for (int v = 0; v < D + 2; ++v) half[v] = W[v] - h * A[v];
    return half;
}

// slopes.face_rule on a face state Wf of the cell with centre Wc: refused (NaN) or first order (Wc) where rho or p <= 0 (audit C8),
// then the RAMSES floors (ramses_face) when p.floors. Used by: face_flux().
template <int D>
KOKKOS_INLINE_FUNCTION State<D> face_rule(State<D> W, const State<D>& Wc, const Params& p) {
    constexpr int P = D + 1;
    const bool bad = W[0] <= 0.0 || W[P] <= 0.0;
    if (!(p.floors && !p.allow_first_order) && bad)
        LUNA_GRID_UNROLL
        for (int v = 0; v < D + 2; ++v) W[v] = p.allow_first_order ? Wc[v] : Kokkos::Experimental::quiet_NaN_v<double>;
    if (!p.floors) return W;
    double rho = W[0] < p.smallr ? Wc[0] : W[0];
    const double pressure = W[P] < p.smallr_c2 ? Wc[P] : W[P];
    rho = nan_max(rho, p.smallr);
    W[0] = rho;
    W[P] = nan_max(pressure, rho * p.c2_over_gamma);
    return W;
}

// The flux across the face between a left and a right cell on axis d (Pallas _face): face states W_half +- slope / 2, the face rule,
// the Riemann core in the face's frame (fluxes.rotate_primitive_pair: the other axes in increasing order, 0 past the dimension).
// Used by: cell_step().
template <int D, class Riemann>
KOKKOS_INLINE_FUNCTION State<D> face_flux(const State<D>& halfL, const State<D>& slopeL, const State<D>& centreL, const State<D>& halfR,
                                          const State<D>& slopeR, const State<D>& centreR, int d, const Params& p) {
    constexpr int P = D + 1;
    State<D> L, R;
    LUNA_GRID_UNROLL
    for (int v = 0; v < D + 2; ++v) {
        L[v] = halfL[v] + 0.5 * slopeL[v];
        R[v] = halfR[v] - 0.5 * slopeR[v];
    }
    L = face_rule<D>(L, centreL, p);
    R = face_rule<D>(R, centreR, p);
    const int t1 = d == 0 ? 1 : 0, t2 = d == 2 ? 1 : 2;  // the transverse axes in increasing order
    auto side = [&](const State<D>& W) {
        return AxisSide<double>{W[0], W[1 + d], W[P], t1 < D ? W[1 + t1] : 0.0, t2 < D ? W[1 + t2] : 0.0};
    };
    const AxisFlux<double> F = Riemann::axis_core(side(L), side(R), p.gamma);
    State<D> out;
    out[0] = F[0];
    out[1 + d] = F[1];
    if (t1 < D) out[1 + t1] = F[2];
    if (t2 < D) out[1 + t2] = F[3];
    out[P] = F[4];
    return out;
}

// Ghost index on an axis of n cells (c in [-2, n + 1]) and what it does to the axis's velocity: 0 nothing, 1 negate (reflective),
// 2 min(v, 0) (diode, low side), 3 max(v, 0) (diode, high side). Used by: Field::operator().
KOKKOS_INLINE_FUNCTION int ghost(int c, int n, Boundary b, int& velocity) {
    velocity = 0;
    if (c >= 0 && c < n) return c;
    const bool low = c < 0;
    switch (b) {
        case Boundary::periodic:
            return low ? c + n : c - n;
        case Boundary::reflective:
            velocity = 1;
            return low ? -1 - c : 2 * n - 1 - c;
        case Boundary::diode:
            velocity = low ? 2 : 3;
            return low ? 0 : n - 1;
        default:
            return low ? 0 : n - 1;
    }
}

// The primitives W on the grid, structure of arrays (variable v of cell (i, j, k) at v cells + (i ny + j) nz + k: z fastest, as LUNA's
// (nvar, Nx, Ny, Nz) arrays), read through the boundary's ghosts. Used by: cell_step(), the kernels.
template <int D>
struct Field {
    const double* W = nullptr;
    long cells = 0;
    int n[3] = {1, 1, 1};
    Boundary boundary = Boundary::outflow;

    KOKKOS_INLINE_FUNCTION State<D> operator()(int i, int j, int k) const {
        int fix[3] = {0, 0, 0};
        const int c[3] = {i, j, k};
        int m[3] = {0, 0, 0};
        LUNA_GRID_UNROLL
        for (int a = 0; a < D; ++a) m[a] = ghost(c[a], n[a], boundary, fix[a]);
        const long at = (static_cast<long>(m[0]) * n[1] + m[1]) * n[2] + m[2];
        State<D> w;
        LUNA_GRID_UNROLL
        for (int v = 0; v < D + 2; ++v) w[v] = W[v * cells + at];
        LUNA_GRID_UNROLL
        for (int a = 0; a < D; ++a) {
            double& u = w[1 + a];
            if (fix[a] == 1) u = -u;
            if (fix[a] == 2) u = nan_min(u, 0.0);
            if (fix[a] == 3) u = nan_max(u, 0.0);
        }
        return w;
    }
};

// -div F of cell (i, j, k) (Pallas _call, the fused kernel: the cell's predictor and its 2 D neighbours', its 2 D faces; dU summed
// x, then y, then z, each (F_minus - F_plus) / dx_d). h = 0.5 dt. Used by: cell_step().
template <int D, class Riemann>
KOKKOS_INLINE_FUNCTION State<D> divergence(const Field<D>& W, int i, int j, int k, const Params& p, double h) {
    const int o[3] = {i, j, k};
    auto at = [&](int a, int s, int b = 0, int t = 0) {  // W at o + s e_a + t e_b
        int c[3] = {o[0], o[1], o[2]};
        c[a] += s;
        c[b] += t;
        return W(c[0], c[1], c[2]);
    };
    const State<D> W0 = W(i, j, k);
    State<D> near[D][2];  // o - e_d, o + e_d
    LUNA_GRID_UNROLL
    for (int d = 0; d < D; ++d) {
        near[d][0] = at(d, -1);
        near[d][1] = at(d, 1);
    }
    State<D> slope0[D];
    LUNA_GRID_UNROLL
    for (int d = 0; d < D; ++d)
        LUNA_GRID_UNROLL
        for (int v = 0; v < D + 2; ++v) slope0[d][v] = limit(W0[v] - near[d][0][v], near[d][1][v] - W0[v], p.limiter);
    const State<D> half0 = predict<D>(W0, slope0, p, h);
    State<D> dU;
    LUNA_GRID_UNROLL
    for (int d = 0; d < D; ++d) {
        State<D> F[2];
        LUNA_GRID_UNROLL
        for (int s = 0; s < 2; ++s) {
            const int step = 2 * s - 1;
            const State<D>& Wq = near[d][s];
            State<D> slope[D];
            LUNA_GRID_UNROLL
            for (int e = 0; e < D; ++e) {
                if (e == d) {
                    const State<D> far = at(d, 2 * step);
                    LUNA_GRID_UNROLL
                    for (int v = 0; v < D + 2; ++v)
                        slope[e][v] = s == 0 ? limit(Wq[v] - far[v], W0[v] - Wq[v], p.limiter) : limit(Wq[v] - W0[v], far[v] - Wq[v], p.limiter);
                } else {
                    const State<D> lo = at(d, step, e, -1), hi = at(d, step, e, 1);
                    LUNA_GRID_UNROLL
                    for (int v = 0; v < D + 2; ++v) slope[e][v] = limit(Wq[v] - lo[v], hi[v] - Wq[v], p.limiter);
                }
            }
            const State<D> half = predict<D>(Wq, slope, p, h);
            F[s] = s == 0 ? face_flux<D, Riemann>(half, slope[d], Wq, half0, slope0[d], W0, d, p)
                          : face_flux<D, Riemann>(half0, slope0[d], W0, half, slope[d], Wq, d, p);
        }
        LUNA_GRID_UNROLL
        for (int v = 0; v < D + 2; ++v) {
            const double contribution = (F[0][v] - F[1][v]) * p.inv[d];
            dU[v] = d == 0 ? contribution : dU[v] + contribution;
        }
    }
    return dU;
}

// The half-step states of the split step (Pallas _predict_call): structure of arrays on the grid plus one ghost ring per axis, cell
// (i, j, k) at index (i + 1, j + 1, k + 1) on the grid's axes. Written by: predictor_cell(). Used by: divergence_split().
template <int D>
struct HalfField {
    double* H = nullptr;
    long cells = 0;  // the padded cells
    int m[3] = {1, 1, 1};

    KOKKOS_INLINE_FUNCTION long index(int i, int j, int k) const {
        const int c[3] = {i, j, k};
        int q[3] = {0, 0, 0};
        LUNA_GRID_UNROLL
        for (int a = 0; a < D; ++a) q[a] = c[a] + 1;
        return (static_cast<long>(q[0]) * m[1] + q[1]) * m[2] + q[2];
    }
    KOKKOS_INLINE_FUNCTION State<D> operator()(int i, int j, int k) const {
        const long at = index(i, j, k);
        State<D> h;
        LUNA_GRID_UNROLL
        for (int v = 0; v < D + 2; ++v) h[v] = H[v * cells + at];
        return h;
    }
};

// Pass A of the split step: the half-step state of cell (i, j, k), i, j, k from -1 to n on the grid's axes (the cells and one ghost
// ring), from all of its slopes; the same predict() as divergence(). h = 0.5 dt. Used by: the split step's predictor kernels.
template <int D>
KOKKOS_INLINE_FUNCTION void predictor_cell(const Field<D>& W, const HalfField<D>& out, int i, int j, int k, const Params& p, double h) {
    const int o[3] = {i, j, k};
    const State<D> W0 = W(i, j, k);
    State<D> slope[D];
    LUNA_GRID_UNROLL
    for (int d = 0; d < D; ++d) {
        int lo[3] = {o[0], o[1], o[2]}, hi[3] = {o[0], o[1], o[2]};
        lo[d] -= 1;
        hi[d] += 1;
        const State<D> Wl = W(lo[0], lo[1], lo[2]), Wh = W(hi[0], hi[1], hi[2]);
        LUNA_GRID_UNROLL
        for (int v = 0; v < D + 2; ++v) slope[d][v] = limit(W0[v] - Wl[v], Wh[v] - W0[v], p.limiter);
    }
    const State<D> half = predict<D>(W0, slope, p, h);
    const long at = out.index(i, j, k);
    LUNA_GRID_UNROLL
    for (int v = 0; v < D + 2; ++v) out.H[v * out.cells + at] = half[v];
}

// Pass B of the split step (Pallas _faces_call): -div F of cell (i, j, k) from the stored half-step states and the slopes along each
// axis, recomputed (each cell's own and its two neighbours'); the same faces, sums and order as divergence(). Used by: cell_step().
template <int D, class Riemann>
KOKKOS_INLINE_FUNCTION State<D> divergence_split(const Field<D>& W, const HalfField<D>& H, int i, int j, int k, const Params& p) {
    const int o[3] = {i, j, k};
    auto at = [&](int d, int s) {  // W at o + s e_d
        int c[3] = {o[0], o[1], o[2]};
        c[d] += s;
        return W(c[0], c[1], c[2]);
    };
    auto half = [&](int d, int s) {  // the half-step state at o + s e_d
        int c[3] = {o[0], o[1], o[2]};
        c[d] += s;
        return H(c[0], c[1], c[2]);
    };
    const State<D> W0 = W(i, j, k), H0 = H(i, j, k);
    State<D> dU;
    LUNA_GRID_UNROLL
    for (int d = 0; d < D; ++d) {
        const State<D> Wm = at(d, -1), Wp = at(d, 1), Wmm = at(d, -2), Wpp = at(d, 2);
        State<D> s0, sm, sp;
        LUNA_GRID_UNROLL
        for (int v = 0; v < D + 2; ++v) {
            s0[v] = limit(W0[v] - Wm[v], Wp[v] - W0[v], p.limiter);
            sm[v] = limit(Wm[v] - Wmm[v], W0[v] - Wm[v], p.limiter);
            sp[v] = limit(Wp[v] - W0[v], Wpp[v] - Wp[v], p.limiter);
        }
        const State<D> Fm = face_flux<D, Riemann>(half(d, -1), sm, Wm, H0, s0, W0, d, p);
        const State<D> Fp = face_flux<D, Riemann>(H0, s0, W0, half(d, 1), sp, Wp, d, p);
        LUNA_GRID_UNROLL
        for (int v = 0; v < D + 2; ++v) {
            const double contribution = (Fm[v] - Fp[v]) * p.inv[d];
            dU[v] = d == 0 ? contribution : dU[v] + contribution;
        }
    }
    return dU;
}

// Raw arrays of one step: W and U now, U and W next (structure of arrays, `cells` apart). Used by: cell_step(), the kernels.
struct Arrays {
    const double* U = nullptr;
    double* U_next = nullptr;
    double* W_next = nullptr;
};

// One cell's whole step: U_next = U + dt (-div F) (integrators._step_inner), its primitives for the next step, and its next time step
// (returned; -inf for NaN). Split: -div F from the half-step states in H (divergence_split), else the fused divergence().
// Used by: GridFiniteVolume's Kokkos kernel, grid_cuda.hpp's kernel.
template <int D, class Riemann, bool Split = false>
KOKKOS_INLINE_FUNCTION double cell_step(const Field<D>& W, const Arrays& a, int i, int j, int k, const Params& p, double dt,
                                        const HalfField<D>& H = {}) {
    const long at = (static_cast<long>(i) * W.n[1] + j) * W.n[2] + k;
    State<D> dU;
    if constexpr (Split)
        dU = divergence_split<D, Riemann>(W, H, i, j, k, p);
    else
        dU = divergence<D, Riemann>(W, i, j, k, p, 0.5 * dt);
    State<D> U;
    LUNA_GRID_UNROLL
    for (int v = 0; v < D + 2; ++v) {
        U[v] = a.U[v * W.cells + at] + dt * dU[v];
        a.U_next[v * W.cells + at] = U[v];
    }
    const State<D> Wn = to_primitive<D>(U, p);
    LUNA_GRID_UNROLL
    for (int v = 0; v < D + 2; ++v) a.W_next[v * W.cells + at] = Wn[v];
    return cell_dt<D>(U, p);
}

// A double's bits as an unsigned key in the same order (for an atomic minimum), and back. Used by: grid_cuda.hpp.
KOKKOS_INLINE_FUNCTION std::uint64_t order_key(double x) {
    const auto b = Kokkos::bit_cast<std::uint64_t>(x);
    return b >> 63 ? ~b : b | (std::uint64_t(1) << 63);
}
KOKKOS_INLINE_FUNCTION double from_order_key(std::uint64_t k) {
    return Kokkos::bit_cast<double>(k >> 63 ? k & ~(std::uint64_t(1) << 63) : ~k);
}

}  // namespace hydro::grid
