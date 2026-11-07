#pragma once

#include <Kokkos_Core.hpp>

#include <cstdint>
#include <type_traits>
#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_fp16.h>
#endif

// Precision policy, stage 0 (compile time): picks the storage types of hydro::FiniteVolume<D, Exec, Riemann, P> (finite_volume.hpp).
// Chosen by --precision in hydro_sedov / hydro_wave3d (run_precision, main); Precision<> is all double.
namespace hydro {

// Tag for the Gradient slot of Precision: FP16 storage of float gradients.
// Used by: Precision (half, gradient_store), FiniteVolume::Sweep (stores), FiniteVolume::FaceFlux (loads).
struct Half {};

// The precision policy P of FiniteVolume; Q, volumes, centroids, W and the sums of the face fluxes are double in every policy.
//   Geometry  face rows the hydro reads (area vector, centroid, owner, neighbour): double, or float rows of the mesh
//   Weight    void: the least-squares fit is solved in the gradient sweep; float or double: the mesh stores each face
//             side's fit weight and the gradient is a weighted sum
//   Gradient  gradients, limiter and face-state change: double, float, or Half (float arithmetic, FP16 storage with one
//             power-of-two exponent per group rho, v, P)
//   Flux      the stored face fluxes (per-face loop): double or float, summed in double
//   Riemann   the Riemann solve: double, or float in the frame of the face (flux back to the lab frame in double)
// Used by: FiniteVolume (typedefs Mesh, G, GStore, FluxStore), hydro_sedov run<P, R>, hydro_wave3d run<P>.
template <class Geometry = double, class Weight = void, class Gradient = double, class Flux = double,
          class Riemann = double>
struct Precision {
    static_assert(std::is_same_v<Geometry, double> || std::is_same_v<Geometry, float>, "geometry: double or float");
    static_assert(std::is_void_v<Weight> || std::is_same_v<Weight, double> || std::is_same_v<Weight, float>,
                  "weight: void, double or float");
    using weight = Weight;
    static_assert(std::is_same_v<Gradient, double> || std::is_same_v<Gradient, float> || std::is_same_v<Gradient, Half>,
                  "gradient: double, float or Half");
    static_assert(std::is_same_v<Flux, double> || std::is_same_v<Flux, float>, "flux: double or float");
    static_assert(std::is_same_v<Riemann, double> || std::is_same_v<Riemann, float>, "riemann: double or float");
    using geometry = Geometry;
    using gradient = std::conditional_t<std::is_same_v<Gradient, Half>, float, Gradient>;
    using gradient_store = std::conditional_t<std::is_same_v<Gradient, Half>, std::uint16_t, gradient>;
    static constexpr bool half = std::is_same_v<Gradient, Half>;
    using flux = Flux;
    using riemann = Riemann;
};

// The gradients, limiter and face-state change in float, the rest double (`--precision=grad32`).
// Used by: DefaultPrecision<3>, run_precision (hydro_sedov), main (hydro_wave3d).
using Grad32 = Precision<double, void, float>;

// The default P of FiniteVolume: Grad32 in 3D (L1 unchanged to ~1e-3, +8 to +12 %, 64 B/cell less; README); double in 2D and 1D.
// Used by: FiniteVolume<D> default argument, run_precision / main (`--precision=default`).
template <int D>
using DefaultPrecision = std::conditional_t<D == 3, Grad32, Precision<>>;

namespace half {

// float -> IEEE binary16 bits, round to nearest even.
// Used by: FiniteVolume::Sweep (stores each gradient component, scaled by power_of_two(k)).
KOKKOS_INLINE_FUNCTION std::uint16_t from_float(float f) {
#if defined(__CUDA_ARCH__)
    return __half_as_ushort(__float2half_rn(f));
#else
    const std::uint32_t x = Kokkos::bit_cast<std::uint32_t>(f);
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    const int exponent = static_cast<int>((x >> 23) & 0xff) - 127 + 15;
    std::uint32_t mantissa = x & 0x7fffffu;
    if (((x >> 23) & 0xff) == 0xff) return static_cast<std::uint16_t>(sign | 0x7c00u | (mantissa ? 0x200u : 0u));
    if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    if (exponent <= 0) {
        if (exponent < -10) return static_cast<std::uint16_t>(sign);
        mantissa |= 0x800000u;
        const int shift = 14 - exponent;
        std::uint32_t h = mantissa >> shift;
        const std::uint32_t rest = mantissa & ((1u << shift) - 1), halfway = 1u << (shift - 1);
        if (rest > halfway || (rest == halfway && (h & 1))) ++h;
        return static_cast<std::uint16_t>(sign | h);
    }
    std::uint32_t h = static_cast<std::uint32_t>(exponent) << 10 | mantissa >> 13;
    const std::uint32_t rest = mantissa & 0x1fffu;
    if (rest > 0x1000u || (rest == 0x1000u && (h & 1))) ++h;
    return static_cast<std::uint16_t>(sign | h);
#endif
}

// IEEE binary16 bits -> float, the inverse of from_float.
// Used by: FiniteVolume::FaceFlux (reads gradients, then multiplies by power_of_two(-k)).
KOKKOS_INLINE_FUNCTION float to_float(std::uint16_t h) {
#if defined(__CUDA_ARCH__)
    return __half2float(__ushort_as_half(h));
#else
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    const int exponent = (h >> 10) & 0x1f;
    std::uint32_t mantissa = h & 0x3ffu;
    if (exponent == 0x1f) return Kokkos::bit_cast<float>(sign | 0x7f800000u | mantissa << 13);
    if (exponent == 0) {
        if (mantissa == 0) return Kokkos::bit_cast<float>(sign);
        int e = -1;
        do {
            mantissa <<= 1;
            ++e;
        } while (!(mantissa & 0x400u));
        return Kokkos::bit_cast<float>(sign | static_cast<std::uint32_t>(127 - 15 - e) << 23 | (mantissa & 0x3ffu) << 13);
    }
    return Kokkos::bit_cast<float>(sign | static_cast<std::uint32_t>(exponent - 15 + 127) << 23 | mantissa << 13);
#endif
}

// 2^k for k in [-126, 127].
// Used by: FiniteVolume::Sweep (scale 2^k on store), FiniteVolume::FaceFlux (2^-k on load).
KOKKOS_INLINE_FUNCTION float power_of_two(int k) { return Kokkos::bit_cast<float>(static_cast<std::uint32_t>(k + 127) << 23); }

// The exponent k of a group with largest magnitude m: m 2^k in [2^14, 2^15) (k = 0 for m = 0), within [-114, 126].
// Used by: FiniteVolume::Sweep (m is the largest |gradient| of the group in the cell; k goes to power_of_two).
KOKKOS_INLINE_FUNCTION int group_exponent(float m) {
    if (!(m > 0)) return 0;
    const int e = static_cast<int>((Kokkos::bit_cast<std::uint32_t>(m) >> 23) & 0xff) - 127;
    return Kokkos::clamp(14 - e, -114, 126);
}

// The group (0 rho, 1 velocity, 2 pressure) of gradient component c = D v + a, v the variable, a the axis.
// Used by: FiniteVolume::Sweep and FaceFlux, to pick the scale of component c.
template <int D>
KOKKOS_INLINE_FUNCTION int group(int c) {
    return c < D ? 0 : c < D + D * D ? 1 : 2;
}

}  // namespace half

// Entries of a row of `count` values of type T, padded to whole 32-byte sectors.
// Used by: FiniteVolume (gradient_width, the row length of the gradient array).
template <class T>
constexpr int row_width(int count) {
    constexpr int per_sector = 32 / static_cast<int>(sizeof(T));
    return (count + per_sector - 1) / per_sector * per_sector;
}

}  // namespace hydro
