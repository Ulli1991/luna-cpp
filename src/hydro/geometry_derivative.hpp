#pragma once

#include <Kokkos_Core.hpp>

namespace hydro {

// The derivatives of a Voronoi mesh's geometry with respect to its generator positions, at fixed topology (the same faces and
// cell lists), as the moving step's tangent and adjoint read them. The mesh library is to provide these (voronoi-cpp issue:
// "Geometry derivatives at fixed topology"); until it does, the derivative tests use a reference implementation of their own.
//
// The quantities are those FiniteVolume::Geometry reads:
//  - V_i: cell volumes;
//  - c_i: cell centroids relative to the generator;
//  - A_f: face area vectors (owner to neighbour);
//  - x_f: face centroids relative to the owner.
// The mesh computes them from offsets on its integer grid; the derivative is that of the same formulas with the offsets
// continuous. The separations x_j - x_i the hydro also reads have the derivative dx_j - dx_i, which the hydro takes itself.
template <int D, class Memory>
struct GeometryDerivative {
    using Points = Kokkos::View<double* [D], Kokkos::LayoutRight, Memory>;  // one row per cell or per face
    using Scalars = Kokkos::View<double*, Memory>;
    virtual ~GeometryDerivative() = default;
    // dx (per cell) -> dV, dc (per cell), dA, dx_f (per face), written.
    virtual void tangent(const Points& dx, const Scalars& dV, const Points& dc, const Points& dA, const Points& dxf) const = 0;
    // The transpose: from the bars of V, c (per cell), A, x_f (per face), adds to xbar (per cell).
    virtual void adjoint(const Scalars& Vbar, const Points& cbar, const Points& Abar, const Points& xfbar, const Points& xbar) const = 0;
};

}  // namespace hydro
