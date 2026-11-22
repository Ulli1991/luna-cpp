#pragma once

// Test-only: the derivatives of a 2D Voronoi mesh's geometry in the generator positions at fixed topology (hydro::GeometryDerivative),
// on the host from the mesh's triangles, until voronoi-cpp provides them. Used by: hydro_derivatives_test.

#include "geometry_derivative.hpp"

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace reference {

using V2 = std::array<double, 2>;
inline V2 operator+(V2 a, V2 b) { return {a[0] + b[0], a[1] + b[1]}; }
inline V2 operator-(V2 a, V2 b) { return {a[0] - b[0], a[1] - b[1]}; }
inline V2 operator*(V2 a, double s) { return {a[0] * s, a[1] * s}; }
inline double dot2(V2 a, V2 b) { return a[0] * b[0] + a[1] * b[1]; }

// The circumcentre of the origin, a and b, relative to the origin (tri_mesh::CircleAs), and its tangent.
inline V2 circle(V2 a, V2 b) {
    const double la = dot2(a, a), lb = dot2(b, b), det = a[0] * b[1] - a[1] * b[0];
    return {(la * b[1] - lb * a[1]) / (2 * det), (lb * a[0] - la * b[0]) / (2 * det)};
}
inline V2 circle_tangent(V2 a, V2 b, V2 da, V2 db) {
    const double la = dot2(a, a), lb = dot2(b, b), det = a[0] * b[1] - a[1] * b[0];
    const double dla = 2 * dot2(a, da), dlb = 2 * dot2(b, db);
    const double ddet = da[0] * b[1] + a[0] * db[1] - da[1] * b[0] - a[1] * db[0];
    const V2 c = circle(a, b);
    return {(dla * b[1] + la * db[1] - dlb * a[1] - lb * da[1]) / (2 * det) - c[0] * ddet / det,
            (dlb * a[0] + lb * da[0] - dla * b[0] - la * db[0]) / (2 * det) - c[1] * ddet / det};
}

// Voronoi vertex of triangle t: its circumcentre, p0 + C(a, b) with a, b the nearest-image offsets of corners 1, 2 from corner 0.
// Face f, the edge opposite corner k of t with t' across it (as tri_mesh::faces_as): e = c_t - c_t', A = sign (e_y, -e_x),
// x_f = c_t - x_owner - e / 2. Cell i (as tet_cells.hpp): p_f = |A_f . d_f| with d_f the separation to the other cell,
// P = sum p_f, V = P / 4, c = (2/3) sum p_f x_f,i / P with x_f,i the face centroid relative to i.
template <class Mesh, class Memory>
class Geometry2D : public hydro::GeometryDerivative<2, Memory> {
public:
    using Base = hydro::GeometryDerivative<2, Memory>;
    using typename Base::Points;
    using typename Base::Scalars;

    explicit Geometry2D(const Mesh& mesh) {
        const auto& tri = mesh.triangles();
        const auto corners = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tri.corners);
        const auto links = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tri.links);
        const auto x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.positions());
        const auto owner = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.owner());
        const auto neighbour = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.neighbour());
        const auto start = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.cell_face_start());
        const auto list = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.cell_face());
        const auto& box = mesh.sides();
        for (int a = 0; a < 2; ++a) side_[a] = box.side[a];
        n_ = mesh.particles();
        faces_ = mesh.faces();
        x_.resize(n_);
        for (int i = 0; i < n_; ++i) x_[i] = {x(i, 0), x(i, 1)};
        // Triangles: corners, offsets a, b, the circumcentre's absolute position (near corner 0).
        const int nt = tri.count();
        tri_.resize(nt);
        for (int t = 0; t < nt; ++t) {
            Tri& T = tri_[t];
            for (int k = 0; k < 3; ++k) T.v[k] = corners(t, k);
            T.a = wrap(x_[T.v[1]] - x_[T.v[0]]);
            T.b = wrap(x_[T.v[2]] - x_[T.v[0]]);
            T.c = x_[T.v[0]] + circle(T.a, T.b);
        }
        // Faces: the two triangles, the sign, the owner.
        face_.assign(faces_, Face{});
        for (int t = 0; t < nt; ++t)
            for (int k = 0; k < 3; ++k) {
                if (corners(t, k) >= corners(t, 3 + k)) continue;
                const int a = corners(t, (k + 1) % 3), b = corners(t, (k + 2) % 3), f = links(t, 3 + k);
                Face& F = face_[f];
                F.t = t;
                F.u = links(t, k);
                F.sign = a < b ? 1.0 : -1.0;
                F.o = a < b ? a : b;
                F.j = a < b ? b : a;
                const V2 e = wrap(tri_[t].c - tri_[F.u].c);
                F.A = {F.sign * e[1], -F.sign * e[0]};
                F.xf = wrap(tri_[t].c - x_[F.o]) - e * 0.5;
            }
        for (int f = 0; f < faces_; ++f)
            if (face_[f].o != owner(f) || face_[f].j != neighbour(f)) throw std::logic_error("reference::Geometry2D: face numbering");
        // Cells: the list, p_f, the face centroid relative to the cell, P and the moment.
        cell_.resize(n_);
        for (int i = 0; i < n_; ++i) {
            Cell& C = cell_[i];
            for (int k = start(i); k < start(i + 1); ++k) {
                const int f = list(k) & 0x7fffffff;
                const Face& F = face_[f];
                const int other = F.o == i ? F.j : F.o;
                const V2 d = wrap(x_[other] - x_[i]);
                const double Ad = dot2(F.A, d);
                Entry E{f, other, Ad >= 0 ? 1.0 : -1.0, std::fabs(Ad), F.o == i ? F.xf : F.xf + wrap(x_[F.o] - x_[i]), d};
                C.P += E.p;
                C.M = C.M + E.xf * E.p;
                C.entries.push_back(E);
            }
        }
    }

    // The forward values, for the test's comparison with the mesh: V, c (per cell), A, x_f (per face).
    double volume(int i) const { return cell_[i].P / 4; }
    V2 centroid(int i) const { return cell_[i].M * (kappa / cell_[i].P); }
    V2 area(int f) const { return face_[f].A; }
    V2 face_centroid(int f) const { return face_[f].xf; }

    void tangent(const Points& dx_d, const Scalars& dV_d, const Points& dc_d, const Points& dA_d, const Points& dxf_d) const override {
        const auto dx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dx_d);
        auto at = [&](int i) { return V2{dx(i, 0), dx(i, 1)}; };
        std::vector<V2> dct(tri_.size()), dA(faces_), dxf(faces_);
        for (std::size_t t = 0; t < tri_.size(); ++t) {
            const Tri& T = tri_[t];
            const V2 d0 = at(T.v[0]);
            dct[t] = d0 + circle_tangent(T.a, T.b, at(T.v[1]) - d0, at(T.v[2]) - d0);
        }
        for (int f = 0; f < faces_; ++f) {
            const Face& F = face_[f];
            const V2 de = dct[F.t] - dct[F.u];
            dA[f] = {F.sign * de[1], -F.sign * de[0]};
            dxf[f] = dct[F.t] - at(F.o) - de * 0.5;
        }
        const auto dV = Kokkos::create_mirror_view(dV_d);
        const auto dc = Kokkos::create_mirror_view(dc_d);
        const auto hA = Kokkos::create_mirror_view(dA_d);
        const auto hxf = Kokkos::create_mirror_view(dxf_d);
        for (int i = 0; i < n_; ++i) {
            const Cell& C = cell_[i];
            double dP = 0;
            V2 dM{0, 0};
            for (const Entry& E : C.entries) {
                const Face& F = face_[E.f];
                const V2 dd = at(E.other) - at(i);
                const double dp = E.s * (dot2(dA[E.f], E.d) + dot2(F.A, dd));
                const V2 dxfi = F.o == i ? dxf[E.f] : dxf[E.f] + at(F.o) - at(i);
                dP += dp;
                dM = dM + E.xf * dp + dxfi * E.p;
            }
            dV(i) = dP / 4;
            const V2 g = (dM - C.M * (dP / C.P)) * (kappa / C.P);
            dc(i, 0) = g[0];
            dc(i, 1) = g[1];
        }
        for (int f = 0; f < faces_; ++f)
            for (int a = 0; a < 2; ++a) {
                hA(f, a) = dA[f][a];
                hxf(f, a) = dxf[f][a];
            }
        Kokkos::deep_copy(dV_d, dV);
        Kokkos::deep_copy(dc_d, dc);
        Kokkos::deep_copy(dA_d, hA);
        Kokkos::deep_copy(dxf_d, hxf);
    }

    void adjoint(const Scalars& Vbar_d, const Points& cbar_d, const Points& Abar_d, const Points& xfbar_d, const Points& xbar_d) const override {
        const auto Vbar = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Vbar_d);
        const auto cbar = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cbar_d);
        const auto Ab = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Abar_d);
        const auto xfb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, xfbar_d);
        const auto xbar = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, xbar_d);
        std::vector<V2> xb(n_, V2{0, 0}), Abar(faces_), xfbar(faces_), ctbar(tri_.size(), V2{0, 0});
        for (int f = 0; f < faces_; ++f) {
            Abar[f] = {Ab(f, 0), Ab(f, 1)};
            xfbar[f] = {xfb(f, 0), xfb(f, 1)};
        }
        for (int i = 0; i < n_; ++i) {
            const Cell& C = cell_[i];
            const V2 cb{cbar(i, 0), cbar(i, 1)};
            // V = P / 4, c = kappa M / P
            const double Pbar = Vbar(i) / 4 - kappa * dot2(cb, C.M) / (C.P * C.P);
            const V2 Mbar = cb * (kappa / C.P);
            for (const Entry& E : C.entries) {
                const Face& F = face_[E.f];
                const double pbar = Pbar + dot2(Mbar, E.xf);
                const V2 xfibar = Mbar * E.p;
                // dp = s (dA . d + A . dd), dd = dx_other - dx_i
                Abar[E.f] = Abar[E.f] + E.d * (pbar * E.s);
                const V2 ddbar = F.A * (pbar * E.s);
                xb[E.other] = xb[E.other] + ddbar;
                xb[i] = xb[i] - ddbar;
                // dx_f,i = dx_f (+ dx_owner - dx_i when i is not the owner)
                xfbar[E.f] = xfbar[E.f] + xfibar;
                if (F.o != i) {
                    xb[F.o] = xb[F.o] + xfibar;
                    xb[i] = xb[i] - xfibar;
                }
            }
        }
        for (int f = 0; f < faces_; ++f) {
            const Face& F = face_[f];
            // dA = sign (de_y, -de_x); dx_f = dc_t - dx_o - de / 2; de = dc_t - dc_u
            V2 debar{-F.sign * Abar[f][1], F.sign * Abar[f][0]};
            debar = debar - xfbar[f] * 0.5;
            ctbar[F.t] = ctbar[F.t] + xfbar[f] + debar;
            ctbar[F.u] = ctbar[F.u] - debar;
            xb[F.o] = xb[F.o] - xfbar[f];
        }
        for (std::size_t t = 0; t < tri_.size(); ++t) {
            const Tri& T = tri_[t];
            // dc_t = dx_0 + C'(dx_1 - dx_0, dx_2 - dx_0): C' column by column
            const V2 unit[2] = {{1, 0}, {0, 1}};
            V2 abar{0, 0}, bbar{0, 0};
            for (int a = 0; a < 2; ++a) {
                abar[a] = dot2(ctbar[t], circle_tangent(T.a, T.b, unit[a], V2{0, 0}));
                bbar[a] = dot2(ctbar[t], circle_tangent(T.a, T.b, V2{0, 0}, unit[a]));
            }
            xb[T.v[0]] = xb[T.v[0]] + ctbar[t] - abar - bbar;
            xb[T.v[1]] = xb[T.v[1]] + abar;
            xb[T.v[2]] = xb[T.v[2]] + bbar;
        }
        for (int i = 0; i < n_; ++i)
            for (int a = 0; a < 2; ++a) xbar(i, a) += xb[i][a];
        Kokkos::deep_copy(xbar_d, xbar);
    }

private:
    static constexpr double kappa = 2.0 / 3.0;  // the pyramid centroid's fraction in 2D, D / (D + 1)
    struct Tri {
        int v[3];
        V2 a, b, c;
    };
    struct Face {
        int t = -1, u = -1, o = -1, j = -1;
        double sign = 1;
        V2 A{0, 0}, xf{0, 0};
    };
    struct Entry {
        int f, other;
        double s, p;
        V2 xf, d;  // the face centroid relative to the cell, the separation to the other cell
    };
    struct Cell {
        double P = 0;
        V2 M{0, 0};
        std::vector<Entry> entries;
    };
    V2 wrap(V2 d) const {
        for (int a = 0; a < 2; ++a) d[a] -= side_[a] * std::nearbyint(d[a] / side_[a]);
        return d;
    }
    double side_[2] = {1, 1};
    int n_ = 0, faces_ = 0;
    std::vector<V2> x_;
    std::vector<Tri> tri_;
    std::vector<Face> face_;
    std::vector<Cell> cell_;
};

}  // namespace reference
