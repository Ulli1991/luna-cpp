#pragma once

#include "euler.hpp"
#include "precision.hpp"
#include "voronoi_mesh.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

// Solver stage (D = 2, 3): driver calls time_step() then step(); this class owns Q and turns the voronoi::Mesh<D> into updated Q.
// The state is the extensive Q per cell; the mesh (static or moving) is the only thing taken from voronoi-cpp. Design: docs/design.md.
namespace hydro {

enum class Update { per_cell, per_face };

// Used by: Sweep (limiter). Where the gradient limiter keeps the extrapolated values within the neighbours' range: face centroids, or nowhere.
// minmod and van_leer: the 1D hydro only, LUNA's limiters 0 and 2.
enum class Limiter { face_centroid, none, minmod, van_leer };

enum class Motion { static_mesh, moving };

// The time step rule: AREPO next generation's, or LUNA's (2D, RAMSES): the smallest of courant / sum_d (|v_d - w_d| + c) / dx.
enum class StepRule { arepo, luna };

// Used by: phase_end(), filled by time_step() and step(). Phase times in ms, summed over steps (fenced); per-cell loop: fluxes covers the update too.
struct Timings {
    double primitives = 0, sweep = 0, fluxes = 0, update = 0, drift = 0, mesh = 0, cells = 0;
};

// Used by: move(), mesh_stats(). What the mesh updates of a moving run did, summed over steps.
struct MeshStats {
    long long updates = 0, rebuilds = 0, stuck = 0, host_flips = 0, device_flips = 0, resorts = 0;
    long long stuck_steps = 0, local_repairs = 0, local_cells = 0;  // updates with stuck entries; local repairs (tet_local)
};

template <int D, class ExecSpace = Kokkos::DefaultExecutionSpace, class Riemann = HLLC,
          class Precision = hydro::DefaultPrecision<D>>
class FiniteVolume {
public:
    static_assert(D == 2 || D == 3, "hydro::FiniteVolume: D = 1 is finite_volume_1d.hpp");
    static constexpr int V = D + 2;
    using P = Precision;
    using Memory = typename ExecSpace::memory_space;
    using Mesh = voronoi::Mesh<D, ExecSpace, typename P::geometry>;
    using State = hydro::State<D>;
    using Vec = hydro::Vec<D>;
    using G = typename P::gradient;
    using GStore = typename P::gradient_store;
    using FluxStore = typename P::flux;
    static constexpr int gradient_width = row_width<GStore>(V * D);
    using Cells = Kokkos::View<double* [V], Kokkos::LayoutRight, Memory>;
    using Points = Kokkos::View<double* [D], Kokkos::LayoutRight, Memory>;
    // Whole steps of the mesh grid each point moves over a moving step (int32: exact, 12 B a cell in 3D).
    using Steps = Kokkos::View<int* [D], Kokkos::LayoutRight, Memory>;
    using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    // Per cell and step, written by primitives(): W = primitive(Q / V), the sound speed, the radius.
    using Prims = Kokkos::View<double* [V + 2], Kokkos::LayoutRight, Memory, Unmanaged>;
    using Gradients = Kokkos::View<GStore* [gradient_width], Kokkos::LayoutRight, Memory, Unmanaged>;
    using Exponents = Kokkos::View<std::uint32_t*, Memory, Unmanaged>;  // FP16 gradients: one byte per group
    using FaceFluxes = Kokkos::View<FluxStore* [V], Kokkos::LayoutRight, Memory, Unmanaged>;

    // AREPO next generation's mesh regularisation, as its Sedov examples set it (docs/design.md).
    static constexpr double cell_shaping_speed = 0.5, cell_max_angle_factor = 2.25;

    // mesh: built, update_cells() done. Static: its geometry must not change. Moving: this object owns the positions.
    // Moving at order 1: MUSCL-Hancock on a moving mesh, step_moving().
    // owner_layout (3D): the lists carry the owner bit (voronoi::Layout::owner_bit), set here if the mesh lacks it. Same bits.
    FiniteVolume(const ExecSpace& exec, Mesh& mesh, const Gas& gas, int order, Update update,
                 Limiter limiter = Limiter::face_centroid, Motion motion = Motion::static_mesh, bool owner_layout = true)
        : exec_(exec), mesh_(&mesh), gas_(gas), order_(order), update_(update), limiter_(limiter), motion_(motion),
          both_meshes_(motion == Motion::moving && order == 1), n_(mesh.particles()) {
        if (order != 0 && order != 1) throw std::invalid_argument("hydro::FiniteVolume: order 0 or 1");
        if constexpr (D == 3) {
            if (owner_layout && !mesh.layout().owner_bit) {
                // The caller chose no layout: no re-sort either. The face storage stays as built (float centroids only from a build).
                voronoi::Layout layout = fast_layout(motion, no_resort(mesh.layout()));
                layout.float_centroids = mesh.layout().float_centroids;
                mesh.set_layout(layout);
                mesh.update_cells();
            }
        }
        q_ = Cells("hydro_Q", n_);
        if (order == 0 && update == Update::per_cell)
            next_ = Cells(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("hydro_Q_next")), n_);
        if (motion == Motion::moving) {
            x_ = Points(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("hydro_x")), n_);
            w_ = Points("hydro_w", n_);
            copy_positions();
        }
        if (both_meshes_) {
            held_ = Kokkos::View<char*, Memory>(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("hydro_cell_rows")), cell_row_bytes());
            old_centroid_ = Points(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("hydro_old_centroid")), n_);
            steps_ = Steps(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("hydro_grid_steps")), n_);
        }
        refresh();
    }

    // `layout` without re-sorts. voronoi-cpp re-sorts every 25 updates by default; the hydro keeps it off unless a caller
    // asks (--every): a re-sort costs about a step, and at 256³ moving it does not fit in 40 GB
    // (research_notes/moving_structure_2026-11-18.md). Used by: fast_layout, the constructor, hydro_wave3d.
    static voronoi::Layout no_resort(voronoi::Layout layout = {}) {
        layout.every = 0;
        layout.flips = 0;
        layout.far = 0;
        return layout;
    }

    // The mesh layout the hydro reads fastest: the owner bit, and on a static mesh faces numbered by owner (on a moving mesh
    // that renumbering costs the mesh more than it saves here); no re-sort unless `layout` asks for one. Set it before the
    // mesh's first update (faces are numbered there).
    static voronoi::Layout fast_layout(Motion motion, voronoi::Layout layout = no_resort()) {
        layout.owner_bit = true;
        layout.owner_faces = layout.owner_faces || motion == Motion::static_mesh;
        // Moving: face centroids as float relative to the owner (12 B a face, not 24; the mesh writes and the hydro reads less).
        layout.float_centroids = layout.float_centroids || (motion == Motion::moving && !Mesh::float_rows);
        return layout;
    }

    // Static mesh: from here on the hydro reads only the mesh arrays it holds (faces, face geometry, lists, cell volumes
    // and centroids, positions) and keeps its rows in scratch of its own, so the caller may destroy the voronoi::Mesh and
    // with it what only a moving mesh needs: tetrahedra, circumcentres, repair state, rings, face numbering. Same bits.
    // The arrays stay readable through geometry(). Throws on a moving mesh.
    void release_mesh() {
        if (motion_ != Motion::static_mesh) throw std::logic_error("hydro::FiniteVolume::release_mesh: static meshes only");
        refresh();
        mesh_ = nullptr;
        magnitudes_valid_ = false;  // they lived in the mesh's scratch
    }
    bool holds_mesh() const { return mesh_ != nullptr; }
    int faces() const { return faces_; }

    // The extensive conserved state Q(i, v) = (mass, momentum, energy); set it before stepping.
    // A caller may change Q through it, so the primitive rows of the last update are not trusted after this call.
    Cells state() const {
        prims_ready_ = false;
        return q_;
    }
    // Moving: the positions of the mesh-generating points and their velocities.
    Points positions() const { return x_; }
    Points point_velocities() const { return w_; }
    int order() const { return order_; }
    Update update() const { return update_; }
    Motion motion() const { return motion_; }
    const Gas& gas() const { return gas_; }
    const MeshStats& mesh_stats() const { return mesh_stats_; }

    // Host copy of the limited density gradients of the last time_step(), one row per cell; for output.
    Kokkos::View<double**, Kokkos::HostSpace> density_gradients() const {
        Kokkos::View<double**, Kokkos::HostSpace> rows("density_gradients", n_, D);
        if constexpr (P::half) {  // FP16 rows: decoded as FaceFlux::gradient does (the density group's exponent)
            const auto g = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gradients_);
            const auto k = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, exponents_);
            for (std::ptrdiff_t i = 0; i < std::ptrdiff_t(n_); ++i) {
                const float scale = half::power_of_two(-static_cast<int>(static_cast<std::int8_t>(k(i))));
                for (int a = 0; a < D; ++a) rows(i, a) = static_cast<double>(half::to_float(g(i, a)) * scale);
            }
        } else {
            const auto g = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, gradients_);
            for (std::ptrdiff_t i = 0; i < std::ptrdiff_t(n_); ++i)
                for (int a = 0; a < D; ++a) rows(i, a) = g(i, a);
        }
        return rows;
    }
    void set_step_rule(StepRule rule) { rule_ = rule; }

    // Call order of one step (driver):
    //  1. time_step(courant): refresh(), borrow_scratch(), primitives(), sweep() -> dt
    //  2. step(dt): reuses that sweep, then advance<Order>() -> apply() (fluxes, Q update)
    //     or step_moving() when moving at order 1
    //  3. moving only: move() drifts the points and updates the mesh, then refresh()
    // The global time step: the smallest of AREPO next generation's cell time steps, no power-of-two bins. Returns dt.
    // courant: safety factor; timings: if given, the phase times are added. Also runs the sweep the next step() uses.
    double time_step(double courant, Timings* timings = nullptr) {
        timer_.reset();
        refresh();
        borrow_scratch();
        if (!prims_ready_) primitives();
        prims_ready_ = false;
        phase_end(timings, &Timings::primitives);
        const double dt = sweep(courant);
        swept_ = true;
        phase_end(timings, &Timings::sweep);
        return dt;
    }

    // One step of length dt with the gradients of the last time_step() (swept again if there was none since the last step).
    // Used by: the driver after time_step(); calls advance(), step_moving(), move().
    void step(double dt, Timings* timings = nullptr) {
        timer_.reset();
        refresh();
        if (!swept_) {
            borrow_scratch();
            if (!prims_ready_) primitives();
            prims_ready_ = false;
            phase_end(timings, &Timings::primitives);
            sweep(1.0);
            phase_end(timings, &Timings::sweep);
        }
        swept_ = false;
        if (both_meshes_) {
            step_moving(dt, timings);
            return;
        }
        if (order_ == 1)
            advance<1>(dt, timings);
        else
            advance<0>(dt, timings);
        if (motion_ == Motion::moving) move(dt, timings);
    }

    // Face sides of the last step that took the cell's state because the extrapolated one was not physical.
    // Read back from the device here, once a step (both passes in one copy), so the flux passes need no fence of their own.
    long long fallbacks() const {
        if (!fallbacks_read_) {
            Kokkos::deep_copy(exec_, fallback_host_, fallback_counts_);
            exec_.fence();
            fallbacks_last_ = fallback_host_(0) + fallback_host_(1);
            fallbacks_read_ = true;
        }
        return fallbacks_last_;
    }

    // The radius of a sphere (D = 3) or circle (D = 2) of volume v. Used by: primitives().
    KOKKOS_INLINE_FUNCTION static double radius(double v) {
        if constexpr (D == 3)
            return Kokkos::cbrt(v * (3.0 / (4.0 * M_PI)));
        else
            return Kokkos::sqrt(v * (1.0 / M_PI));
    }
    // The size of a face of area (2D: length) |A| in AREPO's face angle. Used by: Sweep (moving mesh regularisation).
    KOKKOS_INLINE_FUNCTION static double face_angle(double area) {
        if constexpr (D == 3)
            return Kokkos::sqrt(area / M_PI);
        else
            return 0.5 * area;
    }

    // Snapshot of the mesh arrays and per-face/per-cell accessors (float face rows are rounded once by the mesh, read as double).
    // Used by: Sweep, FaceStep, FaceFlux, apply(), primitives(); rebuilt by refresh().
    struct Geometry {
        static constexpr bool float_rows = Mesh::float_rows;
        Kokkos::View<const int*, Memory> owner, neighbour, start, face;
        typename Mesh::Rows area, face_centroid, centroid;
        typename Mesh::FaceRows rows;
        Kokkos::View<const double*, Memory> volume;
        typename Mesh::Positions positions;
        exact_predicates::BoxN<D> box;
        bool bit = false;  // list entries carry the owner bit (voronoi::Layout::owner_bit)
        // Face centroids as float relative to the owner (voronoi::Layout::float_centroids, 3D); face_centroid is then empty.
        Kokkos::View<const float* [3], Kokkos::LayoutRight, Memory> centroid32{};

        template <class Rows>
        KOKKOS_INLINE_FUNCTION Vec row(const Rows& v, int i) const {
            return make_vec<D>([&](int a) { return v(i, a); });
        }
        // The conserved state of cell i per unit volume.
        KOKKOS_INLINE_FUNCTION State per_volume(const Cells& q, int i) const {
            const double inv = 1.0 / volume(i);
            return make_state<D>([&](int v) { return q(i, v) * inv; });
        }
        // Nearest image of generator j seen from generator i.
        KOKKOS_INLINE_FUNCTION Vec separation(int i, int j) const {
            return make_vec<D>(
                [&](int a) { return tet_mesh::wrap(positions(j, a) - positions(i, a), box.side[a], box.inverse[a]); });
        }
        KOKKOS_INLINE_FUNCTION int owner_of(int f) const {
            if constexpr (float_rows)
                return Kokkos::bit_cast<int>(rows(f, 6));
            else
                return owner(f);
        }
        KOKKOS_INLINE_FUNCTION int neighbour_of(int f) const {
            if constexpr (float_rows)
                return Kokkos::bit_cast<int>(rows(f, 7));
            else
                return neighbour(f);
        }
        KOKKOS_INLINE_FUNCTION Vec area_of(int f) const {
            if constexpr (float_rows)
                return make_vec<D>([&](int a) { return static_cast<double>(rows(f, a)); });
            else
                return row(area, f);
        }
        KOKKOS_INLINE_FUNCTION Vec face_centroid_of(int f) const {
            if constexpr (float_rows)
                return make_vec<D>([&](int a) { return static_cast<double>(rows(f, 3 + a)); });
            else if (centroid32.extent(0) != 0)  // not data(): an empty view of zero rows may have a pointer
                return make_vec<D>([&](int a) { return static_cast<double>(centroid32(f, a)); });
            else
                return row(face_centroid, f);
        }
        // Both loads issued at once (no dependent second load).
        KOKKOS_INLINE_FUNCTION int other(int f, int i) const {
            const int o = owner_of(f), n = neighbour_of(f);
            return o == i ? n : o;
        }
        // List entry k of cell i: its face, whether i owns it, the other cell (one load with the mesh's owner bit).
        KOKKOS_INLINE_FUNCTION int face_at(int k) const { return tet_mesh::list_face(face(k)); }
        KOKKOS_INLINE_FUNCTION bool owned_at(int k, int f, int i) const { return bit ? face(k) >= 0 : owner_of(f) == i; }
        KOKKOS_INLINE_FUNCTION int other_at(int k, int f, int i) const {
            if (!bit) return other(f, i);
            return face(k) >= 0 ? neighbour_of(f) : owner_of(f);
        }
        // Face centroid of face f relative to the centroid of cell i (a side of f).
        KOKKOS_INLINE_FUNCTION Vec face_offset(int f, int i) const {
            const int o = owner_of(f);
            const Vec relative = o == i ? face_centroid_of(f) : face_centroid_of(f) + separation(i, o);
            return relative - row(centroid, i);
        }
    };
    // The mesh arrays the hydro reads (after release_mesh() the only copies). Used by: drivers after release_mesh().
    const Geometry& geometry() const { return geometry_; }

    // Public only because CUDA device lambdas may not be defined in private member functions.

    // Fences and adds the elapsed time to Timings::*part. Used by: time_step(), step(), move(), apply().
    void phase_end(Timings* timings, double Timings::*part) {
        if (!timings) return;
        exec_.fence();
        timings->*part += timer_.seconds() * 1e3;
        timer_.reset();
    }

    // The positions start from the mesh's snapped ones. Used by: the constructor (moving).
    void copy_positions() {
        const auto from = mesh_->positions();
        const auto x = x_;
        Kokkos::parallel_for(
            "hydro::positions", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), KOKKOS_LAMBDA(int i) {
                for (int a = 0; a < D; ++a) x(i, a) = from(i, a);
            });
    }

    // The mesh's current arrays (an update may reallocate them). Used by: constructor, time_step(), step(), move().
    // After release_mesh() the arrays in geometry_ are the only ones, and nothing changes them.
    void refresh() {
        if (!mesh_) return;
        const Mesh& mesh = *mesh_;
        faces_ = mesh.faces();
        geometry_ = {mesh.owner(),     mesh.neighbour(), mesh.cell_face_start(), mesh.cell_face(),
                     mesh.area(),      mesh.face_centroid(), mesh.centroid(),    mesh.face_rows(),
                     mesh.volume(),    mesh.positions(),  mesh.sides()};
        if constexpr (D == 3) {
            geometry_.bit = mesh.layout().owner_bit;
            geometry_.centroid32 = mesh.face_centroid32();
        }
    }

    // Sizes of the per-cell rows, each part on a 32-byte boundary. Used by: borrow_scratch(), the constructor.
    static constexpr std::size_t sectors(std::size_t bytes) { return (bytes + 31) / 32 * 32; }
    std::size_t prim_bytes() const { return sectors(sizeof(double) * (V + 2) * static_cast<std::size_t>(n_)); }
    std::size_t gradient_bytes() const { return order_ == 1 ? sizeof(GStore) * gradient_width * static_cast<std::size_t>(n_) : 0; }
    std::size_t exponent_bytes() const { return order_ == 1 && P::half ? sectors(sizeof(std::uint32_t) * static_cast<std::size_t>(n_)) : 0; }
    std::size_t cell_row_bytes() const { return prim_bytes() + gradient_bytes() + exponent_bytes(); }

    // Scratch bytes: the mesh's spare ring pool, or after release_mesh() an array of the hydro's own, grown to fit.
    // Used by: borrow_scratch().
    char* scratch(std::size_t bytes) {
        if (mesh_) return mesh_->scratch(bytes).data();
        if (own_scratch_.extent(0) < bytes)
            own_scratch_ = Kokkos::View<char*, Memory>(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("hydro_scratch")), bytes);
        return own_scratch_.data();
    }

    // Primitive, gradient (and FP16 exponent), then face flux rows in the mesh's scratch, 32-byte aligned; a moving step keeps the cell rows in held_.
    // Used by: time_step(), step(), step_moving(); must precede primitives().
    void borrow_scratch() {
        // After the flux rows: |area| per face, which the sweep reads; kept while the scratch stays where it is.
        const bool per_face = update_ == Update::per_face;
        const std::size_t row_bytes = per_face ? sectors(sizeof(FluxStore) * V * static_cast<std::size_t>(faces_)) : 0;
        const std::size_t flux_bytes = row_bytes + (per_face && cache_magnitudes_ ? sizeof(double) * static_cast<std::size_t>(faces_) : 0);
        const bool held = both_meshes_;
        char* base = held ? held_.data() : scratch(cell_row_bytes() + flux_bytes);
        char* fluxes = held ? mesh_->scratch(flux_bytes).data() : base + cell_row_bytes();
        double* magnitudes = reinterpret_cast<double*>(fluxes + row_bytes);
        if (magnitudes != magnitudes_.data() || faces_ != static_cast<int>(magnitudes_.extent(0))) magnitudes_valid_ = false;
        if (per_face && cache_magnitudes_) magnitudes_ = Magnitudes(magnitudes, faces_);
        prims_ = Prims(reinterpret_cast<double*>(base), n_);
        base += prim_bytes();
        if (order_ == 1) gradients_ = Gradients(reinterpret_cast<GStore*>(base), n_);
        if (exponent_bytes() > 0) exponents_ = Exponents(reinterpret_cast<std::uint32_t*>(base + gradient_bytes()), n_);
        if (update_ == Update::per_face) flux_ = FaceFluxes(reinterpret_cast<FluxStore*>(fluxes), faces_);
    }

    // Stage 1 of a step: per-cell W, sound speed, radius into prims_. Used by: time_step(), step(); feeds sweep().
    void primitives() {
        const auto q = q_;
        const auto prim = prims_;
        const Geometry geo = geometry_;
        const Gas gas = gas_;
        Kokkos::parallel_for(
            "hydro::primitives", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), KOKKOS_LAMBDA(int i) {
                const State W = primitive(geo.per_volume(q, i), gas);
                for (int v = 0; v < V; ++v) prim(i, v) = W[v];
                prim(i, V) = sound_speed(W, gas);
                prim(i, V + 1) = radius(geo.volume(i));
            });
    }
    // W of cell i from prims_. Used by: Sweep, FaceFlux::side(), MovingFlux.
    KOKKOS_INLINE_FUNCTION static State primitive_row(const Prims& prim, int i) {
        return make_state<D>([&](int v) { return prim(i, v); });
    }

    // The sweep of time_step(), one thread per cell: the point velocity (moving), then the limited least-squares gradients (order 1).
    // Reads prims_; writes gradients_/exponents_ (read by FaceFlux::gradient()) and w_ (read by face_velocity()). Used by: sweep().
    template <bool Moving, bool Grad>
    struct Sweep {
        Prims prim;
        Points w;
        Gradients g;
        Exponents exponents;
        Geometry geo;
        Gas gas;
        Limiter limiter;
        Kokkos::View<const double*, Memory> magnitudes;  // |area| per face from the last flux pass on this mesh, or empty

        // |area| of face f: cached (the same bits) or from the area row.
        KOKKOS_INLINE_FUNCTION double magnitude_of(int f) const {
            if (magnitudes.data()) return magnitudes(f);
            const Vec a = geo.area_of(f);
            return Kokkos::sqrt(dot(a, a));
        }

        KOKKOS_INLINE_FUNCTION void operator()(int i) const {
            constexpr bool exact = std::is_same_v<G, double>;
            const State Wi = primitive_row(prim, i);
            double surface = 0, smallest_face = Kokkos::Experimental::infinity_v<double>, max_angle = 0;
            Gradient<D, G> grad;
            // The neighbours' range: W_j, or W_j - W_i in reduced precision (rounded once).
            G lo[V], hi[V];
            for (int v = 0; v < V; ++v) {
                grad[v] = hydro::Vec<D, G>{};
                lo[v] = Kokkos::Experimental::finite_max_v<G>;
                hi[v] = -Kokkos::Experimental::finite_max_v<G>;
            }
            // The normal equations M x = b, summed here and solved after the loop.
            constexpr int U = D + 1;
            G M[U][U] = {};
            G b[V][U] = {};
            const Vec centroid = geo.row(geo.centroid, i);
            // The list walk one entry ahead: the face and neighbour of entry k + 1 load while entry k's rows are summed,
            // so each entry waits for one level of loads, not three (list, face sides, rows).
            const int first = geo.start(i), end = geo.start(i + 1);
            int f_next = first < end ? geo.face_at(first) : 0;
            int j_next = first < end ? geo.other_at(first, f_next, i) : 0;
            for (int k = first; k < end; ++k) {
                const int f = f_next, j = j_next;
                if (k + 1 < end) {
                    f_next = geo.face_at(k + 1);
                    j_next = geo.other_at(k + 1, f_next, i);
                }
                const State Wj = primitive_row(prim, j);
                const Vec d = geo.separation(i, j);
                const double magnitude = magnitude_of(f);
                surface += magnitude;
                smallest_face = Kokkos::fmin(smallest_face, magnitude);
                if constexpr (Moving) max_angle = Kokkos::fmax(max_angle, face_angle(magnitude) / (0.5 * Kokkos::sqrt(dot(d, d))));
                if constexpr (Grad) {
                    G X[U];
                    const Vec between = d + geo.row(geo.centroid, j) - centroid;
                    X[0] = 1;
                    for (int p = 1; p < U; ++p) X[p] = static_cast<G>(between[p - 1]);
                    const G w = static_cast<G>(magnitude);
                    for (int p = 0; p < U; ++p)
                        for (int s = p; s < U; ++s) M[p][s] += w * X[p] * X[s];
                    for (int v = 0; v < V; ++v) {
                        const double dW = Wj[v] - Wi[v];
                        for (int p = 0; p < U; ++p) b[v][p] += w * X[p] * static_cast<G>(dW);
                        const G value = exact ? static_cast<G>(Wj[v]) : static_cast<G>(dW);
                        lo[v] = Kokkos::fmin(lo[v], value);
                        hi[v] = Kokkos::fmax(hi[v], value);
                    }
                }
            }
            // Rare: a face below 1e-6 of the surface does not count for the range or the limiter.
            const double tiny = 1e-6 * surface;
            const bool any_tiny = smallest_face < tiny;
            if constexpr (Grad) {
                if (any_tiny) neighbour_range(i, Wi, tiny, lo, hi);
            }
            if constexpr (Grad) solve(M, b, grad);
            if constexpr (Moving) {
                double fraction = 0;
                if (max_angle > 0.75 * cell_max_angle_factor)
                    fraction = max_angle > cell_max_angle_factor
                                   ? cell_shaping_speed
                                   : cell_shaping_speed * (max_angle - 0.75 * cell_max_angle_factor) / (0.25 * cell_max_angle_factor);
                Vec wi = velocity(Wi);
                const Vec c = centroid;  // centroid - generator = -dx of AREPO
                const double cd = Kokkos::sqrt(dot(c, c));
                if (cd > 0 && fraction > 0) wi = wi + c * (fraction * prim(i, V) / cd);
                for (int a = 0; a < D; ++a) w(i, a) = wi[a];
            }
            if constexpr (Grad) {
                for (int k = geo.start(i); k < geo.start(i + 1); ++k) {
                    if (limiter == Limiter::none) break;
                    const int f = geo.face_at(k);
                    if (any_tiny && magnitude_of(f) < tiny) continue;
                    const Vec offset = geo.face_offset(f, i);
                    const hydro::Vec<D, G> rg = cast<G>(offset);
                    for (int v = 0; v < V; ++v) {
                        const G dp = dot(grad[v], rg);
                        if constexpr (exact) {
                            if (dp > 0 && Wi[v] + dp > hi[v])
                                grad[v] = grad[v] * (hi[v] > Wi[v] ? (hi[v] - Wi[v]) / dp : 0.0);
                            else if (dp < 0 && Wi[v] + dp < lo[v])
                                grad[v] = grad[v] * (lo[v] < Wi[v] ? (lo[v] - Wi[v]) / dp : 0.0);
                        } else {
                            if (dp > 0 && dp > hi[v])
                                grad[v] = grad[v] * (hi[v] > 0 ? hi[v] / dp : G(0));
                            else if (dp < 0 && dp < lo[v])
                                grad[v] = grad[v] * (lo[v] < 0 ? lo[v] / dp : G(0));
                        }
                    }
                }
                store(i, grad);
            }
        }

        // The neighbours' range without the faces below tiny.
        KOKKOS_FUNCTION void neighbour_range(int i, const State& Wi, double tiny, G (&lo)[V], G (&hi)[V]) const {
            constexpr bool exact = std::is_same_v<G, double>;
            for (int v = 0; v < V; ++v) {
                lo[v] = Kokkos::Experimental::finite_max_v<G>;
                hi[v] = -Kokkos::Experimental::finite_max_v<G>;
            }
            for (int k = geo.start(i); k < geo.start(i + 1); ++k) {
                const int f = geo.face_at(k), j = geo.other_at(k, f, i);
                if (magnitude_of(f) < tiny) continue;
                const State Wj = primitive_row(prim, j);
                for (int v = 0; v < V; ++v) {
                    const G value = exact ? static_cast<G>(Wj[v]) : static_cast<G>(Wj[v] - Wi[v]);
                    lo[v] = Kokkos::fmin(lo[v], value);
                    hi[v] = Kokkos::fmax(hi[v], value);
                }
            }
        }

        // The gradient part of the fit: M x_v = b_v for every variable, M = L L^T by Cholesky once.
        template <int U>
        KOKKOS_INLINE_FUNCTION static void solve(const G (&M)[U][U], const G (&b)[V][U], Gradient<D, G>& grad) {
            G L[U][U] = {};
            for (int p = 0; p < U; ++p) {
                for (int s = 0; s <= p; ++s) {
                    G sum = M[s][p];
                    for (int m = 0; m < s; ++m) sum -= L[p][m] * L[s][m];
                    L[p][s] = p == s ? Kokkos::sqrt(sum) : sum / L[s][s];
                }
            }
            for (int v = 0; v < V; ++v) {
                G z[U], x[U];
                for (int p = 0; p < U; ++p) {
                    G sum = b[v][p];
                    for (int m = 0; m < p; ++m) sum -= L[p][m] * z[m];
                    z[p] = sum / L[p][p];
                }
                for (int p = U - 1; p >= 0; --p) {
                    G sum = z[p];
                    for (int m = p + 1; m < U; ++m) sum -= L[m][p] * x[m];
                    x[p] = sum / L[p][p];
                }
                for (int a = 0; a < D; ++a) grad[v][a] = x[1 + a];
            }
        }

        // The row of cell i; FP16: each group (rho, v, P) is scaled by 2^k to put its largest component in [2^14, 2^15).
        KOKKOS_INLINE_FUNCTION void store(int i, const Gradient<D, G>& grad) const {
            if constexpr (P::half) {
                float largest[3] = {0, 0, 0};
                for (int v = 0; v < V; ++v)
                    for (int a = 0; a < D; ++a) {
                        const int group = half::group<D>(D * v + a);
                        largest[group] = Kokkos::fmax(largest[group], Kokkos::fabs(grad[v][a]));
                    }
                int k[3];
                float scale[3];
                for (int s = 0; s < 3; ++s) {
                    k[s] = half::group_exponent(largest[s]);
                    scale[s] = half::power_of_two(k[s]);
                }
                for (int v = 0; v < V; ++v)
                    for (int a = 0; a < D; ++a) g(i, D * v + a) = half::from_float(grad[v][a] * scale[half::group<D>(D * v + a)]);
                exponents(i) = static_cast<std::uint32_t>(static_cast<std::uint8_t>(static_cast<std::int8_t>(k[0]))) |
                               static_cast<std::uint32_t>(static_cast<std::uint8_t>(static_cast<std::int8_t>(k[1]))) << 8 |
                               static_cast<std::uint32_t>(static_cast<std::uint8_t>(static_cast<std::int8_t>(k[2]))) << 16;
            } else {
                for (int v = 0; v < V; ++v)
                    for (int a = 0; a < D; ++a) g(i, D * v + a) = grad[v][a];
            }
        }
    };

    // Sweep's limited gradient of one cell in double, as a function of the cell's and its neighbours' W, with its tangent and adjoint.
    // Entries: the cell's list entries k < count(): W(k), dW(k) (tangent only), between(k) (the fit's offset to the neighbour),
    // weight(k) (|area|) and offset(k) (face centroid minus the cell's centroid). Used by: step_tangent, step_adjoint, the derivative tests.
    static constexpr int fit_size = D + 1;
    // Per variable: the last list entry whose face clipped the gradient (-1: none), the entry holding the bound it clipped to,
    // which bound (upper), and whether it clipped to zero. Clips scale the gradient, so the result is g0 (bound - W_i) / (g0 . o_last).
    struct LimitPattern {
        int last[V], extreme[V];
        bool upper[V], zero[V];
    };
    // The bars of what limited_gradient_adjoint() hands to the neighbours: the fit's right-hand sides b and each variable's bound.
    // The neighbour of entry k gets weight(k) X_k . b[v] + bound[v] if k == entry[v]; self: the cell's own W bar.
    struct GradientAdjoint {
        double b[V][fit_size];
        double bound[V];
        int entry[V];
        State self;
    };
    // The normal equations of the fit, as Sweep sums them. Used by: limited_gradient(), limited_gradient_tangent(), limited_gradient_adjoint().
    template <class Entries>
    KOKKOS_INLINE_FUNCTION static void fit(const State& Wi, const Entries& e, double (&M)[fit_size][fit_size], double (&b)[V][fit_size],
                                           double& surface) {
        constexpr int U = fit_size;
        for (int p = 0; p < U; ++p) {
            for (int s = 0; s < U; ++s) M[p][s] = 0;
            for (int v = 0; v < V; ++v) b[v][p] = 0;
        }
        surface = 0;
        for (int k = 0; k < e.count(); ++k) {
            const Vec between = e.between(k);
            const State Wj = e.W(k);
            double X[U];
            X[0] = 1;
            for (int p = 1; p < U; ++p) X[p] = between[p - 1];
            const double w = e.weight(k);
            surface += w;
            for (int p = 0; p < U; ++p)
                for (int s = p; s < U; ++s) M[p][s] += w * X[p] * X[s];
            for (int v = 0; v < V; ++v) {
                const double dW = Wj[v] - Wi[v];
                for (int p = 0; p < U; ++p) b[v][p] += w * X[p] * dW;
            }
        }
    }
    // x = M^-1 r for every variable, all components (M from fit(), upper triangle). Used by: limited_gradient_tangent(), limited_gradient_adjoint().
    KOKKOS_INLINE_FUNCTION static void solve_all(const double (&M)[fit_size][fit_size], const double (&r)[V][fit_size], double (&x)[V][fit_size]) {
        constexpr int U = fit_size;
        double L[U][U] = {};
        for (int p = 0; p < U; ++p)
            for (int s = 0; s <= p; ++s) {
                double sum = M[s][p];
                for (int m = 0; m < s; ++m) sum -= L[p][m] * L[s][m];
                L[p][s] = p == s ? Kokkos::sqrt(sum) : sum / L[s][s];
            }
        for (int v = 0; v < V; ++v) {
            double z[U];
            for (int p = 0; p < U; ++p) {
                double sum = r[v][p];
                for (int m = 0; m < p; ++m) sum -= L[p][m] * z[m];
                z[p] = sum / L[p][p];
            }
            for (int p = U - 1; p >= 0; --p) {
                double sum = z[p];
                for (int m = p + 1; m < U; ++m) sum -= L[m][p] * x[v][m];
                x[v][p] = sum / L[p][p];
            }
        }
    }
    // The limited gradient (Sweep, double: the same operations); pattern and the unlimited gradient g0 for the derivatives.
    template <class Entries>
    KOKKOS_INLINE_FUNCTION static Gradient<D> limited_gradient(const State& Wi, const Entries& e, Limiter limiter, LimitPattern& pattern,
                                                               Gradient<D>& g0, double (*M_out)[fit_size] = nullptr) {
        static_assert(std::is_same_v<G, double>, "derivatives: double gradients (Precision<>)");
        constexpr int U = fit_size;
        double M[U][U], b[V][U], surface;
        fit(Wi, e, M, b, surface);
        Gradient<D> grad;
        Sweep<false, true>::template solve<U>(M, b, grad);
        g0 = grad;
        if (M_out)
            for (int p = 0; p < U; ++p)
                for (int s = 0; s < U; ++s) M_out[p][s] = M[p][s];
        const double tiny = 1e-6 * surface;
        double lo[V], hi[V];
        int lo_k[V], hi_k[V];
        for (int v = 0; v < V; ++v) {
            lo[v] = Kokkos::Experimental::finite_max_v<double>;
            hi[v] = -Kokkos::Experimental::finite_max_v<double>;
            lo_k[v] = hi_k[v] = -1;
            pattern.last[v] = pattern.extreme[v] = -1;
            pattern.upper[v] = pattern.zero[v] = false;
        }
        for (int k = 0; k < e.count(); ++k) {
            if (e.weight(k) < tiny) continue;
            const State Wj = e.W(k);
            for (int v = 0; v < V; ++v) {
                if (Wj[v] < lo[v]) lo[v] = Wj[v], lo_k[v] = k;
                if (Wj[v] > hi[v]) hi[v] = Wj[v], hi_k[v] = k;
            }
        }
        if (limiter == Limiter::none) return grad;
        for (int k = 0; k < e.count(); ++k) {
            if (e.weight(k) < tiny) continue;
            const Vec offset = e.offset(k);
            for (int v = 0; v < V; ++v) {
                const double dp = dot(grad[v], offset);
                if (dp > 0 && Wi[v] + dp > hi[v]) {
                    grad[v] = grad[v] * (hi[v] > Wi[v] ? (hi[v] - Wi[v]) / dp : 0.0);
                    pattern.last[v] = k, pattern.upper[v] = true, pattern.zero[v] = !(hi[v] > Wi[v]);
                } else if (dp < 0 && Wi[v] + dp < lo[v]) {
                    grad[v] = grad[v] * (lo[v] < Wi[v] ? (lo[v] - Wi[v]) / dp : 0.0);
                    pattern.last[v] = k, pattern.upper[v] = false, pattern.zero[v] = !(lo[v] < Wi[v]);
                }
            }
        }
        for (int v = 0; v < V; ++v) pattern.extreme[v] = pattern.last[v] < 0 ? -1 : pattern.upper[v] ? hi_k[v] : lo_k[v];
        return grad;
    }
    // dg = J (dW_i, dW_j) of limited_gradient() (geometry fixed), at its branch pattern; g: the forward gradient.
    template <class Entries>
    KOKKOS_INLINE_FUNCTION static Gradient<D> limited_gradient_tangent(const State& Wi, const State& dWi, const Entries& e, Limiter limiter,
                                                                       Gradient<D>& g) {
        constexpr int U = fit_size;
        LimitPattern pattern;
        Gradient<D> g0;
        double M[U][U];
        g = limited_gradient(Wi, e, limiter, pattern, g0, M);
        double db[V][U] = {}, dx[V][U];
        for (int k = 0; k < e.count(); ++k) {
            const Vec between = e.between(k);
            const State dWj = e.dW(k);
            const double w = e.weight(k);
            for (int v = 0; v < V; ++v) {
                const double dW = w * (dWj[v] - dWi[v]);
                db[v][0] += dW;
                for (int p = 1; p < U; ++p) db[v][p] += between[p - 1] * dW;
            }
        }
        solve_all(M, db, dx);
        Gradient<D> dg;
        for (int v = 0; v < V; ++v) {
            const Vec dg0 = make_vec<D>([&](int a) { return dx[v][1 + a]; });
            if (pattern.last[v] < 0) {
                dg[v] = dg0;
            } else if (pattern.zero[v]) {
                dg[v] = Vec{};
            } else {
                const Vec o = e.offset(pattern.last[v]);
                const double den = dot(g0[v], o), bound = e.W(pattern.extreme[v])[v], dbound = e.dW(pattern.extreme[v])[v];
                const double s = (bound - Wi[v]) / den, ds = (dbound - dWi[v] - s * dot(dg0, o)) / den;
                dg[v] = dg0 * s + g0[v] * ds;
            }
        }
        return dg;
    }
    // J^T gbar of limited_gradient(): the bars of b and of the bounds, which the neighbours collect, and the cell's own W bar.
    template <class Entries>
    KOKKOS_INLINE_FUNCTION static GradientAdjoint limited_gradient_adjoint(const State& Wi, const Entries& e, Limiter limiter, const Gradient<D>& gbar) {
        constexpr int U = fit_size;
        LimitPattern pattern;
        Gradient<D> g0;
        double M[U][U];
        limited_gradient(Wi, e, limiter, pattern, g0, M);
        GradientAdjoint out;
        double xbar[V][U];
        for (int v = 0; v < V; ++v) {
            out.self[v] = 0;
            out.bound[v] = 0;
            out.entry[v] = -1;
            Vec g0bar = gbar[v];
            if (pattern.zero[v] && pattern.last[v] >= 0) {
                g0bar = Vec{};
            } else if (pattern.last[v] >= 0) {
                const Vec o = e.offset(pattern.last[v]);
                const double den = dot(g0[v], o), bound = e.W(pattern.extreme[v])[v];
                const double s = (bound - Wi[v]) / den, t = dot(gbar[v], g0[v]) / den;
                g0bar = gbar[v] * s - o * (s * t);
                out.bound[v] = t;
                out.entry[v] = pattern.extreme[v];
                out.self[v] -= t;
            }
            xbar[v][0] = 0;
            for (int a = 0; a < D; ++a) xbar[v][1 + a] = g0bar[a];
        }
        solve_all(M, xbar, out.b);  // M is symmetric: b bar = M^-1 x bar
        for (int v = 0; v < V; ++v)
            for (int p = 0; p < U; ++p) out.self[v] -= M[0][p] * out.b[v][p];
        return out;
    }
    // The W bar of entry k's neighbour from limited_gradient_adjoint(). Used by: the derivative tests (step_adjoint gathers it per cell).
    template <class Entries>
    KOKKOS_INLINE_FUNCTION static State neighbour_bar(const Entries& e, int k, const GradientAdjoint& adjoint) {
        const Vec between = e.between(k);
        const double w = e.weight(k);
        return make_state<D>([&](int v) {
            double sum = adjoint.b[v][0];
            for (int p = 1; p < fit_size; ++p) sum += between[p - 1] * adjoint.b[v][p];
            return w * sum + (adjoint.entry[v] == k ? adjoint.bound[v] : 0.0);
        });
    }

    // The cell terms of the time step (AREPO rule). Used by: TimeStep.
    template <bool Moving>
    struct CellStep {
        Prims prim;
        Points w;
        double courant;

        KOKKOS_INLINE_FUNCTION void operator()(int i, double& smallest) const {
            Vec relative = make_vec<D>([&](int a) { return prim(i, 1 + a); });
            if constexpr (Moving) relative = relative - make_vec<D>([&](int a) { return w(i, a); });
            const double step = prim(i, V + 1) / (prim(i, V) + Kokkos::sqrt(dot(relative, relative)));
            smallest = Kokkos::fmin(smallest, courant * step);
        }
    };
    // LUNA's rule (StepRule::luna). Used by: sweep().
    template <bool Moving>
    struct LunaStep {
        static_assert(D == 2, "StepRule::luna: 2D");
        Prims prim;
        Points w;
        double courant;

        KOKKOS_INLINE_FUNCTION void operator()(int i, double& smallest) const {
            const double c = prim(i, V), dx = prim(i, V + 1) * Kokkos::sqrt(M_PI);
            double rate = 0;
            for (int a = 0; a < D; ++a) {
                double relative = prim(i, 1 + a);
                if constexpr (Moving) relative -= w(i, a);
                rate += (Kokkos::abs(relative) + c) / dx;
            }
            smallest = Kokkos::fmin(smallest, courant / rate);
        }
    };
    // The face terms of the time step (AREPO rule). vsig_ij and d_ij are the same from both sides, so each face gives both cells' terms.
    // Used by: TimeStep.
    struct FaceStep {
        Prims prim;
        Geometry geo;
        double courant;

        KOKKOS_INLINE_FUNCTION void operator()(int f, double& smallest) const {
            const int o = geo.owner_of(f), j = geo.neighbour_of(f);
            const Vec d = geo.separation(o, j);
            const double dist = Kokkos::sqrt(dot(d, d));
            const Vec dv = make_vec<D>([&](int a) { return prim(j, 1 + a) - prim(o, 1 + a); });
            const double vsig = prim(o, V) + prim(j, V) - dot(dv, d) / dist;
            if (vsig > 0) {
                smallest = Kokkos::fmin(smallest, courant * ((dist + prim(o, V + 1)) / vsig));
                smallest = Kokkos::fmin(smallest, courant * ((dist + prim(j, V + 1)) / vsig));
            }
        }
    };

    // Both terms of the time step in one reduction (one launch, one read-back): thread k takes face k's terms (k < faces)
    // and cell k's (k < cells). The smallest of the same terms as before, so the same dt. Used by: sweep().
    template <bool Moving>
    struct TimeStep {
        FaceStep face;
        CellStep<Moving> cell;
        int faces, cells;

        KOKKOS_INLINE_FUNCTION void operator()(int k, double& smallest) const {
            if (k < faces) face(k, smallest);
            if (k < cells) cell(k, smallest);
        }
    };

    // Stage 2 of a step: runs Sweep, then the time-step reduction; returns dt. Used by: time_step(), step().
    double sweep(double courant) {
        const bool moving = motion_ == Motion::moving;
        if (order_ == 1)
            return moving ? sweep<true, 1>(courant) : sweep<false, 1>(courant);
        return moving ? sweep<true, 0>(courant) : sweep<false, 0>(courant);
    }
    template <bool Moving, int Order>
    double sweep(double courant) {
        const Kokkos::RangePolicy<ExecSpace> cells(exec_, 0, n_);
        if constexpr (Moving || Order == 1)
            Kokkos::parallel_for("hydro::sweep", cells,
                                 Sweep<Moving, Order == 1>{prims_, w_, gradients_, exponents_, geometry_, gas_, limiter_,
                                                           magnitudes_valid_ ? magnitudes_ : Magnitudes()});
        double smallest = 0;
        if constexpr (D == 2) {
            if (rule_ == StepRule::luna) {
                Kokkos::parallel_reduce("hydro::luna_step", cells, LunaStep<Moving>{prims_, w_, courant}, Kokkos::Min<double>(smallest));
                return smallest;
            }
        }
        Kokkos::parallel_reduce("hydro::time_step", Kokkos::RangePolicy<ExecSpace>(exec_, 0, std::max(faces_, n_)),
                                TimeStep<Moving>{{prims_, geometry_, courant}, {prims_, w_, courant}, faces_, n_},
                                Kokkos::Min<double>(smallest));
        if (dt_trace_) trace_step<Moving>(courant, smallest);
        return smallest;
    }

    // LUNA_DT_TRACE (diagnostic): the time step's limiting cell term and face term, where they are and the rows that set
    // them, one line a step. Used by: sweep().
    template <bool Moving>
    void trace_step(double courant, double dt) {
        using Loc = Kokkos::MinLoc<double, int>;
        using Best = typename Loc::value_type;
        Best cell_best, face_best;
        const CellStep<Moving> cell_term{prims_, w_, courant};
        const FaceStep face_term{prims_, geometry_, courant};
        Kokkos::parallel_reduce(
            "hydro::dt_trace_cells", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_),
            KOKKOS_LAMBDA(int i, Best& best) {
                double s = Kokkos::Experimental::infinity_v<double>;
                cell_term(i, s);
                if (s < best.val) {
                    best.val = s;
                    best.loc = i;
                }
            },
            Loc(cell_best));
        Kokkos::parallel_reduce(
            "hydro::dt_trace_faces", Kokkos::RangePolicy<ExecSpace>(exec_, 0, faces_),
            KOKKOS_LAMBDA(int f, Best& best) {
                double s = Kokkos::Experimental::infinity_v<double>;
                face_term(f, s);
                if (s < best.val) {
                    best.val = s;
                    best.loc = f;
                }
            },
            Loc(face_best));
        // The rows of the limiting cell and of the limiting face's two cells, and their distance from the box centre.
        auto describe = [&](int i) {
            const auto row = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Kokkos::subview(prims_, i, Kokkos::ALL));
            const auto x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Kokkos::subview(geometry_.positions, i, Kokkos::ALL));
            Vec vel{}, wi{};
            double r2 = 0;
            for (int a = 0; a < D; ++a) {
                vel[a] = row(1 + a);
                const double d = x(a) - 0.5 * geometry_.box.side[a];
                r2 += d * d;
            }
            if constexpr (Moving) {
                const auto w = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Kokkos::subview(w_, i, Kokkos::ALL));
                for (int a = 0; a < D; ++a) wi[a] = w(a);
            }
            const Vec rel = vel - wi;
            std::printf(" [cell %d r=%.4g rho=%.4g P=%.4g |v|=%.4g |v-w|=%.4g c=%.4g radius=%.4g]", i, std::sqrt(r2), row(0), row(D + 1),
                        std::sqrt(dot(vel, vel)), std::sqrt(dot(rel, rel)), row(V), row(V + 1));
        };
        std::printf("dt_trace: step=%lld dt=%.6g cell_min=%.6g", ++traced_steps_, dt, cell_best.val);
        describe(cell_best.loc);
        std::printf(" face_min=%.6g face=%d", face_best.val, face_best.loc);
        const auto o = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Kokkos::subview(geometry_.owner, face_best.loc));
        const auto j = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, Kokkos::subview(geometry_.neighbour, face_best.loc));
        describe(o());
        describe(j());
        std::printf("\n");
    }

    // The points drift by dt * w and stay in the box; the mesh is updated. Used by: step(), step_moving().
    void move(double dt, Timings* timings) {
        const auto x = x_;
        const auto w = w_;
        const auto steps = steps_;
        const exact_predicates::BoxN<D> box = mesh_->sides();
        const bool on_grid = both_meshes_;
        magnitudes_valid_ = false;
        Kokkos::parallel_for(
            "hydro::drift", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), KOKKOS_LAMBDA(int i) {
                for (int a = 0; a < D; ++a) {
                    if (on_grid) {  // whole grid steps, as step_moving() set them
                        const long long k = static_cast<long long>(Kokkos::round(x(i, a) * box.per_step)) + steps(i, a);
                        const long long p = box.period[a];
                        x(i, a) = static_cast<double>(((k % p) + p) % p) * box.step;
                        continue;
                    }
                    const double moved = x(i, a) + dt * w(i, a);
                    x(i, a) = moved - box.side[a] * Kokkos::floor(moved / box.side[a]);
                }
            });
        phase_end(timings, &Timings::drift);
        geometry_ = {};  // no reference to what the update may reallocate
        const voronoi::Report report = mesh_->update(x_);
        if (report.resorted) follow_resort();
        mesh_stats_.resorts += report.resorted;
        mesh_stats_.updates += 1;
        mesh_stats_.rebuilds += report.rebuilt;
        mesh_stats_.stuck += report.stuck;
        mesh_stats_.stuck_steps += report.stuck > 0;
        mesh_stats_.local_repairs += report.local_done;
        mesh_stats_.local_cells += report.local_cells;
        mesh_stats_.host_flips += report.host_flips;
        mesh_stats_.device_flips += report.flips;
        phase_end(timings, &Timings::mesh);
        mesh_->update_cells();
        phase_end(timings, &Timings::cells);
        refresh();
    }

    // The mesh renumbered its cells (voronoi::Layout): every per-cell array that outlives the update follows. Used by: move().
    void follow_resort() {
        if constexpr (D == 3) {
            mesh_->permute(q_);
            mesh_->permute(x_);
            mesh_->permute(w_);
            if (!both_meshes_) return;
            mesh_->permute(old_centroid_);
            mesh_->permute(steps_);
            // Pass 2 reads pass 1's cell rows (in held_) by cell: each gathered back from a copy.
            auto carry = [&](const auto& rows) {
                using Rows = std::decay_t<decltype(rows)>;
                Kokkos::View<typename Rows::data_type, typename Rows::array_layout, Memory> copy(
                    Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("hydro_rows_copy")), rows.layout());
                Kokkos::deep_copy(exec_, copy, rows);
                mesh_->permute(copy, rows);
            };
            carry(prims_);
            if (order_ == 1) carry(gradients_);
            if (exponents_.data()) carry(exponents_);
        }
    }

    // The flux through face f (times its area, owner to neighbour); `fallbacks` counts the sides that took the cell's state.
    // Per face: side() x2 -> face_velocity() -> solve(). Reads what primitives() and Sweep wrote. Used by: apply().
    template <int Order, bool Moving>
    struct FaceFlux {
        Cells q;
        Prims prim;
        Gradients g;
        Exponents exponents;
        Points w;
        Geometry geo;
        Gas gas;
        double dt;

        // Cell i's gradient row, decoded from FP16 if needed. Used by: side(), MovingFlux::pass_side().
        KOKKOS_INLINE_FUNCTION Gradient<D, G> gradient(int i) const {
            Gradient<D, G> grad;
            if constexpr (P::half) {
                const std::uint32_t k = exponents(i);
                float scale[3];
                for (int s = 0; s < 3; ++s) scale[s] = half::power_of_two(-static_cast<int>(static_cast<std::int8_t>(k >> (8 * s))));
                for (int v = 0; v < V; ++v)
                    grad[v] = make_vec<D, G>(
                        [&](int a) { return half::to_float(g(i, D * v + a)) * scale[half::group<D>(D * v + a)]; });
            } else {
                for (int v = 0; v < V; ++v) grad[v] = make_vec<D, G>([&](int a) { return g(i, D * v + a); });
            }
            return grad;
        }
        // As AREPO's prepare_state: extrapolate in time and space only if rho and P stay positive, else the cell's state.
        // r: offset from the centroid to the face point. Used by: operator().
        // Order 0 computes W here from Q instead of loading it, or the bits change (docs/design.md).
        KOKKOS_INLINE_FUNCTION State side(int i, Vec r, int& fallbacks) const {
            if constexpr (Order == 0) {
                (void)r;
                (void)fallbacks;
                return primitive(geo.per_volume(q, i), gas);
            } else {
                const State W = primitive_row(prim, i);
                const Gradient<D, G> grad = gradient(i);
                State Wf;
                if constexpr (std::is_same_v<G, double>) {
                    Wf = half_step(W, grad, dt, gas);
                    for (int v = 0; v < V; ++v) Wf[v] += dot(grad[v], r);
                } else {
                    const StateT<D, G> Wg = make_state<D, G>([&](int v) { return static_cast<G>(W[v]); });
                    const StateT<D, G> rate = half_step_rate(Wg, grad, gas);
                    const hydro::Vec<D, G> rg = cast<G>(r);
                    const G h = G(0.5) * static_cast<G>(dt);
                    for (int v = 0; v < V; ++v) Wf[v] = W[v] + static_cast<double>(dot(grad[v], rg) - h * rate[v]);
                }
                if (Wf[0] > 0 && Wf[D + 1] > 0) return Wf;
                ++fallbacks;
                return W;
            }
        }
        // side() at order 1 in double as a function of (W, g), r and dt fixed; fallback: the side took the cell's state.
        // Used by: side_tangent(), side_adjoint(), step_adjoint, the derivative tests.
        KOKKOS_INLINE_FUNCTION static State side_value(const State& W, const Gradient<D>& grad, Vec r, double dt, const Gas& gas, bool& fallback) {
            State Wf = half_step(W, grad, dt, gas);
            for (int v = 0; v < V; ++v) Wf[v] += dot(grad[v], r);
            fallback = !(Wf[0] > 0 && Wf[D + 1] > 0);
            return fallback ? W : Wf;
        }
        // dW_f = J (dW, dg) of side_value(); a side that fell back carries dW. Used by: step_tangent, the derivative tests.
        KOKKOS_INLINE_FUNCTION static State side_tangent(const State& W, const State& dW, const Gradient<D>& grad, const Gradient<D>& dg, Vec r,
                                                         double dt, const Gas& gas) {
            bool fallback;
            side_value(W, grad, r, dt, gas, fallback);
            if (fallback) return dW;
            State dWf = half_step_tangent(W, dW, grad, dg, dt, gas);
            for (int v = 0; v < V; ++v) dWf[v] += dot(dg[v], r);
            return dWf;
        }
        // Adds J^T Wfbar of side_value() to Wbar and gbar. Used by: step_adjoint, the derivative tests.
        KOKKOS_INLINE_FUNCTION static void side_adjoint(const State& W, const Gradient<D>& grad, Vec r, double dt, const Gas& gas, const State& Wfbar,
                                                        State& Wbar, Gradient<D>& gbar) {
            bool fallback;
            side_value(W, grad, r, dt, gas, fallback);
            if (fallback) {
                for (int v = 0; v < V; ++v) Wbar[v] += Wfbar[v];
                return;
            }
            for (int v = 0; v < V; ++v) gbar[v] = gbar[v] + r * Wfbar[v];
            half_step_adjoint(W, grad, dt, gas, Wfbar, Wbar, gbar);
        }
        // AREPO-ng's face velocity: the mean point velocity plus a correction along n, |c| limited to 0.9 |d|. Used by: operator().
        // wL, wR: the point velocities of owner and neighbour (FaceFlux: the rows of w; MovingFlux: from the grid steps).
        KOKKOS_INLINE_FUNCTION Vec face_velocity(int f, int o, int j, Vec wL, Vec wR) const {
            const Vec d = geo.separation(o, j);
            double dist, inv_dist;
            if constexpr (LUNA_FLUX_LEVER == 2) {  // |d| and 1 / |d| from one estimate and a Newton step
                const double d2 = dot(d, d);
                inv_dist = fast::rsqrt(d2);
                dist = d2 * inv_dist;
            } else {
                dist = Kokkos::sqrt(dot(d, d));
                inv_dist = 1.0 / dist;
            }
            const Vec c = geo.face_centroid_of(f) - d * 0.5;
            double facv = dot(c, wL - wR) * inv_dist;
            // |c| only where the limit applies: the squares compared, one double square root a face fewer.
            const double limit = 0.9 * dist, c2 = dot(c, c);
            if (c2 > limit * limit) facv *= limit / Kokkos::sqrt(c2);
            return (wL + wR) * 0.5 + d * (facv * inv_dist);
        }
        // The Riemann solve along the unit normal n; float: in the face's frame, the flux back in the lab frame in double.
        // n: unit normal, wf: face velocity. Used by: operator().
        KOKKOS_INLINE_FUNCTION static State solve(const State& L, const State& R, Vec n, Vec wf, const Gas& gas) {
            if constexpr (std::is_same_v<typename P::riemann, double>) {
                return Riemann::flux(L, R, n, wf, gas);
            } else {
                using T = typename P::riemann;
                auto relative = [&](const State& S) {
                    return make_state<D, T>([&](int v) {
                        return static_cast<T>(v >= 1 && v <= D ? S[v] - wf[v - 1] : S[v]);
                    });
                };
                const StateT<D, T> F = Riemann::flux(relative(L), relative(R), cast<T>(n), hydro::Vec<D, T>{}, gas);
                return detail::to_lab(make_state<D>([&](int v) { return static_cast<double>(F[v]); }), wf);
            }
        }
        // fallbacks_o, fallbacks_j: the sides of owner and neighbour that took the cell's state. Used by: apply().
        KOKKOS_INLINE_FUNCTION State operator()(int f, int& fallbacks_o, int& fallbacks_j) const {
            const int o = geo.owner_of(f), j = geo.neighbour_of(f);
            const Vec a = geo.area_of(f);
            const double magnitude = Kokkos::sqrt(dot(a, a));
            if (magnitude == 0) return State{};  // cospherical or cocircular corners (an exact lattice): no normal, no flux
            Vec wf{};
            if constexpr (Moving) wf = face_velocity(f, o, j, geo.row(w, o), geo.row(w, j));
            State L, R;
            if constexpr (Order == 0) {
                L = side(o, Vec{}, fallbacks_o);
                R = side(j, Vec{}, fallbacks_j);
            } else {
                const Vec shift = wf * (0.5 * dt);
                L = side(o, geo.face_offset(f, o) + shift, fallbacks_o);
                R = side(j, geo.face_offset(f, j) + shift, fallbacks_j);
            }
            State F = solve(L, R, a * (1.0 / magnitude), wf, gas);
            for (int v = 0; v < V; ++v) F[v] *= magnitude;
            return F;
        }
    };

    // The fluxes of a moving step: first on the mesh of t^n with W^n + g (x_f^n - c^n); second on t^n+1 with W^n advanced a whole step.
    // Second: which mesh; old_centroid: centroids of t^n. Used by: step_moving() via apply().
    template <bool Second>
    struct MovingFlux : FaceFlux<1, true> {
        Points old_centroid;
        Steps steps;          // grid steps of each point over the step (int32, half the bytes of w)
        double unit = 0;      // the velocity of one grid step per step: grid step / dt, divided once

        // The point velocity of cell i: its grid steps times the unit (one multiply; a division per component cost 7 %).
        KOKKOS_INLINE_FUNCTION Vec velocity(int i) const {
            return make_vec<D>([&](int a) { return static_cast<double>(steps(i, a)) * unit; });
        }

        // Replaces FaceFlux::side() on both meshes. Used by: operator().
        KOKKOS_INLINE_FUNCTION State pass_side(int i, Vec r, int& fallbacks) const {
            if constexpr (Second) {
                const Vec moved = velocity(i) * (0.5 * this->dt);
                return this->side(i, r + this->geo.row(this->geo.centroid, i) - this->geo.row(old_centroid, i) + moved, fallbacks);
            } else {
                const State W = primitive_row(this->prim, i);
                const Gradient<D, G> grad = this->gradient(i);
                const hydro::Vec<D, G> rg = cast<G>(r);
                State Wf;
                for (int v = 0; v < V; ++v) Wf[v] = W[v] + static_cast<double>(dot(grad[v], rg));
                if (Wf[0] > 0 && Wf[D + 1] > 0) return Wf;
                ++fallbacks;
                return W;
            }
        }
        KOKKOS_INLINE_FUNCTION State operator()(int f, int& fallbacks_o, int& fallbacks_j) const {
            const int o = this->geo.owner_of(f), j = this->geo.neighbour_of(f);
            const Vec a = this->geo.area_of(f);
            double magnitude, inv_magnitude;
            if constexpr (LUNA_FLUX_LEVER == 2) {  // |A| and 1 / |A| from one estimate and a Newton step
                const double a2 = dot(a, a);
                if (a2 == 0) return State{};
                inv_magnitude = fast::rsqrt(a2);
                magnitude = a2 * inv_magnitude;
            } else {
                magnitude = Kokkos::sqrt(dot(a, a));
                if (magnitude == 0) return State{};
                inv_magnitude = 1.0 / magnitude;
            }
            const Vec wf = this->face_velocity(f, o, j, velocity(o), velocity(j));
            const State L = pass_side(o, this->geo.face_offset(f, o), fallbacks_o);
            const State R = pass_side(j, this->geo.face_offset(f, j), fallbacks_j);
            State F;
            if constexpr (LUNA_FLUX_LEVER == 3 && std::is_same_v<Riemann, HLLC> && std::is_same_v<typename P::riemann, double>)
                F = HLLC::flux_lever<D>(L, R, a * inv_magnitude, wf, this->gas, this->prim(o, V), this->prim(j, V));
            else
                F = this->solve(L, R, a * inv_magnitude, wf, this->gas);
            for (int v = 0; v < V; ++v) F[v] *= magnitude;
            return F;
        }
    };

    // MUSCL-Hancock on a moving mesh, AREPO next generation's step, replacing advance()+move() in step():
    //  0. points to the mesh's grid, w to whole grid steps per dt, so each face moves as the snapped points do (luna-cpp #23)
    //  1. keep the centroids of t^n  2. apply() the fluxes of t^n (half dt)  3. move()  4. borrow_scratch()  5. apply() those of t^n+1 (half dt)
    // Used by: step().
    void step_moving(double dt, Timings* timings) {
        const auto old = old_centroid_;
        const Geometry geo = geometry_;
        const auto x = x_;
        const auto w = w_;
        const auto steps = steps_;
        const exact_predicates::BoxN<D> box = mesh_->sides();
        // The points go to the grid the mesh snaps them to, and their move becomes whole grid steps over dt (#23); the fluxes
        // take w = steps · grid step / dt (MovingFlux::velocity), w_ keeps the sweep's velocities.
        Kokkos::parallel_for(
            "hydro::keep_centroids", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), KOKKOS_LAMBDA(int i) {
                for (int a = 0; a < D; ++a) {
                    old(i, a) = geo.centroid(i, a);
                    const int k0 = exact_predicates::snap(x(i, a), box, a), k1 = exact_predicates::snap(x(i, a) + dt * w(i, a), box, a);
                    x(i, a) = static_cast<double>(k0) * box.step;
                    steps(i, a) = exact_predicates::delta(k0, k1, box.period[a]);
                }
            });
        apply<1>(MovingFlux<false>{{q_, prims_, gradients_, exponents_, w_, geometry_, gas_, 0.0}, old_centroid_, steps_, box.step / dt}, 0.5 * dt, timings);
        move(dt, timings);
        borrow_scratch();
        apply<1>(MovingFlux<true>{{q_, prims_, gradients_, exponents_, w_, geometry_, gas_, 2 * dt}, old_centroid_, steps_, box.step / dt}, 0.5 * dt, timings, 1);
    }

    // Picks the Moving instantiation. Used by: step().
    template <int Order>
    void advance(double dt, Timings* timings) {
        if (motion_ == Motion::moving)
            advance<Order, true>(dt, timings);
        else
            advance<Order, false>(dt, timings);
    }

    // Stage 3 of a step: one flux pass over the faces with FaceFlux. Used by: advance<Order>().
    template <int Order, bool Moving>
    void advance(double dt, Timings* timings) {
        apply<Order>(FaceFlux<Order, Moving>{q_, prims_, gradients_, exponents_, w_, geometry_, gas_, dt}, dt, timings);
    }

    // Q -= dt * sum over the cell's faces of the outgoing flux; flux: a FaceFlux or MovingFlux. Used by: advance(), step_moving().
    // pass: which fallback counter the flux pass writes (0, or 1 for a moving step's second pass).
    template <int Order, class Flux>
    void apply(const Flux& flux, double dt, Timings* timings, int pass = 0) {
        if (pass == 0) Kokkos::deep_copy(exec_, Kokkos::subview(fallback_counts_, 1), 0LL);
        fallbacks_read_ = false;
        const Geometry geo = geometry_;
        const auto q = q_;
        if (update_ == Update::per_cell) {
            // In place at order 1 (the fluxes read only the primitive and gradient rows); order 0 reads Q, so into next_.
            const auto next = Order == 0 ? next_ : q_;
            Kokkos::parallel_reduce(
                "hydro::fluxes_and_update", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_),
                KOKKOS_LAMBDA(int i, long long& count) {
                    State sum{};
                    for (int k = geo.start(i); k < geo.start(i + 1); ++k) {
                        const int f = geo.face_at(k);
                        int fallbacks_o = 0, fallbacks_j = 0;
                        const State F = flux(f, fallbacks_o, fallbacks_j);
                        const bool owned = geo.owned_at(k, f, i);
                        count += owned ? fallbacks_o : fallbacks_j;
                        const double sign = owned ? 1.0 : -1.0;
                        for (int v = 0; v < V; ++v) sum[v] += sign * F[v];
                    }
                    for (int v = 0; v < V; ++v) next(i, v) = q(i, v) - dt * sum[v];
                },
                Kokkos::subview(fallback_counts_, pass));
            phase_end(timings, &Timings::fluxes);
            if constexpr (Order == 0) std::swap(q_, next_);
            return;
        }
        const auto F = flux_;
        // A moving step's second pass is on the next step's mesh: its update also writes the next primitive rows (as primitives()).
        const bool next_prims = pass == 1;
        const auto prim = prims_;
        const Gas gas = gas_;
        // The magnitudes the next sweep reads: from a moving step's second pass, or once on a static mesh.
        const bool keep = cache_magnitudes_ && (pass == 1 || (motion_ == Motion::static_mesh && !magnitudes_valid_));
        const auto mag = magnitudes_;
        Kokkos::parallel_reduce(
            "hydro::fluxes", Kokkos::RangePolicy<ExecSpace>(exec_, 0, faces_),
            KOKKOS_LAMBDA(int f, long long& count) {
                int fallbacks = 0;
                const State value = flux(f, fallbacks, fallbacks);
                count += fallbacks;
                for (int v = 0; v < V; ++v) F(f, v) = static_cast<FluxStore>(value[v]);
                if (keep) {
                    const Vec a = geo.area_of(f);
                    mag(f) = Kokkos::sqrt(dot(a, a));
                }
            },
            Kokkos::subview(fallback_counts_, pass));
        phase_end(timings, &Timings::fluxes);
        Kokkos::parallel_for(
            "hydro::update", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), KOKKOS_LAMBDA(int i) {
                State sum{};
                for (int k = geo.start(i); k < geo.start(i + 1); ++k) {
                    const int f = geo.face_at(k);
                    const double sign = geo.owned_at(k, f, i) ? 1.0 : -1.0;
                    for (int v = 0; v < V; ++v) sum[v] += sign * static_cast<double>(F(f, v));
                }
                State next;
                for (int v = 0; v < V; ++v) q(i, v) = next[v] = q(i, v) - dt * sum[v];
                if (next_prims) {
                    const double inv = 1.0 / geo.volume(i);
                    const State W = primitive(make_state<D>([&](int v) { return next[v] * inv; }), gas);
                    for (int v = 0; v < V; ++v) prim(i, v) = W[v];
                    prim(i, V) = sound_speed(W, gas);
                    prim(i, V + 1) = radius(geo.volume(i));
                }
            });
        if (keep) magnitudes_valid_ = true;
        prims_ready_ = next_prims;
        phase_end(timings, &Timings::update);
    }

    // Derivatives of a step (docs/derivatives.md): a static mesh, order 1, the per-face loop, double throughout (Precision<>), dt fixed.
    // Throws otherwise. Used by: step_tangent(), step_adjoint().
    void require_derivatives(const char* what) const {
        static_assert(std::is_same_v<G, double> && std::is_same_v<FluxStore, double> && std::is_same_v<typename P::riemann, double> &&
                          !Mesh::float_rows,
                      "derivatives of a step: Precision<> (double throughout)");
        if (motion_ != Motion::static_mesh || order_ != 1 || update_ != Update::per_face)
            throw std::logic_error(std::string("hydro::FiniteVolume::") + what + ": a static mesh, order 1, the per-face loop");
    }
    using Rows = Kokkos::View<double**, Kokkos::LayoutRight, Memory>;
    // A Rows view of at least rows x width, kept between steps. Used by: step_tangent(), step_adjoint().
    static void fit_rows(Rows& r, const char* label, int rows, int width) {
        if (static_cast<int>(r.extent(0)) < rows || static_cast<int>(r.extent(1)) != width)
            r = Rows(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(label)), rows, width);
    }
    // Entries of cell i for limited_gradient*(): its list, the primitive rows and, for the tangent, their tangents.
    struct CellEntries {
        Geometry geo;
        Prims prim;
        Rows dprim;
        int i, first, n;
        Vec centroid;
        KOKKOS_INLINE_FUNCTION CellEntries(const Geometry& g, const Prims& p, const Rows& dp, int cell)
            : geo(g), prim(p), dprim(dp), i(cell), first(g.start(cell)), n(g.start(cell + 1) - g.start(cell)), centroid(g.row(g.centroid, cell)) {}
        KOKKOS_INLINE_FUNCTION int count() const { return n; }
        KOKKOS_INLINE_FUNCTION int face(int k) const { return geo.face_at(first + k); }
        KOKKOS_INLINE_FUNCTION int other(int k) const { return geo.other_at(first + k, face(k), i); }
        KOKKOS_INLINE_FUNCTION State W(int k) const { return primitive_row(prim, other(k)); }
        KOKKOS_INLINE_FUNCTION State dW(int k) const {
            const int j = other(k);
            return make_state<D>([&](int v) { return dprim(j, v); });
        }
        KOKKOS_INLINE_FUNCTION Vec between(int k) const {
            const int j = other(k);
            return geo.separation(i, j) + geo.row(geo.centroid, j) - centroid;
        }
        KOKKOS_INLINE_FUNCTION double weight(int k) const {
            const Vec a = geo.area_of(face(k));
            return Kokkos::sqrt(dot(a, a));
        }
        KOKKOS_INLINE_FUNCTION Vec offset(int k) const { return geo.face_offset(face(k), i); }
    };
    // Gradient row i of a double policy. Used by: step_adjoint.
    KOKKOS_INLINE_FUNCTION static Gradient<D> gradient_row(const Gradients& g, int i) {
        Gradient<D> grad;
        for (int v = 0; v < V; ++v) grad[v] = make_vec<D>([&](int a) { return static_cast<double>(g(i, D * v + a)); });
        return grad;
    }

    // Forward mode: the step of step() (the same Q, bit for bit on the host) and, in the same passes, the tangent dq -> J dq.
    // dq: a tangent of Q (extensive, as state()). Primitives, sweep, fluxes and update each carry their tangent rows
    // (40 + 120 B a cell, 40 B a face). dt is fixed: its dependence on Q (the CFL rule) is not differentiated.
    void step_tangent(double dt, const Cells& dq, Timings* timings = nullptr) {
        require_derivatives("step_tangent");
        timer_.reset();
        refresh();
        borrow_scratch();
        if (!prims_ready_) primitives();
        prims_ready_ = false;
        swept_ = false;
        fit_rows(d_prims_, "hydro_d_prims", n_, V);
        fit_rows(d_gradients_, "hydro_d_gradients", n_, V * D);
        fit_rows(d_fluxes_, "hydro_d_fluxes", faces_, V);
        const auto q = q_;
        const auto prim = prims_;
        const auto g = gradients_;
        const auto F = flux_;
        const auto dprim = d_prims_, dg = d_gradients_, dF = d_fluxes_;
        const Geometry geo = geometry_;
        const Gas gas = gas_;
        const Limiter limiter = limiter_;
        Kokkos::parallel_for(
            "hydro::primitives_tangent", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), KOKKOS_LAMBDA(int i) {
                const State dW = primitive_tangent(geo.per_volume(q, i), geo.per_volume(dq, i), gas);
                for (int v = 0; v < V; ++v) dprim(i, v) = dW[v];
            });
        phase_end(timings, &Timings::primitives);
        Kokkos::parallel_for(
            "hydro::sweep_tangent", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), KOKKOS_LAMBDA(int i) {
                const CellEntries e(geo, prim, dprim, i);
                const State dWi = make_state<D>([&](int v) { return dprim(i, v); });
                Gradient<D> grad;
                const Gradient<D> dgrad = limited_gradient_tangent(primitive_row(prim, i), dWi, e, limiter, grad);
                for (int v = 0; v < V; ++v)
                    for (int a = 0; a < D; ++a) {
                        g(i, D * v + a) = grad[v][a];
                        dg(i, D * v + a) = dgrad[v][a];
                    }
            });
        phase_end(timings, &Timings::sweep);
        Kokkos::parallel_for(
            "hydro::fluxes_tangent", Kokkos::RangePolicy<ExecSpace>(exec_, 0, faces_), KOKKOS_LAMBDA(int f) {
                const int o = geo.owner_of(f), j = geo.neighbour_of(f);
                const Vec a = geo.area_of(f);
                const double magnitude = Kokkos::sqrt(dot(a, a));
                State value{}, tangent{};
                if (magnitude != 0) {
                    const Vec n = a * (1.0 / magnitude);
                    auto sides = [&](int c, State& Wf, State& dWf) {
                        const State W = primitive_row(prim, c), dW = make_state<D>([&](int v) { return dprim(c, v); });
                        const Gradient<D> grad = gradient_row(g, c);
                        Gradient<D> dgrad;
                        for (int v = 0; v < V; ++v) dgrad[v] = make_vec<D>([&](int k) { return dg(c, D * v + k); });
                        const Vec r = geo.face_offset(f, c) + Vec{};
                        bool fallback;
                        Wf = FaceFlux<1, false>::side_value(W, grad, r, dt, gas, fallback);
                        dWf = FaceFlux<1, false>::side_tangent(W, dW, grad, dgrad, r, dt, gas);
                    };
                    State L, R, dL, dR;
                    sides(o, L, dL);
                    sides(j, R, dR);
                    value = Riemann::flux(L, R, n, Vec{}, gas);
                    tangent = Riemann::flux_tangent(L, R, dL, dR, n, Vec{}, gas);
                    for (int v = 0; v < V; ++v) {
                        value[v] *= magnitude;
                        tangent[v] *= magnitude;
                    }
                }
                for (int v = 0; v < V; ++v) {
                    F(f, v) = value[v];
                    dF(f, v) = tangent[v];
                }
            });
        phase_end(timings, &Timings::fluxes);
        Kokkos::parallel_for(
            "hydro::update_tangent", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), KOKKOS_LAMBDA(int i) {
                State sum{}, dsum{};
                for (int k = geo.start(i); k < geo.start(i + 1); ++k) {
                    const int f = geo.face_at(k);
                    const double sign = geo.owned_at(k, f, i) ? 1.0 : -1.0;
                    for (int v = 0; v < V; ++v) {
                        sum[v] += sign * F(f, v);
                        dsum[v] += sign * dF(f, v);
                    }
                }
                for (int v = 0; v < V; ++v) {
                    q(i, v) = q(i, v) - dt * sum[v];
                    dq(i, v) = dq(i, v) - dt * dsum[v];
                }
            });
        phase_end(timings, &Timings::update);
    }

    // Reverse mode: qbar = Q bar after the step on entry, before it on return (J^T qbar); Q stays the state at the start of the step,
    // which the caller restores from a checkpoint. Recomputes the forward primitives and gradients, then three passes:
    //  fluxes: per cell, each of its faces' flux adjoint (each face from both sides, no face rows) -> the cell's W and g bars
    //  sweep:  per cell, the gradient's adjoint -> the fit's b bars and bound bars (rows), the cell's own W bar
    //  update: per cell, the neighbours' b and bound bars gathered, then the primitive adjoint into qbar. No atomics: the same bits every run.
    void step_adjoint(double dt, const Cells& qbar, Timings* timings = nullptr) {
        require_derivatives("step_adjoint");
        timer_.reset();
        refresh();
        borrow_scratch();
        primitives();
        prims_ready_ = true;  // Q is unchanged
        swept_ = false;
        Kokkos::parallel_for("hydro::sweep", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_),
                             Sweep<false, true>{prims_, w_, gradients_, exponents_, geometry_, gas_, limiter_, Magnitudes()});
        phase_end(timings, &Timings::primitives);
        fit_rows(bar_w_, "hydro_bar_w", n_, V);
        fit_rows(bar_g_, "hydro_bar_g", n_, V * D);
        fit_rows(bar_b_, "hydro_bar_b", n_, V * fit_size);
        fit_rows(bar_bound_, "hydro_bar_bound", n_, V);
        if (static_cast<int>(bound_face_.extent(0)) < n_)
            bound_face_ = Kokkos::View<int* [V], Kokkos::LayoutRight, Memory>(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("hydro_bound_face")), n_);
        const auto q = q_;
        const auto prim = prims_;
        const auto g = gradients_;
        const auto wbar = bar_w_, gbar = bar_g_, bbar = bar_b_, boundbar = bar_bound_;
        const auto bound_face = bound_face_;
        const Geometry geo = geometry_;
        const Gas gas = gas_;
        const Limiter limiter = limiter_;
        Kokkos::parallel_for(
            "hydro::fluxes_adjoint", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), KOKKOS_LAMBDA(int i) {
                State Wi_bar{};
                Gradient<D> gi_bar;
                for (int v = 0; v < V; ++v) gi_bar[v] = Vec{};
                for (int k = geo.start(i); k < geo.start(i + 1); ++k) {
                    const int f = geo.face_at(k);
                    const Vec a = geo.area_of(f);
                    const double magnitude = Kokkos::sqrt(dot(a, a));
                    if (magnitude == 0) continue;
                    const int o = geo.owner_of(f), j = geo.neighbour_of(f);
                    const Vec n = a * (1.0 / magnitude);
                    auto side = [&](int c) {
                        bool fallback;
                        return FaceFlux<1, false>::side_value(primitive_row(prim, c), gradient_row(g, c), geo.face_offset(f, c) + Vec{}, dt, gas, fallback);
                    };
                    const State L = side(o), R = side(j);
                    // Q_o -= dt F, Q_j += dt F, F = |a| flux(L, R)
                    const State Fbar = make_state<D>([&](int v) { return magnitude * dt * (qbar(j, v) - qbar(o, v)); });
                    State Lbar{}, Rbar{};
                    Riemann::flux_adjoint(L, R, n, Vec{}, gas, Fbar, Lbar, Rbar);
                    const bool owned = geo.owned_at(k, f, i);
                    FaceFlux<1, false>::side_adjoint(primitive_row(prim, i), gradient_row(g, i), geo.face_offset(f, i) + Vec{}, dt, gas,
                                                     owned ? Lbar : Rbar, Wi_bar, gi_bar);
                }
                for (int v = 0; v < V; ++v) {
                    wbar(i, v) = Wi_bar[v];
                    for (int d = 0; d < D; ++d) gbar(i, D * v + d) = gi_bar[v][d];
                }
            });
        phase_end(timings, &Timings::fluxes);
        Kokkos::parallel_for(
            "hydro::sweep_adjoint", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), KOKKOS_LAMBDA(int i) {
                const CellEntries e(geo, prim, Rows(), i);
                Gradient<D> bar;
                for (int v = 0; v < V; ++v) bar[v] = make_vec<D>([&](int d) { return gbar(i, D * v + d); });
                const GradientAdjoint adjoint = limited_gradient_adjoint(primitive_row(prim, i), e, limiter, bar);
                for (int v = 0; v < V; ++v) {
                    wbar(i, v) += adjoint.self[v];
                    for (int p = 0; p < fit_size; ++p) bbar(i, fit_size * v + p) = adjoint.b[v][p];
                    boundbar(i, v) = adjoint.bound[v];
                    bound_face(i, v) = adjoint.entry[v] < 0 ? -1 : e.face(adjoint.entry[v]);
                }
            });
        phase_end(timings, &Timings::sweep);
        Kokkos::parallel_for(
            "hydro::update_adjoint", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), KOKKOS_LAMBDA(int j) {
                State W_bar = make_state<D>([&](int v) { return wbar(j, v); });
                const Vec cj = geo.row(geo.centroid, j);
                for (int k = geo.start(j); k < geo.start(j + 1); ++k) {
                    const int f = geo.face_at(k), i = geo.other_at(k, f, j);
                    const Vec between = geo.separation(i, j) + cj - geo.row(geo.centroid, i);  // j as cell i's fit sees it
                    const Vec a = geo.area_of(f);
                    const double w = Kokkos::sqrt(dot(a, a));
                    for (int v = 0; v < V; ++v) {
                        double sum = bbar(i, fit_size * v);
                        for (int p = 1; p < fit_size; ++p) sum += between[p - 1] * bbar(i, fit_size * v + p);
                        W_bar[v] += w * sum + (bound_face(i, v) == f ? boundbar(i, v) : 0.0);
                    }
                }
                const double inv = 1.0 / geo.volume(j);
                const State U_bar = primitive_adjoint(geo.per_volume(q, j), W_bar, gas);
                for (int v = 0; v < V; ++v) qbar(j, v) += U_bar[v] * inv;
            });
        phase_end(timings, &Timings::update);
    }

private:
    ExecSpace exec_;
    Mesh* mesh_;
    Gas gas_;
    int order_;
    Update update_;
    Limiter limiter_;
    Motion motion_;
    bool both_meshes_;  // moving at order 1: step_moving()
    int n_, faces_ = 0;
    bool swept_ = false;
    mutable bool prims_ready_ = false;  // the last update wrote prims_ for the current Q and mesh
    StepRule rule_ = StepRule::arepo;
    // |area| per face kept for the sweep (LUNA_NO_MAGNITUDES turns it off, for measurements).
    using Magnitudes = Kokkos::View<double*, Memory, Unmanaged>;
    Magnitudes magnitudes_;
    bool magnitudes_valid_ = false;
    bool cache_magnitudes_ = std::getenv("LUNA_NO_MAGNITUDES") == nullptr;
    bool dt_trace_ = std::getenv("LUNA_DT_TRACE") != nullptr;  // trace_step(): the limiting terms of each time step (diagnostic)
    long long traced_steps_ = 0;
    // Fallbacks of the two flux passes of the last step (one pass: the first), read back by fallbacks() once a step.
    Kokkos::View<long long[2], Memory> fallback_counts_{"hydro_fallbacks"};
    Kokkos::View<long long[2], Kokkos::HostSpace> fallback_host_{"hydro_fallbacks_host"};
    mutable bool fallbacks_read_ = false;
    mutable long long fallbacks_last_ = 0;
    Kokkos::Timer timer_;
    Cells q_, next_;
    Prims prims_;
    Points x_, w_;
    Points old_centroid_;              // moving step: the centroids of t^n
    Steps steps_;                      // moving step: each point's whole grid steps over dt (keep_centroids)
    Kokkos::View<char*, Memory> held_;  // moving step: the rows of prims_, gradients_, exponents_
    Kokkos::View<char*, Memory> own_scratch_;  // after release_mesh(): the cell rows and face fluxes
    Gradients gradients_;
    Exponents exponents_;
    FaceFluxes flux_;
    Geometry geometry_;
    MeshStats mesh_stats_;
    // Derivative rows, allocated by the first step_tangent() / step_adjoint() (empty otherwise): tangents of W, g, the face fluxes;
    // bars of W, g, the fit's b, each variable's bound and the face it came through.
    Rows d_prims_, d_gradients_, d_fluxes_;
    Rows bar_w_, bar_g_, bar_b_, bar_bound_;
    Kokkos::View<int* [V], Kokkos::LayoutRight, Memory> bound_face_;
};

}  // namespace hydro
