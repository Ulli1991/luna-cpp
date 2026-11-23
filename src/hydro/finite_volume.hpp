#pragma once

#include "euler.hpp"
#include "mesh_derivative.hpp"
#include "precision.hpp"
#include "voronoi_mesh.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <map>
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
    // A caller set the positions and updated the mesh itself (a checkpoint restored): the cached face magnitudes and rows go.
    void mesh_moved() {
        magnitudes_valid_ = false;
        prims_ready_ = false;
        swept_ = false;
    }
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

    // No extra work in the sweep: the production instantiation of Sweep. Used by: Sweep (the default Hook).
    struct NoSweepHook {
        struct Cell {};
        KOKKOS_INLINE_FUNCTION Cell begin(int, const State&) const { return {}; }
        template <class... A>
        KOKKOS_INLINE_FUNCTION void entry(Cell&, const A&...) const {}
        template <class... A>
        KOKKOS_INLINE_FUNCTION void solve(Cell&, const A&...) const {}
        KOKKOS_INLINE_FUNCTION void clip(Cell&, int, const Vec&) const {}
        KOKKOS_INLINE_FUNCTION void clipped(Cell&, int, bool, bool, int) const {}
        KOKKOS_INLINE_FUNCTION void finish(Cell&, int) const {}
    };
    // The sweep of time_step(), one thread per cell: the point velocity (moving), then the limited least-squares gradients (order 1).
    // Reads prims_; writes gradients_/exponents_ (read by FaceFlux::gradient()) and w_ (read by face_velocity()). Hook: work riding
    // in the same walks (TangentSweepHook: the tangent of the gradients), nothing by default. Used by: sweep(), time_step_tangent().
    template <bool Moving, bool Grad, class Hook = NoSweepHook>
    struct Sweep {
        Prims prim;
        Points w;
        Gradients g;
        Exponents exponents;
        Geometry geo;
        Gas gas;
        Limiter limiter;
        Kokkos::View<const double*, Memory> magnitudes;  // |area| per face from the last flux pass on this mesh, or empty
        Hook hook{};

        // |area| of face f: cached (the same bits) or from the area row.
        KOKKOS_INLINE_FUNCTION double magnitude_of(int f) const {
            if (magnitudes.data()) return magnitudes(f);
            const Vec a = geo.area_of(f);
            return Kokkos::sqrt(dot(a, a));
        }

        KOKKOS_INLINE_FUNCTION void operator()(int i) const {
            constexpr bool exact = std::is_same_v<G, double>;
            const State Wi = primitive_row(prim, i);
            [[maybe_unused]] auto cell = hook.begin(i, Wi);  // the hook's per-cell state (NoSweepHook: empty)
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
                    hook.entry(cell, f, j, Wj, between, magnitude);
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
            if constexpr (Grad) {
                solve(M, b, grad);
                hook.solve(cell, i, tiny, any_tiny, M, grad);
            }
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
                    hook.clip(cell, f, offset);
                    const hydro::Vec<D, G> rg = cast<G>(offset);
                    for (int v = 0; v < V; ++v) {
                        const G dp = dot(grad[v], rg);
                        if constexpr (exact) {
                            if (dp > 0 && Wi[v] + dp > hi[v]) {
                                grad[v] = grad[v] * (hi[v] > Wi[v] ? (hi[v] - Wi[v]) / dp : 0.0);
                                hook.clipped(cell, v, true, !(hi[v] > Wi[v]), f);
                            } else if (dp < 0 && Wi[v] + dp < lo[v]) {
                                grad[v] = grad[v] * (lo[v] < Wi[v] ? (lo[v] - Wi[v]) / dp : 0.0);
                                hook.clipped(cell, v, false, !(lo[v] < Wi[v]), f);
                            }
                        } else {
                            if (dp > 0 && dp > hi[v])
                                grad[v] = grad[v] * (hi[v] > 0 ? hi[v] / dp : G(0));
                            else if (dp < 0 && dp < lo[v])
                                grad[v] = grad[v] * (lo[v] < 0 ? lo[v] / dp : G(0));
                        }
                    }
                }
                hook.finish(cell, i);
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
    // Entries: the cell's list entries k < count(): load<Tangent>(k) (an Entry), weight(k) (|area|), offset(k) (face centroid minus
    // the cell's centroid), and for the tests W(k), between(k). Used by: step_tangent, step_adjoint, the derivative tests.
    static constexpr int fit_size = D + 1;
    // Per variable: the last list entry whose face clipped the gradient (-1: none), the entry holding the bound it clipped to,
    // which bound (upper), whether it clipped to zero, the last clip's face offset, the bound and its tangent.
    // Clips scale the gradient, so the result is g0 (bound - W_i) / (g0 . o_last).
    struct LimitPattern {
        int last[V], extreme[V];
        bool upper[V], zero[V];
        Vec offset[V];
        double bound[V], dbound[V];
    };
    // The bars of what limited_gradient_adjoint() hands to the neighbours: the fit's right-hand sides b and each variable's bound.
    // The neighbour of entry k gets weight(k) X_k . b[v] + bound[v] if k == entry[v]; self: the cell's own W bar.
    struct GradientAdjoint {
        double b[V][fit_size];
        double bound[V];
        int entry[V];
        State self;
    };
    // One list entry as the fit reads it: the neighbour's W (and dW for the tangent), the fit's offset to it, |area|.
    // Used by: Entries::load(), limited_core().
    struct Entry {
        State W, dW;
        Vec between;
        double weight;
    };
    // The Cholesky factor of the fit's M (upper triangle), and x = M^-1 r with it for every variable.
    // Used by: limited_core(), limited_gradient_adjoint().
    KOKKOS_INLINE_FUNCTION static void cholesky(const double (&M)[fit_size][fit_size], double (&L)[fit_size][fit_size]) {
        constexpr int U = fit_size;
        for (int p = 0; p < U; ++p)
            for (int s = 0; s < U; ++s) L[p][s] = 0;
        for (int p = 0; p < U; ++p)
            for (int s = 0; s <= p; ++s) {
                double sum = M[s][p];
                for (int m = 0; m < s; ++m) sum -= L[p][m] * L[s][m];
                L[p][s] = p == s ? Kokkos::sqrt(sum) : sum / L[s][s];
            }
    }
    KOKKOS_INLINE_FUNCTION static void solve_factored(const double (&L)[fit_size][fit_size], const double (&r)[V][fit_size], double (&x)[V][fit_size]) {
        constexpr int U = fit_size;
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
    // The fit, the neighbours' range and the clips in two walks over the cell's list, each entry loaded once a walk (a third walk
    // only when a face is below 1e-6 of the surface); with Tangent also the tangent's right-hand side and fit. Returns the limited
    // gradient; g0, dg0: the unlimited fit and its tangent; Lc: M's Cholesky factor; M[0][p] in M0.
    // visit(k): called once per entry in the first walk (AdjointCell adds each face's side adjoint there).
    // Used by: limited_gradient(), limited_gradient_tangent(), limited_gradient_adjoint().
    struct NoVisit {
        KOKKOS_INLINE_FUNCTION void operator()(int) const {}
    };
    template <bool Tangent, class Entries, class Visit = NoVisit>
    KOKKOS_INLINE_FUNCTION static Gradient<D> limited_core(const State& Wi, const State& dWi, const Entries& e, Limiter limiter, LimitPattern& pattern,
                                                           Gradient<D>& g0, Gradient<D>& dg0, double (&Lc)[fit_size][fit_size],
                                                           double (&M0)[fit_size], const Visit& visit = Visit{}) {
        constexpr int U = fit_size;
        double M[U][U] = {}, b[V][U] = {}, db[V][U] = {};
        double surface = 0, smallest = Kokkos::Experimental::infinity_v<double>;
        double lo[V], hi[V], dlo[V], dhi[V];
        int lo_k[V], hi_k[V];
        for (int v = 0; v < V; ++v) {
            lo[v] = Kokkos::Experimental::finite_max_v<double>;
            hi[v] = -Kokkos::Experimental::finite_max_v<double>;
            dlo[v] = dhi[v] = 0;
            lo_k[v] = hi_k[v] = -1;
        }
        for (int k = 0; k < e.count(); ++k) {
            const Entry x = e.template load<Tangent>(k);
            visit(k);
            double X[U];
            X[0] = 1;
            for (int p = 1; p < U; ++p) X[p] = x.between[p - 1];
            const double w = x.weight;
            surface += w;
            smallest = Kokkos::fmin(smallest, w);
            for (int p = 0; p < U; ++p)
                for (int s = p; s < U; ++s) M[p][s] += w * X[p] * X[s];
            for (int v = 0; v < V; ++v) {
                const double dW = x.W[v] - Wi[v];
                for (int p = 0; p < U; ++p) b[v][p] += w * X[p] * dW;
                if constexpr (Tangent) {
                    const double ddW = w * (x.dW[v] - dWi[v]);
                    for (int p = 0; p < U; ++p) db[v][p] += X[p] * ddW;
                }
                if (x.W[v] < lo[v]) lo[v] = x.W[v], lo_k[v] = k, dlo[v] = Tangent ? x.dW[v] : 0.0;
                if (x.W[v] > hi[v]) hi[v] = x.W[v], hi_k[v] = k, dhi[v] = Tangent ? x.dW[v] : 0.0;
            }
        }
        // Rare: a face below 1e-6 of the surface does not count for the range or the limiter.
        const double tiny = 1e-6 * surface;
        const bool any_tiny = smallest < tiny;
        if (any_tiny) {
            for (int v = 0; v < V; ++v) {
                lo[v] = Kokkos::Experimental::finite_max_v<double>;
                hi[v] = -Kokkos::Experimental::finite_max_v<double>;
                lo_k[v] = hi_k[v] = -1;
            }
            for (int k = 0; k < e.count(); ++k) {
                const Entry x = e.template load<Tangent>(k);
                if (x.weight < tiny) continue;
                for (int v = 0; v < V; ++v) {
                    if (x.W[v] < lo[v]) lo[v] = x.W[v], lo_k[v] = k, dlo[v] = Tangent ? x.dW[v] : 0.0;
                    if (x.W[v] > hi[v]) hi[v] = x.W[v], hi_k[v] = k, dhi[v] = Tangent ? x.dW[v] : 0.0;
                }
            }
        }
        cholesky(M, Lc);
        for (int p = 0; p < U; ++p) M0[p] = M[0][p];
        double xs[V][U];
        solve_factored(Lc, b, xs);
        for (int v = 0; v < V; ++v) g0[v] = make_vec<D>([&](int a) { return xs[v][1 + a]; });
        if constexpr (Tangent) {
            solve_factored(Lc, db, xs);
            for (int v = 0; v < V; ++v) dg0[v] = make_vec<D>([&](int a) { return xs[v][1 + a]; });
        }
        Gradient<D> grad = g0;
        for (int v = 0; v < V; ++v) {
            pattern.last[v] = pattern.extreme[v] = -1;
            pattern.upper[v] = pattern.zero[v] = false;
        }
        if (limiter != Limiter::none)
            for (int k = 0; k < e.count(); ++k) {
                if (any_tiny && e.weight(k) < tiny) continue;
                const Vec offset = e.offset(k);
                for (int v = 0; v < V; ++v) {
                    const double dp = dot(grad[v], offset);
                    if (dp > 0 && Wi[v] + dp > hi[v]) {
                        grad[v] = grad[v] * (hi[v] > Wi[v] ? (hi[v] - Wi[v]) / dp : 0.0);
                        pattern.last[v] = k, pattern.upper[v] = true, pattern.zero[v] = !(hi[v] > Wi[v]), pattern.offset[v] = offset;
                    } else if (dp < 0 && Wi[v] + dp < lo[v]) {
                        grad[v] = grad[v] * (lo[v] < Wi[v] ? (lo[v] - Wi[v]) / dp : 0.0);
                        pattern.last[v] = k, pattern.upper[v] = false, pattern.zero[v] = !(lo[v] < Wi[v]), pattern.offset[v] = offset;
                    }
                }
            }
        for (int v = 0; v < V; ++v) {
            pattern.extreme[v] = pattern.last[v] < 0 ? -1 : pattern.upper[v] ? hi_k[v] : lo_k[v];
            pattern.bound[v] = pattern.upper[v] ? hi[v] : lo[v];
            pattern.dbound[v] = pattern.upper[v] ? dhi[v] : dlo[v];
        }
        return grad;
    }
    // The limited gradient in double, as Sweep's double path computes it, for every precision policy (the derivatives are those of the
    // double scheme); pattern and the unlimited gradient g0 for the derivatives. Used by: the derivative tests.
    template <class Entries>
    KOKKOS_INLINE_FUNCTION static Gradient<D> limited_gradient(const State& Wi, const Entries& e, Limiter limiter, LimitPattern& pattern,
                                                               Gradient<D>& g0) {
        Gradient<D> dg0;
        double Lc[fit_size][fit_size], M0[fit_size];
        return limited_core<false>(Wi, State{}, e, limiter, pattern, g0, dg0, Lc, M0);
    }
    // dg = J (dW_i, dW_j) of limited_gradient() (geometry fixed), at its branch pattern; g: the forward gradient.
    template <class Entries>
    KOKKOS_INLINE_FUNCTION static Gradient<D> limited_gradient_tangent(const State& Wi, const State& dWi, const Entries& e, Limiter limiter,
                                                                       Gradient<D>& g) {
        LimitPattern pattern;
        Gradient<D> g0, dg0;
        double Lc[fit_size][fit_size], M0[fit_size];
        g = limited_core<true>(Wi, dWi, e, limiter, pattern, g0, dg0, Lc, M0);
        Gradient<D> dg;
        for (int v = 0; v < V; ++v) {
            if (pattern.last[v] < 0) {
                dg[v] = dg0[v];
            } else if (pattern.zero[v]) {
                dg[v] = Vec{};
            } else {
                const Vec o = pattern.offset[v];
                const double den = dot(g0[v], o);
                const double s = (pattern.bound[v] - Wi[v]) / den, ds = (pattern.dbound[v] - dWi[v] - s * dot(dg0[v], o)) / den;
                dg[v] = dg0[v] * s + g0[v] * ds;
            }
        }
        return dg;
    }
    // J^T gbar of limited_gradient(): the bars of b and of the bounds, which the neighbours collect, and the cell's own W bar.
    // visit: as limited_core's; it may complete gbar, which is read only after the first walk.
    template <class Entries, class Visit = NoVisit>
    KOKKOS_INLINE_FUNCTION static GradientAdjoint limited_gradient_adjoint(const State& Wi, const Entries& e, Limiter limiter, const Gradient<D>& gbar,
                                                                           const Visit& visit = Visit{}) {
        constexpr int U = fit_size;
        LimitPattern pattern;
        Gradient<D> g0, dg0;
        double Lc[U][U], M0[U];
        limited_core<false>(Wi, State{}, e, limiter, pattern, g0, dg0, Lc, M0, visit);  // gbar may be filled by visit: read after
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
                const Vec o = pattern.offset[v];
                const double den = dot(g0[v], o);
                const double s = (pattern.bound[v] - Wi[v]) / den, t = dot(gbar[v], g0[v]) / den;
                g0bar = gbar[v] * s - o * (s * t);
                out.bound[v] = t;
                out.entry[v] = pattern.extreme[v];
                out.self[v] -= t;
            }
            xbar[v][0] = 0;
            for (int a = 0; a < D; ++a) xbar[v][1 + a] = g0bar[a];
        }
        solve_factored(Lc, xbar, out.b);  // M is symmetric: b bar = M^-1 x bar
        for (int v = 0; v < V; ++v)
            for (int p = 0; p < U; ++p) out.self[v] -= M0[p] * out.b[v][p];
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
        return smallest;
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
        // Cell i's gradient row as stored (decoded from FP16 if needed), in double. Used by: TangentFluxes, AdjointFaces, AdjointCell.
        KOKKOS_INLINE_FUNCTION Gradient<D> stored_gradient(int i) const {
            const Gradient<D, G> grad = gradient(i);
            Gradient<D> out;
            for (int v = 0; v < V; ++v) out[v] = cast<double>(grad[v]);
            return out;
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
            return side_tangent(W, dW, grad, dg, r, dt, gas, fallback);
        }
        // The same with the forward's fallback given (side() in the policy's precision decides it; the derivative is the double one at
        // the stored gradient). Used by: TangentFluxes, side_tangent() above.
        KOKKOS_INLINE_FUNCTION static State side_tangent(const State& W, const State& dW, const Gradient<D>& grad, const Gradient<D>& dg, Vec r,
                                                         double dt, const Gas& gas, bool fallback) {
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
            side_adjoint(W, grad, r, dt, gas, Wfbar, Wbar, gbar, fallback);
        }
        // The same with the forward's fallback given. Used by: AdjointCell, side_adjoint() above.
        KOKKOS_INLINE_FUNCTION static void side_adjoint(const State& W, const Gradient<D>& grad, Vec r, double dt, const Gas& gas, const State& Wfbar,
                                                        State& Wbar, Gradient<D>& gbar, bool fallback) {
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
            const double dist = Kokkos::sqrt(dot(d, d));
            const Vec c = geo.face_centroid_of(f) - d * 0.5;
            const double inv_dist = 1.0 / dist;
            double facv = dot(c, wL - wR) * inv_dist;
            const double cc = Kokkos::sqrt(dot(c, c));
            if (cc > 0.9 * dist) facv *= 0.9 * dist / cc;
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
            const double magnitude = Kokkos::sqrt(dot(a, a));
            if (magnitude == 0) return State{};
            const Vec wf = this->face_velocity(f, o, j, velocity(o), velocity(j));
            const State L = pass_side(o, this->geo.face_offset(f, o), fallbacks_o);
            const State R = pass_side(j, this->geo.face_offset(f, j), fallbacks_j);
            State F = this->solve(L, R, a * (1.0 / magnitude), wf, this->gas);
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

    // Derivatives of a step: a static mesh, order 1 (MUSCL-Hancock), the per-face loop, dt fixed, every gradient policy (double, grad32,
    // grad16). The forward is the production step (its kernels, its rows as stored); the derivative is that of the double scheme at
    // those rows. Flux rows, Riemann solve and geometry must be double. Throws otherwise (research_notes/autodiff_l1_2026-11-19.md).
    // Used by: step_tangent(), step_adjoint().
    void require_derivatives(const char* what) const {
        static_assert(std::is_same_v<FluxStore, double> && std::is_same_v<typename P::riemann, double> && !Mesh::float_rows,
                      "derivatives of a step: double flux rows, Riemann solve and geometry");
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
        // Entry k with its face and neighbour looked up once.
        template <bool Tangent>
        KOKKOS_INLINE_FUNCTION Entry load(int k) const {
            const int f = face(k), j = geo.other_at(first + k, f, i);
            const Vec a = geo.area_of(f);
            Entry x;
            x.W = primitive_row(prim, j);
            if constexpr (Tangent) x.dW = make_state<D>([&](int v) { return dprim(j, v); });
            x.between = geo.separation(i, j) + geo.row(geo.centroid, j) - centroid;
            x.weight = Kokkos::sqrt(dot(a, a));
            return x;
        }
    };
    // Row i of a Rows of V entries as a state, and of V * D entries as a gradient. Used by: the tangent and adjoint passes.
    KOKKOS_INLINE_FUNCTION static State state_row(const Rows& r, int i) { return make_state<D>([&](int v) { return r(i, v); }); }
    KOKKOS_INLINE_FUNCTION static Gradient<D> gradient_row(const Rows& r, int i) {
        Gradient<D> grad;
        for (int v = 0; v < V; ++v) grad[v] = make_vec<D>([&](int a) { return r(i, D * v + a); });
        return grad;
    }

    // Tangent of primitives(): dW = J dU per cell. Used by: step_tangent().
    struct TangentPrimitives {
        Cells q, dq;
        Rows dprim;
        Geometry geo;
        Gas gas;
        KOKKOS_INLINE_FUNCTION void operator()(int i) const {
            const State dW = primitive_tangent(geo.per_volume(q, i), geo.per_volume(dq, i), gas);
            for (int v = 0; v < V; ++v) dprim(i, v) = dW[v];
        }
    };
    // Tangent of the sweep: dg per cell, the derivative of the double limited gradient at W. Used by: step_tangent().
    struct TangentSweep {
        Prims prim;
        Rows dprim, dg;
        Geometry geo;
        Limiter limiter;
        KOKKOS_INLINE_FUNCTION void operator()(int i) const {
            const CellEntries e(geo, prim, dprim, i);
            Gradient<D> grad;
            const Gradient<D> dgrad = limited_gradient_tangent(primitive_row(prim, i), state_row(dprim, i), e, limiter, grad);
            for (int v = 0; v < V; ++v)
                for (int a = 0; a < D; ++a) dg(i, D * v + a) = dgrad[v][a];
        }
    };
    // The double fit riding in the forward Sweep (Sweep<false, true, SweepFitHook<...>>): the same walks, the neighbours' rows loaded
    // once. A double policy reuses the forward's M, unlimited gradient and clips; grad32 and grad16 keep their own fit and clips in double
    // (the derivative of the double scheme, as TangentSweep). Tangent (time_step_tangent()): writes dg. Not (step_adjoint()): stores what
    // the gradient's adjoint needs, so AdjointCell walks no neighbours for it: M's Cholesky factor, M's row 0, g0, and per variable the
    // last clip's offset, bound, flags and the face its bound came through (fit: 55 doubles, flags: 2 V ints a cell).
    template <bool Tangent>
    struct SweepFitHook {
        Rows dprim, dg;  // Tangent
        Prims prim;
        Geometry geo;
        Rows fit{};                                                // not Tangent
        Kokkos::View<int**, Kokkos::LayoutRight, Memory> flags{};  // not Tangent
        static constexpr int U = fit_size;
        static constexpr bool exact = std::is_same_v<G, double>;
        static constexpr int UM = exact ? 1 : U, VM = exact ? 1 : V;  // the own fit, only for a reduced policy
        static constexpr int VT = Tangent ? V : 1;
        static constexpr int VG = exact ? 1 : V;  // the clip's working gradient: a double policy takes the forward's clips (clipped())
        struct Cell {
            State Wi, dWi;
            double db[VT][U];
            double M[UM][UM], b[VM][UM];
            double lo[V], hi[V], dlo[V], dhi[V];
            int lo_f[V], hi_f[V];
            double Lc[U][U], M0[U];
            Gradient<D> g0, dg0;
            Vec grad[VG];
            bool clipped[V], upper[V], zero[V];
            int clip_f[V];  // the face of the last clip (its offset recomputed in finish())
        };
        KOKKOS_INLINE_FUNCTION Cell begin(int i, const State& Wi) const {
            Cell c;
            c.Wi = Wi;
            if constexpr (Tangent) c.dWi = state_row(dprim, i);
            for (int v = 0; v < VT; ++v)
                for (int p = 0; p < U; ++p) c.db[v][p] = 0;
            for (int v = 0; v < V; ++v) {
                c.lo[v] = Kokkos::Experimental::finite_max_v<double>;
                c.hi[v] = -Kokkos::Experimental::finite_max_v<double>;
                c.dlo[v] = c.dhi[v] = 0;
                c.lo_f[v] = c.hi_f[v] = -1;
                c.clipped[v] = c.upper[v] = c.zero[v] = false;
            }
            for (int p = 0; p < UM; ++p)
                for (int s = 0; s < UM; ++s) c.M[p][s] = 0;
            for (int v = 0; v < VM; ++v)
                for (int p = 0; p < UM; ++p) c.b[v][p] = 0;
            return c;
        }
        // Neighbour j (face f) of the walk: the range with the bounds' tangents or faces, the tangent's right-hand side, the own fit.
        KOKKOS_INLINE_FUNCTION void entry(Cell& c, int f, int j, const State& Wj, const Vec& between, double w) const {
            double X[U];
            X[0] = 1;
            for (int p = 1; p < U; ++p) X[p] = between[p - 1];
            State dWj{};
            if constexpr (Tangent) dWj = state_row(dprim, j);
            for (int v = 0; v < V; ++v) {
                if constexpr (Tangent) {
                    const double ddW = w * (dWj[v] - c.dWi[v]);
                    for (int p = 0; p < U; ++p) c.db[v][p] += X[p] * ddW;
                }
                if (Wj[v] < c.lo[v]) c.lo[v] = Wj[v], c.dlo[v] = dWj[v], c.lo_f[v] = f;
                if (Wj[v] > c.hi[v]) c.hi[v] = Wj[v], c.dhi[v] = dWj[v], c.hi_f[v] = f;
            }
            if constexpr (!exact) {
                for (int p = 0; p < U; ++p)
                    for (int s = p; s < U; ++s) c.M[p][s] += w * X[p] * X[s];
                for (int v = 0; v < V; ++v)
                    for (int p = 0; p < U; ++p) c.b[v][p] += w * X[p] * (Wj[v] - c.Wi[v]);
            }
        }
        // After the walk: the range without faces below tiny (rare, another walk), the unlimited fit and its tangent.
        template <class Matrix, class Grad>
        KOKKOS_INLINE_FUNCTION void solve(Cell& c, int i, double tiny, bool any_tiny, const Matrix& M, const Grad& grad) const {
            if (any_tiny) {
                for (int v = 0; v < V; ++v) {
                    c.lo[v] = Kokkos::Experimental::finite_max_v<double>;
                    c.hi[v] = -Kokkos::Experimental::finite_max_v<double>;
                }
                for (int k = geo.start(i); k < geo.start(i + 1); ++k) {
                    const int f = geo.face_at(k), j = geo.other_at(k, f, i);
                    const Vec a = geo.area_of(f);
                    if (Kokkos::sqrt(dot(a, a)) < tiny) continue;
                    const State Wj = primitive_row(prim, j);
                    for (int v = 0; v < V; ++v) {
                        const double dWj = Tangent ? dprim(j, v) : 0.0;
                        if (Wj[v] < c.lo[v]) c.lo[v] = Wj[v], c.dlo[v] = dWj, c.lo_f[v] = f;
                        if (Wj[v] > c.hi[v]) c.hi[v] = Wj[v], c.dhi[v] = dWj, c.hi_f[v] = f;
                    }
                }
            }
            double xs[V][U];
            if constexpr (exact) {
                cholesky(M, c.Lc);
                for (int p = 0; p < U; ++p) c.M0[p] = M[0][p];
                for (int v = 0; v < V; ++v) c.g0[v] = grad[v];
            } else {
                cholesky(c.M, c.Lc);
                for (int p = 0; p < U; ++p) c.M0[p] = c.M[0][p];
                solve_factored(c.Lc, c.b, xs);
                for (int v = 0; v < V; ++v) c.g0[v] = make_vec<D>([&](int a) { return xs[v][1 + a]; });
            }
            if constexpr (Tangent) {
                double rhs[V][U];
                for (int v = 0; v < V; ++v)
                    for (int p = 0; p < U; ++p) rhs[v][p] = c.db[v < VT ? v : 0][p];
                solve_factored(c.Lc, rhs, xs);
                for (int v = 0; v < V; ++v) c.dg0[v] = make_vec<D>([&](int a) { return xs[v][1 + a]; });
            }
            for (int v = 0; v < VG; ++v) c.grad[v] = c.g0[v];
        }
        // A face (f) of the limiter walk. A reduced policy: the clip of limited_core() on its own double gradient. A double policy takes
        // the forward's clips instead (clipped(): the same decisions, no copy of the gradient).
        KOKKOS_INLINE_FUNCTION void clip(Cell& c, int f, const Vec& offset) const {
            if constexpr (!exact) {
                for (int v = 0; v < V; ++v) {
                    const double dp = dot(c.grad[v], offset);
                    if (dp > 0 && c.Wi[v] + dp > c.hi[v]) {
                        c.grad[v] = c.grad[v] * (c.hi[v] > c.Wi[v] ? (c.hi[v] - c.Wi[v]) / dp : 0.0);
                        c.clipped[v] = true, c.upper[v] = true, c.zero[v] = !(c.hi[v] > c.Wi[v]), c.clip_f[v] = f;
                    } else if (dp < 0 && c.Wi[v] + dp < c.lo[v]) {
                        c.grad[v] = c.grad[v] * (c.lo[v] < c.Wi[v] ? (c.lo[v] - c.Wi[v]) / dp : 0.0);
                        c.clipped[v] = true, c.upper[v] = false, c.zero[v] = !(c.lo[v] < c.Wi[v]), c.clip_f[v] = f;
                    }
                }
            }
        }
        KOKKOS_INLINE_FUNCTION void clipped(Cell& c, int v, bool upper, bool zero, int f) const {
            if constexpr (exact) c.clipped[v] = true, c.upper[v] = upper, c.zero[v] = zero, c.clip_f[v] = f;
        }
        // Tangent: the limited gradient's tangent at the clip pattern, as limited_gradient_tangent(), into dg.
        // Not: the stored fit (FitRow layout) and flags (clipped | upper << 1 | zero << 2, then the bound's face).
        KOKKOS_INLINE_FUNCTION void finish(Cell& c, int i) const {
            if constexpr (Tangent) {
                for (int v = 0; v < V; ++v) {
                    Vec dgv = c.dg0[v];
                    if (c.clipped[v] && c.zero[v]) {
                        dgv = Vec{};
                    } else if (c.clipped[v]) {
                        const Vec o = geo.face_offset(c.clip_f[v], i);
                        const double bound = c.upper[v] ? c.hi[v] : c.lo[v], dbound = c.upper[v] ? c.dhi[v] : c.dlo[v];
                        const double den = dot(c.g0[v], o);
                        const double s = (bound - c.Wi[v]) / den, ds = (dbound - c.dWi[v] - s * dot(c.dg0[v], o)) / den;
                        dgv = c.dg0[v] * s + c.g0[v] * ds;
                    }
                    for (int a = 0; a < D; ++a) dg(i, D * v + a) = dgv[a];
                }
            } else {
                for (int p = 0; p < U; ++p)
                    for (int s = 0; s <= p; ++s) fit(i, FitRow::factor + p * U + s) = c.Lc[p][s];
                for (int p = 0; p < U; ++p) fit(i, FitRow::row0 + p) = c.M0[p];
                for (int v = 0; v < V; ++v) {
                    const Vec offset = c.clipped[v] ? geo.face_offset(c.clip_f[v], i) : Vec{};
                    for (int a = 0; a < D; ++a) {
                        fit(i, FitRow::g0 + D * v + a) = c.g0[v][a];
                        fit(i, FitRow::offset + D * v + a) = offset[a];
                    }
                    fit(i, FitRow::bound + v) = c.upper[v] ? c.hi[v] : c.lo[v];
                    flags(i, v) = (c.clipped[v] ? 1 : 0) | (c.upper[v] ? 2 : 0) | (c.zero[v] ? 4 : 0);
                    flags(i, V + v) = c.upper[v] ? c.hi_f[v] : c.lo_f[v];
                }
            }
        }
    };
    using TangentSweepHook = SweepFitHook<true>;
    // Columns of the stored fit of a cell (SweepFitHook<false>). Used by: SweepFitHook::finish, AdjointCell.
    struct FitRow {
        static constexpr int factor = 0, row0 = factor + fit_size * fit_size, g0 = row0 + fit_size, offset = g0 + V * D,
                             bound = offset + V * D, width = bound + V;
    };
    // The global time step of the sweep's rows (the reductions of sweep(), without its Sweep launch). Used by: time_step_tangent().
    double reduce_time_step(double courant) {
        const Kokkos::RangePolicy<ExecSpace> cells(exec_, 0, n_);
        double smallest = 0;
        if constexpr (D == 2) {
            if (rule_ == StepRule::luna) {
                Kokkos::parallel_reduce("hydro::luna_step", cells, LunaStep<false>{prims_, w_, courant}, Kokkos::Min<double>(smallest));
                return smallest;
            }
        }
        Kokkos::parallel_reduce("hydro::time_step", Kokkos::RangePolicy<ExecSpace>(exec_, 0, std::max(faces_, n_)),
                                TimeStep<false>{{prims_, geometry_, courant}, {prims_, w_, courant}, faces_, n_}, Kokkos::Min<double>(smallest));
        return smallest;
    }
    // time_step() with the tangent of its sweep (TangentSweepHook in the same Sweep): primitives, their tangents, the fused sweep, dt.
    // The next step_tangent(dt, dq) with the same dq reuses both. Returns dt, the same as time_step()'s.
    double time_step_tangent(double courant, const Cells& dq, Timings* timings = nullptr) {
        require_derivatives("time_step_tangent");
        timer_.reset();
        refresh();
        borrow_scratch();
        if (!prims_ready_) primitives();
        prims_ready_ = false;
        fit_rows(d_prims_, "hydro_d_prims", n_, V);
        fit_rows(d_gradients_, "hydro_d_gradients", n_, V * D);
        const Kokkos::RangePolicy<ExecSpace> cells(exec_, 0, n_);
        Kokkos::parallel_for("hydro::primitives_tangent", cells, TangentPrimitives{q_, dq, d_prims_, geometry_, gas_});
        phase_end(timings, &Timings::primitives);
        Kokkos::parallel_for("hydro::sweep_with_tangent", cells,
                             Sweep<false, true, TangentSweepHook>{prims_, w_, gradients_, exponents_, geometry_, gas_, limiter_,
                                                                  magnitudes_valid_ ? magnitudes_ : Magnitudes(),
                                                                  TangentSweepHook{d_prims_, d_gradients_, prims_, geometry_}});
        const double dt = reduce_time_step(courant);
        phase_end(timings, &Timings::sweep);
        swept_ = true;
        tangent_swept_ = true;
        return dt;
    }
    // The face state the derivatives linearise at: side() in double from the rows as stored (for a double policy exactly side();
    // for grad32/grad16 the float face state recomputed in double, so the tangent and the adjoint use the same point whatever each
    // kernel's multiply-add contraction). fallback: as side_value(). Used by: TangentFluxes, AdjointFaces.
    KOKKOS_INLINE_FUNCTION static State linear_side(const FaceFlux<1, false>& flux, int c, Vec r, bool& fallback) {
        return FaceFlux<1, false>::side_value(primitive_row(flux.prim, c), flux.stored_gradient(c), r, flux.dt, flux.gas, fallback);
    }
    // The face flux of step() and its tangent; `count`: fallbacks, as apply(). A double policy computes each side once and the flux
    // from it (side() and solve() are those operations); a reduced one takes the value from FaceFlux (its float sides).
    // Used by: step_tangent().
    struct TangentFluxes {
        FaceFlux<1, false> flux;
        Rows dprim, dg;
        Rows FdF;  // per face: F (the flux of step()) then dF, one 80-B row (the update reads both: 3 sectors, not 4)
        KOKKOS_INLINE_FUNCTION void operator()(int f, long long& count) const {
            const Geometry& geo = flux.geo;
            const Vec a = geo.area_of(f);
            const double magnitude = Kokkos::sqrt(dot(a, a));
            State value{}, tangent{};
            if (magnitude != 0) {
                const int o = geo.owner_of(f), j = geo.neighbour_of(f);
                const Vec shift = Vec{} * (0.5 * flux.dt);
                const Vec ro = geo.face_offset(f, o) + shift, rj = geo.face_offset(f, j) + shift;
                bool fallback_o, fallback_j;
                const State L = linear_side(flux, o, ro, fallback_o), R = linear_side(flux, j, rj, fallback_j);
                const State dL = FaceFlux<1, false>::side_tangent(primitive_row(flux.prim, o), state_row(dprim, o), flux.stored_gradient(o),
                                                                  gradient_row(dg, o), ro, flux.dt, flux.gas, fallback_o);
                const State dR = FaceFlux<1, false>::side_tangent(primitive_row(flux.prim, j), state_row(dprim, j), flux.stored_gradient(j),
                                                                  gradient_row(dg, j), rj, flux.dt, flux.gas, fallback_j);
                const Vec n = a * (1.0 / magnitude);
                if constexpr (std::is_same_v<G, double>) {  // the value from the tangent's own intermediates: one Riemann solve
                    tangent = Riemann::flux_tangent(L, R, dL, dR, n, Vec{}, flux.gas, &value);
                    for (int v = 0; v < V; ++v) value[v] *= magnitude;
                    count += fallback_o + fallback_j;
                } else {
                    int fallbacks = 0;
                    value = flux(f, fallbacks, fallbacks);
                    count += fallbacks;
                    tangent = Riemann::flux_tangent(L, R, dL, dR, n, Vec{}, flux.gas);
                }
                for (int v = 0; v < V; ++v) tangent[v] *= magnitude;
            }
            for (int v = 0; v < V; ++v) {
                FdF(f, v) = static_cast<double>(static_cast<FluxStore>(value[v]));
                FdF(f, V + v) = tangent[v];
            }
        }
    };
    // The update of apply() (the same operations) and its tangent. Used by: step_tangent().
    struct TangentUpdate {
        Cells q, dq;
        Rows FdF;  // TangentFluxes' rows: F then dF
        Geometry geo;
        double dt;
        KOKKOS_INLINE_FUNCTION void operator()(int i) const {
            State sum{}, dsum{};
            for (int k = geo.start(i); k < geo.start(i + 1); ++k) {
                const int f = geo.face_at(k);
                const double sign = geo.owned_at(k, f, i) ? 1.0 : -1.0;
                for (int v = 0; v < V; ++v) {
                    sum[v] += sign * FdF(f, v);
                    dsum[v] += sign * FdF(f, V + v);
                }
            }
            for (int v = 0; v < V; ++v) {
                q(i, v) = q(i, v) - dt * sum[v];
                dq(i, v) = dq(i, v) - dt * dsum[v];
            }
        }
    };

    // Forward mode: the step of step() (its kernels for the forward, so the same Q) and the tangent dq -> J dq beside it.
    // dq: a tangent of Q (extensive, as state()). Rows: dW 40 B and dg 120 B a cell, F and dF 80 B a face. Reuses the sweep of time_step()
    // as step() does, and after time_step_tangent(courant, dq) also its tangents (the fused sweep). dt is fixed: its dependence on Q
    // (the CFL rule) is not differentiated.
    void step_tangent(double dt, const Cells& dq, Timings* timings = nullptr) {
        require_derivatives("step_tangent");
        timer_.reset();
        refresh();
        const bool fused = swept_ && tangent_swept_;  // time_step_tangent() ran: the sweep and the tangents are there
        if (!swept_) {
            borrow_scratch();
            if (!prims_ready_) primitives();
            phase_end(timings, &Timings::primitives);
            Kokkos::parallel_for("hydro::sweep", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_),
                                 Sweep<false, true>{prims_, w_, gradients_, exponents_, geometry_, gas_, limiter_,
                                                    magnitudes_valid_ ? magnitudes_ : Magnitudes()});
            phase_end(timings, &Timings::sweep);
        }
        prims_ready_ = false;
        swept_ = false;
        tangent_swept_ = false;
        fit_rows(d_prims_, "hydro_d_prims", n_, V);
        fit_rows(d_gradients_, "hydro_d_gradients", n_, V * D);
        fit_rows(d_fluxes_, "hydro_d_fluxes", faces_, 2 * V);
        const Kokkos::RangePolicy<ExecSpace> cells(exec_, 0, n_);
        if (!fused) {
            Kokkos::parallel_for("hydro::primitives_tangent", cells, TangentPrimitives{q_, dq, d_prims_, geometry_, gas_});
            phase_end(timings, &Timings::primitives);
            Kokkos::parallel_for("hydro::sweep_tangent", cells, TangentSweep{prims_, d_prims_, d_gradients_, geometry_, limiter_});
            phase_end(timings, &Timings::sweep);
        }
        Kokkos::deep_copy(exec_, Kokkos::subview(fallback_counts_, 1), 0LL);
        fallbacks_read_ = false;
        const FaceFlux<1, false> flux{q_, prims_, gradients_, exponents_, w_, geometry_, gas_, dt};
        Kokkos::parallel_reduce("hydro::fluxes_tangent", Kokkos::RangePolicy<ExecSpace>(exec_, 0, faces_),
                                TangentFluxes{flux, d_prims_, d_gradients_, d_fluxes_}, Kokkos::subview(fallback_counts_, 0));
        phase_end(timings, &Timings::fluxes);
        Kokkos::parallel_for("hydro::update_tangent", cells, TangentUpdate{q_, dq, d_fluxes_, geometry_, dt});
        phase_end(timings, &Timings::update);
    }

    // Reverse pass over the faces, once per face: the Riemann adjoint of face f -> the bars of its two face states, stored as one row
    // (L bar, R bar; 80 B a face). The face states as the tangent takes them (linear_side). Used by: step_adjoint().
    struct AdjointFaces {
        FaceFlux<1, false> flux;
        Cells qbar;
        Rows face_bar;
        KOKKOS_INLINE_FUNCTION void operator()(int f) const {
            const Geometry& geo = flux.geo;
            const Vec a = geo.area_of(f);
            const double magnitude = Kokkos::sqrt(dot(a, a));
            State Lbar{}, Rbar{};
            if (magnitude != 0) {
                const int o = geo.owner_of(f), j = geo.neighbour_of(f);
                const Vec shift = Vec{} * (0.5 * flux.dt);
                bool fallback_o, fallback_j;
                const State L = linear_side(flux, o, geo.face_offset(f, o) + shift, fallback_o);
                const State R = linear_side(flux, j, geo.face_offset(f, j) + shift, fallback_j);
                // Q_o -= dt F, Q_j += dt F, F = |a| flux(L, R)
                const State Fbar = make_state<D>([&](int v) { return magnitude * flux.dt * (qbar(j, v) - qbar(o, v)); });
                Riemann::flux_adjoint(L, R, a * (1.0 / magnitude), Vec{}, flux.gas, Fbar, Lbar, Rbar);
            }
            for (int v = 0; v < V; ++v) {
                face_bar(f, v) = Lbar[v];
                face_bar(f, V + v) = Rbar[v];
            }
        }
    };
    // Reverse pass per cell, one walk of its list: for each face, the bar of the cell's side (AdjointFaces' row, read in list order: no
    // atomics, the same bits every run) through the face-state adjoint -> the cell's W and g bars; then the gradient's adjoint from the
    // fit the forward recompute stored (SweepFitHook<false>), no neighbour rows -> the fit's b bars and bound bars (rows), the cell's own
    // W bar. Used by: step_adjoint().
    struct AdjointCell {
        FaceFlux<1, false> flux;
        Rows face_bar, wbar, bbar, boundbar;
        Kokkos::View<int* [V], Kokkos::LayoutRight, Memory> bound_face;
        Rows fit;
        Kokkos::View<int**, Kokkos::LayoutRight, Memory> flags;
        KOKKOS_INLINE_FUNCTION void operator()(int i) const {
            constexpr int U = fit_size;
            const Geometry& geo = flux.geo;
            const State Wi = primitive_row(flux.prim, i);
            const Gradient<D> gi = flux.stored_gradient(i);
            const Vec shift = Vec{} * (0.5 * flux.dt);
            State Wi_bar{};
            Gradient<D> gi_bar;
            for (int v = 0; v < V; ++v) gi_bar[v] = Vec{};
            for (int k = geo.start(i); k < geo.start(i + 1); ++k) {
                const int f = geo.face_at(k);
                const Vec a = geo.area_of(f);
                if (dot(a, a) == 0) continue;
                const int column = geo.owned_at(k, f, i) ? 0 : V;
                const State mine = make_state<D>([&](int v) { return face_bar(f, column + v); });
                const Vec r = geo.face_offset(f, i) + shift;
                bool fallback;
                FaceFlux<1, false>::side_value(Wi, gi, r, flux.dt, flux.gas, fallback);
                FaceFlux<1, false>::side_adjoint(Wi, gi, r, flux.dt, flux.gas, mine, Wi_bar, gi_bar, fallback);
            }
            // The gradient's adjoint, as limited_gradient_adjoint(), from the stored fit.
            double Lc[U][U], xbar[V][U], b[V][U];
            for (int p = 0; p < U; ++p)
                for (int s = 0; s < U; ++s) Lc[p][s] = s <= p ? fit(i, FitRow::factor + p * U + s) : 0.0;
            double self[V], bound[V];
            int face[V];
            for (int v = 0; v < V; ++v) {
                const int flag = flags(i, v);
                const Vec g0 = make_vec<D>([&](int a) { return fit(i, FitRow::g0 + D * v + a); });
                Vec g0bar = gi_bar[v];
                self[v] = 0;
                bound[v] = 0;
                face[v] = -1;
                if ((flag & 1) && (flag & 4)) {
                    g0bar = Vec{};
                } else if (flag & 1) {
                    const Vec o = make_vec<D>([&](int a) { return fit(i, FitRow::offset + D * v + a); });
                    const double den = dot(g0, o);
                    const double s = (fit(i, FitRow::bound + v) - Wi[v]) / den, t = dot(gi_bar[v], g0) / den;
                    g0bar = gi_bar[v] * s - o * (s * t);
                    bound[v] = t;
                    face[v] = flags(i, V + v);
                    self[v] -= t;
                }
                xbar[v][0] = 0;
                for (int a = 0; a < D; ++a) xbar[v][1 + a] = g0bar[a];
            }
            solve_factored(Lc, xbar, b);  // M is symmetric: b bar = M^-1 x bar
            for (int v = 0; v < V; ++v) {
                for (int p = 0; p < U; ++p) self[v] -= fit(i, FitRow::row0 + p) * b[v][p];
                wbar(i, v) = Wi_bar[v] + self[v];
                for (int p = 0; p < U; ++p) bbar(i, U * v + p) = b[v][p];
                boundbar(i, v) = bound[v];
                bound_face(i, v) = face[v];
            }
        }
    };
    // Gather, per cell: the neighbours' b and bound bars, then the primitive adjoint into qbar. Used by: step_adjoint().
    struct AdjointGather {
        Cells q, qbar;
        Rows wbar, bbar, boundbar;
        Kokkos::View<int* [V], Kokkos::LayoutRight, Memory> bound_face;
        Geometry geo;
        Gas gas;
        KOKKOS_INLINE_FUNCTION void operator()(int j) const {
            State W_bar = state_row(wbar, j);
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
        }
    };

    // Reverse mode: qbar = Q bar after the step on entry, before it on return (J^T qbar); Q stays the state at the start of the step,
    // which the caller restores from a checkpoint. Recomputes the forward primitives and the sweep (production Sweep, rows as stored;
    // its SweepFitHook stores the double fit, 440 B a cell), then AdjointFaces (once per face), AdjointCell, AdjointGather.
    // No atomics: the same bits every run.
    void step_adjoint(double dt, const Cells& qbar, Timings* timings = nullptr) {
        require_derivatives("step_adjoint");
        timer_.reset();
        refresh();
        borrow_scratch();
        primitives();
        prims_ready_ = true;  // Q is unchanged
        swept_ = false;
        const Kokkos::RangePolicy<ExecSpace> cells(exec_, 0, n_);
        fit_rows(bar_fit_, "hydro_bar_fit", n_, FitRow::width);
        if (static_cast<int>(fit_flags_.extent(0)) < n_)
            fit_flags_ = Kokkos::View<int**, Kokkos::LayoutRight, Memory>(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("hydro_fit_flags")), n_, 2 * V);
        Kokkos::parallel_for("hydro::sweep_with_fit", cells,
                             Sweep<false, true, SweepFitHook<false>>{prims_, w_, gradients_, exponents_, geometry_, gas_, limiter_,
                                                                     magnitudes_valid_ ? magnitudes_ : Magnitudes(),
                                                                     SweepFitHook<false>{Rows(), Rows(), prims_, geometry_, bar_fit_, fit_flags_}});
        phase_end(timings, &Timings::primitives);
        fit_rows(bar_w_, "hydro_bar_w", n_, V);
        fit_rows(bar_b_, "hydro_bar_b", n_, V * fit_size);
        fit_rows(bar_bound_, "hydro_bar_bound", n_, V);
        if (static_cast<int>(bound_face_.extent(0)) < n_)
            bound_face_ = Kokkos::View<int* [V], Kokkos::LayoutRight, Memory>(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("hydro_bound_face")), n_);
        const FaceFlux<1, false> flux{q_, prims_, gradients_, exponents_, w_, geometry_, gas_, dt};
        fit_rows(bar_face_, "hydro_bar_face", faces_, 2 * V);
        Kokkos::parallel_for("hydro::faces_adjoint", Kokkos::RangePolicy<ExecSpace>(exec_, 0, faces_), AdjointFaces{flux, qbar, bar_face_});
        phase_end(timings, &Timings::fluxes);
        Kokkos::parallel_for("hydro::cell_adjoint", cells, AdjointCell{flux, bar_face_, bar_w_, bar_b_, bar_bound_, bound_face_, bar_fit_, fit_flags_});
        phase_end(timings, &Timings::sweep);
        Kokkos::parallel_for("hydro::update_adjoint", cells, AdjointGather{q_, qbar, bar_w_, bar_b_, bar_bound_, bound_face_, geometry_, gas_});
        phase_end(timings, &Timings::update);
    }

    // ---- The moving step at fixed topology, order 0 (advance<0, true>() + move()). The state is (Q, x): the geometry depends on the
    // positions, and the positions move with the point velocities, which depend on W. The geometry's derivative in the positions
    // comes from a GeometryDerivative (the mesh library's, or the tests' reference). Correctness first: plain kernels, per-call
    // arrays. The forward is step() itself. Quantisation of the positions to the mesh grid is taken as the identity.
    // d face_angle / d |A|. Used by: Regularised.
    KOKKOS_INLINE_FUNCTION static double face_angle_slope(double area) {
        if constexpr (D == 3)
            return area > 0 ? 0.5 / (M_PI * Kokkos::sqrt(area / M_PI)) : 0.0;
        else
            return 0.5;
    }

    // Sweep<true, ...>'s point velocity of cell i: v + c (fraction c_s / |c|), fraction from the largest face angle (face fmax).
    // Its tangent and adjoint. Used by: step_moving_tangent(), step_moving_adjoint().
    struct Regularised {
        Vec w{}, c{};
        double cs = 0, cd = 0, amax = 0, fraction = 0, slope = 0;
        int fmax = -1, other = -1;
        bool active = false;
        KOKKOS_INLINE_FUNCTION Regularised(const Geometry& geo, int i, const State& W, double sound) : cs(sound) {
            for (int k = geo.start(i); k < geo.start(i + 1); ++k) {
                const int f = geo.face_at(k), j = geo.other_at(k, f, i);
                const Vec d = geo.separation(i, j), a = geo.area_of(f);
                const double angle = face_angle(Kokkos::sqrt(dot(a, a))) / (0.5 * Kokkos::sqrt(dot(d, d)));
                if (angle > amax) {
                    amax = angle;
                    fmax = f;
                    other = j;
                }
            }
            constexpr double F = cell_max_angle_factor;
            if (amax > 0.75 * F) {
                fraction = amax > F ? cell_shaping_speed : cell_shaping_speed * (amax - 0.75 * F) / (0.25 * F);
                slope = amax > F ? 0.0 : cell_shaping_speed / (0.25 * F);
            }
            w = velocity(W);
            c = geo.row(geo.centroid, i);
            cd = Kokkos::sqrt(dot(c, c));
            active = cd > 0 && fraction > 0;
            if (active) w = w + c * (fraction * cs / cd);
        }
        // d amax from the tangents of fmax's area vector and of the separation to the other cell.
        KOKKOS_INLINE_FUNCTION double angle_tangent(const Geometry& geo, int i, Vec dA, Vec dd) const {
            const Vec d = geo.separation(i, other), a = geo.area_of(fmax);
            const double mag = Kokkos::sqrt(dot(a, a)), dist = Kokkos::sqrt(dot(d, d));
            return face_angle_slope(mag) * dot(a, dA) / mag / (0.5 * dist) - amax * dot(d, dd) / (dist * dist);
        }
        KOKKOS_INLINE_FUNCTION Vec tangent(const Geometry& geo, int i, const State& dW, double dcs, Vec dc, Vec dA, Vec dd) const {
            Vec dw = velocity(dW);
            if (!active) return dw;
            const double dfraction = slope != 0 ? slope * angle_tangent(geo, i, dA, dd) : 0.0;
            const double k = fraction * cs / cd, dcd = dot(c, dc) / cd;
            const double dk = (dfraction * cs + fraction * dcs) / cd - k * dcd / cd;
            return dw + dc * k + c * dk;
        }
        // From wbar: adds to Wbar (velocity), csbar, cbar; the bars of fmax's area vector and of the separation to `other`.
        KOKKOS_INLINE_FUNCTION void adjoint(const Geometry& geo, int i, Vec wbar, State& Wbar, double& csbar, Vec& cbar, Vec& Abar, Vec& dbar) const {
            for (int a = 0; a < D; ++a) Wbar[1 + a] += wbar[a];
            if (!active) return;
            const double k = fraction * cs / cd;
            cbar = cbar + wbar * k;
            const double kbar = dot(wbar, c);
            const double fractionbar = kbar * cs / cd;
            csbar += kbar * fraction / cd;
            const double cdbar = -kbar * k / cd;
            cbar = cbar + c * (cdbar / cd);
            if (slope == 0) return;
            const double abar = slope * fractionbar;
            const Vec d = geo.separation(i, other), a = geo.area_of(fmax);
            const double mag = Kokkos::sqrt(dot(a, a)), dist = Kokkos::sqrt(dot(d, d));
            Abar = Abar + a * (abar * face_angle_slope(mag) / mag / (0.5 * dist));
            dbar = dbar - d * (abar * amax / (dist * dist));
        }
    };

    // FaceFlux::face_velocity() as a function of d (owner to neighbour), x_f (relative to the owner), w_L, w_R; tangent; adjoint.
    // Used by: step_moving_tangent(), step_moving_adjoint().
    struct FaceVelocity {
        // The same operations as FaceFlux::face_velocity().
        KOKKOS_INLINE_FUNCTION static Vec value(Vec d, Vec xf, Vec wL, Vec wR) {
            const double dist = Kokkos::sqrt(dot(d, d));
            const Vec c = xf - d * 0.5;
            const double inv_dist = 1.0 / dist;
            double facv = dot(c, wL - wR) * inv_dist;
            const double cc = Kokkos::sqrt(dot(c, c));
            if (cc > 0.9 * dist) facv *= 0.9 * dist / cc;
            return (wL + wR) * 0.5 + d * (facv * inv_dist);
        }
        KOKKOS_INLINE_FUNCTION static Vec tangent(Vec d, Vec xf, Vec wL, Vec wR, Vec dd, Vec dxf, Vec dwL, Vec dwR) {
            const double dist = Kokkos::sqrt(dot(d, d)), ddist = dot(d, dd) / dist;
            const Vec c = xf - d * 0.5, dc = dxf - dd * 0.5, u = wL - wR, du = dwL - dwR;
            const double f0 = dot(c, u) / dist, df0 = (dot(dc, u) + dot(c, du)) / dist - f0 * ddist / dist;
            const double cc = Kokkos::sqrt(dot(c, c));
            double facv = f0, dfacv = df0;
            if (cc > 0.9 * dist) {
                const double s = 0.9 * dist / cc, ds = 0.9 * ddist / cc - s * dot(c, dc) / (cc * cc);
                facv = f0 * s;
                dfacv = df0 * s + f0 * ds;
            }
            return (dwL + dwR) * 0.5 + dd * (facv / dist) + d * (dfacv / dist - facv * ddist / (dist * dist));
        }
        KOKKOS_INLINE_FUNCTION static void adjoint(Vec d, Vec xf, Vec wL, Vec wR, Vec wfbar, Vec& dbar, Vec& xfbar, Vec& wLbar, Vec& wRbar) {
            const double dist = Kokkos::sqrt(dot(d, d));
            const Vec c = xf - d * 0.5, u = wL - wR;
            const double f0 = dot(c, u) / dist, cc = Kokkos::sqrt(dot(c, c));
            const bool clipped = cc > 0.9 * dist;
            const double s = clipped ? 0.9 * dist / cc : 1.0, facv = f0 * s;
            wLbar = wLbar + wfbar * 0.5;
            wRbar = wRbar + wfbar * 0.5;
            // d (facv / dist)
            dbar = dbar + wfbar * (facv / dist);
            const double qbar = dot(wfbar, d);
            const double facvbar = qbar / dist;
            double distbar = -qbar * facv / (dist * dist);
            Vec cbar{};
            const double f0bar = facvbar * s;
            if (clipped) {  // s = 0.9 dist / cc
                const double sbar = facvbar * f0;
                distbar += sbar * 0.9 / cc;
                cbar = cbar + c * (-sbar * s / (cc * cc));
            }
            // f0 = c.u / dist
            cbar = cbar + u * (f0bar / dist);
            const Vec ubar = c * (f0bar / dist);
            distbar -= f0bar * f0 / dist;
            wLbar = wLbar + ubar;
            wRbar = wRbar - ubar;
            // c = x_f - d / 2, dist = |d|
            xfbar = xfbar + cbar;
            dbar = dbar - cbar * 0.5 + d * (distbar / dist);
        }
    };

    using Scalars = Kokkos::View<double*, Memory>;
    using Derivative = GeometryDerivative<D, Memory>;
    using PrimRows = Kokkos::View<double* [V + 2], Kokkos::LayoutRight, Memory>;  // W, c_s (and room for the radius): readable as Prims
    KOKKOS_INLINE_FUNCTION static State row_state(const Rows& r, int i) { return make_state<D>([&](int v) { return r(i, v); }); }
    KOKKOS_INLINE_FUNCTION static State row_state(const PrimRows& r, int i) { return make_state<D>([&](int v) { return r(i, v); }); }
    // A work array of the moving derivatives, kept between steps by label: rows x width entries (width: a Rows view's; the others
    // have it in their type), allocated without a fill when it must grow, zeroed only if asked. Used by: the moving derivatives.
    template <class View>
    View work(const std::string& label, std::size_t rows, std::size_t width = 1, bool zero = false) {
        using T = typename View::value_type;
        std::size_t per_row = 1;
        if constexpr (View::rank == 2) per_row = View::static_extent(1) != 0 ? View::static_extent(1) : width;
        const std::size_t bytes = sizeof(T) * rows * per_row;
        Kokkos::View<char*, Memory>& raw = work_[label];
        if (raw.extent(0) < bytes) raw = Kokkos::View<char*, Memory>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "hydro_work_" + label), bytes);
        T* data = reinterpret_cast<T*>(raw.data());
        View v;
        if constexpr (View::rank == 2) {
            if constexpr (View::static_extent(1) == 0)
                v = View(data, rows, width);
            else
                v = View(data, rows);
        } else {
            v = View(data, rows);
        }
        if (zero) Kokkos::deep_copy(exec_, v, T(0));
        return v;
    }
    // Frees the moving derivatives' work arrays (they stay between steps otherwise).
    void release_derivative_work() { work_.clear(); }
    void require_moving_derivatives(const char* what) const {
        static_assert(std::is_same_v<FluxStore, double> && std::is_same_v<typename P::riemann, double> && !Mesh::float_rows,
                      "derivatives of a step: double flux rows, Riemann solve and geometry");
        if (motion_ != Motion::moving || update_ != Update::per_face)
            throw std::logic_error(std::string("hydro::FiniteVolume::") + what + ": a moving mesh, the per-face loop");
    }

    // W, c_s of every cell and, with dq and dV given, their tangents (cell rows: W in [0, V), c_s in V). Used by: the moving derivatives.
    void moving_primitives(const PrimRows& W, const Rows& dW, const Cells& dq, const Scalars& dV) {
        const Geometry geo = geometry_;
        const auto q = q_;
        const Gas gas = gas_;
        const bool tangent = dW.extent(0) > 0;
        Kokkos::parallel_for(
            "hydro::moving_primitives", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), KOKKOS_LAMBDA(int i) {
                const State U = geo.per_volume(q, i);
                const State Wi = primitive(U, gas);
                const double cs = sound_speed(Wi, gas);
                for (int v = 0; v < V; ++v) W(i, v) = Wi[v];
                W(i, V) = cs;
                if (!tangent) return;
                const double inv = 1.0 / geo.volume(i);
                const State dU = make_state<D>([&](int v) { return (dq(i, v) - U[v] * dV(i)) * inv; });
                const State dWi = primitive_tangent(U, dU, gas);
                for (int v = 0; v < V; ++v) dW(i, v) = dWi[v];
                dW(i, V) = cs > 0 ? 0.5 * gas.gamma * (dWi[D + 1] / Wi[0] - Wi[D + 1] * dWi[0] / (Wi[0] * Wi[0])) / cs : 0.0;
            });
    }

    // One moving step at order 0 and its tangent at fixed topology: dq, dx (of Q, x before) -> (of Q, x after), in place.
    // geometry: the derivative of the current mesh's geometry. The forward is step(dt) (the same bits as without the tangent).
    void step_moving_tangent(double dt, const Cells& dq, const Points& dx, const Derivative& geometry, bool steps_fixed = false) {
        require_moving_derivatives("step_moving_tangent");
        if (order_ == 1) return step_moving_tangent_1(dt, dq, dx, geometry, steps_fixed);
        refresh();
        const Geometry geo = geometry_;
        const Gas gas = gas_;
        Scalars dV = work<Scalars>("hydro_dV", n_);
        Points dc = work<Points>("hydro_dc", n_), dA = work<Points>("hydro_dA", faces_), dxf = work<Points>("hydro_dxf", faces_),
               dw = work<Points>("hydro_dw", n_), w = work<Points>("hydro_w_tangent", n_);
        geometry.tangent(dx, dV, dc, dA, dxf);
        PrimRows W = work<PrimRows>("hydro_moving_W", n_);
        Rows dW = work<Rows>("hydro_moving_dW", n_, V + 1), dG = work<Rows>("hydro_moving_dG", faces_, V);
        moving_primitives(W, dW, dq, dV);
        Kokkos::parallel_for(
            "hydro::moving_velocity_tangent", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), KOKKOS_LAMBDA(int i) {
                const Regularised r(geo, i, row_state(W, i), W(i, V));
                Vec dAf{}, ddf{};
                if (r.fmax >= 0) {
                    dAf = geo.row(dA, r.fmax);
                    ddf = geo.row(dx, r.other) - geo.row(dx, i);
                }
                const Vec t = r.tangent(geo, i, row_state(dW, i), dW(i, V), geo.row(dc, i), dAf, ddf);
                for (int a = 0; a < D; ++a) {
                    dw(i, a) = t[a];
                    w(i, a) = r.w[a];
                }
            });
        Kokkos::parallel_for(
            "hydro::moving_faces_tangent", Kokkos::RangePolicy<ExecSpace>(exec_, 0, faces_), KOKKOS_LAMBDA(int f) {
                const int o = geo.owner_of(f), j = geo.neighbour_of(f);
                const Vec a = geo.area_of(f);
                if (dot(a, a) == 0) {
                    for (int v = 0; v < V; ++v) dG(f, v) = 0;
                    return;
                }
                const Vec d = geo.separation(o, j), xf = geo.face_centroid_of(f), wL = geo.row(w, o), wR = geo.row(w, j);
                const Vec dd = geo.row(dx, j) - geo.row(dx, o);
                const Vec wf = FaceVelocity::value(d, xf, wL, wR);
                const Vec dwf = FaceVelocity::tangent(d, xf, wL, wR, dd, geo.row(dxf, f), geo.row(dw, o), geo.row(dw, j));
                const State dGf = MovingFace<Riemann, D>::tangent(row_state(W, o), row_state(W, j), a, wf, row_state(dW, o), row_state(dW, j),
                                                                  geo.row(dA, f), dwf, gas);
                for (int v = 0; v < V; ++v) dG(f, v) = dGf[v];
            });
        Kokkos::parallel_for(
            "hydro::moving_update_tangent", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), KOKKOS_LAMBDA(int i) {
                State sum{};
                for (int k = geo.start(i); k < geo.start(i + 1); ++k) {
                    const int f = geo.face_at(k);
                    const double sign = geo.owned_at(k, f, i) ? 1.0 : -1.0;
                    for (int v = 0; v < V; ++v) sum[v] += sign * dG(f, v);
                }
                for (int v = 0; v < V; ++v) dq(i, v) -= dt * sum[v];
                for (int a = 0; a < D; ++a) dx(i, a) += dt * dw(i, a);
            });
        step(dt);
    }

    // Reverse mode of step_moving_tangent(): qbar, xbar (of Q, x after the step) on entry, of Q, x before on return. Q and the
    // positions are the state at the start of the step (the caller restores them and updates the mesh); nothing steps.
    void step_moving_adjoint(double dt, const Cells& qbar, const Points& xbar, const Derivative& geometry, bool steps_fixed = false) {
        require_moving_derivatives("step_moving_adjoint");
        if (order_ == 1) return step_moving_adjoint_1(dt, qbar, xbar, geometry, steps_fixed);
        refresh();
        const Geometry geo = geometry_;
        const Gas gas = gas_;
        const auto q = q_;
        PrimRows W = work<PrimRows>("hydro_moving_W", n_);
        Rows face_bar = work<Rows>("hydro_moving_face_bar", faces_, 2 * V + 4 * D);
        moving_primitives(W, Rows(), Cells(), Scalars());
        Points w = work<Points>("hydro_w_adjoint", n_), Abar = work<Points>("hydro_Abar", faces_), xfbar = work<Points>("hydro_xfbar", faces_),
               cbar = work<Points>("hydro_cbar", n_);
        Points angle_Abar = work<Points>("hydro_angle_Abar", n_), angle_dbar = work<Points>("hydro_angle_dbar", n_);
        Scalars Vbar = work<Scalars>("hydro_Vbar", n_);
        Kokkos::View<int*, Memory> fmax = work<Kokkos::View<int*, Memory>>("hydro_fmax", n_);
        Kokkos::parallel_for(
            "hydro::moving_velocity", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), KOKKOS_LAMBDA(int i) {
                const Regularised r(geo, i, row_state(W, i), W(i, V));
                for (int a = 0; a < D; ++a) w(i, a) = r.w[a];
            });
        // Per face: G's bar from both cells' Q bars, then the bars of W_L, W_R, A, w_f, and through w_f of d, x_f, w_L, w_R.
        // face_bar row: W_L bar, W_R bar, w_L bar, w_R bar, d bar, A bar (the separation's bar is gathered by the cells).
        Kokkos::parallel_for(
            "hydro::moving_faces_adjoint", Kokkos::RangePolicy<ExecSpace>(exec_, 0, faces_), KOKKOS_LAMBDA(int f) {
                for (int s = 0; s < 2 * V + 4 * D; ++s) face_bar(f, s) = 0;
                for (int a = 0; a < D; ++a) xfbar(f, a) = 0;
                const int o = geo.owner_of(f), j = geo.neighbour_of(f);
                const Vec a = geo.area_of(f);
                if (dot(a, a) == 0) return;
                const Vec d = geo.separation(o, j), xf = geo.face_centroid_of(f), wL = geo.row(w, o), wR = geo.row(w, j);
                const Vec wf = FaceVelocity::value(d, xf, wL, wR);
                const State Gbar = make_state<D>([&](int v) { return -dt * (qbar(o, v) - qbar(j, v)); });
                State Lbar{}, Rbar{};
                Vec Ab{}, wfbar{}, db{}, xfb{}, wLb{}, wRb{};
                MovingFace<Riemann, D>::adjoint(row_state(W, o), row_state(W, j), a, wf, gas, Gbar, Lbar, Rbar, Ab, wfbar);
                FaceVelocity::adjoint(d, xf, wL, wR, wfbar, db, xfb, wLb, wRb);
                for (int v = 0; v < V; ++v) {
                    face_bar(f, v) = Lbar[v];
                    face_bar(f, V + v) = Rbar[v];
                }
                for (int c = 0; c < D; ++c) {
                    face_bar(f, 2 * V + c) = wLb[c];
                    face_bar(f, 2 * V + D + c) = wRb[c];
                    face_bar(f, 2 * V + 2 * D + c) = db[c];
                    face_bar(f, 2 * V + 3 * D + c) = Ab[c];
                    xfbar(f, c) = xfb[c];
                }
            });
        // Per cell: w's bar (dt xbar and the faces'), through the point velocity to W, c_s, c and the angle face; W's bar from the
        // faces; Q's bar and V's bar through the primitives.
        Kokkos::parallel_for(
            "hydro::moving_cells_adjoint", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), KOKKOS_LAMBDA(int i) {
                Vec wbar = geo.row(xbar, i) * dt;
                State Wbar{};
                for (int k = geo.start(i); k < geo.start(i + 1); ++k) {
                    const int f = geo.face_at(k);
                    const bool owned = geo.owned_at(k, f, i);
                    for (int v = 0; v < V; ++v) Wbar[v] += face_bar(f, (owned ? 0 : V) + v);
                    for (int a = 0; a < D; ++a) wbar[a] += face_bar(f, 2 * V + (owned ? 0 : D) + a);
                }
                const Regularised r(geo, i, row_state(W, i), W(i, V));
                double csbar = 0;
                Vec cb{}, Ab{}, db{};
                r.adjoint(geo, i, wbar, Wbar, csbar, cb, Ab, db);
                fmax(i) = r.fmax;
                for (int a = 0; a < D; ++a) {
                    cbar(i, a) = cb[a];
                    angle_Abar(i, a) = Ab[a];
                    angle_dbar(i, a) = db[a];
                }
                // c_s = sqrt(gamma P / rho)
                const State Wi = row_state(W, i);
                if (W(i, V) > 0) {
                    const double c2bar = csbar / (2 * W(i, V));
                    Wbar[D + 1] += c2bar * gas.gamma / Wi[0];
                    Wbar[0] -= c2bar * gas.gamma * Wi[D + 1] / (Wi[0] * Wi[0]);
                }
                // W = primitive(Q / V)
                const State U = geo.per_volume(q, i);
                const State Ubar = primitive_adjoint(U, Wbar, gas);
                const double inv = 1.0 / geo.volume(i);
                double vb = 0;
                for (int v = 0; v < V; ++v) {
                    qbar(i, v) += Ubar[v] * inv;
                    vb -= Ubar[v] * U[v] * inv;
                }
                Vbar(i) = vb;
            });
        // Per face: A's bar with the angle faces of its two cells. Per cell: the separations' bars (the faces' and the angle face's).
        Kokkos::parallel_for(
            "hydro::moving_area_bar", Kokkos::RangePolicy<ExecSpace>(exec_, 0, faces_), KOKKOS_LAMBDA(int f) {
                const int o = geo.owner_of(f), j = geo.neighbour_of(f);
                for (int a = 0; a < D; ++a)
                    Abar(f, a) = face_bar(f, 2 * V + 3 * D + a) + (fmax(o) == f ? angle_Abar(o, a) : 0.0) + (fmax(j) == f ? angle_Abar(j, a) : 0.0);
            });
        Kokkos::parallel_for(
            "hydro::moving_separation_bar", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), KOKKOS_LAMBDA(int i) {
                Vec sum = geo.row(angle_dbar, i) * -1.0;  // d = x_other - x_i of i's own angle face
                for (int k = geo.start(i); k < geo.start(i + 1); ++k) {
                    const int f = geo.face_at(k), j = geo.other_at(k, f, i);
                    const bool owned = geo.owned_at(k, f, i);
                    const Vec db = make_vec<D>([&](int a) { return face_bar(f, 2 * V + 2 * D + a); });  // d = x_j - x_o
                    sum = owned ? sum - db : sum + db;
                    if (fmax(j) == f) sum = sum + geo.row(angle_dbar, j);  // j's angle face: d = x_i - x_j
                }
                for (int a = 0; a < D; ++a) xbar(i, a) += sum[a];
            });
        geometry.adjoint(Vbar, cbar, Abar, xfbar, xbar);
    }

    // ---- Order 1 (step_moving(): MUSCL-Hancock in two passes, on the meshes of t^n and t^n+1). The derivative also goes through the
    // limited fit's geometry (centroid offsets, weights |A|, the limiter's face offsets) and the second pass's geometry.
    // The tangents of the geometry cell i's fit and limiter read. Used by: moving_gradient_tangent().
    struct FitGeometry {
        Geometry geo;
        Points dx, dc, dA, dxf;
        int i;
        // between = x_j - x_i + c_j - c_i; weight = |A_f|; offset = x_f relative to i, minus c_i (face_offset()).
        KOKKOS_INLINE_FUNCTION Vec dbetween(int j) const { return geo.row(dx, j) - geo.row(dx, i) + geo.row(dc, j) - geo.row(dc, i); }
        KOKKOS_INLINE_FUNCTION double dweight(int f) const {
            const Vec a = geo.area_of(f);
            return dot(a, geo.row(dA, f)) / Kokkos::sqrt(dot(a, a));
        }
        KOKKOS_INLINE_FUNCTION Vec doffset(int f) const { return offset_tangent(geo, f, i, dx, dc, dxf); }
    };
    // The tangent of face_offset(f, i) from the geometry's tangents. Used by: FitGeometry, the moving passes.
    KOKKOS_INLINE_FUNCTION static Vec offset_tangent(const Geometry& geo, int f, int i, const Points& dx, const Points& dc, const Points& dxf) {
        const int o = geo.owner_of(f);
        Vec d = geo.row(dxf, f) - geo.row(dc, i);
        if (o != i) d = d + geo.row(dx, o) - geo.row(dx, i);
        return d;
    }
    // The global list position of face f in cell i's list. Used by: step_moving_adjoint_1().
    KOKKOS_INLINE_FUNCTION static int entry_of(const Geometry& geo, int i, int f) {
        for (int k = geo.start(i); k < geo.start(i + 1); ++k)
            if (geo.face_at(k) == f) return k;
        return -1;
    }

    // dg of limited_gradient() in W and in the geometry, at its branch pattern; g: the forward gradient. The fit x = M^-1 b moves by
    // M^-1 (db - dM x); the last clip's scale s = (bound - W_i) / (g0 . o) also by o's tangent.
    KOKKOS_INLINE_FUNCTION static Gradient<D> moving_gradient_tangent(const State& Wi, const State& dWi, const CellEntries& e, const FitGeometry& t,
                                                                      Limiter limiter, Gradient<D>& g) {
        constexpr int U = fit_size;
        LimitPattern pattern;
        Gradient<D> g0, dg0;
        double Lc[U][U], M0[U];
        g = limited_core<true>(Wi, dWi, e, limiter, pattern, g0, dg0, Lc, M0);  // dg0: the part in W
        double b[V][U] = {}, rhs[V][U] = {}, dM[U][U] = {};
        for (int k = 0; k < e.count(); ++k) {
            const int f = e.face(k), j = e.other(k);
            const Entry x = e.template load<false>(k);
            const Vec dB = t.dbetween(j);
            const double w = x.weight, dw = t.dweight(f);
            double X[U], dX[U];
            X[0] = 1;
            dX[0] = 0;
            for (int p = 1; p < U; ++p) {
                X[p] = x.between[p - 1];
                dX[p] = dB[p - 1];
            }
            for (int p = 0; p < U; ++p)
                for (int s = 0; s < U; ++s) dM[p][s] += dw * X[p] * X[s] + w * (dX[p] * X[s] + X[p] * dX[s]);
            for (int v = 0; v < V; ++v) {
                const double dW = x.W[v] - Wi[v];
                for (int p = 0; p < U; ++p) {
                    b[v][p] += w * X[p] * dW;
                    rhs[v][p] += (dw * X[p] + w * dX[p]) * dW;
                }
            }
        }
        double xs[V][U], dxs[V][U];
        solve_factored(Lc, b, xs);
        for (int v = 0; v < V; ++v)
            for (int p = 0; p < U; ++p)
                for (int s = 0; s < U; ++s) rhs[v][p] -= dM[p][s] * xs[v][s];
        solve_factored(Lc, rhs, dxs);
        for (int v = 0; v < V; ++v) dg0[v] = dg0[v] + make_vec<D>([&](int a) { return dxs[v][1 + a]; });
        Gradient<D> dg;
        for (int v = 0; v < V; ++v) {
            if (pattern.last[v] < 0) {
                dg[v] = dg0[v];
            } else if (pattern.zero[v]) {
                dg[v] = Vec{};
            } else {
                const Vec o = pattern.offset[v], dof = t.doffset(e.face(pattern.last[v]));
                const double den = dot(g0[v], o);
                const double s = (pattern.bound[v] - Wi[v]) / den;
                const double ds = (pattern.dbound[v] - dWi[v] - s * (dot(dg0[v], o) + dot(g0[v], dof))) / den;
                dg[v] = dg0[v] * s + g0[v] * ds;
            }
        }
        return dg;
    }
    // The transpose: out as limited_gradient_adjoint() (the neighbours' and the cell's own W bars); emit(k, between_bar, weight_bar,
    // offset_bar) once per entry with the geometric bars (Mbar = -sum_v beta_v x_v^T, beta_v = M^-1 xbar_v).
    template <class Emit>
    KOKKOS_INLINE_FUNCTION static GradientAdjoint moving_gradient_adjoint(const State& Wi, const CellEntries& e, Limiter limiter, const Gradient<D>& gbar,
                                                                          const Emit& emit) {
        constexpr int U = fit_size;
        LimitPattern pattern;
        Gradient<D> g0, dg0;
        double Lc[U][U], M0[U];
        limited_core<false>(Wi, State{}, e, limiter, pattern, g0, dg0, Lc, M0);
        GradientAdjoint out;
        double xbar[V][U];
        Vec obar[V];
        for (int v = 0; v < V; ++v) {
            out.self[v] = 0;
            out.bound[v] = 0;
            out.entry[v] = -1;
            obar[v] = Vec{};
            Vec g0bar = gbar[v];
            if (pattern.zero[v] && pattern.last[v] >= 0) {
                g0bar = Vec{};
            } else if (pattern.last[v] >= 0) {
                const Vec o = pattern.offset[v];
                const double den = dot(g0[v], o);
                const double s = (pattern.bound[v] - Wi[v]) / den, tt = dot(gbar[v], g0[v]) / den;
                g0bar = gbar[v] * s - o * (s * tt);
                obar[v] = g0[v] * (-s * tt);
                out.bound[v] = tt;
                out.entry[v] = pattern.extreme[v];
                out.self[v] -= tt;
            }
            xbar[v][0] = 0;
            for (int a = 0; a < D; ++a) xbar[v][1 + a] = g0bar[a];
        }
        solve_factored(Lc, xbar, out.b);
        for (int v = 0; v < V; ++v)
            for (int p = 0; p < U; ++p) out.self[v] -= M0[p] * out.b[v][p];
        // The fit itself (with its intercept) for Mbar.
        double b[V][U] = {}, xs[V][U], Mbar[U][U];
        for (int k = 0; k < e.count(); ++k) {
            const Entry x = e.template load<false>(k);
            double X[U];
            X[0] = 1;
            for (int p = 1; p < U; ++p) X[p] = x.between[p - 1];
            for (int v = 0; v < V; ++v)
                for (int p = 0; p < U; ++p) b[v][p] += x.weight * X[p] * (x.W[v] - Wi[v]);
        }
        solve_factored(Lc, b, xs);
        for (int p = 0; p < U; ++p)
            for (int s = 0; s < U; ++s) {
                Mbar[p][s] = 0;
                for (int v = 0; v < V; ++v) Mbar[p][s] -= out.b[v][p] * xs[v][s];
            }
        for (int k = 0; k < e.count(); ++k) {
            const Entry x = e.template load<false>(k);
            const double w = x.weight;
            double X[U];
            X[0] = 1;
            for (int p = 1; p < U; ++p) X[p] = x.between[p - 1];
            double weight_bar = 0, Xbar[U] = {};
            for (int v = 0; v < V; ++v) {
                const double dW = x.W[v] - Wi[v];
                double bX = 0;
                for (int p = 0; p < U; ++p) bX += out.b[v][p] * X[p];
                weight_bar += bX * dW;
                for (int p = 0; p < U; ++p) Xbar[p] += w * dW * out.b[v][p];
            }
            for (int p = 0; p < U; ++p)
                for (int s = 0; s < U; ++s) {
                    weight_bar += X[p] * Mbar[p][s] * X[s];
                    Xbar[p] += w * (Mbar[p][s] + Mbar[s][p]) * X[s];
                }
            Vec ob{};
            for (int v = 0; v < V; ++v)
                if (pattern.last[v] == k && !pattern.zero[v]) ob = ob + obar[v];
            emit(k, make_vec<D>([&](int a) { return Xbar[1 + a]; }), weight_bar, ob);
        }
        return out;
    }

    // Order 1: tangents of the fit's rows at t^n, pass 1 on t^n, step(), then pass 2 on t^n+1. The point velocities the passes take
    // are the forward's whole grid steps per dt (w-hat); their tangent is dw (the quantisation taken as the identity). steps_fixed: the steps
    // held (derivative 0), which central differences small enough to change no step can check (the tests).
    void step_moving_tangent_1(double dt, const Cells& dq, const Points& dx, const Derivative& geometry, bool steps_fixed) {
        refresh();
        const Gas gas = gas_;
        const Limiter limiter = limiter_;
        const exact_predicates::BoxN<D> box = mesh_->sides();
        const auto x = x_;
        const Kokkos::RangePolicy<ExecSpace> cells(exec_, 0, n_);
        Scalars dV = work<Scalars>("hydro_dV", n_);
        Points dc = work<Points>("hydro_dc", n_), dA = work<Points>("hydro_dA", faces_), dxf = work<Points>("hydro_dxf", faces_);
        geometry.tangent(dx, dV, dc, dA, dxf);
        PrimRows W = work<PrimRows>("hydro_moving_W", n_);
        Rows dW = work<Rows>("hydro_moving_dW", n_, V + 1), g = work<Rows>("hydro_moving_g", n_, V * D), dg = work<Rows>("hydro_moving_dg", n_, V * D);
        Points wh = work<Points>("hydro_w_hat", n_), dw = work<Points>("hydro_dw", n_), c0 = work<Points>("hydro_c0", n_),
               dc0 = work<Points>("hydro_dc0", n_);
        moving_primitives(W, dW, dq, dV);
        const double unit = box.step / dt;
        {
            const Geometry geo = geometry_;
            Kokkos::parallel_for(
                "hydro::moving_cells_tangent_1", cells, KOKKOS_LAMBDA(int i) {
                    const State Wi = row_state(W, i), dWi = row_state(dW, i);
                    const Regularised r(geo, i, Wi, W(i, V));
                    Vec dAf{}, ddf{};
                    if (r.fmax >= 0) {
                        dAf = geo.row(dA, r.fmax);
                        ddf = geo.row(dx, r.other) - geo.row(dx, i);
                    }
                    const Vec t = r.tangent(geo, i, dWi, dW(i, V), geo.row(dc, i), dAf, ddf);
                    for (int a = 0; a < D; ++a) {
                        const int k0 = exact_predicates::snap(x(i, a), box, a), k1 = exact_predicates::snap(x(i, a) + dt * r.w[a], box, a);
                        wh(i, a) = static_cast<double>(exact_predicates::delta(k0, k1, box.period[a])) * unit;
                        dw(i, a) = t[a];
                        c0(i, a) = geo.centroid(i, a);
                        dc0(i, a) = dc(i, a);
                    }
                    const CellEntries e(geo, Prims(W), dW, i);
                    Gradient<D> gi;
                    const Gradient<D> dgi = moving_gradient_tangent(Wi, dWi, e, FitGeometry{geo, dx, dc, dA, dxf, i}, limiter, gi);
                    for (int v = 0; v < V; ++v)
                        for (int a = 0; a < D; ++a) {
                            g(i, D * v + a) = gi[v][a];
                            dg(i, D * v + a) = dgi[v][a];
                        }
                });
            if (steps_fixed) Kokkos::deep_copy(exec_, dw, 0.0);  // the grid steps held: no derivative through w-hat
            moving_pass_tangent(false, dt, geo, W, dW, g, dg, wh, dw, c0, dc0, dx, dc, dA, dxf, dq);
        }
        step(dt);
        Kokkos::parallel_for(
            "hydro::moving_drift_tangent", cells, KOKKOS_LAMBDA(int i) {
                for (int a = 0; a < D; ++a) dx(i, a) += dt * dw(i, a);
            });
        refresh();
        Scalars dV2 = work<Scalars>("hydro_dV2", n_);
        Points dc2 = work<Points>("hydro_dc2", n_), dA2 = work<Points>("hydro_dA2", faces_), dxf2 = work<Points>("hydro_dxf2", faces_);
        geometry.tangent(dx, dV2, dc2, dA2, dxf2);
        moving_pass_tangent(true, dt, geometry_, W, dW, g, dg, wh, dw, c0, dc0, dx, dc2, dA2, dxf2, dq);
    }
    // One pass's tangent: dq -= dt/2 sum dG. First: W + g r at t^n (r: face_offset). Second: on t^n+1, W advanced a whole step and
    // extrapolated from the old centroid to the new face (r = face_offset + c - c^n + dt w-hat), as MovingFlux<true>.
    void moving_pass_tangent(bool second, double dt, const Geometry& geo, const PrimRows& W, const Rows& dW, const Rows& g, const Rows& dg,
                             const Points& wh, const Points& dw, const Points& c0, const Points& dc0, const Points& dx, const Points& dc,
                             const Points& dA, const Points& dxf, const Cells& dq) {
        const Gas gas = gas_;
        Rows dG = work<Rows>("hydro_moving_dG", faces_, V);
        Kokkos::parallel_for(
            "hydro::moving_pass_tangent", Kokkos::RangePolicy<ExecSpace>(exec_, 0, faces_), KOKKOS_LAMBDA(int f) {
                for (int v = 0; v < V; ++v) dG(f, v) = 0;
                const int o = geo.owner_of(f), j = geo.neighbour_of(f);
                const Vec a = geo.area_of(f);
                if (dot(a, a) == 0) return;
                const Vec d = geo.separation(o, j), xf = geo.face_centroid_of(f), wL = geo.row(wh, o), wR = geo.row(wh, j);
                const Vec dd = geo.row(dx, j) - geo.row(dx, o);
                const Vec wf = FaceVelocity::value(d, xf, wL, wR);
                const Vec dwf = FaceVelocity::tangent(d, xf, wL, wR, dd, geo.row(dxf, f), geo.row(dw, o), geo.row(dw, j));
                State side[2], dside[2];
                for (int s = 0; s < 2; ++s) {
                    const int i = s == 0 ? o : j;
                    const State Wi = row_state(W, i), dWi = row_state(dW, i);
                    const Gradient<D> gi = gradient_row(g, i), dgi = gradient_row(dg, i);
                    Vec r = geo.face_offset(f, i), dr = offset_tangent(geo, f, i, dx, dc, dxf);
                    if (second) {
                        r = r + geo.row(geo.centroid, i) - geo.row(c0, i) + geo.row(wh, i) * dt;
                        dr = dr + geo.row(dc, i) - geo.row(dc0, i) + geo.row(dw, i) * dt;
                    }
                    bool fallback;
                    const State Wf = FaceFlux<1, true>::side_value(Wi, gi, r, second ? 2 * dt : 0.0, gas, fallback);
                    side[s] = Wf;
                    dside[s] = FaceFlux<1, true>::side_tangent(Wi, dWi, gi, dgi, r, second ? 2 * dt : 0.0, gas, fallback);
                    if (!fallback)
                        for (int v = 0; v < V; ++v) dside[s][v] += dot(gi[v], dr);
                }
                const State dGf = MovingFace<Riemann, D>::tangent(side[0], side[1], a, wf, dside[0], dside[1], geo.row(dA, f), dwf, gas);
                for (int v = 0; v < V; ++v) dG(f, v) = dGf[v];
            });
        Kokkos::parallel_for(
            "hydro::moving_pass_update_tangent", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), KOKKOS_LAMBDA(int i) {
                State sum{};
                for (int k = geo.start(i); k < geo.start(i + 1); ++k) {
                    const int f = geo.face_at(k);
                    const double sign = geo.owned_at(k, f, i) ? 1.0 : -1.0;
                    for (int v = 0; v < V; ++v) sum[v] += sign * dG(f, v);
                }
                for (int v = 0; v < V; ++v) dq(i, v) -= 0.5 * dt * sum[v];
            });
    }

    // A moving pass's face row: 2V + 2D doubles (moving_pass_adjoint()).
    static constexpr int pass_row = 2 * V + 2 * D;
    // The offset r of cell i's side of face f (pass 2: from the old centroid to the new face, + dt w-hat). Used by: the passes.
    KOKKOS_INLINE_FUNCTION static Vec pass_offset(bool second, double dt, const Geometry& geo, int f, int i, const Points& wh, const Points& c0) {
        Vec r = geo.face_offset(f, i);
        if (second) r = r + geo.row(geo.centroid, i) - geo.row(c0, i) + geo.row(wh, i) * dt;
        return r;
    }
    // One pass's adjoint per face, from qbar. Row (2V + 2D): the two sides' bars (of the face states), the face velocity's bar, and
    // the neighbour side's bar of its offset (its offset carries x_owner). A bar and x_f bar (both sides' offsets in it) go to the
    // face arrays Ab, xfb. The sides' own adjoints (W, g, c, w-hat, x) are each cell's, in moving_pass_gather().
    // Used by: step_moving_adjoint_1().
    void moving_pass_adjoint(bool second, double dt, const Geometry& geo, const PrimRows& W, const Rows& g, const Points& wh, const Points& c0,
                             const Cells& qbar, const Rows& rows, const Points& Ab, const Points& xfb) {
        const Gas gas = gas_;
        Kokkos::parallel_for(
            "hydro::moving_pass_adjoint", Kokkos::RangePolicy<ExecSpace>(exec_, 0, faces_), KOKKOS_LAMBDA(int f) {
                const int o = geo.owner_of(f), j = geo.neighbour_of(f);
                const Vec a = geo.area_of(f);
                if (dot(a, a) == 0) {  // no flux: no bars
                    for (int s = 0; s < pass_row; ++s) rows(f, s) = 0;
                    for (int c = 0; c < D; ++c) Ab(f, c) = xfb(f, c) = 0;
                    return;
                }
                const Vec d = geo.separation(o, j), xf = geo.face_centroid_of(f), wL = geo.row(wh, o), wR = geo.row(wh, j);
                const Vec wf = FaceVelocity::value(d, xf, wL, wR);
                State side[2];
                Vec rbar[2];
                bool fallback[2];
                Gradient<D> gs[2];
                for (int s = 0; s < 2; ++s) {
                    const int i = s == 0 ? o : j;
                    gs[s] = gradient_row(g, i);
                    side[s] = FaceFlux<1, true>::side_value(row_state(W, i), gs[s], pass_offset(second, dt, geo, f, i, wh, c0), second ? 2 * dt : 0.0,
                                                           gas, fallback[s]);
                }
                const State Gbar = make_state<D>([&](int v) { return -0.5 * dt * (qbar(o, v) - qbar(j, v)); });
                State Sbar[2] = {State{}, State{}};
                Vec A{}, wfbar{}, db{}, xfbar{}, wLb{}, wRb{};
                MovingFace<Riemann, D>::adjoint(side[0], side[1], a, wf, gas, Gbar, Sbar[0], Sbar[1], A, wfbar);
                FaceVelocity::adjoint(d, xf, wL, wR, wfbar, db, xfbar, wLb, wRb);  // x_f's share here; the cells redo theirs
                for (int s = 0; s < 2; ++s) {
                    rbar[s] = Vec{};
                    if (!fallback[s])
                        for (int v = 0; v < V; ++v) rbar[s] = rbar[s] + gs[s][v] * Sbar[s][v];
                    xfbar = xfbar + rbar[s];  // r = x_f relative to i (...) - c_i
                }
                for (int v = 0; v < V; ++v) {
                    rows(f, v) = Sbar[0][v];
                    rows(f, V + v) = Sbar[1][v];
                }
                for (int c = 0; c < D; ++c) {
                    rows(f, 2 * V + c) = wfbar[c];
                    rows(f, 2 * V + D + c) = rbar[1][c];
                    Ab(f, c) = A[c];
                    xfb(f, c) = xfbar[c];
                }
            });
    }
    // Each cell's sides of one pass, from the face rows: the side's adjoint (W, g), its offset's bar (c^n, x, and w-hat on pass 2),
    // its share of the face velocity's adjoint (w-hat, x); summed in registers and added once. Used by: step_moving_adjoint_1().
    void moving_pass_gather(bool second, double dt, const Geometry& geo, const PrimRows& W, const Rows& g, const Points& wh, const Points& c0,
                            const Rows& rows, const Rows& Wb, const Rows& gb, const Points& whb, const Points& cb, const Points& xb) {
        const Gas gas = gas_;
        Kokkos::parallel_for(
            "hydro::moving_pass_gather", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), KOKKOS_LAMBDA(int i) {
                const State Wi = row_state(W, i);
                const Gradient<D> gi = gradient_row(g, i);
                State Wbar{};
                Gradient<D> gbar;
                for (int v = 0; v < V; ++v) gbar[v] = Vec{};
                Vec whbar{}, cbar{}, xbar{};
                for (int k = geo.start(i); k < geo.start(i + 1); ++k) {
                    const int f = geo.face_at(k);
                    const Vec a = geo.area_of(f);
                    if (dot(a, a) == 0) continue;
                    const bool owned = geo.owned_at(k, f, i);
                    const int o = owned ? i : geo.owner_of(f), j = owned ? geo.neighbour_of(f) : i;
                    const State Sbar = make_state<D>([&](int v) { return rows(f, (owned ? 0 : V) + v); });
                    const Vec r = pass_offset(second, dt, geo, f, i, wh, c0);
                    bool fallback;
                    FaceFlux<1, true>::side_value(Wi, gi, r, second ? 2 * dt : 0.0, gas, fallback);
                    FaceFlux<1, true>::side_adjoint(Wi, gi, r, second ? 2 * dt : 0.0, gas, Sbar, Wbar, gbar, fallback);
                    Vec rbar{};
                    if (!fallback)
                        for (int v = 0; v < V; ++v) rbar = rbar + gi[v] * Sbar[v];
                    cbar = cbar - rbar;
                    if (second) whbar = whbar + rbar * dt;
                    if (!owned) xbar = xbar - rbar;                                                     // own offset: + x_o - x_i
                    if (owned) xbar = xbar + make_vec<D>([&](int c) { return rows(f, 2 * V + D + c); });  // the neighbour's
                    // The face velocity's adjoint, this side's share: w-hat and the separation d = x_j - x_o.
                    const Vec d = geo.separation(o, j), xf = geo.face_centroid_of(f);
                    const Vec wfbar = make_vec<D>([&](int c) { return rows(f, 2 * V + c); });
                    Vec db{}, xfb{}, wLb{}, wRb{};
                    FaceVelocity::adjoint(d, xf, geo.row(wh, o), geo.row(wh, j), wfbar, db, xfb, wLb, wRb);
                    whbar = whbar + (owned ? wLb : wRb);
                    xbar = owned ? xbar - db : xbar + db;
                }
                for (int v = 0; v < V; ++v) {
                    Wb(i, v) += Wbar[v];
                    for (int c = 0; c < D; ++c) gb(i, D * v + c) += gbar[v][c];
                }
                for (int c = 0; c < D; ++c) {
                    whb(i, c) += whbar[c];
                    cb(i, c) += cbar[c];
                    xb(i, c) += xbar[c];
                }
            });
    }

    // Reverse mode at order 1: qbar, xbar of (Q, x) after the step on entry, before on return. Q and the positions are the state at the
    // start of the step (the caller restores them and updates the mesh). It steps to t^n+1 for the second pass's mesh, then restores
    // Q, x and the mesh of t^n for the first pass, the fit and the primitives.
    void step_moving_adjoint_1(double dt, const Cells& qbar, const Points& xbar, const Derivative& geometry, bool steps_fixed) {
        refresh();
        const Gas gas = gas_;
        const Limiter limiter = limiter_;
        const exact_predicates::BoxN<D> box = mesh_->sides();
        const Kokkos::RangePolicy<ExecSpace> cells(exec_, 0, n_);
        Cells q_saved = work<Cells>("hydro_q_saved", n_);
        Points x_saved = work<Points>("hydro_x_saved", n_);
        Kokkos::deep_copy(exec_, q_saved, q_);
        Kokkos::deep_copy(exec_, x_saved, x_);
        PrimRows W = work<PrimRows>("hydro_moving_W", n_);
        Rows g = work<Rows>("hydro_moving_g", n_, V * D);
        Points wh = work<Points>("hydro_w_hat", n_), c0 = work<Points>("hydro_c0", n_);
        moving_primitives(W, Rows(), Cells(), Scalars());
        const double unit = box.step / dt;
        {
            const Geometry geo = geometry_;
            const auto x = x_;
            Kokkos::parallel_for(
                "hydro::moving_cells_1", cells, KOKKOS_LAMBDA(int i) {
                    const State Wi = row_state(W, i);
                    const Regularised r(geo, i, Wi, W(i, V));
                    for (int a = 0; a < D; ++a) {
                        const int k0 = exact_predicates::snap(x(i, a), box, a), k1 = exact_predicates::snap(x(i, a) + dt * r.w[a], box, a);
                        wh(i, a) = static_cast<double>(exact_predicates::delta(k0, k1, box.period[a])) * unit;
                        c0(i, a) = geo.centroid(i, a);
                    }
                    LimitPattern pattern;
                    Gradient<D> g0;
                    const Gradient<D> gi = limited_gradient(Wi, CellEntries(geo, Prims(W), Rows(), i), limiter, pattern, g0);
                    for (int v = 0; v < V; ++v)
                        for (int a = 0; a < D; ++a) g(i, D * v + a) = gi[v][a];
                });
        }
        Rows Wb = work<Rows>("hydro_Wbar", n_, V, true), gb = work<Rows>("hydro_gbar", n_, V * D, true);  // summed into: zeroed
        Points whb = work<Points>("hydro_whbar", n_, 1, true), cb = work<Points>("hydro_cbar1", n_, 1, true), xb = work<Points>("hydro_xbar_next", n_);
        Kokkos::deep_copy(exec_, xb, xbar);
        // Pass 2 on t^n+1.
        step(dt);
        refresh();
        {
            Rows rows = work<Rows>("hydro_pass_rows", faces_, pass_row);
            Points Ab = work<Points>("hydro_Abar2", faces_), xfb = work<Points>("hydro_xfbar2", faces_), cb2 = work<Points>("hydro_cbar2", n_, 1, true);
            Scalars Vb2 = work<Scalars>("hydro_Vbar2", n_, 1, true);
            moving_pass_adjoint(true, dt, geometry_, W, g, wh, c0, qbar, rows, Ab, xfb);
            moving_pass_gather(true, dt, geometry_, W, g, wh, c0, rows, Wb, gb, whb, cb, xb);
            geometry.adjoint(Vb2, cb2, Ab, xfb, xb);  // x^n+1's bar, complete
        }
        // x^n+1 = x^n + dt w-hat: x^n's bar and w-hat's.
        Kokkos::parallel_for(
            "hydro::moving_drift_adjoint", cells, KOKKOS_LAMBDA(int i) {
                for (int a = 0; a < D; ++a) whb(i, a) += dt * xb(i, a);
            });
        // Back to t^n.
        Kokkos::deep_copy(exec_, q_, q_saved);
        Kokkos::deep_copy(exec_, x_, x_saved);
        geometry_ = {};
        mesh_->update(x_);
        mesh_->update_cells();
        refresh();
        prims_ready_ = false;
        swept_ = false;
        magnitudes_valid_ = false;
        const Geometry geo = geometry_;
        {
            Rows rows = work<Rows>("hydro_pass_rows", faces_, pass_row);
            Points Ab = work<Points>("hydro_Abar", faces_), xfb = work<Points>("hydro_xfbar", faces_);
            moving_pass_adjoint(false, dt, geo, W, g, wh, c0, qbar, rows, Ab, xfb);
            moving_pass_gather(false, dt, geo, W, g, wh, c0, rows, Wb, gb, whb, cb, xb);
            const int entries = [&] {
                int total = 0;
                Kokkos::deep_copy(total, Kokkos::subview(geo.start, n_));
                return total;
            }();
            constexpr int E = V + 2 * D + 1;  // per list entry: the neighbour's W bar, between bar, offset bar, weight bar
            Rows entry_bar = work<Rows>("hydro_entry_bar", entries, E);
            Points angle_Abar = work<Points>("hydro_angle_Abar", n_), angle_dbar = work<Points>("hydro_angle_dbar", n_);
            Kokkos::View<int*, Memory> fmax = work<Kokkos::View<int*, Memory>>("hydro_fmax", n_);
            Scalars Vb = work<Scalars>("hydro_Vbar", n_);
            // Per cell: w-hat's bar through the point velocity; g's bar through the fit (entries' bars); the own W bar.
            Kokkos::parallel_for(
                "hydro::moving_cells_adjoint_1", cells, KOKKOS_LAMBDA(int i) {
                    const State Wi = row_state(W, i);
                    State Wbar = make_state<D>([&](int v) { return Wb(i, v); });
                    Vec cbar = geo.row(cb, i);
                    const Regularised r(geo, i, Wi, W(i, V));
                    double csbar = 0;
                    Vec Aab{}, adb{};
                    r.adjoint(geo, i, steps_fixed ? Vec{} : geo.row(whb, i), Wbar, csbar, cbar, Aab, adb);
                    fmax(i) = r.fmax;
                    for (int a = 0; a < D; ++a) {
                        angle_Abar(i, a) = Aab[a];
                        angle_dbar(i, a) = adb[a];
                    }
                    if (W(i, V) > 0) {
                        const double c2bar = csbar / (2 * W(i, V));
                        Wbar[D + 1] += c2bar * gas.gamma / Wi[0];
                        Wbar[0] -= c2bar * gas.gamma * Wi[D + 1] / (Wi[0] * Wi[0]);
                    }
                    const CellEntries e(geo, Prims(W), Rows(), i);
                    const int first = geo.start(i);
                    const GradientAdjoint out = moving_gradient_adjoint(Wi, e, limiter, gradient_row(gb, i), [&](int k, Vec bb, double wb, Vec ob) {
                        for (int a = 0; a < D; ++a) {
                            entry_bar(first + k, V + a) = bb[a];
                            entry_bar(first + k, V + D + a) = ob[a];
                        }
                        entry_bar(first + k, V + 2 * D) = wb;
                    });
                    for (int k = 0; k < e.count(); ++k) {
                        const State nb = neighbour_bar(e, k, out);
                        for (int v = 0; v < V; ++v) entry_bar(first + k, v) = nb[v];
                    }
                    for (int v = 0; v < V; ++v) Wb(i, v) = Wbar[v] + out.self[v];
                    for (int a = 0; a < D; ++a) cb(i, a) = cbar[a];
                });
            // Per cell: the neighbours' W bars, the primitives' adjoint (Q bar, V bar); the c and x bars of the fit's offsets.
            const auto q = q_;
            Kokkos::parallel_for(
                "hydro::moving_gather_adjoint_1", cells, KOKKOS_LAMBDA(int i) {
                    State Wbar = make_state<D>([&](int v) { return Wb(i, v); });
                    Vec cbar = geo.row(cb, i), xsum = geo.row(angle_dbar, i) * -1.0;
                    for (int k = geo.start(i); k < geo.start(i + 1); ++k) {
                        const int f = geo.face_at(k), j = geo.other_at(k, f, i), o = geo.owner_of(f);
                        const int kj = entry_of(geo, j, f);
                        for (int v = 0; v < V; ++v) Wbar[v] += entry_bar(kj, v);
                        const Vec bb_own = make_vec<D>([&](int a) { return entry_bar(k, V + a); });
                        const Vec bb_j = make_vec<D>([&](int a) { return entry_bar(kj, V + a); });
                        const Vec ob_own = make_vec<D>([&](int a) { return entry_bar(k, V + D + a); });
                        const Vec ob_j = make_vec<D>([&](int a) { return entry_bar(kj, V + D + a); });
                        // between (own) = x_j - x_i + c_j - c_i; j's between = x_i - x_j + c_i - c_j
                        cbar = cbar - bb_own + bb_j;
                        xsum = xsum - bb_own + bb_j;
                        // offset (own) = x_f rel o (+ x_o - x_i if i is not o) - c_i
                        cbar = cbar - ob_own;
                        if (o != i) xsum = xsum - ob_own;
                        if (o == i) xsum = xsum + ob_j;  // j's offset carries x_o - x_j
                        if (fmax(j) == f) xsum = xsum + geo.row(angle_dbar, j);
                    }
                    const State U = geo.per_volume(q, i);
                    const State Ubar = primitive_adjoint(U, Wbar, gas);
                    const double inv = 1.0 / geo.volume(i);
                    double vb = 0;
                    for (int v = 0; v < V; ++v) {
                        qbar(i, v) += Ubar[v] * inv;
                        vb -= Ubar[v] * U[v] * inv;
                    }
                    Vb(i) = vb;
                    for (int a = 0; a < D; ++a) {
                        cb(i, a) = cbar[a];
                        xb(i, a) += xsum[a];
                    }
                });
            Kokkos::parallel_for(
                "hydro::moving_faces_bar_1", Kokkos::RangePolicy<ExecSpace>(exec_, 0, faces_), KOKKOS_LAMBDA(int f) {
                    const int o = geo.owner_of(f), j = geo.neighbour_of(f);
                    const int ko = entry_of(geo, o, f), kj = entry_of(geo, j, f);
                    const Vec a = geo.area_of(f);
                    const double mag = Kokkos::sqrt(dot(a, a));
                    const double wbar = mag > 0 ? (entry_bar(ko, V + 2 * D) + entry_bar(kj, V + 2 * D)) / mag : 0.0;
                    for (int c = 0; c < D; ++c) {
                        Ab(f, c) += a[c] * wbar + (fmax(o) == f ? angle_Abar(o, c) : 0.0) +
                                   (fmax(j) == f ? angle_Abar(j, c) : 0.0);
                        xfb(f, c) += entry_bar(ko, V + D + c) + entry_bar(kj, V + D + c);
                    }
                });
            geometry.adjoint(Vb, cb, Ab, xfb, xb);
        }
        Kokkos::deep_copy(exec_, xbar, xb);
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
    bool tangent_swept_ = false;  // time_step_tangent() also swept the tangents (dW, dg) for the next step_tangent()
    std::map<std::string, Kokkos::View<char*, Memory>> work_;  // the moving derivatives' work arrays by label (work())
    mutable bool prims_ready_ = false;  // the last update wrote prims_ for the current Q and mesh
    StepRule rule_ = StepRule::arepo;
    // |area| per face kept for the sweep (LUNA_NO_MAGNITUDES turns it off, for measurements).
    using Magnitudes = Kokkos::View<double*, Memory, Unmanaged>;
    Magnitudes magnitudes_;
    bool magnitudes_valid_ = false;
    bool cache_magnitudes_ = std::getenv("LUNA_NO_MAGNITUDES") == nullptr;
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
    // bars of W, the fit's b, each variable's bound and the face it came through, of each face's two face states; the stored fit.
    Rows d_prims_, d_gradients_, d_fluxes_;
    Rows bar_w_, bar_b_, bar_bound_, bar_face_, bar_fit_;
    Kokkos::View<int**, Kokkos::LayoutRight, Memory> fit_flags_;
    Kokkos::View<int* [V], Kokkos::LayoutRight, Memory> bound_face_;
};

}  // namespace hydro
