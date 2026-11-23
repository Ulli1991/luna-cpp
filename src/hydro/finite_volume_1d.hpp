#pragma once

#include "finite_volume.hpp"

// Solver stage (D = 1): the driver calls time_step() then step() (or step_and_time_step()); one fused kernel per step on voronoi::Mesh<1>.
// Same scheme as FiniteVolume<D> in finite_volume.hpp; an inflow wall supplies cells split from the end cell. Design: docs/design.md.
namespace hydro {

enum class Wall { reflective, outflow, inflow };
// Used by: Stencil (ghost cells), set_inflow_spacing().
struct Walls {
    Wall left = Wall::outflow, right = Wall::outflow;
    State<1> left_inflow{}, right_inflow{};  // primitives (rho, v, P) of an inflow
};

// Used by: time_step(), run_fused(). The global time step's rule: arepo is AREPO next generation's, luna the RAMSES rule (docs/design.md).
enum class TimeStep { arepo, luna };

// The defaults of a 1D run: static, LUNA's rule at 0.8 (reproduces LUNA); moving, AREPO's rule at 0.4 (not 3D's 0.3).
inline constexpr double courant_static_1d = 0.8, courant_moving_1d = 0.4;

// What makes a moving 1D cell refine or derefine (FiniteVolume<1>::set_adapt). mass, the default on a moving line: a cell splits when
// its mass exceeds refine_mass times its target mass (its mass at the start) and goes when it falls below derefine_mass times it,
// as AREPO; length thresholds, if set, act as a safeguard on top. length: only the length thresholds (set_refine, set_derefine). off:
// neither. research_notes/derefine_1d_2026-11-20.md.
enum class Adapt { off, mass, length };
inline constexpr double refine_mass_default = 2.0, derefine_mass_default = 0.5;
// Adapt::mass's size bounds, in initial spacings h: a cell narrower than derefine_length_default goes whatever its mass (the safeguard
// that fixes #30, where the cells keep their mass); mass splits only cells wider than split_min_default, mass removes only cells
// narrower than merge_max_default (a light cell next to dense gas would otherwise split every step, and open walls would empty).
inline constexpr double derefine_length_default = 0.03, split_min_default = 0.12, merge_max_default = 2.0;

template <class ExecSpace, class Riemann>
class FiniteVolume<1, ExecSpace, Riemann, Precision<>> {
public:
    static constexpr int D = 1, V = 3;
    static constexpr double cell_shaping_speed = 0.5;
    using Memory = typename ExecSpace::memory_space;
    using Mesh = voronoi::Mesh<1, ExecSpace>;
    using State = hydro::State<1>;
    using Cells = Kokkos::View<double* [V], Kokkos::LayoutRight, Memory>;
    using Points = Kokkos::View<double* [1], Kokkos::LayoutRight, Memory>;
    using Line = typename Mesh::Line;
    using i64 = exact_predicates::i64;
    using Keys = Kokkos::View<sorted_mesh::Key*, Memory>;

    // Call order of one step (driver):
    //  1. time_step(courant, rule) -> dt
    //  2. step(dt): advance() -> run() Kernel<..., false>, swap Q, mesh_->order(), supply_cells() if due()
    //  or: step_and_time_step(dt, courant): run_fused() does 2. and 1. of the next step in one launch
    // mesh: built (update()); walls exactly when the mesh has them. Moving: this object owns the positions.
    FiniteVolume(const ExecSpace& exec, Mesh& mesh, const Gas& gas, int order, Limiter limiter = Limiter::face_centroid,
                 Motion motion = Motion::static_mesh, const Walls& walls = {}, bool regularise = true)
        : exec_(exec), mesh_(&mesh), gas_(gas), order_(order), limiter_(limiter), motion_(motion), walls_(walls),
          regularise_(regularise), n_(mesh.particles()), capacity_(n_) {
        if (order != 0 && order != 1) throw std::invalid_argument("hydro::FiniteVolume<1>: order 0 or 1");
        set_inflow_spacing(mesh.box() / n_);
        q_ = Cells("hydro_Q", n_);
        next_ = Cells(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("hydro_Q_next")), n_);
        if (motion == Motion::moving) {
            x_ = Points(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("hydro_x")), n_);
            Kokkos::deep_copy(exec_, x_, mesh.positions());
            adapt_ = Adapt::mass;
            const double h = mesh.box() / n_;
            set_mass_bounds(split_min_default * h, merge_max_default * h, derefine_length_default * h);
        }
    }

    // Both views may be longer than cells(): cells can be added.
    Cells state() const { return q_; }
    Points positions() const { return x_; }
    int cells() const { return n_; }
    long long cells_added() const { return added_; }
    // delta of the head of this file; 0: a wall supplies no cells.
    double inflow_spacing() const { return spacing_; }
    void set_inflow_spacing(double spacing) {
        const auto open = [&](Wall wall) { return motion_ == Motion::moving && mesh_->line().periodic == false && wall != Wall::reflective && spacing > 0; };
        supply_left_ = open(walls_.left);
        supply_right_ = open(walls_.right);
        spacing_ = spacing;
    }
    // Refinement and derefinement (moving only; luna-cpp #30). After each step a cell may lose its generator (derefine(): its two
    // neighbours share its mass, momentum and energy) or get a new one (refine()). Which cells: set_adapt (default Adapt::mass on a
    // moving line, off on a static one). Mass: below derefine_mass or above refine_mass times the cell's target mass (its mass at the
    // first step; a split's children inherit the parent's, cells from an inflow wall get their mass at birth; cells with target 0 never
    // adapt by mass). Mass's size bounds (set_mass_bounds; defaults 0.12 h, 2 h, 0.03 h, h the initial spacing): it splits only
    // cells wider than split_min and removes only cells narrower than merge_max; and any cell narrower than the safeguard goes
    // whatever its mass (set_derefine replaces the safeguard). Length (Adapt::length): narrower than min_volume, wider than max_volume
    // (set_derefine, set_refine; 0: none). The split takes minmod-limited linear profiles (default) or the old cells' means
    // (set_split(false)). Factors and thresholds need hysteresis: refine_mass >= 4 derefine_mass, max_volume > 4 min_volume.
    void set_adapt(Adapt mode) {
        adapt_ = motion_ == Motion::moving ? mode : Adapt::off;
        targets_ready_ = false;
    }
    Adapt adapt() const { return adapt_; }
    void set_adapt_mass(double refine_mass, double derefine_mass) {
        refine_mass_ = refine_mass;
        deref_mass_ = derefine_mass;
        check_hysteresis();
    }
    void set_mass_bounds(double split_min, double merge_max, double safeguard) {
        split_min_ = split_min;
        merge_max_ = merge_max;
        safeguard_ = safeguard;
    }
    void set_derefine(double min_volume) {
        deref_volume_ = motion_ == Motion::moving ? min_volume : 0.0;
        check_hysteresis();
    }
    void set_refine(double max_volume) {
        refine_volume_ = motion_ == Motion::moving ? max_volume : 0.0;
        check_hysteresis();
    }
    void set_split(bool linear) { refine_linear_ = linear; }
    // Steering (moving only; off by default): see Stencil::steering. ratio e.g. 0.25.
    void set_steer(double ratio) { steer_ = motion_ == Motion::moving ? ratio : 0.0; }
    double derefine_volume() const { return deref_volume_; }
    double refine_volume() const { return refine_volume_; }
    long long cells_removed() const { return removed_; }
    long long cells_refined() const { return refined_; }
    int order() const { return order_; }
    Motion motion() const { return motion_; }
    const Gas& gas() const { return gas_; }
    const MeshStats& mesh_stats() const { return mesh_stats_; }
    long long fallbacks() const { return -1; }  // not counted (as the per-cell loop of finite_volume.hpp)

    // Read-only view of the line (mesh, Q, walls) for the kernels; stencil() rebuilds it per call.
    // Used by: time_step(), supply_cells(), Kernel.
    struct Stencil {
        Line line;
        Cells q;
        Gas gas;
        Walls walls;
        Limiter limiter;
        bool regularise, supply_left, supply_right;
        double spacing;
        double steer = 0;  // set_steer(): a generator moves away from a neighbour cell narrower than steer times its own; 0: off

        // Used by: wants_cell(). Whether end slot s has room for a cell from its wall, d grid steps from it.
        KOKKOS_INLINE_FUNCTION bool room_at(double volume, i64 d) const {
            return volume > 2 * spacing && static_cast<double>(d) * line.step >= 1.5 * spacing;
        }
        // 0..2 cells an end slot asks its wall for. Used by: time_step(), supply_cells(), Kernel::operator().
        KOKKOS_INLINE_FUNCTION int wants_cell(int s) const {
            int wants = 0;
            if (s == 0 && supply_left) wants += room_at(line.volume(0), static_cast<i64>(line.key(0)));
            if (s == line.n - 1 && supply_right) wants += room_at(line.volume(s), line.period - static_cast<i64>(line.key(s)));
            return wants;
        }

        // Used by: load(), load_inner(). A cell of a stencil: its primitives and centroid relative to the generator; real = false for a ghost behind a wall.
        struct Cell {
            State W{};
            double c = 0, V = 0;
            bool real = true;
        };

        // Used by: load(), time_step(). Distance between the generators of slots s and s + 1.
        KOKKOS_INLINE_FUNCTION double length(int s) const { return static_cast<double>(line.gap(s)) * line.step; }
        // W of slot s. Used by: load(), time_step(), supply_cells().
        KOKKOS_INLINE_FUNCTION State primitive_of(int s) const {
            const int i = line.order(s);
            const double inv = 1.0 / line.volume(s);
            return primitive(State{q(i, 0) * inv, q(i, 1) * inv, q(i, 2) * inv}, gas);
        }
        KOKKOS_INLINE_FUNCTION static State mirror(State W) {
            W[1] = -W[1];
            return W;
        }
        // The ghost cell behind wall `left` of the end slot whose state is W. Used by: load(), ghost_face().
        KOKKOS_INLINE_FUNCTION State ghost(const State& W, bool left) const {
            const Wall wall = left ? walls.left : walls.right;
            if (wall == Wall::reflective) return mirror(W);
            if (wall == Wall::inflow) return left ? walls.left_inflow : walls.right_inflow;
            return W;
        }
        // The ghost's face state at the wall for end cell W and face state Wf: mirror of Wf if reflective, else the ghost's own W. Used by: Kernel::body().
        KOKKOS_INLINE_FUNCTION State ghost_face(const State& Wf, const State& W, bool left) const {
            const Wall wall = left ? walls.left : walls.right;
            if (wall == Wall::reflective) return mirror(Wf);
            return ghost(W, left);
        }

        // Cells s - R .. s + R into cell[] (a ghost behind a wall) and their generator distances into gap[] (twice the wall's to a mirror).
        // Used by: time_step() (moving), Kernel::body<false>().
        template <int R>
        KOKKOS_INLINE_FUNCTION void load(int s, Cell (&cell)[2 * R + 1], double (&gap)[2 * R]) const {
            constexpr int M = 2 * R + 1;
            int slot[M];
            for (int k = 0; k < M; ++k) slot[k] = line.slot(s, k - R);
            for (int k = 0; k < M; ++k) {
                if (slot[k] >= 0) {
                    cell[k].W = primitive_of(slot[k]);
                    cell[k].c = line.centroid(slot[k]);
                    cell[k].V = line.volume(slot[k]);
                }
            }
            for (int k = 0; k < M; ++k) {
                if (slot[k] >= 0) continue;
                cell[k].real = false;
                const int end = k < R ? k + 1 : k - 1;
                if (slot[end] < 0) continue;
                cell[k].W = ghost(cell[end].W, k < R);
                cell[k].c = -cell[end].c;
            }
            for (int k = 0; k < M - 1; ++k) {
                const int a = slot[k], b = slot[k + 1];
                gap[k] = a >= 0 ? length(a) : b >= 0 ? length(-1) : 0.0;
            }
        }

        // load<R> far from the ends of the arrays (R + 1 <= s <= n - R - 2): the same bits from whole-step gaps read once.
        // Used by: time_step() (moving), Kernel::body<true>().
        template <int R>
        KOKKOS_INLINE_FUNCTION void load_inner(int s, Cell (&cell)[2 * R + 1], double (&gap)[2 * R]) const {
            constexpr int M = 2 * R + 1;
            double steps[M + 1];  // steps[k]: grid steps from slot s - R - 1 + k to the next
            i64 previous = static_cast<i64>(line.key(s - R - 1));
            for (int k = 0; k <= M; ++k) {
                const i64 here = static_cast<i64>(line.key(s - R + k));
                const i64 d = here - previous;
                steps[k] = static_cast<double>(d < 0 ? d + line.period : d);  // a periodic line's seam
                previous = here;
            }
            for (int k = 0; k < M; ++k) {
                const int i = line.order(s - R + k);
                cell[k].V = (steps[k] + steps[k + 1]) * (0.5 * line.step);
                const double inv = 1.0 / cell[k].V;
                cell[k].W = primitive(State{q(i, 0) * inv, q(i, 1) * inv, q(i, 2) * inv}, gas);
                cell[k].c = (steps[k + 1] - steps[k]) * (0.25 * line.step);
            }
            for (int k = 0; k < M - 1; ++k) gap[k] = steps[k + 1] * line.step;
        }

        // The limited gradient of variable v of the cell at *cell, its neighbours at distances gapL and gapR.
        // Used by: time_step(), Kernel::body().
        KOKKOS_INLINE_FUNCTION double gradient(const Cell* cell, double gapL, double gapR, int v) const {
            const Cell& me = *cell;
            const double dL = -gapL + cell[-1].c - me.c, dR = gapR + cell[1].c - me.c;
            const double Wi = me.W[v], left = cell[-1].W[v], right = cell[1].W[v];
            if (limiter == Limiter::minmod || limiter == Limiter::van_leer) {
                const double sL = (left - Wi) / dL, sR = (right - Wi) / dR;
                if (!(sL * sR > 0)) return 0.0;
                if (limiter == Limiter::van_leer) return 2 * sL * sR / (sL + sR);
                return Kokkos::fabs(sL) < Kokkos::fabs(sR) ? sL : sR;
            }
            double g = ((right - Wi) - (left - Wi)) / (dR - dL);
            if (limiter == Limiter::none) return g;
            const double r[2] = {-0.5 * gapL - me.c, 0.5 * gapR - me.c};
            const double lo = Kokkos::fmin(left, right), hi = Kokkos::fmax(left, right);
            for (int f = 0; f < 2; ++f) {
                const double dp = g * r[f];
                if (dp > 0 && Wi + dp > hi)
                    g = g * (hi > Wi ? (hi - Wi) / dp : 0.0);
                else if (dp < 0 && Wi + dp < lo)
                    g = g * (lo < Wi ? (lo - Wi) / dp : 0.0);
            }
            return g;
        }

        // The point velocity's difference from the fluid's (AREPO-ng's regularisation) for primitives W, density gradient g, centroid c, volume.
        // Used by: time_step(), Kernel::body().
        KOKKOS_INLINE_FUNCTION double drift(const State& W, double g, double c, double volume) const {
            const double r = 0.5 * volume;
            if (!regularise || Kokkos::fabs(c) <= 0.5 * r) return 0.0;
            double offset = c;
            const double dgrad = Kokkos::fabs(g);
            if (dgrad > 0) {
                const double tmp = 3 * r + W[0] / dgrad;
                const double x = (tmp - Kokkos::sqrt(tmp * tmp - 8 * r * r)) / 4;
                if (x < 0.25 * r) offset += x / dgrad * g;
            }
            const double d = Kokkos::fabs(offset);
            if (d <= 0.75 * r) return 0.0;
            const double fraction = d > r ? cell_shaping_speed : cell_shaping_speed * (d - 0.75 * r) / (0.25 * r);
            return offset * (fraction * sound_speed(W, gas) / d);
        }
        // Steering after plume's face-angle trigger (Doug Rennehan's code): a cell that flattens symmetrically keeps its centroid on
        // its generator, so drift()'s offset trigger never fires. In 1D a cell's length is set by its neighbours' generators alone, so
        // moving its own generator cannot widen it: the neighbours move instead. A generator next to a cell narrower than steer times
        // its own cell moves away from it, at cell_shaping_speed times the sound speed, scaled by how far below that ratio the neighbour
        // is (both sides add). Off unless set_steer(ratio > 0). Used by: point_velocity().
        KOKKOS_INLINE_FUNCTION double steering(const Cell* cell, double volume) const {
            double v = 0;
            for (int side = -1; side <= 1; side += 2) {
                const double ratio = cell[side].V / volume;
                if (cell[side].V > 0 && ratio < steer) v -= side * (1 - ratio / steer);
            }
            return v * cell_shaping_speed * sound_speed(cell->W, gas);
        }
        // The point velocity relative to the cell's mean velocity: drift(), plus, for a generator beyond 0.75 r of its centroid (where the drift starts),
        // up to the fluid's velocity at the generator, linear between the centroids of this cell and its neighbour there (luna-cpp #25).
        // Used by: time_step(), Kernel::body().
        KOKKOS_INLINE_FUNCTION double point_velocity(const Cell* cell, double gapL, double gapR, double g_rho, double volume) const {
            const Cell& me = *cell;
            const double r = 0.5 * volume, d = Kokkos::fabs(me.c), weight = Kokkos::fmin(1.0, (d - 0.75 * r) / (0.25 * r));
            double drifting = drift(me.W, g_rho, me.c, volume);
            if (steer > 0) drifting += steering(cell, volume);
            if (!(weight > 0)) return drifting;
            const int side = me.c < 0 ? 1 : -1;
            const double between = side > 0 ? gapR + cell[1].c - me.c : gapL + me.c - cell[-1].c;
            return drifting + weight * (cell[side].W[1] - me.W[1]) * (d / between);
        }
    };

    // The global time step (see TimeStep). |v - w|: point_velocity().
    // Used by: the driver before step(); run_fused() when cells were added or the order changed.
    double time_step(double courant, TimeStep rule = TimeStep::arepo, Timings* timings = nullptr) {
        timer_.reset();
        const Stencil st = stencil();
        const bool moving = motion_ == Motion::moving, luna = rule == TimeStep::luna;
        const bool second_order = order_ == 1;
        double dt = 0;
        const auto rule_at = KOKKOS_LAMBDA(int s, double& smallest) {
                typename Stencil::Cell cell[3];
                double gap[2] = {0, 0};
                if (moving && s >= 2 && s <= st.line.n - 3)
                    st.template load_inner<1>(s, cell, gap);
                else if (moving)
                    st.template load<1>(s, cell, gap);
                else
                    cell[1].W = st.primitive_of(s);
                const State& Wi = cell[1].W;
                const double ci = sound_speed(Wi, st.gas);
                const double volume = st.line.volume(s);
                double vi = Kokkos::fabs(Wi[1]);
                if (moving) {
                    const double g = second_order ? st.gradient(cell + 1, gap[0], gap[1], 0) : 0.0;
                    vi = Kokkos::fabs(st.point_velocity(cell + 1, gap[0], gap[1], g, volume));
                }
                if (luna) {
                    smallest = Kokkos::fmin(smallest, courant * volume / (vi + ci));
                    return;
                }
                const double r = 0.5 * volume;
                double step = r / (ci + vi);
                for (int side = -1; side <= 1; side += 2) {
                    // Behind a wall the ghost, at the end generator's mirror (cold gas into a wall, Noh, luna-cpp #15, #27).
                    const int t = st.line.slot(s, side);
                    const State Wj = t >= 0 ? st.primitive_of(t) : st.ghost(Wi, side < 0);
                    const double dist = st.length(side < 0 ? s - 1 : s);
                    const double vsig = ci + sound_speed(Wj, st.gas) - (Wj[1] - Wi[1]) * side;
                    if (vsig > 0) step = Kokkos::fmin(step, (dist + r) / vsig);
                }
                smallest = Kokkos::fmin(smallest, courant * step);
        };
        const Kokkos::RangePolicy<ExecSpace> slots(exec_, 0, n_);
        if (watching_) {
            Kokkos::parallel_reduce(
                "hydro1::time_step_ends", slots,
                KOKKOS_LAMBDA(int s, double& smallest, int& wants) {
                    rule_at(s, smallest);
                    wants += st.wants_cell(s);
                },
                Kokkos::Min<double>(dt), Kokkos::Sum<int>(wants_));
            looked_ = true;
        } else {
            Kokkos::parallel_reduce("hydro1::time_step", slots, rule_at, Kokkos::Min<double>(dt));
        }
        phase_end(timings, &Timings::sweep);
        return dt;
    }

    // One step of length dt (moving: the points then drift and the mesh's order is updated). Used by: the driver after time_step().
    void step(double dt, Timings* timings = nullptr) {
        timer_.reset();
        ensure_targets();
        reserve_next_points();
        if (order_ == 1)
            advance<1>(dt);
        else
            advance<0>(dt);
        phase_end(timings, &Timings::fluxes);
        std::swap(q_, next_);
        if (motion_ == Motion::moving) {
            std::swap(x_, x_next_);
            const voronoi::Report report = mesh_->order(live_points());
            mesh_stats_.updates += 1;
            mesh_stats_.rebuilds += report.rebuilt;
            if (due()) supply_cells();
            derefine();
            refine();
            looked_ = false;
            phase_end(timings, &Timings::mesh);
        }
    }

    // One step of length dt in one launch; returns the next time step by LUNA's rule at this Courant number.
    // Used by: the driver instead of step() + time_step().
    // Static: time_step(courant, luna) after step(dt), to rounding.
    double step_and_time_step(double dt, double courant) {
        ensure_targets();
        reserve_next_points();
        const bool moving = motion_ == Motion::moving;
        if (order_ == 1) return moving ? run_fused<1, true>(dt, courant) : run_fused<1, false>(dt, courant);
        return moving ? run_fused<0, true>(dt, courant) : run_fused<0, false>(dt, courant);
    }

    // Adapt::mass: the target masses, each cell's mass before its first step (once; set_adapt starts over). Used by: step(),
    // step_and_time_step(). Public because CUDA lambdas cannot be in private members.
    void ensure_targets() {
        if (!mass_on() || targets_ready_) return;
        if (target_.extent(0) < static_cast<std::size_t>(capacity_))
            target_ = Kokkos::View<double*, Memory>(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("hydro_target")), capacity_);
        const auto target = target_;
        const auto q = q_;
        Kokkos::parallel_for("hydro1::targets", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), KOKKOS_LAMBDA(int i) { target(i) = q(i, 0); });
        targets_ready_ = true;
    }

    // Adds the cells the walls have room for; runs only when one is due. Used by: step(), run_fused(). Public because CUDA lambdas cannot be in private members.
    bool supply_cells() {
        const Stencil st = stencil();
        const auto ends = ends_;
        const bool left = supply_left_, right = supply_right_;
        Kokkos::parallel_for(
            "hydro1::ends", Kokkos::RangePolicy<ExecSpace>(exec_, 0, 1), KOKKOS_LAMBDA(int) {
                const int last = st.line.n - 1;
                ends(0) = static_cast<double>(st.line.key(0)) * st.line.step;
                ends(1) = st.wants_cell(0);
                ends(2) = static_cast<double>(st.line.period - static_cast<i64>(st.line.key(last))) * st.line.step;
                ends(3) = st.wants_cell(last);
                ends(4) = (left && st.primitive_of(0)[1] > 0) || (right && st.primitive_of(last)[1] < 0);
            });
        Kokkos::deep_copy(exec_, ends_host_, ends);
        watching_ = watching_ || ends_host_(4) > 0;
        int requests = 0;
        const auto points_host = points_host_;
        if (ends_host_(1) > 0) points_host(requests++, 0) = ends_host_(0) - spacing_;
        if (ends_host_(3) > 0) points_host(requests++, 0) = st.line.side() - (ends_host_(2) - spacing_);
        if (requests == 0) return false;
        reserve_cells(n_ + requests);
        Kokkos::deep_copy(exec_, points_, points_host_);
        const auto wanted = Kokkos::subview(points_, std::make_pair(0, requests), Kokkos::ALL);
        const typename Mesh::Changes& change = mesh_->insert(wanted);
        const auto q = q_;
        const auto x = x_;
        const auto id = change.id;
        const auto neighbour = change.neighbour;
        const auto overlap = change.overlap;
        const auto volume = change.volume;
        const bool targets = mass_on() && targets_ready_;
        const auto target = target_;
        const double spacing = spacing_;
        Kokkos::parallel_for(
            "hydro1::split", Kokkos::RangePolicy<ExecSpace>(exec_, 0, change.requested), KOKKOS_LAMBDA(int r) {
                const int i = id(r);
                if (i < 0) return;
                for (int v = 0; v < V; ++v) q(i, v) = 0;
                for (int side = 0; side < 2; ++side) {
                    const int old = neighbour(r, side);
                    if (old < 0) continue;
                    const double fraction = overlap(r, side) / volume(r, side);
                    for (int v = 0; v < V; ++v) {
                        const double part = fraction * q(old, v);
                        q(old, v) -= part;
                        q(i, v) += part;
                    }
                }
                x(i, 0) = wanted(r, 0);
                // A cell from the wall is born small and grows to the set spacing: its target, the mass of a lattice cell at its density.
                if (targets) target(i) = q(i, 0) / (overlap(r, 0) + overlap(r, 1)) * spacing;
            });
        n_ += change.done;
        added_ += change.done;
        return change.done > 0;
    }

    // Removes the generators of cells narrower than deref_volume_ (luna-cpp #30: a slab of tiny mass squeezed to a layer keeps all
    // its generators, and the cells and dt shrink with it). As AREPO's derefinement: the removed cell's Voronoi length goes to its
    // two neighbours (Mesh<1>::remove's overlaps), and so do its conserved totals, in the same proportions; mass, momentum and energy
    // are summed, so conserved to rounding. A neighbour that is a wall gives all to the other side. Mesh<1>::remove defers a cell
    // next to another one going in the same call (the next step takes it) and keeps 3 cells. Returns whether any went.
    // Used by: step(), run_fused(). Public because CUDA lambdas cannot be in private members.
    bool derefine() {
        // Mass: the safeguard unless set_derefine gave a length; length: that length.
        const double limit = !length_on() ? 0.0 : mass_on() && !(deref_volume_ > 0) ? safeguard_ : deref_volume_;
        const double factor = mass_on() && targets_ready_ ? deref_mass_ : 0.0;
        const double merge_max = merge_max_ > 0 ? merge_max_ : 1e300;
        if (!(limit > 0) && !(factor > 0)) return false;
        const Line line = mesh_->line();
        const int n = n_;
        if (deref_ids_.extent(0) < static_cast<std::size_t>(n))
            deref_ids_ = Kokkos::View<int*, Memory>(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("hydro_deref_ids")), capacity_);
        const auto ids = deref_ids_;
        const auto q = q_;
        const auto target = target_;
        int count = 0;
        Kokkos::parallel_scan(
            "hydro1::deref_find", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n),
            KOKKOS_LAMBDA(int s, int& at, bool final) {
                const int i = line.order(s);
                const bool light = factor > 0 && target(i) > 0 && q(i, 0) < factor * target(i) && line.volume(s) < merge_max;
                if (line.volume(s) < limit || light) {
                    if (final) ids(at) = i;
                    ++at;
                }
            },
            count);
        if (count == 0) return false;
        const auto wanted = Kokkos::subview(ids, std::make_pair(0, count));
        const typename Mesh::Changes& change = mesh_->remove(wanted);
        if (change.done == 0) return false;
        const auto x = x_;
        const bool targets = factor > 0;
        const auto id = change.id;
        const auto neighbour = change.neighbour;
        const auto overlap = change.overlap;
        const auto moved = change.moved;
        // One pass per side: within a pass no two removed cells share a neighbour (remove() never takes two adjacent cells), so no
        // atomics, and a cell between two removed ones sums left share then right share, the same bits every run.
        for (int side = 0; side < 2; ++side)
            Kokkos::parallel_for(
                "hydro1::deref_share", Kokkos::RangePolicy<ExecSpace>(exec_, 0, change.requested), KOKKOS_LAMBDA(int r) {
                    const int i = id(r), to = neighbour(r, side);
                    if (i < 0 || to < 0) return;  // deferred, or a wall: overlap 0 on its side
                    const double fraction = overlap(r, side) / (overlap(r, 0) + overlap(r, 1));
                    for (int v = 0; v < V; ++v) q(to, v) += fraction * q(i, v);
                });
        Kokkos::parallel_for(
            "hydro1::deref_move", Kokkos::RangePolicy<ExecSpace>(exec_, 0, change.moves), KOKKOS_LAMBDA(int m) {
                const int from = moved(m, 0), to = moved(m, 1);
                for (int v = 0; v < V; ++v) q(to, v) = q(from, v);
                x(to, 0) = x(from, 0);
                if (targets) target(to) = target(from);  // the neighbours that took the mass keep their own targets
            });
        n_ -= change.done;
        removed_ += change.done;
        return true;
    }

    // Gives every cell above refine_mass times its target mass (Adapt::mass) or wider than refine_volume_ a new generator, in its
    // larger gap (to the farther neighbour generator, or to the mirror behind a wall), at delta = max((far - near) / 2, far / 10) from
    // its generator. The parent then keeps (near + delta) / 2 >= half its length, the new cell gets far / 2 >= half, and the cell
    // across the gap loses delta / 2 <= far / 4: a cell above twice its target mass splits into two near it. (Its centre of mass
    // would not do: in a regular cell it is the generator itself.) The new cell inherits the parent's target mass. Mesh<1>::insert
    // reports the length the new cell takes from each of its two old neighbours, the parent and the cell across the gap. The new cell
    // gets what lies on those lengths and the old cells keep the rest, so the totals are kept by construction:
    //  - linear (default): the old cell's minmod-limited linear profile of each conserved density integrated over the piece (slope 0
    //    next to a wall); a uniform state has slope 0, so it stays uniform;
    //  - constant: the fraction overlap / volume of the old cell's totals (exact for a uniform state too).
    // Three passes (pieces, then the old cells left of new ones, then those right of them): no old cell is touched twice in a pass,
    // so no atomics and the same bits every run. One request per gap (Mesh<1>::insert defers a second). Returns whether any went in.
    // Used by: step(), run_fused(). Public because CUDA lambdas cannot be in private members.
    bool refine() {
        const double limit = length_on() ? refine_volume_ : 0.0;
        const double factor = mass_on() && targets_ready_ ? refine_mass_ : 0.0;
        const double split_min = split_min_;
        if (!(limit > 0) && !(factor > 0)) return false;
        const Line line = mesh_->line();
        const int n = n_;
        if (refine_points_.extent(0) < static_cast<std::size_t>(n)) {
            refine_points_ = Points(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("hydro_refine_points")), capacity_);
            refine_parent_ = Kokkos::View<int*, Memory>(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("hydro_refine_parent")), capacity_);
        }
        const auto points = refine_points_;
        const auto parent = refine_parent_;
        const auto q0 = q_;
        const auto target0 = target_;
        int count = 0;
        Kokkos::parallel_scan(
            "hydro1::refine_find", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n),
            KOKKOS_LAMBDA(int s, int& at, bool final) {
                const int i = line.order(s);
                const bool heavy = factor > 0 && target0(i) > 0 && q0(i, 0) > factor * target0(i) && line.volume(s) > split_min;
                if (!(limit > 0 && line.volume(s) > limit) && !heavy) return;
                if (final) {
                    // Behind a wall the gap runs to the generator's mirror, so the same rule keeps the point inside the line.
                    const i64 left = line.gap(s - 1), right = line.gap(s), k = static_cast<i64>(line.key(s));
                    const i64 far = right >= left ? right : left, near = right >= left ? left : right;
                    const i64 delta = Kokkos::max((far - near) / 2, far / 10);
                    i64 key = right >= left ? k + delta : k - delta;
                    if (line.periodic) {
                        if (key < 0) key += line.period;
                        if (key >= line.period) key -= line.period;
                    } else {
                        key = Kokkos::min(Kokkos::max(key, i64(0)), line.period - 1);
                    }
                    points(at, 0) = (static_cast<double>(key) + 0.5) * line.step;  // inside grid step `key`
                    parent(at) = i;
                }
                ++at;
            },
            count);
        if (count == 0) return false;
        reserve_cells(n_ + count);
        const bool linear = refine_linear_;
        if (linear) {
            if (slope_.extent(0) < static_cast<std::size_t>(capacity_))
                slope_ = Kokkos::View<double* [V], Memory>(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("hydro_refine_slope")), capacity_);
            const auto q = q_;
            const auto slope = slope_;
            Kokkos::parallel_for(
                "hydro1::refine_slopes", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n), KOKKOS_LAMBDA(int s) {
                    const int i = line.order(s), a = line.slot(s, -1), b = line.slot(s, 1);
                    const double Vs = line.volume(s);
                    for (int v = 0; v < V; ++v) slope(i, v) = 0;
                    if (a < 0 || b < 0) return;
                    const double Va = line.volume(a), Vb = line.volume(b);
                    for (int v = 0; v < V; ++v) {
                        const double u = q(i, v) / Vs;
                        const double dl = (u - q(line.order(a), v) / Va) / (0.5 * (Va + Vs));
                        const double dr = (q(line.order(b), v) / Vb - u) / (0.5 * (Vb + Vs));
                        slope(i, v) = dl * dr <= 0 ? 0.0 : (Kokkos::fabs(dl) < Kokkos::fabs(dr) ? dl : dr);
                    }
                });
        }
        if (piece_.extent(0) < static_cast<std::size_t>(count))
            piece_ = Kokkos::View<double* [2][V], Memory>(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("hydro_refine_piece")), capacity_);
        const auto wanted = Kokkos::subview(points, std::make_pair(0, count), Kokkos::ALL);
        const typename Mesh::Changes& change = mesh_->insert(wanted);
        if (change.done == 0) return false;
        const auto q = q_;
        const auto x = x_;
        const auto slope = slope_;
        const auto piece = piece_;
        const auto target = target_;
        const bool targets = mass_on() && targets_ready_;
        const auto id = change.id;
        const auto neighbour = change.neighbour;
        const auto overlap = change.overlap;
        const auto volume = change.volume;
        Kokkos::parallel_for(
            "hydro1::refine_split", Kokkos::RangePolicy<ExecSpace>(exec_, 0, change.requested), KOKKOS_LAMBDA(int r) {
                const int i = id(r);
                if (i < 0) return;
                if (targets) target(i) = target(parent(r));  // the children inherit the parent's target mass
                for (int v = 0; v < V; ++v) q(i, v) = 0;
                for (int side = 0; side < 2; ++side) {
                    const int old = neighbour(r, side);
                    for (int v = 0; v < V; ++v) piece(r, side, v) = 0;
                    if (old < 0) continue;  // a wall: nothing taken on that side
                    const double l = overlap(r, side), Vo = volume(r, side);
                    // The piece lies at the old cell's end next to the new generator: right end for side 0, left end for side 1.
                    const double offset = (side == 0 ? 0.5 : -0.5) * (Vo - l);
                    for (int v = 0; v < V; ++v) {
                        const double part = linear ? (q(old, v) / Vo + slope(old, v) * offset) * l : q(old, v) * (l / Vo);
                        piece(r, side, v) = part;
                        q(i, v) += part;
                    }
                }
                x(i, 0) = wanted(r, 0);
            });
        for (int side = 0; side < 2; ++side)
            Kokkos::parallel_for(
                "hydro1::refine_take", Kokkos::RangePolicy<ExecSpace>(exec_, 0, change.requested), KOKKOS_LAMBDA(int r) {
                    const int old = neighbour(r, side);
                    if (id(r) < 0 || old < 0) return;
                    for (int v = 0; v < V; ++v) q(old, v) -= piece(r, side, v);
                });
        n_ += change.done;
        refined_ += change.done;
        return true;
    }

    // Fences and adds the elapsed time to Timings::*part. Used by: time_step(), step().
    void phase_end(Timings* timings, double Timings::*part) {
        if (!timings) return;
        exec_.fence();
        timings->*part += timer_.seconds() * 1e3;
        timer_.reset();
    }

    // The fused step, one thread per slot, also giving the next time step and the slots out of order; public for CUDA lambdas.
    // Fused: also snap the moved points to keys and reduce the next dt. Used by: run(), run_fused().
    template <int Order, bool Moving, bool Fused>
    struct Kernel {
        Stencil st;
        Cells next;
        Points x, x_next;
        Keys keys;
        sorted_mesh::Grid grid;
        double dt, courant;

        // Static: a cell's state at a face at offset r from its centroid, half a step on. Used by: body().
        KOKKOS_INLINE_FUNCTION State face_state(const State& W, const State& g, double r) const {
            if constexpr (Order == 0) {
                (void)g;
                (void)r;
                return W;
            } else {
                Gradient<1> grad;
                for (int v = 0; v < V; ++v) grad[v][0] = g[v];
                State Wf = half_step(W, grad, dt, st.gas);
                for (int v = 0; v < V; ++v) Wf[v] += g[v] * r;
                if (Wf[0] > 0 && Wf[2] > 0) return Wf;
                return W;
            }
        }

        // Moving: W + g r, a whole step on (MUSCL-Hancock) when `second`; the cell's own W if rho or P would not stay positive.
        // Used by: body(), on both meshes.
        KOKKOS_INLINE_FUNCTION State extrapolated(const State& W, const State& g, double r, int second) const {
            if constexpr (Order == 0) {
                (void)g;
                (void)r;
                (void)second;
                return W;
            } else {
                State Wf = W;
                if (second) {
                    Gradient<1> grad;
                    for (int v = 0; v < V; ++v) grad[v][0] = g[v];
                    Wf = half_step(W, grad, 2 * dt, st.gas);
                }
                for (int v = 0; v < V; ++v) Wf[v] += g[v] * r;
                if (Wf[0] > 0 && Wf[2] > 0) return Wf;
                return W;
            }
        }

        KOKKOS_INLINE_FUNCTION void operator()(int s) const {
            double smallest = 0;
            int descents = 0;
            (*this)(s, smallest, descents);
        }
        KOKKOS_INLINE_FUNCTION void operator()(int s, double& smallest, int& descents) const {
            if (s >= 3 && s <= st.line.n - 4)
                body<true>(s, smallest, descents);
            else
                body<false>(s, smallest, descents);
        }
        // Moving and fused, with the end cells' wish for a new one.
        KOKKOS_INLINE_FUNCTION void operator()(int s, double& smallest, int& descents, int& wants) const {
            (*this)(s, smallest, descents);
            if (s == 0 || s == st.line.n - 1) wants += st.wants_cell(s);
        }

        // Where a point at x0 ends up after dt. Used by: body().
        KOKKOS_INLINE_FUNCTION double moved(double x0, double velocity) const {
            const double side = st.line.side();
            const double y = Kokkos::fma(dt, velocity, x0);
            return st.line.periodic ? y - side * Kokkos::floor(y / side) : Kokkos::fmin(Kokkos::fmax(y, 0.0), side);
        }

        // Per slot: load, gradients, point velocities (moving: the new keys), two face fluxes, Q update (fused: next dt).
        // Moving: each flux is the mean of those on the meshes of t^n and t^n+1, as finite_volume.hpp's step_moving().
        // Inner: no ghost in reach, nothing to wrap. Used by: operator().
        template <bool Inner>
        KOKKOS_INLINE_FUNCTION void body(int s, double& smallest, int& descents) const {
            const Line& line = st.line;
            typename Stencil::Cell cell[5];
            double gap[4];
            if constexpr (Inner)
                st.template load_inner<2>(s, cell, gap);
            else
                st.template load<2>(s, cell, gap);
            // Gradients of slots s - 1, s, s + 1 (k = 1, 2, 3) and their point velocities.
            State grad[5];
            double point[5] = {0, 0, 0, 0, 0};
            for (int k = 1; k <= 3; ++k) {
                for (int v = 0; v < V; ++v) grad[k][v] = 0;
                if (!Inner && !cell[k].real) continue;
                if constexpr (Order == 1)
                    for (int v = 0; v < V; ++v) grad[k][v] = st.gradient(cell + k, gap[k - 1], gap[k], v);
                if constexpr (Moving)
                    point[k] = cell[k].W[1] + st.point_velocity(cell + k, gap[k - 1], gap[k], grad[k][0], 0.5 * (gap[k - 1] + gap[k]));
            }
            // Moving: each point's new key and its move in grid steps; a face moves by the mean of its generators' moves.
            sorted_mesh::Key key[5] = {0, 0, 0, 0, 0};
            double moves[5] = {0, 0, 0, 0, 0};
            if constexpr (Moving) {
                for (int k = 1; k <= 3; ++k) {
                    if (!Inner && !cell[k].real) continue;
                    const int t = Inner ? s - 2 + k : line.slot(s, k - 2);
                    key[k] = sorted_mesh::snap(moved(x(line.order(t), 0), point[k]), grid, line.periodic);
                    i64 d = static_cast<i64>(key[k]) - static_cast<i64>(line.key(t));
                    if (line.periodic) d = d > line.period / 2 ? d - line.period : d < -line.period / 2 ? d + line.period : d;
                    moves[k] = static_cast<double>(d);
                }
            }
            // The fluxes through the left (k = 1 | 2) and right (2 | 3) faces of s, positive along +x.
            State F[2];
            for (int f = 0; f < 2; ++f) {
                const int a = 1 + f, b = 2 + f;
                if constexpr (Moving) {
                    // A wall face is static; between cells, the face's move on the grid sets w, so V^n+1 - V^n = dt (w_R - w_L) to rounding.
                    const bool inside = Inner || (cell[a].real && cell[b].real);
                    const double travel = inside ? 0.5 * (moves[a] + moves[b]) * line.step : 0.0;
                    const double w = travel / dt;
                    const double ra = 0.5 * gap[a] - cell[a].c, rb = -0.5 * gap[a] - cell[b].c;
                    // On the mesh of t^n (second = 0), then on that of t^n+1 (W^n a whole step on, to where the face is then).
                    const auto on_mesh = [&](int second) {
                        State L, R;
                        if (Inner || cell[a].real) L = extrapolated(cell[a].W, grad[a], ra + second * travel, second);
                        if (Inner || cell[b].real) R = extrapolated(cell[b].W, grad[b], rb + second * travel, second);
                        if constexpr (!Inner) {
                            if (!cell[a].real) L = st.ghost_face(R, cell[b].W, true);
                            if (!cell[b].real) R = st.ghost_face(L, cell[a].W, false);
                        }
                        return Riemann::flux(L, R, Vec<1>{1.0}, Vec<1>{w}, st.gas);
                    };
                    if constexpr (Order == 0) {
                        F[f] = on_mesh(0);
                    } else {
                        const State first = on_mesh(0), second = on_mesh(1);
                        for (int v = 0; v < V; ++v) F[f][v] = 0.5 * (first[v] + second[v]);
                    }
                } else {
                    const double w = 0, shift = 0.5 * dt * w;
                    State L, R;
                    if (Inner || cell[a].real) L = face_state(cell[a].W, grad[a], 0.5 * gap[a] - cell[a].c + shift);
                    if (Inner || cell[b].real) R = face_state(cell[b].W, grad[b], -0.5 * gap[a] - cell[b].c + shift);
                    if constexpr (!Inner) {
                        if (!cell[a].real) L = st.ghost_face(R, cell[b].W, true);
                        if (!cell[b].real) R = st.ghost_face(L, cell[a].W, false);
                    }
                    F[f] = Riemann::flux(L, R, Vec<1>{1.0}, Vec<1>{w}, st.gas);
                }
            }
            const int i = line.order(s);
            State Unew;
            for (int v = 0; v < V; ++v) {
                Unew[v] = st.q(i, v) - dt * (F[1][v] - F[0][v]);
                next(i, v) = Unew[v];
            }
            if constexpr (Moving) x_next(i, 0) = moved(x(i, 0), point[2]);
            if constexpr (Fused) {
                double volume = cell[2].V, relative = 0;
                if constexpr (Moving) {
                    keys(s) = key[2];
                    double left, right;
                    if (Inner || cell[1].real) {
                        const i64 d = static_cast<i64>(key[2]) - static_cast<i64>(key[1]);
                        left = static_cast<double>(d < 0 && line.periodic ? d + line.period : d);
                    } else {
                        left = static_cast<double>(2 * static_cast<i64>(key[2]));
                    }
                    if (Inner || cell[3].real) {
                        const int t = Inner ? s + 1 : line.slot(s, 1);
                        const i64 d = static_cast<i64>(key[3]) - static_cast<i64>(key[2]);
                        right = static_cast<double>(d < 0 && line.periodic ? d + line.period : d);
                        descents += !sorted_mesh::before(key[2], i, key[3], line.order(t));
                    } else {
                        right = static_cast<double>(2 * (line.period - static_cast<i64>(key[2])));
                    }
                    volume = (left + right) * (0.5 * line.step);
                    relative = Kokkos::fabs(point[2] - cell[2].W[1]);
                }
                const double inv = 1.0 / volume;
                const State Wnew = primitive(State{Unew[0] * inv, Unew[1] * inv, Unew[2] * inv}, st.gas);
                if constexpr (!Moving) relative = Kokkos::fabs(Wnew[1]);
                smallest = Kokkos::fmin(smallest, courant * volume / (relative + sound_speed(Wnew, st.gas)));
            }
        }
    };

    // Picks the Moving instantiation. Used by: step().
    template <int Order>
    void advance(double dt) {
        if (motion_ == Motion::moving)
            run<Order, true>(dt);
        else
            run<Order, false>(dt);
    }
    // Launches the unfused Kernel. Used by: advance().
    template <int Order, bool Moving>
    void run(double dt) {
        const Kernel<Order, Moving, false> kernel{stencil(), next_, x_, x_next_, {}, grid(), dt, 0.0};
        Kokkos::parallel_for("hydro1::step", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), kernel);
    }
    // Launches the fused Kernel, then reorders/supplies cells; returns the next dt. Used by: step_and_time_step().
    template <int Order, bool Moving>
    double run_fused(double dt, double courant) {
        const Line& line = mesh_->line();
        const Kernel<Order, Moving, true> kernel{stencil(), next_, x_, x_next_, mesh_->spare_keys(), grid(), dt, courant};
        double smallest = 0;
        int descents = 0, wants = 0;
        const Kokkos::RangePolicy<ExecSpace> slots(exec_, 0, n_);
        if (Moving && watching_)
            Kokkos::parallel_reduce("hydro1::step_fused_ends", slots, kernel, Kokkos::Min<double>(smallest), Kokkos::Sum<int>(descents),
                                    Kokkos::Sum<int>(wants));
        else
            Kokkos::parallel_reduce("hydro1::step_fused", slots, kernel, Kokkos::Min<double>(smallest), Kokkos::Sum<int>(descents));
        std::swap(q_, next_);
        if constexpr (Moving) {
            std::swap(x_, x_next_);
            mesh_->swap_keys();
            mesh_stats_.updates += 1;
            bool again = false;
            if (descents != (line.periodic ? 1 : 0)) {
                mesh_stats_.rebuilds += mesh_->order(live_points()).rebuilt;
                again = true;
            }
            wants_ = wants;
            looked_ = watching_;
            if (due()) again = supply_cells() || again;
            again = derefine() || again;
            again = refine() || again;
            looked_ = false;
            if (again) return time_step(courant, TimeStep::luna);
        }
        return smallest;
    }

private:
    // The line's grid, for the kernels' snap of the moved points. Used by: run(), run_fused().
    sorted_mesh::Grid grid() const {
        sorted_mesh::Grid g;
        g.side = mesh_->line().side();
        g.step = mesh_->line().step;
        return g;
    }
    // A moving step writes the new positions beside the old ones. Used by: step(), step_and_time_step().
    void reserve_next_points() {
        if (x_.extent(0) > 0 && x_next_.extent(0) == 0)
            x_next_ = Points(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("hydro_x_next")), capacity_);
    }
    // Used by: time_step(), supply_cells(), run(), run_fused().
    Stencil stencil() const {
        return {mesh_->line(), q_, gas_, walls_, limiter_, regularise_, supply_left_, supply_right_, spacing_, steer_};
    }
    // Whether to look at the end cells after this step: as the kernels asked while gas enters, else every 64 steps. Used by: step(), run_fused().
    bool due() {
        if (!supply_left_ && !supply_right_) return false;
        if (watching_) return !looked_ || wants_ > 0;
        if (--until_look_ > 0) return false;
        until_look_ = 64;
        return true;
    }
    // Used by: step(), run_fused() for mesh_->order().
    auto live_points() const { return Kokkos::subview(x_, std::make_pair(0, n_), Kokkos::ALL); }

    // Room for `count` cells, by a quarter more when full. Used by: supply_cells().
    void reserve_cells(int count) {
        if (count <= capacity_) return;
        capacity_ = std::max(count, capacity_ + capacity_ / 4);
        const auto used = std::make_pair(0, n_);
        const Cells q(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("hydro_Q")), capacity_);
        const Points x(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("hydro_x")), capacity_);
        Kokkos::deep_copy(exec_, Kokkos::subview(q, used, Kokkos::ALL), Kokkos::subview(q_, used, Kokkos::ALL));
        Kokkos::deep_copy(exec_, Kokkos::subview(x, used, Kokkos::ALL), Kokkos::subview(x_, used, Kokkos::ALL));
        q_ = q;
        x_ = x;
        if (target_.extent(0) > 0) {
            const Kokkos::View<double*, Memory> target(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("hydro_target")), capacity_);
            Kokkos::deep_copy(exec_, Kokkos::subview(target, used), Kokkos::subview(target_, used));
            target_ = target;
        }
        next_ = Cells(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("hydro_Q_next")), capacity_);
        if (x_next_.extent(0) > 0)
            x_next_ = Points(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("hydro_x_next")), capacity_);
    }

    ExecSpace exec_;
    Mesh* mesh_;
    Gas gas_;
    int order_;
    Limiter limiter_;
    Motion motion_;
    Walls walls_;
    bool regularise_;
    int n_, capacity_;
    Cells q_, next_;
    Points x_, x_next_;
    MeshStats mesh_stats_;
    // Cells from the walls: what the last time_step() saw at the ends, and a request at each wall at most.
    bool supply_left_ = false, supply_right_ = false, watching_ = false, looked_ = false;
    double spacing_ = 0;
    int wants_ = 0, until_look_ = 0;
    long long added_ = 0;
    double deref_volume_ = 0;  // set_derefine(); 0: off
    long long removed_ = 0;
    Kokkos::View<int*, Memory> deref_ids_;
    double refine_volume_ = 0;  // set_refine(); 0: off
    bool refine_linear_ = true;
    long long refined_ = 0;
    Adapt adapt_ = Adapt::off;  // set_adapt(); the constructor makes it mass on a moving line
    double refine_mass_ = refine_mass_default, deref_mass_ = derefine_mass_default;
    double steer_ = 0;  // set_steer(); 0: off
    double split_min_ = 0, merge_max_ = 0, safeguard_ = 0;  // set_mass_bounds(); the constructor sets the defaults on a moving line
    bool targets_ready_ = false;
    Kokkos::View<double*, Memory> target_;        // per id: the target mass (Adapt::mass)
    Points refine_points_;                        // the new generators' positions, one per wide cell
    Kokkos::View<int*, Memory> refine_parent_;    // per request: the id of the cell that splits
    Kokkos::View<double* [V], Memory> slope_;    // linear split: limited slopes of the conserved densities, per old id
    Kokkos::View<double* [2][V], Memory> piece_;  // what each new cell takes from its two old neighbours

    void check_hysteresis() const {
        if (refine_volume_ > 0 && deref_volume_ > 0 && !(refine_volume_ > 4 * deref_volume_))
            throw std::invalid_argument("FiniteVolume<1>: refinement needs max_volume > 4 min_volume of derefinement");
        if (refine_mass_ > 0 && deref_mass_ > 0 && !(refine_mass_ >= 4 * deref_mass_))
            throw std::invalid_argument("FiniteVolume<1>: refinement needs refine_mass >= 4 derefine_mass");
    }
    bool mass_on() const { return adapt_ == Adapt::mass; }
    bool length_on() const { return adapt_ != Adapt::off; }  // length thresholds: the criterion (length) or a safeguard (mass)
    Kokkos::View<double[5], Memory> ends_{"hydro_ends"};
    typename Kokkos::View<double[5], Memory>::host_mirror_type ends_host_ = Kokkos::create_mirror_view(ends_);
    Points points_{"hydro_new_points", 2};
    typename Points::host_mirror_type points_host_ = Kokkos::create_mirror_view(points_);
    Kokkos::Timer timer_;
};

}  // namespace hydro
