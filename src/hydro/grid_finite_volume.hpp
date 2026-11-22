#pragma once

#include "grid.hpp"
#include "grid_cuda.hpp"

#include <Kokkos_Core.hpp>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

// The fixed-grid solver (research_notes/fixed_grid_2026-11-19.md): owns U and its primitives W on an N_x x N_y x N_z grid and steps
// them with grid.hpp's cell_step(), as a Kokkos kernel (also the host reference CI runs) or a CUDA kernel. Driver: refresh() once after
// setting state(), then time_step() and step(dt) in a loop. Used by: hydro_sedov --grid, grid_test.
namespace hydro::grid {

// Which kernel runs step(). Used by: Options.
enum class Kernel { kokkos, cuda };

// Used by: GridFiniteVolume (constructor).
struct Options {
    Limiter limiter = Limiter::mc;       // LUNA's default LIMITER 1
    Boundary boundary = Boundary::outflow;
    double cfl = 0.8;                    // LUNA's default CFL
    bool floors = false;                 // preserve_face_states
    double smallr = 1e-10, smallc = 1e-10;
    bool allow_first_order = false;      // allow_first_order_faces
    Kernel kernel = Kernel::kokkos;
    std::array<int, 3> tile{0, 0, 0};    // cells per block along x, y, z; 0: the default (LUNA's fused tiles on a GPU, one cell on a host)
    bool split = true;                   // LUNA's split step: a predictor pass, then the faces pass (+(D + 2) doubles a padded cell); false: one kernel
};

template <int D, class Riemann = HLLC, class Exec = Kokkos::DefaultExecutionSpace>
class GridFiniteVolume {
public:
    using Memory = typename Exec::memory_space;
    using Fields = Kokkos::View<double**, Kokkos::LayoutRight, Memory>;  // (D + 2, cells): variable-major, cells z fastest
    static constexpr bool device = !Kokkos::SpaceAccessibility<Kokkos::HostSpace, Memory>::accessible;

    // n cells and edge lengths per axis, ideal gas gamma. dx = length / n, as LUNA's.
    GridFiniteVolume(const Exec& exec, std::array<int, D> n, std::array<double, D> length, double gamma, const Options& o = {})
        : exec_(exec), o_(o) {
        double dx[3] = {1, 1, 1};
        for (int a = 0; a < D; ++a) {
            if (n[a] < 2) throw std::invalid_argument("GridFiniteVolume: at least 2 cells per axis");
            n_[a] = n[a];
            dx[a] = length[a] / n[a];
        }
        p_ = make_params(gamma, dx, o.cfl, o.limiter, o.floors, o.smallr, o.smallc, o.allow_first_order);
        cells_ = static_cast<long>(n_[0]) * n_[1] * n_[2];
        // LUNA's FUSED_TILE (Pallas tile_for: the largest power of two up to the cap that divides the extent); a host runs one cell a team.
        const int caps[3][3] = {{64, 1, 1}, {4, 32, 1}, {1, 4, 64}};
        for (int a = 0; a < 3; ++a) {
            int t = 1;
            const int cap = o.tile[a] > 0 ? o.tile[a] : device ? caps[D - 1][a] : 1;
            while (t * 2 <= cap && n_[a] % (t * 2) == 0) t *= 2;
            tile_[a] = t;
            tiles_[a] = n_[a] / t;
        }
        if (tile_[0] * tile_[1] * tile_[2] > 1024) throw std::invalid_argument("GridFiniteVolume: more than 1024 cells in a tile");
#if !defined(KOKKOS_ENABLE_CUDA)
        if (o.kernel == Kernel::cuda) throw std::invalid_argument("GridFiniteVolume: Kernel::cuda needs Kokkos's CUDA backend");
#endif
        U_ = Fields("grid_U", D + 2, cells_);
        U_next_ = Fields("grid_U_next", D + 2, cells_);
        W_ = Fields("grid_W", D + 2, cells_);
        W_next_ = Fields("grid_W_next", D + 2, cells_);
        if (o.split) {
            half_cells_ = 1;
            for (int a = 0; a < 3; ++a) {
                half_m_[a] = a < D ? n_[a] + 2 : 1;
                half_cells_ *= half_m_[a];
            }
            H_ = Fields("grid_half", D + 2, half_cells_);
        }
        dt_ = Kokkos::View<double, Memory>("grid_dt");
        key_ = Kokkos::View<unsigned long long, Memory>("grid_dt_key");
    }

    // U (conserved per volume); set it, then refresh(). After step(), the new U.
    Fields state() const { return U_; }
    // The primitives of state() (valid after refresh() or step()).
    Fields primitives() const { return W_; }
    const Params& params() const { return p_; }
    long cells() const { return cells_; }
    int extent(int a) const { return n_[a]; }
    const int* tile() const { return tile_; }

    // W = to_primitive(U) and the next time step, after state() was set from outside.
    void refresh() {
        const Fields U = U_, W = W_;
        const Params p = p_;
        Kokkos::parallel_reduce(
            "grid_refresh", Kokkos::RangePolicy<Exec>(exec_, 0, cells_),
            KOKKOS_LAMBDA(long c, double& low) {
                State<D> u;
                for (int v = 0; v < D + 2; ++v) u[v] = U(v, c);
                const State<D> w = to_primitive<D>(u, p);
                for (int v = 0; v < D + 2; ++v) W(v, c) = w[v];
                const double dt = cell_dt<D>(u, p);
                if (dt < low) low = dt;
            },
            Kokkos::Min<double, Memory>(dt_));
        from_key_ = false;
    }

    // LUNA's CFL time step of the current state (integrators._make_compute_dt_fn; -inf if a cell is NaN). Fences.
    double time_step() const {
        exec_.fence();
        if (from_key_) {
            unsigned long long k = 0;
            Kokkos::deep_copy(k, key_);
            return from_order_key(k);
        }
        double dt = 0;
        Kokkos::deep_copy(dt, dt_);
        return dt;
    }

    // One MUSCL-Hancock step of length dt; leaves the next time step for time_step().
    void step(double dt) {
        Field<D> W;
        W.W = W_.data();
        W.cells = cells_;
        for (int a = 0; a < 3; ++a) W.n[a] = n_[a];
        W.boundary = o_.boundary;
        const Arrays a{U_.data(), U_next_.data(), W_next_.data()};
        HalfField<D> H;
        H.H = H_.data();
        H.cells = half_cells_;
        for (int b = 0; b < 3; ++b) H.m[b] = half_m_[b];
        if (o_.kernel == Kernel::cuda) {
#if defined(KOKKOS_ENABLE_CUDA)
            Kokkos::deep_copy(exec_, key_, ~0ULL);
            step_cuda<D, Riemann>(exec_.cuda_stream(), W, H, o_.split, a, p_, dt, tile_, tiles_, key_.data());
            from_key_ = true;
#endif
        } else {
            const Kokkos::TeamPolicy<Exec> teams(exec_, tiles_[0] * tiles_[1] * tiles_[2], tile_[0] * tile_[1] * tile_[2]);
            if (o_.split) {
                const Params p = p_;
                const double h = 0.5 * dt;
                Kokkos::parallel_for(
                    "grid_predictor", Kokkos::RangePolicy<Exec>(exec_, 0, half_cells_), KOKKOS_LAMBDA(long c) {
                        const int k = static_cast<int>(c % H.m[2]), j = static_cast<int>((c / H.m[2]) % H.m[1]);
                        const int i = static_cast<int>(c / (static_cast<long>(H.m[2]) * H.m[1]));
                        predictor_cell<D>(W, H, i - 1, D > 1 ? j - 1 : j, D > 2 ? k - 1 : k, p, h);
                    });
                const StepTeam<true> f{W, H, a, p_, dt, {tile_[0], tile_[1], tile_[2]}, {tiles_[0], tiles_[1], tiles_[2]}};
                Kokkos::parallel_reduce("grid_faces", teams, f, Kokkos::Min<double, Memory>(dt_));
            } else {
                const StepTeam<false> f{W, H, a, p_, dt, {tile_[0], tile_[1], tile_[2]}, {tiles_[0], tiles_[1], tiles_[2]}};
                Kokkos::parallel_reduce("grid_step", teams, f, Kokkos::Min<double, Memory>(dt_));
            }
            from_key_ = false;
        }
        std::swap(U_, U_next_);
        std::swap(W_, W_next_);
    }

    // The Kokkos form of step_kernel (grid_cuda.hpp): one team per tile, one thread per cell, the same numbering and cell_step().
    // Public: CUDA kernels cannot take a private type. Split: the faces pass of the split step.
    template <bool Split>
    struct StepTeam {
        Field<D> W;
        HalfField<D> H;
        Arrays a;
        Params p;
        double dt;
        int tile[3], tiles[3];

        KOKKOS_INLINE_FUNCTION void operator()(const typename Kokkos::TeamPolicy<Exec>::member_type& m, double& best) const {
            const int r = m.league_rank();
            const int tz = r % tiles[2], ty = (r / tiles[2]) % tiles[1], tx = r / (tiles[2] * tiles[1]);
            double tile_low = Kokkos::Experimental::infinity_v<double>;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(m, tile[0] * tile[1] * tile[2]),
                [&](int t, double& low) {
                    const int k = tz * tile[2] + t % tile[2], j = ty * tile[1] + (t / tile[2]) % tile[1], i = tx * tile[0] + t / (tile[2] * tile[1]);
                    const double cell = cell_step<D, Riemann, Split>(W, a, i, j, k, p, dt, H);
                    if (cell < low) low = cell;
                },
                Kokkos::Min<double>(tile_low));
            Kokkos::single(Kokkos::PerTeam(m), [&] {
                if (tile_low < best) best = tile_low;
            });
        }
    };

private:
    Exec exec_;
    Options o_;
    Params p_;
    int n_[3] = {1, 1, 1};
    int tile_[3] = {1, 1, 1}, tiles_[3] = {1, 1, 1};
    long cells_ = 0;
    Fields U_, U_next_, W_, W_next_, H_;
    long half_cells_ = 0;
    int half_m_[3] = {1, 1, 1};
    Kokkos::View<double, Memory> dt_;
    Kokkos::View<unsigned long long, Memory> key_;
    bool from_key_ = false;
};

}  // namespace hydro::grid
