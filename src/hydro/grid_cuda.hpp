#pragma once

// The fixed-grid step as hand-written CUDA kernels (research_notes/fixed_grid_2026-11-19.md): the per-cell numerics are grid.hpp's
// cell_step() and predictor_cell(), the same functions the Kokkos kernels call; only the launch and the block's minimum of the time step
// differ. Built only with Kokkos's CUDA backend (nvcc_wrapper compiles every source as CUDA). Used by: GridFiniteVolume::step (Kernel::cuda).
#if defined(KOKKOS_ENABLE_CUDA)

#include "grid.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

// Launch bounds of step_kernel for a register sweep: -DLUNA_GRID_MIN_BLOCKS=b -DLUNA_GRID_MAX_THREADS=t caps the registers so that b
// blocks of t threads fit an SM. Off by default (caps are measured both ways, research_notes/fixed_grid_2026-11-19.md).
#if defined(LUNA_GRID_MIN_BLOCKS)
#define LUNA_GRID_BOUNDS __launch_bounds__(LUNA_GRID_MAX_THREADS, LUNA_GRID_MIN_BLOCKS)
#else
#define LUNA_GRID_BOUNDS
#endif

namespace hydro::grid {

// The smaller of two time steps (a NaN cell's -inf wins). Used by: block_min().
__device__ inline double smaller(double a, double b) { return b < a ? b : a; }

// The block's minimum of x, valid in thread 0 (lever 6): a shuffle tree in each warp, the warps' minima through 32 doubles of shared
// memory, a shuffle tree again; blocks under 32 threads (powers of two) take a shared-memory tree. The same minimum as any order.
// Used by: step_kernel.
__device__ inline double block_min(double x) {
    __shared__ double low[32];
    const int t = static_cast<int>(threadIdx.x), n = static_cast<int>(blockDim.x);
    if (n < 32) {
        low[t] = x;
        __syncthreads();
        for (int w = n / 2; w > 0; w >>= 1) {
            if (t < w) low[t] = smaller(low[t], low[t + w]);
            __syncthreads();
        }
        return low[0];
    }
    for (int offset = 16; offset > 0; offset >>= 1) x = smaller(x, __shfl_down_sync(0xffffffffu, x, offset));
    if ((t & 31) == 0) low[t >> 5] = x;
    __syncthreads();
    if (t < 32) {
        x = t < n / 32 ? low[t] : Kokkos::Experimental::infinity_v<double>;
        for (int offset = 16; offset > 0; offset >>= 1) x = smaller(x, __shfl_down_sync(0xffffffffu, x, offset));
    }
    return x;
}

// One block per tile, one thread per cell (z fastest within the tile, tiles numbered z fastest as the Kokkos kernel's league), the
// tile's minimum time step by block_min(), then one atomicMin on its order key. Split: the faces pass of the split step, reading the
// half-step states H. S: the options fixed at compile time (Fixed).
template <int D, class Riemann, bool Split, class S>
__device__ inline void step_body(Field<D> W, HalfField<D> H, Arrays a, Params p, double dt, int3 tile, int3 tiles, unsigned long long* key) {
    S::apply(p, W);
    const int r = static_cast<int>(blockIdx.x), t = static_cast<int>(threadIdx.x);
    const int tz = r % tiles.z, ty = (r / tiles.z) % tiles.y, tx = r / (tiles.z * tiles.y);
    const int k = tz * tile.z + t % tile.z, j = ty * tile.y + (t / tile.z) % tile.y, i = tx * tile.x + t / (tile.z * tile.y);
    const double low = block_min(cell_step<D, Riemann, Split>(W, a, i, j, k, p, dt, H));
    if (t == 0) atomicMin(key, static_cast<unsigned long long>(order_key(low)));
}
template <int D, class Riemann, bool Split, class S = Fixed<>>
__global__ void LUNA_GRID_BOUNDS step_kernel(Field<D> W, HalfField<D> H, Arrays a, Params p, double dt, int3 tile, int3 tiles, unsigned long long* key) {
    step_body<D, Riemann, Split, S>(W, H, a, p, dt, tile, tiles, key);
}
// The specialised faces pass with registers capped for 4 blocks of 128 threads per SM (lever 8: 142 -> 128 registers, 24 B of stack,
// 16 warps); blocks of at most 128 threads. Used by: split_cuda() (Launch::bounded).
template <int D, class Riemann, class S>
__global__ void __launch_bounds__(128, 4) faces_kernel_128x4(Field<D> W, HalfField<D> H, Arrays a, Params p, double dt, int3 tile, int3 tiles,
                                                             unsigned long long* key) {
    step_body<D, Riemann, true, S>(W, H, a, p, dt, tile, tiles, key);
}

// The predictor pass of the split step: one thread per cell of the grid plus its ghost ring, numbered z fastest.
template <int D, class S = Fixed<>>
__global__ void predictor_kernel(Field<D> W, HalfField<D> H, Params p, double h) {
    S::apply(p, W);
    const long c = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= H.cells) return;
    const int k = static_cast<int>(c % H.m[2]), j = static_cast<int>((c / H.m[2]) % H.m[1]);
    const int i = static_cast<int>(c / (static_cast<long>(H.m[2]) * H.m[1]));
    predictor_any<D>(W, H, i - 1, D > 1 ? j - 1 : j, D > 2 ? k - 1 : k, p, h);
}

// One step's launch arguments. Used by: step_cuda() and its dispatch.
template <int D>
struct Launch {
    cudaStream_t stream;
    Field<D> W;
    HalfField<D> H;
    Arrays a;
    Params p;
    double dt;
    int3 tile, tiles;
    int threads, blocks;
    unsigned long long* key;
    bool bounded;  // the specialised faces pass with launch bounds (faces_kernel_128x4) where the block has at most 128 threads
};

// The split step with the options S fixed at compile time: the predictor pass (256 threads a block), then the faces pass.
template <int D, class Riemann, class S>
void split_cuda(const Launch<D>& l) {
    predictor_kernel<D, S><<<static_cast<unsigned>((l.H.cells + 255) / 256), 256, 0, l.stream>>>(l.W, l.H, l.p, 0.5 * l.dt);
    if constexpr (!std::is_same_v<S, Fixed<>>) {
        if (l.bounded && l.threads <= 128) {
            faces_kernel_128x4<D, Riemann, S><<<l.blocks, l.threads, 0, l.stream>>>(l.W, l.H, l.a, l.p, l.dt, l.tile, l.tiles, l.key);
            return;
        }
    }
    step_kernel<D, Riemann, true, S><<<l.blocks, l.threads, 0, l.stream>>>(l.W, l.H, l.a, l.p, l.dt, l.tile, l.tiles, l.key);
}
// The dispatch over the boundary for limiter L (lever 5). Used by: step_cuda().
template <int D, class Riemann, int L>
void split_cuda_boundary(const Launch<D>& l) {
    switch (l.W.boundary) {
        case Boundary::periodic:
            return split_cuda<D, Riemann, Fixed<L, 0, 0>>(l);
        case Boundary::outflow:
            return split_cuda<D, Riemann, Fixed<L, 1, 0>>(l);
        case Boundary::reflective:
            return split_cuda<D, Riemann, Fixed<L, 2, 0>>(l);
        default:
            return split_cuda<D, Riemann, Fixed<L, 4, 0>>(l);
    }
}

// Launches the step on `stream`: tile cells per block, tiles per axis; with split, first the predictor pass. specialise: the split step
// runs a kernel instantiated for the limiter and boundary (and no floors or first-order faces; with them, the run-time kernel).
// Used by: GridFiniteVolume::step.
template <int D, class Riemann>
void step_cuda(cudaStream_t stream, const Field<D>& W, const HalfField<D>& H, bool split, bool specialise, bool bounded, const Arrays& a,
               const Params& p, double dt, const int (&tile)[3], const int (&tiles)[3], unsigned long long* key) {
    const Launch<D> l{stream, W, H, a, p, dt, make_int3(tile[0], tile[1], tile[2]), make_int3(tiles[0], tiles[1], tiles[2]),
                      tile[0] * tile[1] * tile[2], tiles[0] * tiles[1] * tiles[2], key, bounded};
    if (!split) {
        step_kernel<D, Riemann, false><<<l.blocks, l.threads, 0, stream>>>(W, H, a, p, dt, l.tile, l.tiles, key);
        return;
    }
#if defined(LUNA_GRID_NO_SPECIALISE)  // the instantiations left out, to measure their build cost
    (void)specialise;
    return split_cuda<D, Riemann, Fixed<>>(l);
#else
    if (!specialise || p.floors || p.allow_first_order) return split_cuda<D, Riemann, Fixed<>>(l);
    switch (p.limiter) {
        case Limiter::minmod:
            return split_cuda_boundary<D, Riemann, 0>(l);
        case Limiter::mc:
            return split_cuda_boundary<D, Riemann, 1>(l);
        default:
            return split_cuda_boundary<D, Riemann, 2>(l);
    }
#endif
}

}  // namespace hydro::grid

#endif
