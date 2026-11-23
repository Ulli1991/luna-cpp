#pragma once

// The fixed-grid step as hand-written CUDA kernels (research_notes/fixed_grid_2026-11-19.md): the per-cell numerics are grid.hpp's
// cell_step() and predictor_cell(), the same functions the Kokkos kernels call; only the launch and the block's minimum of the time step
// differ. Built only with Kokkos's CUDA backend (nvcc_wrapper compiles every source as CUDA). Used by: GridFiniteVolume::step (Kernel::cuda).
#if defined(KOKKOS_ENABLE_CUDA)

#include "grid.hpp"

#include <cuda_runtime.h>

#include <cstdint>

// Launch bounds of step_kernel for a register sweep: -DLUNA_GRID_MIN_BLOCKS=b -DLUNA_GRID_MAX_THREADS=t caps the registers so that b
// blocks of t threads fit an SM. Off by default (caps are measured both ways, research_notes/fixed_grid_2026-11-19.md).
#if defined(LUNA_GRID_MIN_BLOCKS)
#define LUNA_GRID_BOUNDS __launch_bounds__(LUNA_GRID_MAX_THREADS, LUNA_GRID_MIN_BLOCKS)
#else
#define LUNA_GRID_BOUNDS
#endif

namespace hydro::grid {

// One block per tile, one thread per cell (z fastest within the tile, tiles numbered z fastest as the Kokkos kernel's league), the
// tile's minimum time step by a shared-memory tree (block sizes are powers of two), then one atomicMin on its order key.
// Split: the faces pass of the split step, reading the half-step states H.
template <int D, class Riemann, bool Split>
__global__ void LUNA_GRID_BOUNDS step_kernel(Field<D> W, HalfField<D> H, Arrays a, Params p, double dt, int3 tile, int3 tiles, unsigned long long* key) {
    __shared__ double low[1024];
    const int r = static_cast<int>(blockIdx.x), t = static_cast<int>(threadIdx.x);
    const int tz = r % tiles.z, ty = (r / tiles.z) % tiles.y, tx = r / (tiles.z * tiles.y);
    const int k = tz * tile.z + t % tile.z, j = ty * tile.y + (t / tile.z) % tile.y, i = tx * tile.x + t / (tile.z * tile.y);
    low[t] = cell_step<D, Riemann, Split>(W, a, i, j, k, p, dt, H);
    __syncthreads();
    for (int w = static_cast<int>(blockDim.x) / 2; w > 0; w >>= 1) {
        if (t < w && low[t + w] < low[t]) low[t] = low[t + w];
        __syncthreads();
    }
    if (t == 0) atomicMin(key, static_cast<unsigned long long>(order_key(low[0])));
}

// The predictor pass of the split step: one thread per cell of the grid plus its ghost ring, numbered z fastest.
template <int D>
__global__ void predictor_kernel(Field<D> W, HalfField<D> H, Params p, double h) {
    const long c = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= H.cells) return;
    const int k = static_cast<int>(c % H.m[2]), j = static_cast<int>((c / H.m[2]) % H.m[1]);
    const int i = static_cast<int>(c / (static_cast<long>(H.m[2]) * H.m[1]));
    predictor_any<D>(W, H, i - 1, D > 1 ? j - 1 : j, D > 2 ? k - 1 : k, p, h);
}

// Launches the step on `stream`: tile cells per block, tiles per axis; with split, first the predictor pass (256 threads a block).
// Used by: GridFiniteVolume::step.
template <int D, class Riemann>
void step_cuda(cudaStream_t stream, const Field<D>& W, const HalfField<D>& H, bool split, const Arrays& a, const Params& p, double dt,
               const int (&tile)[3], const int (&tiles)[3], unsigned long long* key) {
    const int threads = tile[0] * tile[1] * tile[2];
    const int blocks = tiles[0] * tiles[1] * tiles[2];
    const int3 t = make_int3(tile[0], tile[1], tile[2]), n = make_int3(tiles[0], tiles[1], tiles[2]);
    if (split) {
        predictor_kernel<D><<<static_cast<unsigned>((H.cells + 255) / 256), 256, 0, stream>>>(W, H, p, 0.5 * dt);
        step_kernel<D, Riemann, true><<<blocks, threads, 0, stream>>>(W, H, a, p, dt, t, n, key);
    } else {
        step_kernel<D, Riemann, false><<<blocks, threads, 0, stream>>>(W, H, a, p, dt, t, n, key);
    }
}

}  // namespace hydro::grid

#endif
