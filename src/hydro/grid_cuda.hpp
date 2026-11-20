#pragma once

// The fixed-grid step as a hand-written CUDA kernel (research_notes/fixed_grid_2026-11-19.md, phase 1): the per-cell numerics are
// grid.hpp's cell_step(), the same function the Kokkos kernel calls; only the launch and the block's minimum of the time step differ.
// Built only with Kokkos's CUDA backend (nvcc_wrapper compiles every source as CUDA). Used by: GridFiniteVolume::step (Kernel::cuda).
#if defined(KOKKOS_ENABLE_CUDA)

#include "grid.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace hydro::grid {

// One block per tile, one thread per cell (z fastest within the tile, tiles numbered z fastest as the Kokkos kernel's league), the
// tile's minimum time step by a shared-memory tree (block sizes are powers of two), then one atomicMin on its order key.
template <int D, class Riemann>
__global__ void step_kernel(Field<D> W, Arrays a, Params p, double dt, int3 tile, int3 tiles, unsigned long long* key) {
    __shared__ double low[1024];
    const int r = static_cast<int>(blockIdx.x), t = static_cast<int>(threadIdx.x);
    const int tz = r % tiles.z, ty = (r / tiles.z) % tiles.y, tx = r / (tiles.z * tiles.y);
    const int k = tz * tile.z + t % tile.z, j = ty * tile.y + (t / tile.z) % tile.y, i = tx * tile.x + t / (tile.z * tile.y);
    low[t] = cell_step<D, Riemann>(W, a, i, j, k, p, dt);
    __syncthreads();
    for (int w = static_cast<int>(blockDim.x) / 2; w > 0; w >>= 1) {
        if (t < w && low[t + w] < low[t]) low[t] = low[t + w];
        __syncthreads();
    }
    if (t == 0) atomicMin(key, static_cast<unsigned long long>(order_key(low[0])));
}

// Launches step_kernel on `stream`: tile cells per block, tiles per axis. Used by: GridFiniteVolume::step.
template <int D, class Riemann>
void step_cuda(cudaStream_t stream, const Field<D>& W, const Arrays& a, const Params& p, double dt, const int (&tile)[3],
               const int (&tiles)[3], unsigned long long* key) {
    const int threads = tile[0] * tile[1] * tile[2];
    const int blocks = tiles[0] * tiles[1] * tiles[2];
    step_kernel<D, Riemann><<<blocks, threads, 0, stream>>>(W, a, p, dt, make_int3(tile[0], tile[1], tile[2]),
                                                            make_int3(tiles[0], tiles[1], tiles[2]), key);
}

}  // namespace hydro::grid

#endif
