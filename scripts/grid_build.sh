#!/bin/bash
# Builds of the fixed grid (research_notes/fixed_grid_2026-11-19.md) in this worktree, on a compute node:
#   srun -p interactive -n1 -c8 --mem=32G --time=01:00:00 scripts/grid_build.sh <host|cuda|cuda-exact> [targets...]
# host: gcc 15, Kokkos OpenMP+Serial (/u/uli/kokkos-gcc15), build-grid-host;
# cuda: gcc 14, CUDA 13.0, Kokkos CUDA (/u/uli/kokkos-cuda), build-grid-cuda; cuda-exact: the same with LUNA_BIT_EXACT=ON.
set -euo pipefail
source /etc/profile.d/modules.sh
module purge
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
kind="$1"
shift
targets=("${@:-grid_test}")
case "$kind" in
host)
    module load gcc/15 cmake hdf5-serial
    cfg=(-DCMAKE_BUILD_TYPE=Release -DKokkos_DIR=/u/uli/kokkos-gcc15/lib64/cmake/Kokkos -DCMAKE_EXPORT_COMPILE_COMMANDS=ON)
    dir="$root/build-grid-host" ;;
cuda | cuda-exact)
    module load gcc/14 cuda/13.0 cmake hdf5-serial
    cfg=(-DCMAKE_BUILD_TYPE=Release -DKokkos_DIR=/u/uli/kokkos-cuda/lib64/cmake/Kokkos -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper)
    dir="$root/build-grid-$kind"
    [[ "$kind" == cuda-exact ]] && cfg+=(-DLUNA_BIT_EXACT=ON) ;;
*)
    echo "kind: host, cuda or cuda-exact" >&2
    exit 2 ;;
esac
# A variant for a head-to-head: GRID_DEFINES="-DNAME ..." builds into <dir>-<GRID_VARIANT> with those definitions.
if [[ -n "${GRID_VARIANT:-}" ]]; then
    dir="$dir-$GRID_VARIANT"
    cfg+=("-DCMAKE_CXX_FLAGS=${GRID_DEFINES:-}")
fi
mkdir -p "$dir"
cmake -S "$root" -B "$dir" "${cfg[@]}" >"$dir/configure.log" 2>&1 || { tail -30 "$dir/configure.log"; exit 1; }
cmake --build "$dir" --parallel "${SLURM_CPUS_PER_TASK:-4}" --target "${targets[@]}" 2>&1 | grep -E "error|warning|Error" | grep -v "^make" | head -60 || true
echo "build of ${targets[*]} in $dir done (errors and warnings above, if any)"
