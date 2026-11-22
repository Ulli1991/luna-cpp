#!/bin/bash
# Builds hydro_1d (and, with TARGETS, more): super_tube_v2_build.sh <omp|cuda>.
# omp: Release, gcc 15, Kokkos OpenMP (/u/uli/kokkos-gcc15), -march=icelake-server, in $B/build-omp; on a general or interactive node.
# cuda: Release, gcc 14, CUDA 13.0, Kokkos CUDA AMPERE80 (/u/uli/kokkos-cuda), in $B/build-cuda; on a GPU node (nvcc_wrapper).
# R: the source tree; B (default R): where the build directories go, e.g. /ptmp/uli/<worktree>.
set -euo pipefail
source /etc/profile.d/modules.sh
R=${R:-/raven/u/uli/luna-cpp-tube-v2}   # not dirname $0: sbatch runs a copy of the script
B=${B:-$R}
T=${TARGETS:-luna_hydro_1d}
module purge
mkdir -p $B/build-$1
if [[ $1 == omp ]]; then
  module load gcc/15 cmake
  cmake -S $R -B $B/build-omp -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-march=icelake-server \
    -DKokkos_DIR=/u/uli/kokkos-gcc15/lib64/cmake/Kokkos -DCMAKE_CXX_COMPILER="$(which g++)" > $B/build-omp/configure.log 2>&1
  make -C $B/build-omp -j"${SLURM_CPUS_PER_TASK:-8}" $T 2>&1 | grep -E "error|Built" || true
else
  module load gcc/14 cuda/13.0 cmake
  cmake -S $R -B $B/build-cuda -DCMAKE_BUILD_TYPE=Release -DKokkos_DIR=/u/uli/kokkos-cuda/lib64/cmake/Kokkos \
    -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper > $B/build-cuda/configure.log 2>&1
  make -C $B/build-cuda -j"${SLURM_CPUS_PER_TASK:-8}" $T 2>&1 | grep -E "error|Built" || true
fi
