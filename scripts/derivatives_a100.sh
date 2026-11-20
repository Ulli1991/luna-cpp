#!/bin/bash
# The derivative test and its costs on one A100: CUDA build of the autodiff-l1 worktree with LUNA_DERIVATIVE_TESTS=ON,
# hydro_derivatives_test, then --cost at 64^3 and 128^3 cells (mode scan: --scan instead).
#   sbatch -o <logs>/deriv-a100-%j.log scripts/derivatives_a100.sh [scan]
#SBATCH -J deriv_a100
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:45:00
set -uo pipefail
source /etc/profile.d/modules.sh; module purge; module load gcc/14 cuda/13.0 cmake hdf5-serial
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm --format=csv,noheader
repo=/raven/u/uli/luna-cpp-ad
b=${BUILD:-/ptmp/uli/luna-cpp-ad/build-ad-cuda}  # builds on /ptmp (the /raven/u quota)
cd $repo && git log --oneline -1
cmake -S . -B $b -DCMAKE_BUILD_TYPE=Release -DKokkos_DIR=/u/uli/kokkos-cuda/lib64/cmake/Kokkos \
    -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper -DLUNA_DERIVATIVE_TESTS=ON > $b.cfg.log 2>&1 || { echo "configure failed"; tail -20 $b.cfg.log; exit 1; }
cmake --build $b --parallel 16 --target hydro_derivatives_test > $b.build.log 2>&1; echo "build exit $?"
grep -E "warning|error" $b.build.log | grep -v "extern/" | head -40
[[ -x $b/hydro_derivatives_test ]] || exit 1
time $b/hydro_derivatives_test
if [[ "${1:-}" == scan ]]; then
    $b/hydro_derivatives_test --scan | grep -v "^  ok"
else
    for n in 262144 2097152; do $b/hydro_derivatives_test --cost $n; done
fi
