#!/bin/bash
# Sod theta on one A100: CUDA build of sod_theta, luna-cpp's run (scripts/examples/sod_theta.py), the cost at larger N, then LUNA's
# JAX run of the same problem (LUNA worktree's scripts/examples/autodiff/sod_theta.py) on the same GPU.
#   sbatch -o <logs>/sod-%j.log scripts/examples/sod_theta_a100.sh [minmod | mc | vanleer]
#SBATCH -J sod_theta
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:50:00
set -uo pipefail
source /etc/profile.d/modules.sh; module purge; module load gcc/14 cuda/13.0 cmake hdf5-serial
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm --format=csv,noheader
lim=${1:-minmod}
repo=${REPO:-/raven/u/uli/luna-cpp-adx}
luna=${LUNA_REPO:-/raven/u/uli/LUNA-worktrees/autodiff-examples}
py=$HOME/jax-env/bin/python
b=/ptmp/uli/luna-cpp-adx/build/cuda
out=/ptmp/uli/luna-cpp-adx/runs/sod_theta
mkdir -p $out
cd $repo && git log --oneline -1
cmake -S . -B $b -DCMAKE_BUILD_TYPE=Release -DKokkos_DIR=/u/uli/kokkos-cuda/lib64/cmake/Kokkos \
    -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper -DLUNA_DERIVATIVE_EXAMPLES=ON > $b.cfg.log 2>&1 || { echo "configure failed"; tail -20 $b.cfg.log; exit 1; }
cmake --build $b --parallel 16 --target luna_sod_theta > $b.build.log 2>&1; echo "build exit $?"
grep -E "warning|error" $b.build.log | grep -v "extern/" | head -40
[[ -x $b/sod_theta ]] || exit 1
export PYTHONDONTWRITEBYTECODE=1
[[ ${SKIP_CPP:-0} == 1 ]] || $py -u $repo/scripts/examples/sod_theta.py $b/sod_theta $out/sod_theta_cpp_$lim.npz --limiter=$lim
[[ ${SKIP_CPP:-0} == 1 ]] || $b/sod_theta --limiter=$lim --cost=200,16384,131072,1048576,8388608
case $lim in minmod) lnum=0 ;; mc) lnum=1 ;; *) lnum=2 ;; esac
cd $out && git -C $luna log --oneline -1
PYTHONPATH=$luna:$luna/src XLA_PYTHON_CLIENT_PREALLOCATE=false $py -u -m scripts.examples.autodiff.sod_theta limiter=$lnum out=$out/sod_theta_luna_$lim.npz
