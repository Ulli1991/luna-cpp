#!/bin/bash
# The KH twin experiment on one A100: luna-cpp static and moving (scripts/examples/kh_twin.py on the rollout binary), then LUNA's
# JAX run of the same problem at luna-cpp's dt and steps (LUNA worktree's scripts/examples/autodiff/kh_twin.py).
#   sbatch -o <logs>/kh-%j.log scripts/examples/kh_twin_a100.sh [n] [starts] [maxiter]
#SBATCH -J kh_twin
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=02:30:00
set -uo pipefail
source /etc/profile.d/modules.sh; module purge; module load gcc/14 cuda/13.0 cmake hdf5-serial
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm --format=csv,noheader
n=${1:-256}; starts=${2:-4}; maxiter=${3:-60}
repo=${REPO:-/raven/u/uli/luna-cpp-adx}
luna=${LUNA_REPO:-/raven/u/uli/LUNA-worktrees/autodiff-examples}
py=$HOME/jax-env/bin/python
b=/ptmp/uli/luna-cpp-adx/build/cuda
out=/ptmp/uli/luna-cpp-adx/runs/kh_twin
work=/ptmp/uli/luna-cpp-adx/tmp/kh-${SLURM_JOB_ID:-0}
mkdir -p $out $work
cd $repo && git log --oneline -1
cmake -S . -B $b -DCMAKE_BUILD_TYPE=Release -DKokkos_DIR=/u/uli/kokkos-cuda/lib64/cmake/Kokkos \
    -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper -DLUNA_DERIVATIVE_EXAMPLES=ON > $b.cfg.log 2>&1 || { echo "configure failed"; tail -20 $b.cfg.log; exit 1; }
cmake --build $b --parallel 16 --target luna_rollout > $b.rollout.log 2>&1; echo "rollout build exit $?"
grep -E "error" -A3 $b.rollout.log | head -40
[[ -x $b/rollout ]] || exit 1
export PYTHONDONTWRITEBYTECODE=1
cd $repo/scripts/examples
for mode in ${MODES:-static moving}; do
    flag=$([[ $mode == moving ]] && echo --moving || true)
    $py -u kh_twin.py $b/rollout $out/kh_twin_cpp_${mode}_$n.npz $work $flag --n=$n --starts=$starts --maxiter=$maxiter
done
read steps dt < <($py -c "import numpy as np; d=np.load('$out/kh_twin_cpp_static_$n.npz'); print(int(d['steps']), repr(float(d['dt'])))")
cd $work && git -C $luna log --oneline -1
PYTHONPATH=$luna:$luna/src XLA_PYTHON_CLIENT_PREALLOCATE=false $py -u -m scripts.examples.autodiff.kh_twin nx=$n steps=$steps dt=$dt \
    starts=$starts maxiter=$maxiter out=$out/kh_twin_luna_$n.npz
