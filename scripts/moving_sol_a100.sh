#!/bin/bash
# Speed of light of the moving step's hydro kernels (branch moving-sol): Nsight Compute, full set, of every FiniteVolume
# kernel of one moving Sedov step (about step 10), then the per-kernel table (scripts/moving_sol_table.py).
#   sbatch --export=ALL,BIN=<build dir>,OUT=<dir on /ptmp> -o <logs>/sol-%j.out scripts/moving_sol_a100.sh <side> [label]
#SBATCH -J moving_sol
#SBATCH -p gpudev
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0 hdf5-serial nsight_compute/2026
side="${1:-128}" label="${2:-main}"
here="$(cd "$(dirname "$0")" && pwd)"
[[ -n "${HERE:-}" ]] && here="$HERE"
ncu=/mpcdf/soft/SLE_15/packages/x86_64/nsight_compute/2026/ncu
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,clocks.sm,clocks.applications.graphics --format=csv,noheader
rep="$OUT/ncu_${label}_${side}"
dcgmi profile --pause 2>/dev/null
trap 'dcgmi profile --resume 2>/dev/null' EXIT
# Eight FiniteVolume kernels a moving step (sweep, time step, keep_centroids, fluxes and update twice, drift): skip nine steps.
"$ncu" --set full --target-processes all --clock-control none --kernel-name-base demangled --kernel-name "regex:FiniteVolume" \
    --launch-skip 72 --launch-count 8 -f -o "$rep" "$BIN/hydro_sedov" "$side" jittered --moving --steps=12 > "$rep.out" 2>&1
echo "ncu exit $?"
"$ncu" --import "$rep.ncu-rep" --page raw --csv > "$rep.raw.csv" 2>/dev/null
"${PY:-$HOME/jax-env/bin/python}" "$here/moving_sol_table.py" "$rep.raw.csv" "$side"
