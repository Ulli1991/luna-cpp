#!/bin/bash
# LUNA's fixed-grid Sedov 3D throughput on one A100: whole steps per wall second over 10-step windows, snapshots and diagnostics off.
#   sbatch -o <logs>/luna-%j.out scripts/luna_speed_a100.sh [sides] [kernels]
# Each line: side, kernel, precision, the perf lines (cells x steps / wall s of the window); the first window holds the compile.
#SBATCH -J luna_speed
#SBATCH -p gpudev
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/13 cuda/12
luna_dir="${LUNA_DIR:-/raven/u/uli/LUNA}"
luna_env="${LUNA_ENV:-$HOME/jax-env}"
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm --format=csv,noheader
(cd "$luna_dir" && git log --oneline -1)
for side in ${1:-128 256}; do
    for kernel in ${2:-xla pallas}; do
        for x64 in 0 1; do
            echo "== LUNA ${side}^3 kernel=$kernel x64=$x64"
            (cd "$luna_dir" && timeout 240 "$luna_env/bin/python" -u -m scripts.validation.runs.sedov3d_run nx="$side" ORDER=1 LIMITER=0 riemann_solver=hllc \
                enable_x64=$x64 hydro_kernel=$kernel write_snapshots=0 enable_diagnostics=0 max_steps=${STEPS:-80} output_cadence=10 \
                output_dir="/tmp/luna_speed_$side" 2>&1 | grep -E "perf =|Performance|Total steps|Error|error|refuse" | tail -9)
        done
    done
done
