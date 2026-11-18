#!/bin/bash
# Local-memory traffic, registers, warps active and DRAM of the hydro kernels (Nsight Compute), Sedov 128^3 jittered,
# static and moving, 14 steps; 12 hydro kernel launches from step 11 on are measured (docs/tried.md, "local memory").
#   sbatch -p gpu1 -o <logs>/ncu-%j.out scripts/hydro_ncu_local_a100.sh <label=hydro_sedov> ...
# Summary per kernel by scripts/hydro_ncu_summary.py.
#SBATCH -J hydro_ncu
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:30:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0 cmake hdf5-serial nsight_compute/2026
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -n "${HERE:-}" ]] && here="$HERE"
nvidia-smi --query-gpu=name,clocks.sm,clocks.applications.graphics --format=csv,noheader
out=${OUT:-ncu-${SLURM_JOB_ID:-local}}
mkdir -p "$out"
ncu=${NCU:-/mpcdf/soft/SLE_15/packages/x86_64/nsight_compute/2026/ncu}
metrics=gpu__time_duration.sum,launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active
metrics+=,l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum,dram__bytes_read.sum,dram__bytes_write.sum
# The node's DCGM profiling holds the counters ncu needs: paused for the job, resumed on exit.
dcgmi profile --pause 2>/dev/null
trap 'dcgmi profile --resume 2>/dev/null' EXIT
for item in "$@"; do
    label="${item%%=*}" bin="${item#*=}"
    for mesh in static moving; do
        move=""
        skip=50
        [[ "$mesh" == moving ]] && move=--moving && skip=70
        "$ncu" --target-processes all --clock-control none --kernel-name-base demangled \
            --kernel-name "regex:Sweep|FaceFlux|MovingFlux|FaceStep|CellStep|TimeStep" \
            --launch-skip $skip --launch-count 12 --metrics $metrics --csv --print-units base \
            "$bin" 128 jittered --steps=14 $move > "$out/$label-$mesh.csv" 2> "$out/$label-$mesh.err"
        echo "== $label $mesh: $(grep -c '"' "$out/$label-$mesh.csv") csv lines"
    done
done
echo "== summary"
for f in "$out"/*.csv; do python3 "$here/hydro_ncu_summary.py" 2097152 "$f"; done
