#!/bin/bash
# Phase 1 of the fixed grid (research_notes/fixed_grid_2026-11-19.md): the CUDA kernel, the Kokkos kernel and LUNA's production step
# (Pallas, FP64) on LUNA's Sedov 3D, in one job on one A100, with the node and SM clock logged every second.
#   sbatch --export=ALL,BIN=<dir with hydro_sedov>,OUT=<dir> -o <logs>/phase1-%j.out scripts/grid_phase1_a100.sh [speed|ncu|both]
# speed: G cell updates/s, median step after warm-up, sides SIDES (default 128 256 384), limiters minmod and MC, both kernels and LUNA.
# ncu: per kernel at 256^3 (two steps each): time, DRAM bytes, registers, occupancy, spills, FP64 pipe, instructions, stalls; ours and LUNA's.
#SBATCH -J grid_phase1
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=01:00:00
set -uo pipefail
source /etc/profile.d/modules.sh
mode="${1:-both}"
BIN="${BIN:?BIN: the directory with hydro_sedov}"
OUT="${OUT:?OUT: the output directory}"
SIDES="${SIDES:-128 256 384}"
STEPS="${STEPS:-60}"
LUNA_DIR="${LUNA_DIR:-/raven/u/uli/LUNA}"
LUNA_ENV="${LUNA_ENV:-$HOME/jax-env}"
NCU=/mpcdf/soft/SLE_15/packages/x86_64/nsight_compute/2026/ncu
job="${SLURM_JOB_ID:-0}"
mkdir -p "$OUT"
echo "node $(hostname) job $job"
nvidia-smi --query-gpu=name,uuid,clocks.sm,clocks.max.sm,clocks.applications.graphics,power.limit --format=csv,noheader
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader -l 1 >"$OUT/clock-$job.csv" &
logger=$!
trap 'kill $logger 2>/dev/null' EXIT
echo "luna-cpp $(git -C "$BIN/.." log --oneline -1 2>/dev/null | cut -c1-70)"
echo "LUNA $(git -C "$LUNA_DIR" log --oneline -1 | cut -c1-70)"

ours() {  # side kernel limiter [extra flags]
    (module purge; module load gcc/14 cuda/13.0 hdf5-serial
     "$BIN/hydro_sedov" "$1" grid --grid --kernel="$2" --limiter="$3" --steps="$STEPS" "${@:4}" 2>&1 | grep -E "^(grid|run|conservation|checksum)")
}
luna() {  # side LIMITER
    (module purge; module load gcc/13 cuda/12
     cd "$LUNA_DIR" && PYTHONDONTWRITEBYTECODE=1 timeout 600 "$LUNA_ENV/bin/python" -u -m scripts.validation.runs.sedov3d_run nx="$1" ORDER=1 LIMITER="$2" \
        riemann_solver=hllc enable_x64=1 hydro_kernel=pallas write_snapshots=0 enable_diagnostics=0 max_steps="${LUNA_STEPS:-80}" \
        output_cadence=10 output_dir="$OUT/luna_$1" 2>&1 | grep -E "perf =|Performance|Total steps|Error|error|refuse" | tail -9)
}

if [[ "$mode" == speed || "$mode" == both ]]; then
    for side in $SIDES; do
        for lim in minmod mc; do
            l=$([[ $lim == mc ]] && echo 1 || echo 0)
            echo "== ${side}^3 limiter=$lim"
            for kernel in cuda kokkos; do
                echo "-- ours kernel=$kernel $(date +%T)"
                ours "$side" "$kernel" "$lim"
            done
            echo "-- LUNA pallas FP64 LIMITER=$l $(date +%T)"
            luna "$side" "$l"
        done
    done
fi

if [[ "$mode" == ncu || "$mode" == both ]]; then
    M=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,launch__registers_per_thread,launch__block_size,launch__grid_size
    M+=,sm__warps_active.avg.pct_of_peak_sustained_active,launch__occupancy_limit_registers,sm__maximum_warps_per_active_cycle_pct
    M+=,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_bytes.sum,l1tex__t_sector_hit_rate.pct
    M+=,l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum
    M+=,sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_fp64.sum,smsp__inst_executed.sum
    M+=,smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,smsp__sass_thread_inst_executed_op_dadd_pred_on.sum,smsp__sass_thread_inst_executed_op_dmul_pred_on.sum
    M+=,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct
    M+=,smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_barrier_per_warp_active.pct,sm__cycles_elapsed.avg.per_second
    dcgmi profile --pause 2>&1 | tail -1
    for kernel in cuda kokkos; do
        (module purge; module load gcc/14 cuda/13.0 hdf5-serial
         timeout 900 "$NCU" --target-processes all --clock-control none --kernel-name-base demangled --kernel-name "regex:step_kernel|grid_step|ParallelReduce" \
            --launch-skip 2 --launch-count 4 --metrics "$M" --csv --page raw "$BIN/hydro_sedov" 256 grid --grid --kernel=$kernel --limiter=minmod --steps=4 \
            >"$OUT/ncu_ours_$kernel-$job.csv" 2>"$OUT/ncu_ours_$kernel-$job.err"; echo "ncu ours $kernel exit $?")
    done
    (module purge; module load gcc/13 cuda/12
     cd "$LUNA_DIR" && PYTHONDONTWRITEBYTECODE=1 timeout 1200 "$NCU" --target-processes all --clock-control none --kernel-name-base demangled \
        --launch-skip 60 --launch-count 40 --metrics "$M" --csv --page raw \
        "$LUNA_ENV/bin/python" -u -m scripts.validation.runs.sedov3d_run nx=256 ORDER=1 LIMITER=0 riemann_solver=hllc enable_x64=1 hydro_kernel=pallas \
        write_snapshots=0 enable_diagnostics=0 max_steps=8 output_cadence=10 output_dir="$OUT/luna_ncu" \
        >"$OUT/ncu_luna-$job.csv" 2>"$OUT/ncu_luna-$job.err"; echo "ncu LUNA exit $?")
    dcgmi profile --resume 2>&1 | tail -1
fi
