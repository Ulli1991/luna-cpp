#!/bin/bash
# The head-to-head of phase 1 and of every lever (research_notes/fixed_grid_2026-11-19.md, section 9): luna-cpp's CUDA and Kokkos kernels
# of one or more builds and LUNA's production step on LUNA's Sedov 3D, in one job on one A100, in alternating rounds, node and clock logged.
#   sbatch --export=ALL,BINS="<name>=<dir with hydro_sedov> ...",EXACT=<dir: a LUNA_BIT_EXACT hydro_sedov>,OUT=<dir on /ptmp>[,NCU=1] \
#          -o <logs>/compare-%j.out scripts/grid_compare_a100.sh
# Matched in every row: the problem and its initial bits (--ic from LUNA's own state), N, limiter (LIMITER 0 minmod, 1 MC), HLLC, FP64, no
# floors, BOUNDARY 1, CFL 0.8, 100 steps each with its own CFL dt, the median step after 10 warm-up steps, the timed region (the whole step
# with its time step and one host sync, no output, no allocation), the node, GPU and job. Not matched: the kernels themselves (the point).
# Agreement: the 128^3 end states of every build and kernel against LUNA's (rounding), the bit-exact build against LUNA (bits).
# NCU=1: ncu on every build's two kernels at 256^3 (time, DRAM, local memory, registers, occupancy, FP64 pipe, instructions, stalls).
#SBATCH -J grid_compare
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=01:30:00
set -uo pipefail
source /etc/profile.d/modules.sh
BINS="${BINS:?}" OUT="${OUT:?}"
SIDES="${SIDES:-128 256 384}" ROUNDS="${ROUNDS:-3}" STEPS="${STEPS:-100}" LIMITERS="${LIMITERS:-minmod mc}"
here="${SLURM_SUBMIT_DIR:-.}/scripts"
py="${LUNA_ENV:-$HOME/jax-env}/bin/python"
NCU_BIN=/mpcdf/soft/SLE_15/packages/x86_64/nsight_compute/2026/ncu
job="${SLURM_JOB_ID:-0}"
mkdir -p "$OUT"
echo "node $(hostname) job $job"
nvidia-smi --query-gpu=name,uuid,clocks.sm,clocks.max.sm,clocks.applications.graphics --format=csv,noheader
nvidia-smi --query-gpu=timestamp,clocks.sm,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader -l 1 >"$OUT/clock-$job.csv" &
logger=$!
trap 'kill $logger 2>/dev/null' EXIT
for b in $BINS; do echo "build ${b%%=*}: ${b#*=}"; done
luna() {  # N LIMITER [out=...]
    (module purge; module load gcc/13 cuda/12
     cd "$OUT" && PYTHONDONTWRITEBYTECODE=1 "$py" -u "$here/luna_grid_time.py" "$1" "$2" steps="$STEPS" warm=10 "${@:3}" 2>&1 | grep -E "^luna_|Error|error")
}
ours() {  # binary kernel N limiter ic [extra]
    (module purge; module load gcc/14 cuda/13.0 hdf5-serial
     "$1" "$3" grid --grid --kernel="$2" --limiter="$4" --ic="$5" --steps="$STEPS" --warm=10 "${@:6}" 2>&1 | grep -E "^(run|checksum)" | tr '\n' ' ')
}
compare() {
    (module purge; module load gcc/13; "$py" "$here/grid_compare.py" "$@")
}
for side in $SIDES; do
    for lim in $LIMITERS; do
        l=$([[ $lim == mc ]] && echo 1 || echo 0)
        save=()
        [[ $side == 128 ]] && save=(out="$OUT/luna_${side}_$lim.npy")
        first=$(luna "$side" "$l" "${save[@]}")
        ic=$(echo "$first" | sed -n 's/^luna_ic: rho=\([^ ]*\) p_bg=[^ ]* E_bg=\([^ ]*\) E_blast=\([^ ]*\).*/\1,\2,\3/p')
        echo "== ${side}^3 $lim ic=$ic"
        echo "round 1 $(echo "$first" | grep luna_run)"
        for round in $(seq 1 "$ROUNDS"); do
            for b in $BINS; do
                for kernel in cuda kokkos; do
                    extra=()
                    [[ $side == 128 && $round == 1 ]] && extra=(--out="$OUT/${b%%=*}_${kernel}_${side}_$lim.npy")
                    echo "round $round ${b%%=*} $kernel: $(ours "${b#*=}/hydro_sedov" $kernel "$side" $lim "$ic" "${extra[@]}")"
                done
            done
            [[ $round -gt 1 ]] && echo "round $round $(luna "$side" "$l" | grep luna_run)"
        done
        if [[ $side == 128 ]]; then
            if [[ -n "${EXACT:-}" ]]; then
                for kernel in cuda kokkos; do
                    echo "exact $kernel: $(ours "$EXACT/hydro_sedov" $kernel "$side" $lim "$ic" --out="$OUT/exact_${kernel}_${side}_$lim.npy")"
                    compare "$OUT/exact_${kernel}_${side}_$lim.npy" "$OUT/luna_${side}_$lim.npy" "exact $kernel vs LUNA $lim"
                done
            fi
            for b in $BINS; do
                for kernel in cuda kokkos; do
                    compare "$OUT/${b%%=*}_${kernel}_${side}_$lim.npy" "$OUT/luna_${side}_$lim.npy" "${b%%=*} $kernel vs LUNA $lim"
                done
                compare "$OUT/${b%%=*}_cuda_${side}_$lim.npy" "$OUT/${b%%=*}_kokkos_${side}_$lim.npy" "${b%%=*} cuda vs kokkos $lim"
            done
        fi
    done
done
if [[ -n "${NCU:-}" ]]; then
    M=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,launch__registers_per_thread,launch__block_size,launch__grid_size
    M+=,sm__warps_active.avg.pct_of_peak_sustained_active,sm__maximum_warps_per_active_cycle_pct,dram__throughput.avg.pct_of_peak_sustained_elapsed
    M+=,lts__t_bytes.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum
    M+=,sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.sum
    M+=,smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,smsp__sass_thread_inst_executed_op_dadd_pred_on.sum,smsp__sass_thread_inst_executed_op_dmul_pred_on.sum
    M+=,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct
    M+=,smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_barrier_per_warp_active.pct,sm__cycles_elapsed.avg.per_second
    dcgmi profile --pause 2>&1 | tail -1
    for b in $BINS; do
        for kernel in cuda kokkos; do
            (module purge; module load gcc/14 cuda/13.0 hdf5-serial
             timeout 900 "$NCU_BIN" --target-processes all --clock-control none --kernel-name-base demangled --kernel-name "regex:step_kernel|predictor_kernel|StepTeam|ParallelFor" \
                --launch-skip 4 --launch-count 6 --metrics "$M" --csv --page raw "${b#*=}/hydro_sedov" 256 grid --grid --kernel=$kernel --limiter=minmod \
                --steps=5 >"$OUT/ncu_${b%%=*}_$kernel-$job.csv" 2>"$OUT/ncu_${b%%=*}_$kernel-$job.err"; echo "ncu ${b%%=*} $kernel exit $?")
        done
    done
    dcgmi profile --resume 2>&1 | tail -1
    (module purge; module load gcc/13; "$py" "$here/ncu_table.py" 16777216 "$OUT"/ncu_*-"$job".csv)
fi
