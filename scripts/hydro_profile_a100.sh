#!/bin/bash
# One moving step of the hydro, kernel by kernel, on one A100 (docs/measurements.md, "Hydro step profile").
#   sbatch --export=ALL,BIN=<dir with hydro_sedov, hydro_wave3d>,OUT=<dir> -o <logs>/prof-%j.out scripts/hydro_profile_a100.sh <nsys|ncu> <sedov|vortex>
# nsys: kernel and API summaries of 20 steps. ncu: time, DRAM bytes, registers, occupancy of each FiniteVolume launch of two steps.
#SBATCH -J hydro_profile
#SBATCH -p gpudev
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0 cmake hdf5-serial nsight_systems/2026
mode="$1" problem="$2"
mkdir -p "$OUT"
job="${SLURM_JOB_ID:-0}"
NCU=/mpcdf/soft/SLE_15/packages/x86_64/nsight_compute/2026/ncu
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm --format=csv,noheader
case "$problem" in
sedov) run=("$BIN/hydro_sedov" 128 jittered --moving --steps=20) ;;
vortex) run=("$BIN/hydro_wave3d" vortex 128 --moving --max-steps=20) ;;
esac
case "$mode" in
nsys)
    "${run[@]}" | grep -E "^(run|phases|throughput|ours3d)"
    nsys profile -t cuda,nvtx -o "$OUT/nsys_${problem}-$job" -f true "${run[@]}" >/dev/null 2>&1
    nsys stats -r cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum -f csv -o "$OUT/nsys_${problem}-$job" "$OUT/nsys_${problem}-$job.nsys-rep" >/dev/null 2>&1
    ls "$OUT" | grep "nsys_${problem}-$job"
    ;;
ncu)
    M=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,launch__registers_per_thread,launch__block_size,launch__grid_size
    M+=,sm__warps_active.avg.pct_of_peak_sustained_active,launch__occupancy_limit_registers,dram__throughput.avg.pct_of_peak_sustained_elapsed
    M+=,lts__t_bytes.sum,l1tex__t_sector_hit_rate.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
    dcgmi profile --pause
    timeout 700 "$NCU" --target-processes all --clock-control none --kernel-name-base demangled --kernel-name "regex:FiniteVolume" \
        --launch-skip "${SKIP:-60}" --launch-count "${COUNT:-24}" --metrics "$M" --csv --page raw "${run[@]}" >"$OUT/ncu_${problem}-$job.csv" 2>"$OUT/ncu_${problem}-$job.err"
    echo "ncu exit $?"
    dcgmi profile --resume
    ;;
esac
