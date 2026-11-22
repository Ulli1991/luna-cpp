#!/bin/bash
# Flux-kernel levers (branch flux-latency) from one binary, each variant an environment setting ("base" = none). Modes:
#   speed  Sedov <side>^3 moving, 40 steps, fenced phases (--phases), ROUNDS rounds, variants alternating
#   tend   Sedov <side>^3 moving to 1e4 yr: steps, GPU s, field L1, checksum
#   ncu    Nsight Compute of the two flux passes at 128^3 (time, registers, warps, DRAM, stalls)
#   sbatch --export=ALL,BIN=<build dir>,OUT=<dir>,VARIANTS="base LUNA_FLUX_BLOCK=128 ..." -o <logs>/fl-%j.out scripts/flux_latency_a100.sh <mode> [side]
#SBATCH -J flux_latency
#SBATCH -p gpudev
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0 cmake hdf5-serial
mode="${1:-speed}" side="${2:-128}"
here="${HERE:-$(cd "$(dirname "$0")" && pwd)}"
py="${PY:-$HOME/jax-env/bin/python}"
export PYTHONPATH="${LUNA_DIR:-/raven/u/uli/LUNA}/src"
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,clocks.sm,clocks.applications.graphics --format=csv,noheader
envs() { [[ "$1" == base ]] && echo "" || echo "$1" | tr '+' ' '; }
case "$mode" in
speed)
    for round in ${ROUNDS:-1 2}; do
        for v in ${VARIANTS:-base}; do
            echo "== $v ${side}^3 round $round"
            env $(envs "$v") "$BIN/hydro_sedov" "$side" jittered --moving --steps=40 --phases | grep -E "^(phases_ms_median|checksum)"
        done
    done
    ;;
tend)
    for v in ${VARIANTS:-base}; do
        echo "== $v ${side}^3 to t_end"
        env $(envs "$v") "$BIN/hydro_sedov" "$side" jittered --moving --snapshots=1 --out="$OUT/tend${side}_${v//[=+]/_}" | grep -E "^(run|checksum|fallbacks)"
        "$py" "$here/sedov_field_l1.py" "$OUT/tend${side}_${v//[=+]/_}" | sed "s/^/$v /"
    done
    ;;
ncu)
    module load nsight_compute/2026
    ncu=/mpcdf/soft/SLE_15/packages/x86_64/nsight_compute/2026/ncu
    m=gpu__time_duration.sum,launch__registers_per_thread,launch__block_size,sm__warps_active.avg.pct_of_peak_sustained_active
    m+=,dram__bytes_read.sum,dram__bytes_write.sum,smsp__inst_executed.sum,sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active
    m+=,smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio,l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum
    dcgmi profile --pause 2>/dev/null
    trap 'dcgmi profile --resume 2>/dev/null' EXIT
    for v in ${VARIANTS:-base}; do
        echo "== ncu $v"
        env $(envs "$v") "$ncu" --target-processes all --clock-control none --kernel-name-base demangled --kernel-name "regex:MovingFlux" \
            --launch-skip 40 --launch-count 2 --metrics $m --csv --print-units base "$BIN/hydro_sedov" 128 jittered --moving --steps=14 2>/dev/null |
            grep -E '^"' | awk -F'","' 'NR>1 {gsub(/"/,"",$15); k=($5~/lambda\(int, long long/)?"fluxes":"update"; print $1"\" "k" "$13"="$15}' | tr -d '"'
    done
    ;;
esac
