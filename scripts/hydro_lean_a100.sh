#!/bin/bash
# The hydro-speed levers of 2026-11-18 (docs/tried.md), each against main in one A100 job.
#   sbatch --export=ALL,BINS="main=<build dir> lever=<build dir>@--flag+--flag",OUT=<dir> -o <logs>/lean-%j.out scripts/hydro_lean_a100.sh <mode>
# Modes:
#   speed  Sedov jittered, 40 steps, static and moving, SIDES (128 256), ROUNDS (1 2), the binaries alternating:
#          once with --phases (fenced phase times; trees before the lever print them always) and once without (the wall
#          time of each step: step_wall_ms_median; trees before it print s_per_step only). --memory each time.
#   bits   hashes of Q, x and volume after 50 steps at 128^3: static, moving, order 0, per-cell moving, HLL moving,
#          double moving (scripts/hydro_levers_a100.sh bits, which also hashes the 3D vortex, 1D Sod and KH 2D).
#   ctest  ctest of each binary's build directory.
# The SM clock is sampled every second into $OUT/clock-<job>.csv.
#SBATCH -J hydro_lean
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
mode="${1:-speed}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -n "${HERE:-}" ]] && here="$HERE"
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,clocks.sm,clocks.applications.graphics,clocks.mem --format=csv,noheader
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,utilization.gpu,power.draw --format=csv -l 1 > "$OUT/clock-${SLURM_JOB_ID:-local}.csv" &
smi=$!
parse() {  # parse name=dir[@flag+flag]: sets name, dir, flags
    name="${1%%=*}" rest="${1#*=}" dir="${rest%%@*}" flags=""
    [[ "$rest" == *@* ]] && flags="$(echo "${rest#*@}" | tr '+' ' ')"
}
keep="^(run|phases_ms_median|throughput_M_per_s|memory: phase=(steps|released|hydro_setup)|checksum|fallbacks)"
case "$mode" in
speed)
    for side in ${SIDES:-128 256}; do
        for round in ${ROUNDS:-1 2}; do
            for mesh in ${MESHES:-static moving}; do
                for timing in --phases ""; do
                    for b in $BINS; do
                        parse "$b"
                        echo "== $name ${side}^3 $mesh ${timing:-wall} round $round $flags"
                        move=""
                        [[ "$mesh" == moving ]] && move=--moving
                        "$dir/hydro_sedov" "$side" jittered --steps=40 --memory $timing $move $flags | grep -E "$keep"
                    done
                done
            done
        done
    done
    ;;
bits)
    HERE="$here" bash "$here/hydro_levers_a100.sh" bits
    ;;
ctest)
    for b in $BINS; do
        parse "$b"
        echo "== ctest $name"
        ctest --test-dir "$dir" --output-on-failure 2>&1 | tail -12
    done
    ;;
esac
kill $smi 2>/dev/null
