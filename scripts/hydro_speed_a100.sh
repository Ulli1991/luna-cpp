#!/bin/bash
# Hydro speed and precision options on one A100 (research_notes/hydro_speed_2026-11-04.md). MAIN and FIX are two CUDA build directories:
#   MAIN=<build> FIX=<build> sbatch --export=ALL,MAIN=...,FIX=... -p gpudev -o <logs>/speed-%j.out scripts/hydro_speed_a100.sh <out> <mode>
# The modes are the cases below; PRECISIONS selects the options.
#SBATCH -J hydro_speed
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
out="$(realpath -m "${1:-speed-runs}")"
mode="${2:-speed128}"
mkdir -p "$out"
nvidia-smi -L
all="default double grad32 grad16 mixed low"

speed() {  # speed <build> <side> [flags]: the phase lines of a 40-step run
    local build="$1" side="$2"
    shift 2
    "$build/hydro_sedov" "$side" jittered --steps=40 --memory "$@" | grep -E "^(mesh|run|phases|throughput|memory|conservation|fallbacks)"
}

case "$mode" in
speed128 | speed256)
    side=128
    options="${PRECISIONS:-$all}"
    rounds="1 2"
    if [[ "$mode" == speed256 ]]; then
        side=256
        options="${PRECISIONS:-mixed low}"
        rounds="1"
    fi
    # Particle order: Morton (default) and as generated (--input-order: lattice rows).
    for round in $rounds; do
        for mesh in "" "--moving"; do
            for order in "" "--input-order"; do
                for variant in MAIN FIX; do
                    echo "== $variant ${side}^3 ${mesh:-static} ${order:-morton} round $round"
                    speed "${!variant}" $side $mesh $order
                done
            done
        done
    done
    for round in $rounds; do
        for p in $options; do
            for mesh in "" "--moving"; do
                for loop in ${UPDATES:-face}; do
                    echo "== FIX ${side}^3 ${mesh:-static} precision=$p update=$loop round $round"
                    speed "$FIX" $side $mesh --precision=$p --update=$loop
                done
            done
        done
    done
    ;;
bits)
    # MAIN and FIX, 50 steps at 128^3: hashes of the final U, x and volume (scripts/snapshot_hash.sh).
    # An old tree takes MAIN_FLAGS empty, FIX_FLAGS=--precision=double.
    for variant in MAIN FIX; do
        extra="${variant}_FLAGS"
        for flags in "" "--moving" "--order=0" "--update=cell --moving" "--riemann=hll --moving"; do
            tag="$(echo "$flags" | tr -d ' -=')"
            "${!variant}/hydro_sedov" 128 jittered $flags ${!extra:-} --snapshots=1 --steps=50 --out="$out/bits_${tag:-static}_$variant" > /dev/null
        done
    done
    bash scripts/snapshot_hash.sh "$out"/bits_* | sort
    ;;
ctest)
    ctest --test-dir "$FIX" --output-on-failure | tail -5
    ;;
esac
