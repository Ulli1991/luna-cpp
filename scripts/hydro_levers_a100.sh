#!/bin/bash
# Hydro speed levers against main in one A100 job (docs/measurements.md, "Hydro speed levers"). Modes: speed (Sedov 40 steps, static and moving),
# bits (hashes: Sedov 128^3 cases, 1D Sod, 3D vortex, KH 2D), tend (moving Sedov to t_end: GPU s, field L1). EXTRA: flags for every run.
#   sbatch --export=ALL,BINS="main=<dir> fix=<dir>@--flag+--flag ...",OUT=<dir> -o <logs>/levers-%j.out scripts/hydro_levers_a100.sh <mode> [side]
#SBATCH -J hydro_levers
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
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm --format=csv,noheader
here="${HERE:-/raven/u/uli/luna-cpp-hspeed/scripts}"
parse() {  # parse name=dir[@flag+flag]: sets name, dir, flags
    name="${1%%=*}" rest="${1#*=}" dir="${rest%%@*}" flags=""
    [[ "$rest" == *@* ]] && flags="$(echo "${rest#*@}" | tr '+' ' ')"
}
case "$mode" in
speed)
    for round in ${ROUNDS:-1 2}; do
        for mesh in ${MESHES:---moving static}; do
            for b in $BINS; do
                parse "$b"
                echo "== $name ${side}^3 $mesh round $round $flags ${EXTRA:-}"
                "$dir/hydro_sedov" "$side" jittered --steps=40 --memory ${mesh/static/} $flags ${EXTRA:-} | grep -E "^(run|phases|throughput|memory: phase=steps)"
            done
        done
    done
    ;;
bits)
    for b in $BINS; do
        parse "$b"
        for run in "" "--moving" "--order=0" "--update=cell --moving" "--riemann=hll --moving" "--precision=double --moving"; do
            tag="$(echo "$run" | tr -d ' -=')"
            "$dir/hydro_sedov" 128 jittered $run $flags ${EXTRA:-} --snapshots=1 --steps=50 --out="$OUT/bits_${tag:-static}_$name" > /dev/null
        done
        for m in "" --moving; do
            t=${m:-static}; t=${t#--}
            [[ -x "$dir/hydro_wave3d" ]] && "$dir/hydro_wave3d" vortex 64 $m $flags --out="$OUT/vortex_${t}_$name.npy" | grep -E "^ours3d" | sed "s/^/$name /"
            [[ -x "$dir/hydro_1d" ]] && "$dir/hydro_1d" sod 1024 $m --repeats=1 | grep -oE "hash=[0-9]+" | sed "s/^/sod $t $name: /"
            [[ -x "$dir/hydro_kh2d" ]] && echo "kh $t $name: $("$dir/hydro_kh2d" 128 $m | grep -E '^(run|totals)' | tr '\n' ' ')"
        done
    done
    bash "$here/snapshot_hash.sh" "$OUT"/bits_* | sort
    for f in "$OUT"/vortex_*.npy; do echo "npy $(basename "$f") $(sha256sum < "$f" | cut -c1-16)"; done
    ;;
tend)
    # Moving Sedov to t_end: GPU seconds and the field L1 (needs LUNA on PYTHONPATH).
    export PYTHONPATH="${LUNA_DIR:-/raven/u/uli/LUNA}/src"
    for b in $BINS; do
        parse "$b"
        echo "== $name ${side}^3 moving to t_end $flags"
        "$dir/hydro_sedov" "$side" jittered --moving --snapshots=1 $flags --out="$OUT/tend_$name" | grep -E "^(run|fallbacks)"
        "${PY:-$HOME/jax-env/bin/python}" "$here/sedov_field_l1.py" "$OUT/tend_$name"
    done
    ;;
esac
