#!/bin/bash
# Precision options on one A100: GPU time and error to t_end (docs/measurements.md, "Precision options").
#   sbatch --export=ALL,BIN=<build>,OUT=<dir> -o <logs>/prec-%j.out scripts/hydro_precision_a100.sh <sedov|waves> [presets]
# sedov: 128^3 static and moving to 1e4 yr, field L1 against Sedov-Taylor; waves: the 3D vortex, sound at amp 1e-3 and 1e-6 (nx 64).
#SBATCH -J hydro_precision
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
mode="$1" presets="${2:-double grad32 grad16 single mixed low}"
here="${HERE:-/raven/u/uli/luna-cpp-hspeed/scripts}"
py="${PY:-$HOME/jax-env/bin/python}"
export PYTHONPATH="${LUNA_DIR:-/raven/u/uli/LUNA}/src"
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm --format=csv,noheader
case "$mode" in
sedov)
    for p in $presets; do
        for m in --moving ""; do
            t=${m:-static}; t=${t#--}
            echo "== sedov 128 $t precision=$p"
            "$BIN/hydro_sedov" 128 jittered $m --precision=$p --snapshots=1 --memory --out="$OUT/sedov_${t}_$p" | grep -E "^(run|memory: phase=steps)"
            "$py" "$here/sedov_field_l1.py" "$OUT/sedov_${t}_$p" &
        done
    done
    wait
    ;;
waves)
    for p in $presets; do
        for m in --moving ""; do
            t=${m:-static}; t=${t#--}
            "$BIN/hydro_wave3d" vortex 64 $m --precision=$p --out="$OUT/vortex_${t}_$p.npy" | grep "^ours3d" | sed "s/^/$p /"
            "$py" "$here/norm3d.py" vortex "$OUT/vortex_${t}_$p.npy" 64 | tail -2 | sed "s/^/vortex $t $p: /"
            for amp in 1e-3 1e-6; do
                "$BIN/hydro_wave3d" sound 64 $m --amp=$amp --precision=$p --out="$OUT/sound${amp}_${t}_$p.npy" | grep "^ours3d" | sed "s/^/$p amp=$amp /"
                "$py" "$here/sound3d_l1.py" "$OUT/sound${amp}_${t}_$p.npy" $amp
            done
        done
    done
    ;;
esac
