#!/bin/bash
# How far a change moves the bits and the accuracy, against a reference build: Sedov 128^3 static and moving, 50 steps
# (scripts/snapshot_diff.py against the first line) and to t_end (steps, GPU s, field L1 by scripts/sedov_field_l1.py).
#   sbatch -p gpu1 -o <logs>/quant-%j.out scripts/hydro_quant_a100.sh <runs file>
# Runs file: one line "label;env;hydro_sedov;flags" per build (env: VAR=value ... or -), the reference first.
# TEND=0 skips the runs to t_end. Needs LUNA on PYTHONPATH for the field L1 (LUNA_DIR, PY).
#SBATCH -J hydro_quant
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:40:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0 cmake hdf5-serial
nvidia-smi --query-gpu=name,clocks.sm,clocks.applications.graphics --format=csv,noheader
out=${OUT:-quant-${SLURM_JOB_ID:-local}}
mkdir -p "$out"
py=${PY:-$HOME/jax-env/bin/python}
export PYTHONPATH="${LUNA_DIR:-/raven/u/uli/LUNA}/src"
scripts="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -n "${HERE:-}" ]] && scripts="$HERE"
ref=""
while IFS=';' read -r label envs bin flags; do
    [[ -z "$label" || "$label" == \#* ]] && continue
    [[ -z "$ref" ]] && ref=$label
    for mesh in static moving; do
        move=""
        [[ "$mesh" == moving ]] && move=--moving
        e=""
        [[ "$envs" != - ]] && e="$envs"
        env $e "$bin" 128 jittered $move $flags --snapshots=1 --steps=50 --out="$out/s50_${mesh}_$label" | grep -E "^checksum" | sed "s/^/$label $mesh 50: /"
        if [[ "${TEND:-1}" != 0 ]]; then
            env $e "$bin" 128 jittered $move $flags --snapshots=1 --out="$out/tend_${mesh}_$label" | grep -E "^(run|checksum)" | sed "s/^/$label $mesh t_end: /"
            "$py" "$scripts/sedov_field_l1.py" "$out/tend_${mesh}_$label" | sed "s/^/$label $mesh t_end: /"
        fi
        if [[ "$label" != "$ref" ]]; then
            "$py" "$scripts/snapshot_diff.py" "$out/s50_${mesh}_$ref" "$out/s50_${mesh}_$label" "$label-vs-$ref $mesh 50 steps"
            [[ "${TEND:-1}" != 0 ]] && "$py" "$scripts/snapshot_diff.py" "$out/tend_${mesh}_$ref" "$out/tend_${mesh}_$label" "$label-vs-$ref $mesh t_end"
        fi
    done
done < "$1"
