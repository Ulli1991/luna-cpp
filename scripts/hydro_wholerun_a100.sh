#!/bin/bash
# Whole runs: Sedov jittered to 1e4 yr, static and moving, SIDES (128 256), for each binary; the run line gives the
# steps, the wall time (mesh build and output excluded) and gpu_s, so the average cell updates per second of a run.
#   sbatch --export=ALL,BINS="name=<build dir>[@flag+flag] ..." -o <logs>/wholerun-%j.out scripts/hydro_wholerun_a100.sh
#SBATCH -J hydro_wholerun
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=01:00:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0 cmake hdf5-serial
nvidia-smi --query-gpu=name,clocks.sm,clocks.applications.graphics --format=csv,noheader
for side in ${SIDES:-128 256}; do
    for mesh in ${MESHES:-moving static}; do
        for b in $BINS; do
            name="${b%%=*}" rest="${b#*=}" dir="${rest%%@*}" flags=""
            [[ "$rest" == *@* ]] && flags="$(echo "${rest#*@}" | tr '+' ' ')"
            move=""
            [[ "$mesh" == moving ]] && move=--moving
            echo "== $name ${side}^3 $mesh to t_end $flags"
            "$dir/hydro_sedov" "$side" jittered --snapshots=1 $move $flags | grep -E "^(run|mesh_updates|fallbacks|checksum)"
        done
    done
done
