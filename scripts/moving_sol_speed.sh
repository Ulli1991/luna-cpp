#!/bin/bash
# Speed and bits of moving-sol levers against main in one A100 job: Sedov <side>^3 moving, 40 steps, fenced phases,
# ROUNDS rounds, variants alternating. A variant is name=<build dir>[@ENV=VALUE+ENV=VALUE].
#   sbatch --export=ALL,VARIANTS="main=<dir> chunk18=<dir>@LUNA_UPDATE_CHUNK=18",ROUNDS="1 2 3" -o <logs>/speed-%j.out scripts/moving_sol_speed.sh <side>
#SBATCH -J moving_sol_speed
#SBATCH -p gpudev
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0 hdf5-serial
side="${1:-128}"
nvidia-smi --query-gpu=name,clocks.sm,clocks.applications.graphics --format=csv,noheader
for round in ${ROUNDS:-1 2 3}; do
    for v in $VARIANTS; do
        name="${v%%=*}" rest="${v#*=}" dir="${rest%%@*}" envs=""
        [[ "$rest" == *@* ]] && envs="$(echo "${rest#*@}" | tr '+' ' ')"
        echo "== $name ${side}^3 round $round"
        env $envs "$dir/hydro_sedov" "$side" jittered --moving --steps=40 --phases --memory | grep -E "^(phases_ms_median|checksum|memory: phase=steps)"
    done
done
