#!/bin/bash
# A voronoi-cpp pin move on one A100, binaries of pin_bits_a100.sh: stuck steps, rebuilds and local repairs in moving
# Sedov 128^3 and 256^3 to t_end (two runs each), then speed (40 steps, --memory) static and moving, old and new
# alternating, three rounds. SM clock sampled every second into clock-<job>.csv.
#   sbatch --export=ALL,OLD=<worktree>,NEW=<worktree> scripts/pin_speed_a100.sh   (TESTING.md)
#SBATCH -J pin_speed
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=01:30:00
set -uo pipefail
source /etc/profile.d/modules.sh; module purge; module load gcc/14 cuda/13.0 cmake hdf5-serial
nvidia-smi --query-gpu=name,clocks.sm,clocks.applications.graphics,clocks.max.sm,clocks.applications.memory --format=csv
new="${NEW:?NEW: the worktree with the new pin}"
old="${OLD:?OLD: the worktree with the old pin}"
declare -A bin=([new]="${NEW_BUILD:-$new/build-cuda}" [old]="${OLD_BUILD:-$old/build-cuda}")  # build dirs (e.g. on /ptmp)
p="${OUT:-$new/build-pin-proof}"; mkdir -p "$p"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,utilization.gpu,temperature.gpu,power.draw --format=csv -l 1 > $p/clock-$SLURM_JOB_ID.csv &
smi=$!
keep="^(run|phases_ms_median|throughput_M_per_s|memory: phase=steps|memory: phase=peak|mesh_updates|checksum)"

echo "== stuck: moving Sedov to t_end"
for side in 128 256; do
    for round in 1 2; do
        for v in old new; do
            d=${!v}
            echo "-- $v ${side}^3 round $round"
            ${bin[$v]}/hydro_sedov $side jittered --moving --snapshots=1 2>/dev/null | grep -E "^(run|mesh_updates|checksum)"
        done
    done
done

echo "== speed: 40 steps"
for round in 1 2 3; do
    for side in 128 256; do
        for mesh in "" "--moving"; do
            for v in old new; do
                d=${!v}
                echo "-- $v ${side}^3 ${mesh:-static} round $round"
                ${bin[$v]}/hydro_sedov $side jittered $mesh --steps=40 --snapshots=1 --memory 2>/dev/null | grep -E "$keep|^memory"
            done
        done
    done
done
kill $smi
