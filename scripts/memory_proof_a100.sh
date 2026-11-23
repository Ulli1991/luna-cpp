#!/bin/bash
# Forward-only peak memory, new against old, with the binaries pin_bits_a100.sh built: Sedov 3D 128^3 static and moving
# (--memory, 40 steps) and KH 2D 256^2 static and moving (--memory, 300 steps). Prints the memory lines of each run.
#   sbatch --export=ALL,NEW_BUILD=<dir>,OLD_BUILD=<dir> -o <log> scripts/memory_proof_a100.sh
#SBATCH -J mem_proof
#SBATCH -p gpudev
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=60G
#SBATCH --time=00:15:00
set -uo pipefail
source /etc/profile.d/modules.sh; module purge; module load gcc/14 cuda/13.0 cmake hdf5-serial
declare -A B=([new]="${NEW_BUILD:?NEW_BUILD}" [old]="${OLD_BUILD:?OLD_BUILD}")
for mesh in "" --moving; do
    for v in old new; do
        echo "-- $v sedov 128^3 [${mesh:-static}]"
        ${B[$v]}/hydro_sedov 128 jittered $mesh --steps=40 --snapshots=1 --memory 2>/dev/null | grep -E "^memory|^checksum"
    done
done
for mesh in "" --moving; do
    for v in old new; do
        echo "-- $v kh 256^2 [${mesh:-static}]"
        ${B[$v]}/hydro_kh2d 256 $mesh --steps=300 --memory 2>&1 | grep -E "memory|hash" | head -6
    done
done
