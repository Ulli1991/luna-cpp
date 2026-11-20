#!/bin/bash
# LUNA's fixed-grid references for luna-reference-solutions/luna/ (research_notes/fixed_grid_2026-11-19.md) on one A100: the full bits set
# (scripts/luna_grid_bits.py full), Sedov 3D 128^3 to t_end and the 3D sound wave at amplitudes 1e-3 and 1e-6 (scripts/luna_grid_refs.py).
#   sbatch -o <logs>/refs-%j.out scripts/luna_grid_refs_a100.sh <out dir>
#SBATCH -J luna_refs
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=01:00:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/13 cuda/12
py="${LUNA_ENV:-$HOME/jax-env}/bin/python"
here="${SLURM_SUBMIT_DIR:-.}/scripts"
out="$1"
mkdir -p "$out/grid_bits" "$out/sedov3d" "$out/sound3d"
hostname
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm --format=csv,noheader
export PYTHONDONTWRITEBYTECODE=1
cd "$out" || exit 1
"$py" -u "$here/luna_grid_bits.py" "$out/grid_bits" full
"$py" -u "$here/luna_grid_refs.py" "$out/sedov3d" sedov 128
for amp in 1e-3 1e-6; do
    for nx in 64 128 256; do
        "$py" -u "$here/luna_grid_refs.py" "$out/sound3d" sound "$nx" "$amp"
    done
done
