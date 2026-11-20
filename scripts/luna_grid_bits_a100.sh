#!/bin/bash
# LUNA's fixed-grid references for the grid bits test (scripts/luna_grid_bits.py) on one A100.
#   sbatch -o <logs>/bits-%j.out scripts/luna_grid_bits_a100.sh <out dir> <ci|full> [pallas|xla]
#SBATCH -J luna_bits
#SBATCH -p gpudev
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=100G
#SBATCH --time=00:15:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/13 cuda/12
luna_env="${LUNA_ENV:-$HOME/jax-env}"
hostname
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm --format=csv,noheader
script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/luna_grid_bits.py"
[[ -f "$script" ]] || script="${SLURM_SUBMIT_DIR:-.}/scripts/luna_grid_bits.py"
mkdir -p "$1"
cd "$1" && PYTHONDONTWRITEBYTECODE=1 "$luna_env/bin/python" -u "$script" "$1" "$2" ${3:+kernel=$3}
