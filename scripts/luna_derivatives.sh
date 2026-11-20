#!/bin/bash
# LUNA's JAX derivatives for hydro_derivatives_test's comparison (scripts/luna_derivatives.py), on a CPU node with LUNA's environment.
#   sbatch -o <logs>/luna-deriv-%j.log scripts/luna_derivatives.sh <out.hpp>
#SBATCH -J luna_deriv
#SBATCH -p general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -uo pipefail
source /etc/profile.d/modules.sh; module purge; module load gcc/13
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
repo=${REPO:-/raven/u/uli/luna-cpp-ad}
luna_env="${LUNA_ENV:-$HOME/jax-env}"
out="${1:?out.hpp}"
work=/ptmp/uli/luna-cpp-ad/luna-deriv-${SLURM_JOB_ID:-0}  # LUNA's working files, if any, off /raven/u
mkdir -p "$work" && cd "$work"
"$luna_env/bin/python" -u "$repo/scripts/luna_derivatives.py" "$out" sod:64:1:5 sod:64:0:5 sound-right:64:1:5 sound-right:64:0:5 sod:64:1:5:1e-3 sod:64:2:5:1e-3 sound-right:64:2:5
