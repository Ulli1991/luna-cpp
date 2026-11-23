#!/bin/bash
# Reruns the built derivative test unbuffered (to see where it stops): sbatch -o <log> scripts/derivatives_rerun_a100.sh [args]
#SBATCH -J deriv_rerun
#SBATCH -p gpudev
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=60G
#SBATCH --time=00:10:00
set -uo pipefail
source /etc/profile.d/modules.sh; module purge; module load gcc/14 cuda/13.0 cmake hdf5-serial
b=${BUILD:-/ptmp/uli/luna-cpp-ad/build-ad-cuda}
stdbuf -o0 -e0 $b/hydro_derivatives_test "$@" 2>&1 | grep -vE "^  ok"
