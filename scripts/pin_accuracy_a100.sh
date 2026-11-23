#!/bin/bash
# A voronoi-cpp pin move: field errors, old against new, with pin_bits_a100.sh's builds (the wave target added where missing):
# Sedov 128^3 static and moving to 1e4 yr (field L1 of rho, v_r, P), the 3D vortex and sound waves at 1e-3 and 1e-6 (nx 64),
# default precision, through hydro_precision_a100.sh. Snapshots (about 0.7 GB a tree) go to OUT on /ptmp.
#   sbatch --export=ALL,NEW_BUILD=<dir>,OLD_BUILD=<dir>,OUT=<dir> -o <log> scripts/pin_accuracy_a100.sh
#SBATCH -J pin_acc
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:45:00
set -uo pipefail
source /etc/profile.d/modules.sh; module purge; module load gcc/14 cuda/13.0 cmake hdf5-serial
here=$(cd "$(dirname "$0")" && pwd)
[[ -f $here/hydro_precision_a100.sh ]] || here=/raven/u/uli/luna-cpp-ad/scripts
declare -A B=([new]="${NEW_BUILD:?NEW_BUILD}" [old]="${OLD_BUILD:?OLD_BUILD}")
for t in old new; do
    cmake --build ${B[$t]} --parallel 16 --target luna_hydro_sedov luna_hydro_wave3d > "$OUT-$t.build.log" 2>&1; echo "build $t exit $?"
done
for t in old new; do
    for mode in sedov waves; do
        echo "#### $t $mode"
        HERE=$here BIN=${B[$t]} OUT=$OUT/acc-$t bash $here/hydro_precision_a100.sh $mode default
    done
done
