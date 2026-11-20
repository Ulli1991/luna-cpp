#!/bin/bash
# The fixed grid's A100 checks (research_notes/fixed_grid_2026-11-19.md): the CUDA build's whole ctest, grid_test in the bit-exact build
# (LUNA_BIT_EXACT=ON) on the stored cases and, given FULL, on luna-reference-solutions' larger ones (luna/grid_bits).
#   sbatch [--export=ALL,FULL=<dir of case directories>] -o <logs>/ctest-%j.out scripts/grid_ctest_a100.sh
#SBATCH -J grid_ctest
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=01:30:00
set -uo pipefail
source /etc/profile.d/modules.sh
root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
hostname
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm --format=csv,noheader
echo "luna-cpp $(git -C "$root" log --oneline -1 | cut -c1-70) $(git -C "$root" status --porcelain src | wc -l) changed files in src"
"$root/scripts/grid_build.sh" cuda all
"$root/scripts/grid_build.sh" cuda-exact grid_test
module purge
module load gcc/14 cuda/13.0 cmake hdf5-serial
ctest --test-dir "$root/build-grid-cuda" --output-on-failure 2>&1 | grep -E "Test +#|tests passed|FAIL|Total Test"
echo "== grid_test, default build (FMA)"
"$root/build-grid-cuda/grid_test" | grep -E "FAIL|LUNA 3d-mc|kernel|cases"
echo "== grid_test, bit-exact build"
"$root/build-grid-cuda-exact/grid_test" | grep -E "FAIL|LUNA|kernel|cases"
if [[ -n "${FULL:-}" ]]; then
    echo "== bit-exact build on $FULL"
    "$root/build-grid-cuda-exact/grid_test" "$FULL" | grep -E "FAIL|LUNA|cases"
    echo "== default build on $FULL"
    "$root/build-grid-cuda/grid_test" "$FULL" | grep -E "FAIL|LUNA|cases"
fi
