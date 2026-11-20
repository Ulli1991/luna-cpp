#!/bin/bash
# The rough step's tangent against central differences for h = 1e-5 .. 1e-9 (hydro_derivatives_test --scan), host build.
#   sbatch -o <logs>/scan-%j.log scripts/derivatives_scan.sh
#SBATCH -J deriv_scan
#SBATCH -p interactive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:40:00
source /etc/profile.d/modules.sh; module purge; module load gcc/15 cmake; module load hdf5-serial 2>/dev/null
export OMP_NUM_THREADS=8 OMP_PROC_BIND=false
repo=/raven/u/uli/luna-cpp-ad
b=${BUILD:-/ptmp/uli/luna-cpp-ad}/build-ad-scan
cd $repo
cmake -S . -B $b -DCMAKE_BUILD_TYPE=Release -DKokkos_DIR=/u/uli/kokkos-gcc15/lib64/cmake/Kokkos -DLUNA_DERIVATIVE_TESTS=ON > $b.cfg.log 2>&1
cmake --build $b --parallel 8 --target hydro_derivatives_test > $b.build.log 2>&1; echo "build exit $?"
grep -E "error" $b.build.log | head -20
$b/hydro_derivatives_test --scan | grep -v "^  ok"
