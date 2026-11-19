#!/bin/bash
# Host builds (gcc 15, Kokkos OpenMP) with LUNA_DERIVATIVE_TESTS=ON, sanitizers OFF and ON, the derivative test in each,
# and CI's lint on the OFF build (check_comments.py, clang-tidy 18 on every configured test).  sbatch scripts/derivatives_host.sh
#SBATCH -J deriv_host
#SBATCH -p interactive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=01:30:00
source /etc/profile.d/modules.sh; module purge; module load gcc/15 cmake; module load hdf5-serial 2>/dev/null
export OMP_NUM_THREADS=2 OMP_PROC_BIND=false UBSAN_OPTIONS=halt_on_error=1 ASAN_OPTIONS=detect_leaks=1
repo=/raven/u/uli/luna-cpp-ad
cd $repo && git log --oneline -1
for san in OFF ON; do
    b=$repo/build-ad-host-$san
    cmake -S . -B $b -DCMAKE_BUILD_TYPE=Release -DKokkos_DIR=/u/uli/kokkos-gcc15/lib64/cmake/Kokkos -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DVORONOI_SANITIZERS=$san -DLUNA_DERIVATIVE_TESTS=ON > $b.cfg.log 2>&1 || { echo "configure $san failed"; tail -20 $b.cfg.log; exit 1; }
    cmake --build $b --parallel 8 --target hydro_derivatives_test > $b.build.log 2>&1; echo "build $san exit $?"
    grep -E "warning|error" $b.build.log | grep -v "extern/" | head -40
    [[ -x $b/hydro_derivatives_test ]] || exit 1
    start=$(date +%s); $b/hydro_derivatives_test; echo "test $san exit $? ($(( $(date +%s) - start )) s)"
done
(module load clang/18; python3 scripts/check_comments.py src; echo "comments exit $?"; clang-tidy -p $repo/build-ad-host-OFF src/test/hydro_derivatives_test.cpp > $repo/build-ad-host-OFF.lint.log 2>&1; echo "clang-tidy exit $?"; grep -E "warning:|error:" $repo/build-ad-host-OFF.lint.log | grep -v "non-user" | head -20)
