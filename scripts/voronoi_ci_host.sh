#!/bin/bash
# CI's host steps (.github/workflows/checks.yml) on the NEW luna-cpp tree: gcc 15, host Kokkos (OpenMP), 2 threads, sanitizers
# OFF and ON, lint (check_comments.py + clang-tidy 18, OFF), ctest --parallel 2. Compute nodes have no network: VORONOI names a
# local voronoi-cpp checkout (-DLUNA_VORONOI_SOURCE), COMMIT the commit it must be at (-DLUNA_VORONOI_COMMIT, optional).
#   sbatch --export=ALL,NEW=<tree>,VORONOI=<checkout>,COMMIT=<sha>,BUILD=<dir>,OUT=<dir> scripts/voronoi_ci_host.sh
#SBATCH -J voronoi_ci_host
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=02:30:00
source /etc/profile.d/modules.sh; module purge; module load gcc/15 cmake; module load hdf5-serial 2>/dev/null
export OMP_NUM_THREADS=2 OMP_PROC_BIND=false UBSAN_OPTIONS=halt_on_error=1 ASAN_OPTIONS=detect_leaks=1
repo="${NEW:?NEW: the luna-cpp tree}"
source_args="-DLUNA_VORONOI_SOURCE=${VORONOI:?VORONOI: a local voronoi-cpp checkout} ${COMMIT:+-DLUNA_VORONOI_COMMIT=$COMMIT}"
builds="${BUILD:-$repo}"
p="${OUT:-$builds/voronoi-proof}"; mkdir -p "$p"
cd $repo && git log --oneline -1
for san in OFF ON; do
    b=$builds/build-ci-$san
    cmake -S . -B $b -DCMAKE_BUILD_TYPE=Release -DKokkos_DIR=/u/uli/kokkos-gcc15/lib64/cmake/Kokkos -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DVORONOI_SANITIZERS=$san $source_args > $p/ci-$san.cfg.log 2>&1
    echo "configure $san exit $?: $(cat $b/luna_build_info.txt)"
    cmake --build $b --parallel 18 > $p/ci-$san.build.log 2>&1; echo "build $san exit $?"; grep -E "warning|error" $p/ci-$san.build.log | grep -v "voronoi-cpp" | head -5
    if [[ $san == OFF ]]; then (module load clang/18; bash scripts/lint_cpp.sh $b > $p/ci-$san.lint.log 2>&1; echo "lint exit $?"; grep -E "warning:|error:" $p/ci-$san.lint.log | grep -v "non-user" | head -10); fi
    ctest --test-dir $b --output-on-failure --parallel 2 > $p/ci-$san.ctest.log 2>&1; echo "ctest $san exit $?"; tail -n 3 $p/ci-$san.ctest.log
    grep -m1 "^build:" $b/Testing/Temporary/LastTest.log
done
