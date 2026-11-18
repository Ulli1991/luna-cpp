#!/bin/bash
# CI's host steps (.github/workflows/checks.yml) on the NEW worktree: gcc 15, host Kokkos (OpenMP), 2 threads, sanitizers
# OFF and ON, lint (check_comments.py + clang-tidy 18, OFF), ctest --parallel 2.  sbatch --export=ALL,NEW=<worktree> scripts/pin_ci_host.sh
#SBATCH -J pin_ci_host
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=02:30:00
source /etc/profile.d/modules.sh; module purge; module load gcc/15 cmake; module load hdf5-serial 2>/dev/null
export OMP_NUM_THREADS=2 OMP_PROC_BIND=false UBSAN_OPTIONS=halt_on_error=1 ASAN_OPTIONS=detect_leaks=1
repo="${NEW:?NEW: the worktree with the new pin}"
p="${OUT:-$repo/build-pin-proof}"; mkdir -p "$p"
cd $repo && git log --oneline -1 && git -C extern/voronoi-cpp log --oneline -1 | cut -c1-60
for san in OFF ON; do
    b=$repo/build-ci-$san
    cmake -S . -B $b -DCMAKE_BUILD_TYPE=Release -DKokkos_DIR=/u/uli/kokkos-gcc15/lib64/cmake/Kokkos -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DVORONOI_SANITIZERS=$san > $p/ci-$san.cfg.log 2>&1
    cmake --build $b --parallel 18 > $p/ci-$san.build.log 2>&1; echo "build $san exit $?"; grep -E "warning|error" $p/ci-$san.build.log | grep -v "extern/" | head -5
    if [[ $san == OFF ]]; then (module load clang/18; bash scripts/lint_cpp.sh $b > $p/ci-$san.lint.log 2>&1; echo "lint exit $?"; grep -E "warning:|error:" $p/ci-$san.lint.log | grep -v "non-user" | head -10); fi
    ctest --test-dir $b --output-on-failure --parallel 2 > $p/ci-$san.ctest.log 2>&1; echo "ctest $san exit $?"; tail -n 3 $p/ci-$san.ctest.log
done
