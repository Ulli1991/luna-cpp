#!/bin/bash
# The 2D hydro on one A100. From the repository root:
#   sbatch --export=ALL,REPO=$PWD,BUILD=<build dir> -o <logs>/dim2d-%j.out scripts/dim2d_a100.sh <mode> [args]
# Modes are the cases below; the clock and the node go into BUILD/runs/clock-<job>.csv.
#SBATCH -J dim2d
#SBATCH -p gpu1,gpudev
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00
set -uo pipefail
source /etc/profile.d/modules.sh
repo="${REPO:-$PWD}"
build="${BUILD:-$repo/build-2d}"
bin="${BIN:-$build/cuda}"
mode="${1:-build}"
mkdir -p "$build/runs"
hostname
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.applications.graphics,power.limit --format=csv,noheader
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,utilization.gpu,memory.used,power.draw --format=csv,noheader -l 5 >"$build/runs/clock-${SLURM_JOB_ID:-0}.csv" &
smi=$!
trap 'kill $smi 2>/dev/null' EXIT
export KOKKOS_NUM_DEVICES=1 CUDA_LAUNCH_BLOCKING=0
use_ours() {
    module purge
    module load gcc/14 cuda/13.0 cmake hdf5-serial
}
luna_dir="${LUNA_DIR:-/raven/u/uli/LUNA}"
luna_env="${LUNA_ENV:-$HOME/jax-env}"
use_luna() {
    module purge
    module load gcc/13 cuda/12
    export PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false LUNA_REFERENCE_SOLUTIONS=/raven/u/uli/luna-reference-solutions
    export PYTHONPATH="$luna_dir/src"
}
luna() {  # luna <N> <out name> [key=value ...]
    local n="$1" name="$2"
    shift 2
    (cd "$build/runs" && "$luna_env/bin/python" -u "$repo/scripts/luna_kh_run.py" "$n" "$build/runs/$name.npz" "$@" 2>&1 | grep -E "^luna|^run|Error|error")
}
do_diag() {  # what limits the moving mesh on KH
    use_ours
    n=256
    for c in 0.8 0.4 0.2 0.1; do "$bin/hydro_kh2d" $n --moving --courant=$c --out="$build/runs/diag_moving_c$c" | grep -E "^ours"; done
    for c in 0.4 0.2 0.1; do "$bin/hydro_kh2d" $n --courant=$c --out="$build/runs/diag_static_c$c" | grep -E "^ours"; done
    "$bin/hydro_kh2d" $n --moving --dt=arepo --out="$build/runs/diag_moving_arepo" | grep -E "^ours"
    "$bin/hydro_kh2d" $n --jitter=0.2 --out="$build/runs/diag_static_j" | grep -E "^ours"
    "$bin/hydro_kh2d" $n --moving --jitter=0.2 --out="$build/runs/diag_moving_j" | grep -E "^ours"
    "$bin/hydro_kh2d" $n --moving --steps=500 --phases --memory | grep -E "^ours|^phase|^memory"
    "$bin/hydro_kh2d" 1024 --moving --steps=500 --phases --memory | grep -E "^ours|^phase|^memory"
    "$bin/hydro_kh2d" 1024 --steps=500 --phases --memory | grep -E "^ours|^phase|^memory"
}
case "$mode" in
all)
    use_ours
    cmake --build "$build/cuda" -j18 --target "${@:2}" 2>&1 | grep -E "warning|error|Error |Built target" | sort | uniq -c | sort -rn | head -30
    ctest --test-dir "$build/cuda" --output-on-failure -R "${TESTS:-hydro2d}" | grep -E "FAIL|tests passed|Failed|Noh"
    [[ -n "${DIAG:-}" ]] && do_diag
    ;;
build)
    use_ours
    [[ -f "$build/cuda/CMakeCache.txt" ]] || cmake -S "$repo" -B "$build/cuda" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH=/u/uli/kokkos-cuda -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper | tail -2
    cmake --build "$build/cuda" -j18 --target "${@:2}" 2>&1 | grep -E "warning|error|Error |Built target" | sort | uniq -c | sort -rn | head -30
    ctest --test-dir "$build/cuda" --output-on-failure -R "${TESTS:-hydro2d}" | tail -30
    ;;
kh)
    for n in "${@:2}"; do
        use_luna
        luna "$n" "luna_$n"
        use_ours
        "$bin/hydro_kh2d" "$n" --memory --out="$build/runs/static_$n" | grep -E "^(ours|memory|phase)"
        "$bin/hydro_kh2d" "$n" --moving --memory --out="$build/runs/moving_$n" | grep -E "^(ours|memory|phase)"
    done
    ;;
khfield)  # khfield <t> <N> [...]
    t="$2"
    for n in "${@:3}"; do
        use_luna
        luna "$n" "field_luna_t${t}_$n" full=1 t_end="$t"
        use_ours
        "$bin/hydro_kh2d" "$n" --t-end="$t" --out="$build/runs/field_static_t${t}_$n" | grep -E "^ours"
        "$bin/hydro_kh2d" "$n" --moving --t-end="$t" --out="$build/runs/field_moving_t${t}_$n" | grep -E "^ours"
    done
    ;;
khbulk)  # khbulk <bulk> <N> [...]: as kh with a bulk velocity along x
    u="$2"
    for n in "${@:3}"; do
        use_luna
        luna "$n" "luna_bulk${u}_$n" bulk="$u"
        use_ours
        "$bin/hydro_kh2d" "$n" --bulk="$u" --memory --out="$build/runs/static_bulk${u}_$n" | grep -E "^(ours|memory|phase)"
        "$bin/hydro_kh2d" "$n" --moving --bulk="$u" --memory --out="$build/runs/moving_bulk${u}_$n" | grep -E "^(ours|memory|phase)"
    done
    ;;
run)
    use_ours
    export PATH="$bin:$PATH"
    "${@:2}"
    ;;
esac
