#!/bin/bash
# One-pass against two-pass on one A100, each mode one job on one node. From the repository root:
#   sbatch --export=ALL,BASE=<dir>,NEW=<dir>,OUT=<runs> -o <logs>/twopass-%j.out scripts/two_pass_a100.sh <mode> [args]
# BASE and NEW hold hydro_sedov (and NEW hydro_wave3d, ctest build): the tree before two passes and this one.
# Modes:
#   bits           hashes of the paths of TESTING.md, BASE and NEW; two-pass moving, twice
#   speed          one-pass, BASE and NEW interleaved, 128^3 static and moving, 256^3 moving, 40 steps, --memory; two-pass
#   sedov <side ...>   moving Sedov to t_end, one-pass and two-pass, final snapshot in OUT
#   mixed          the same at 256^3, --precision=mixed
#   wave <problem> <nx ...>   hydro_wave3d, moving, jitter 0.1, both integrators; end states in OUT
#   ctest          ctest of NEW
# The clock goes into OUT/clock-<job>.csv, one line per 5 s.
#SBATCH -J twopass
#SBATCH -p gpudev,gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0 cmake hdf5-serial
mode="${1:-bits}"
shift || true
mkdir -p "$OUT"
hostname
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.applications.graphics,power.limit --format=csv,noheader
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,utilization.gpu,memory.used,power.draw --format=csv,noheader -l 5 >"$OUT/clock-${SLURM_JOB_ID:-0}.csv" &
smi=$!
trap 'kill $smi 2>/dev/null' EXIT
keep='^(mesh|run|phases|throughput|memory|conservation|fallbacks|checksum|ours3d)'

case "$mode" in
bits)
    for build in "$BASE" "$NEW"; do
        for flags in "" "--moving" "--order=0" "--update=cell --moving" "--riemann=hll --moving"; do
            echo "== $(basename "$build") 128^3 ${flags:-static}"
            # Before the option the moving default was one pass and the precision double; here both are asked for by name.
            extra=""
            [[ "$build" == "$NEW" ]] && extra="--precision=double"
            [[ "$build" == "$NEW" && "$flags" == *--moving* ]] && extra+=" --integrator=one-pass"
            "$build/hydro_sedov" 128 jittered $flags $extra --steps=50 | grep '^checksum'
        done
    done
    for round in 1 2; do
        echo "== two-pass moving, run $round"
        "$NEW/hydro_sedov" 128 jittered --moving --integrator=two-pass --precision=double --steps=50 | grep '^checksum'
    done
    ;;
speed)
    for round in 1 2 3; do
        for mesh in "" "--moving"; do
            for build in "$BASE" "$NEW"; do
                echo "== $(basename "$build") 128^3 ${mesh:-static} one-pass round $round"
                extra=""
                [[ "$build" == "$NEW" ]] && extra="--precision=double"
                [[ "$build" == "$NEW" && -n "$mesh" ]] && extra+=" --integrator=one-pass"
                "$build/hydro_sedov" 128 jittered --steps=40 --memory $mesh $extra | grep -E "$keep"
            done
        done
        echo "== two-pass 128^3 moving round $round"
        "$NEW/hydro_sedov" 128 jittered --steps=40 --memory --moving --integrator=two-pass --precision=double | grep -E "$keep"
    done
    for build in "$BASE" "$NEW"; do
        echo "== $(basename "$build") 256^3 moving one-pass"
        extra=""
        [[ "$build" == "$NEW" ]] && extra="--precision=double --integrator=one-pass"
        "$build/hydro_sedov" 256 jittered --steps=40 --memory --moving $extra | grep -E "$keep"
    done
    echo "== two-pass 256^3 moving"
    "$NEW/hydro_sedov" 256 jittered --steps=40 --memory --moving --integrator=two-pass --precision=double | grep -E "$keep"
    ;;
sedov | mixed)
    sides="$*"
    flags=""
    if [[ "$mode" == mixed ]]; then
        sides=256
        flags="--precision=mixed"
    fi
    for side in $sides; do
        for i in one-pass two-pass; do
            echo "== sedov $side $flags $i"
            run="$OUT/sedov_${side}_${mode}_$i"
            rm -rf "$run"
            "$NEW/hydro_sedov" "$side" jittered --moving --memory $flags --integrator=$i --snapshots=1 --out="$run" | grep -E "$keep"
        done
    done
    ;;
wave)
    problem="$1"
    shift
    for nx in "$@"; do
        for i in one-pass two-pass; do
            "$NEW/hydro_wave3d" "$problem" "$nx" --moving --integrator=$i --out="$OUT/${problem}_${nx}_$i.npy" | grep -E "$keep"
        done
    done
    ;;
ctest)
    ctest --test-dir "$NEW" --output-on-failure | tail -8
    ;;
esac
