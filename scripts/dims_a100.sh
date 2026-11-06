#!/bin/bash
# Throughput against resolution on Sedov in 1D, 2D or 3D on one A100: hydro static and two-pass, mesh only, whole moving step, LUNA.
#   sbatch --export=ALL,BIN=<build>,OUT=<runs> -o <logs>/dims-%j.out scripts/dims_a100.sh <1|2|3>
#SBATCH -J dims
#SBATCH -p gpudev,gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00
set -uo pipefail
source /etc/profile.d/modules.sh
ours_modules() { module purge; module load gcc/14 cuda/13.0 cmake hdf5-serial; }
luna_modules() { module purge; module load gcc/13 cuda/12; }
ours_modules
dim="$1"
job="${SLURM_JOB_ID:-0}"
mkdir -p "$OUT"
LUNA=/raven/u/uli/LUNA
PY=$HOME/jax-env/bin/python
SCRIPT="${REPO:-/raven/u/uli/luna-cpp-roofline}/scripts/luna_sedov_dims.py"
hostname
nvidia-smi --query-gpu=name,uuid,clocks.max.sm,clocks.applications.graphics --format=csv,noheader
nvidia-smi --query-gpu=timestamp,clocks.sm,utilization.gpu,memory.used,power.draw --format=csv,noheader -l 1 >"$OUT/clock-$job.csv" &
smi=$!
trap 'kill $smi 2>/dev/null' EXIT
keep='^(mesh|run|phases|phase|throughput|memory|ours|checksum)|ailed'
case "$dim" in
1)
    sides="$(for e in 12 14 16 18 20 22 24; do echo $((1 << e)); done)"
    for n in $sides; do
        echo "== ours $n static $(date +%T)"
        "$BIN/hydro_1d" sedov "$n" --jitter=0.1 --steps=40 --repeats=1 --phases --memory 2>&1 | grep -E "$keep" | cut -c1-240
        echo "== ours $n one-pass $(date +%T)"
        "$BIN/hydro_1d" sedov "$n" --jitter=0.1 --moving --steps=40 --repeats=1 --phases --memory 2>&1 | grep -E "$keep" | cut -c1-240
    done
    ;;
2)
    sides="256 512 1024 2048 4096 5792"
    for n in $sides; do
        echo "== ours $n static $(date +%T)"
        "$BIN/hydro_kh2d" "$n" --problem=sedov --jitter=0.1 --steps=40 --repeats=2 --phases --memory 2>&1 | grep -E "$keep" | cut -c1-300
        for i in one-pass two-pass; do
            echo "== ours $n $i $(date +%T)"
            "$BIN/hydro_kh2d" "$n" --problem=sedov --jitter=0.1 --moving --integrator="$i" --dt=arepo --courant=0.3 --steps=40 --repeats=2 --phases --memory 2>&1 | grep -E "$keep" | cut -c1-300
        done
    done
    ;;
3)
    sides="32 48 64 96 128 192 256"
    for side in $sides; do
        for flags in "" "--moving --integrator=one-pass" "--moving --integrator=two-pass"; do
            echo "== ours $side ${flags:-static} round 1 $(date +%T)"
            "$BIN/hydro_sedov" "$side" jittered --steps=40 --memory $flags 2>&1 | grep -E "$keep" | cut -c1-300
        done
    done
    ;;
esac
luna_modules
cd "$LUNA"
for round in 1 2 3; do
    for n in $sides; do
        echo "== luna $n round $round $(date +%T)"
        if [[ "$dim" == 3 ]]; then
            "$PY" -u -m scripts.validation.runs.sedov3d_run nx="$n" ORDER=1 LIMITER=1 riemann_solver=hllc enable_x64=1 hydro_kernel=pallas write_snapshots=0 output_dir="$OUT/luna_$job" max_steps=200 2>&1
        else
            "$PY" -u "$SCRIPT" "$dim" "$n" "$OUT/luna_$job" 2>&1
        fi | grep -E "perf =|Performance|Using MUSCL|rror|Error" | cut -c1-140
    done
done
