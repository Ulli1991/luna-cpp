#!/bin/bash
# When voronoi-cpp changes (TESTING.md), on one A100, with the binaries of voronoi_bits_a100.sh (OLD and NEW build dirs):
# moving Sedov 128^3 and 256^3 to t_end, two runs each (stuck steps, rebuilds, local repairs, checksum; the first run's field
# L1 of rho, v_r, P), then speed: 40 steps static and moving, old and new alternating, three rounds, each run once without
# fences (step_wall_ms_median, G_cell_updates_per_s) and once with --phases (the mesh part), with --memory (B/cell).
# SM clock sampled every second into clock-<job>.csv.
#   sbatch --export=ALL,OLD=<build dir>,NEW=<build dir>,OUT=<dir> -o <logs>/voronoi_speed-%j.out scripts/voronoi_speed_a100.sh
#SBATCH -J voronoi_speed
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=01:30:00
set -uo pipefail
source /etc/profile.d/modules.sh; module purge; module load gcc/14 cuda/13.0 cmake hdf5-serial
nvidia-smi --query-gpu=name,clocks.sm,clocks.applications.graphics,clocks.max.sm,clocks.applications.memory --format=csv
new="${NEW:?NEW: the build dir with the new voronoi-cpp}"
old="${OLD:?OLD: the build dir with the old voronoi-cpp}"
p="${OUT:-$new/voronoi-proof}"; mkdir -p "$p"
scripts="${SCRIPTS:-$SLURM_SUBMIT_DIR/scripts}"   # submit from the luna-cpp tree
export PYTHONPATH="${LUNA_DIR:-/raven/u/uli/LUNA}/src"
for v in old new; do d=${!v}; echo "$v $(cat $d/luna_build_info.txt 2>/dev/null || echo "build: no luna_build_info.txt (a submodule tree)")"; done
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,utilization.gpu,temperature.gpu,power.draw --format=csv -l 1 > $p/clock-$SLURM_JOB_ID.csv &
smi=$!
keep="^(run|phases_ms_median|throughput_M_per_s|memory: phase=steps|memory: phase=peak|mesh_updates|checksum)"

echo "== stuck and field L1: moving Sedov to t_end"
for side in 128 256; do
    for round in 1 2; do
        for v in old new; do
            d=${!v}
            echo "-- $v ${side}^3 round $round"
            out=""; [[ $round == 1 ]] && out="--out=$p/tend_${v}_$side"
            $d/hydro_sedov $side jittered --moving --snapshots=1 $out 2>/dev/null | grep -E "^(run|mesh_updates|checksum)"
            [[ $round == 1 ]] && "${PY:-$HOME/jax-env/bin/python}" "$scripts/sedov_field_l1.py" "$p/tend_${v}_$side"
        done
    done
done

echo "== speed: 40 steps"
for round in 1 2 3; do
    for side in 128 256; do
        for mesh in "" "--moving"; do
            for phases in "" "--phases"; do
                for v in old new; do
                    d=${!v}
                    echo "-- $v ${side}^3 ${mesh:-static} ${phases:-wall} round $round"
                    $d/hydro_sedov $side jittered $mesh $phases --steps=40 --snapshots=1 --memory 2>/dev/null | grep -E "$keep"
                done
            done
        done
    done
done
kill $smi
