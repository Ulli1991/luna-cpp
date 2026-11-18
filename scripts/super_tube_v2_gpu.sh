#!/bin/bash
# The super-hard tube v2 on one A100 (research_notes/super_tube_v2_2026-11-18.md), runs given as "<series> <problem> <N> <t>":
#   sbatch [--time=...] super_tube_v2_gpu.sh <out dir> "<series> <problem> <N> <t>" ...
# luna: scripts/luna_1d_run.py (LUNA, read-only), HLLC, MC, CFL 0.8, FLOOR=1: LUNA's floors on; chunked with resume (a later job continues a run);
# moving, static: build-cuda/hydro_1d [--moving] (AREPO's rule, Courant 0.4; static LUNA's at 0.8), --max-wall to the job's end
# (reached = 0 in the JSON if cut).
# Runs one after the other; a finished run (JSON with reached = 1) is skipped.
#SBATCH -J tube2gpu
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=12:00:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge; module load gcc/14 cuda/13.0
R=${R:-/raven/u/uli/luna-cpp-tube-v2}
out=$1; shift
mkdir -p $out; cd $out
export PYTHONPATH=/raven/u/uli/LUNA/src PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
limit=$(squeue -h -j $SLURM_JOB_ID -o %L 2>/dev/null | awk -F'[-:]' '{n=NF; s=$n+60*$(n-1); if(n>2) s+=3600*$(n-2); if(n>3) s+=86400*$(n-3); print s}')
limit=${limit:-3600}
echo "host=$(hostname) date=$(date -Is) job=$SLURM_JOB_ID commit=$(git -C $R rev-parse --short HEAD) seconds_left=$limit"
nvidia-smi --query-gpu=index,name,clocks.sm,clocks.max.sm,power.limit --format=csv,noheader
for job in "$@"; do
  read -r s p n t <<< "$job"
  f=${s}_${p}_${n}_t$t
  grep -qs '"reached": 1' $f.npz.json $f.npy.json && continue
  left=$((limit - SECONDS - 120)); ((left > 60)) || break
  if [[ $s == luna ]]; then
    /u/uli/jax-env/bin/python -u $R/scripts/luna_1d_run.py $p $n hllc $f.npz repeats=1 t=$t series=luna LIMITER=1 CFL=0.8 floor=${FLOOR:-0} \
      max_s=$left resume=$out/state_$f.npz 2>&1 | grep -E "^luna|rror|compile" | sed "s/^/$f: /"
  else
    flags=(); [[ $s == moving ]] && flags=(--moving)
    $R/build-cuda/hydro_1d $p $n "${flags[@]}" --t-end=$t --repeats=1 --series=$s --max-wall=$left --out=$f.npy 2>&1 \
      | grep -E "^ours|rror" | sed "s/^/$f: /"
  fi
done
echo "seconds $SECONDS"
