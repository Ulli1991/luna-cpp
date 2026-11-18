#!/bin/bash
# Runs of the super-hard tube v2 and its stages alone (research_notes/super_tube_v2_2026-11-18.md), one CPU core each:
#   super_tube_v2_run.sh <out dir> <max_s> "<series> <problem> <N> <t>" ...      series: luna, static, moving; up to 4 at once
# luna: scripts/luna_1d_run.py (LUNA /raven/u/uli/LUNA/src, read-only), HLLC, MC, CFL 0.8; static, moving: build-omp/hydro_1d, defaults
# (static LUNA's rule at 0.8; moving AREPO's at 0.4). Files <series>_<problem>_<N>_t<t>.np[yz] and .json; a log line per run on stdout.
#SBATCH -J tube2
#SBATCH -p general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -uo pipefail
R=${R:-/raven/u/uli/luna-cpp-tube-v2}
out=$1 max_s=$2; shift 2
mkdir -p $out; cd $out
export OMP_NUM_THREADS=1 OMP_PROC_BIND=false JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export PYTHONPATH=/raven/u/uli/LUNA/src PYTHONDONTWRITEBYTECODE=1
echo "host=$(hostname) cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2) date=$(date -Is) job=${SLURM_JOB_ID:-none} commit=$(git -C $R rev-parse --short HEAD)"
one() {
  local s=$1 p=$2 n=$3 t=$4 f=$1_$2_$3_t$4
  if [[ $s == luna ]]; then
    timeout $((max_s + 120)) /u/uli/jax-env/bin/python -u $R/scripts/luna_1d_run.py $p $n hllc $f.npz repeats=1 t=$t series=luna LIMITER=1 CFL=0.8 \
      max_s=$max_s resume=$out/state_$f.npz 2>&1 | grep -E "^luna|rror" | sed "s/^/$f: /"
  else
    local flags=(); [[ $s == moving ]] && flags=(--moving --smallest)
    $R/build-omp/hydro_1d $p $n --t-end=$t --repeats=1 --series=$s --max-wall=$max_s --out=$f.npy "${flags[@]}" 2>&1 | grep -E "^ours|rror" | sed "s/^/$f: /"
  fi
}
for job in "$@"; do
  while (($(jobs -rp | wc -l) >= ${SLURM_CPUS_PER_TASK:-4})); do wait -n; done
  one $job &
done
wait
echo "seconds $SECONDS"
