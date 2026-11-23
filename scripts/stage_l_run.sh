#!/bin/bash
# Stage L alone (research_notes/stage_l_2026-11-19.md): stage_l_run.sh <out dir> <max_s> "<series> <k> <N>" ...
#   series luna, static, moving; k = 3, 6, 9 (density ratio 1e3, 1e6, 1e9); runs `jobs` at once (SLURM_CPUS_PER_TASK, or 1 on a GPU).
# static, moving: build-omp/hydro_1d (or build-cuda with BACKEND=cuda) stage-l-<k> <N> --t-end=4 --times=<TIMES>, one run writing
#   <series>_stage-l-<k>_<N>_t<t>.npy snapshots (x, Q) and the final state; moving: AREPO's rule at 0.4, static LUNA's at 0.8.
#   B: where the build directories are (default R). DEREF=f: moving with derefinement, cells narrower than f h removed (luna-cpp #30); default 0, off.
#   REFINE=F: moving with refinement, a new generator in cells wider than F h; SPLIT=linear: limited linear split (default constant).
# luna: scripts/luna_1d_run.py, HLLC, MC, CFL 0.8, LUNA's floors on (floor=1), one run per time of LUNA_TIMES.
#SBATCH -J stagel
#SBATCH -p general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
set -uo pipefail
R=${R:-/raven/u/uli/luna-cpp-stage-l}
out=$1 max_s=$2; shift 2
mkdir -p $out; cd $out
TIMES=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.81,0.9,1,1.25,1.5,1.75,2,2.5,3,3.5,4
LUNA_TIMES="0.1 0.5 0.81 2 4"
export OMP_NUM_THREADS=1 OMP_PROC_BIND=false JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export PYTHONPATH=/raven/u/uli/LUNA/src PYTHONDONTWRITEBYTECODE=1
B=${B:-$R}
bin=$B/build-omp/hydro_1d; par=${SLURM_CPUS_PER_TASK:-4}
if [[ ${BACKEND:-omp} == cuda ]]; then
  source /etc/profile.d/modules.sh; module purge; module load gcc/14 cuda/13.0
  bin=$B/build-cuda/hydro_1d; par=1; unset JAX_PLATFORMS XLA_FLAGS; export XLA_PYTHON_CLIENT_PREALLOCATE=false
fi
echo "host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-none} commit=$(git -C $R rev-parse --short HEAD) bin=$bin"
one() {
  local s=$1 k=$2 n=$3 p=stage-l-$2
  if [[ $s == luna ]]; then
    for t in $LUNA_TIMES; do
      local f=luna_${p}_${n}_t$t
      grep -qs '"reached": 1' $f.npz.json && continue
      timeout $((max_s + 120)) /u/uli/jax-env/bin/python -u $R/scripts/luna_1d_run.py $p $n hllc $f.npz repeats=1 t=$t series=luna LIMITER=1 \
        CFL=0.8 floor=1 max_s=$max_s resume=$out/state_$f.npz 2>&1 | grep -E "^luna|rror" | sed "s/^/$f: /"
    done
  else
    local f=${s}_${p}_${n} flags=(); [[ $s == moving ]] && flags=(--moving --derefine=${DEREF:-0} --refine=${REFINE:-0} --split=${SPLIT:-constant})
    grep -qs '"reached": 1' $f.npy.json && return
    $bin $p $n --t-end=4 --times=$TIMES --repeats=1 --series=$s --max-wall=$max_s --out=$f.npy "${flags[@]}" 2>&1 | grep -E "^ours|rror" \
      | sed "s/^/$f: /"
  fi
}
for job in "$@"; do
  while (($(jobs -rp | wc -l) >= par)); do wait -n; done
  one $job &
done
wait
echo "seconds $SECONDS"
