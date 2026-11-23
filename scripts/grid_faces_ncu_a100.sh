#!/bin/bash
# Our faces kernel against LUNA's (Pallas kernel__1) side by side: ncu's full set on one launch of each, Sedov 3D 256^3 minmod FP64, in one
# job on one A100 (research_notes/fixed_grid_2026-11-19.md, "Why our faces kernel is slower than LUNA's").
#   sbatch --export=ALL,BIN=<dir with hydro_sedov>,OUT=<dir on /ptmp> -o <logs>/faces-ncu-%j.out scripts/grid_faces_ncu_a100.sh
# Writes <OUT>/{ours,luna}-<job>.ncu-rep and, from them, details (all sections), the raw metrics and the instruction mix by opcode.
#SBATCH -J faces_ncu
#SBATCH -p gpudev
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00
set -uo pipefail
source /etc/profile.d/modules.sh
BIN="${BIN:?}" OUT="${OUT:?}"
here="${SLURM_SUBMIT_DIR:-.}/scripts"
py="${LUNA_ENV:-$HOME/jax-env}/bin/python"
NCU=/mpcdf/soft/SLE_15/packages/x86_64/nsight_compute/2026/ncu
job="${SLURM_JOB_ID:-0}"
mkdir -p "$OUT"
echo "node $(hostname) job $job"
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm --format=csv,noheader
dcgmi profile --pause 2>&1 | tail -1
(module purge; module load gcc/13 cuda/12
 cd "$OUT" && PYTHONDONTWRITEBYTECODE=1 timeout 400 "$NCU" --target-processes all --clock-control none --set full --import-source no \
    --kernel-name kernel__1 --launch-skip 4 --launch-count 1 -o "$OUT/luna-$job" -f \
    "$py" -u "$here/luna_grid_time.py" 256 0 steps=8 warm=2 >"$OUT/luna-$job.log" 2>&1; echo "ncu LUNA exit $?")
ic=$(sed -n 's/^luna_ic: rho=\([^ ]*\) p_bg=[^ ]* E_bg=\([^ ]*\) E_blast=\([^ ]*\).*/\1,\2,\3/p' "$OUT/luna-$job.log")
(module purge; module load gcc/14 cuda/13.0 hdf5-serial
 timeout 400 "$NCU" --target-processes all --clock-control none --set full --import-source no --kernel-name-base demangled \
    --kernel-name "regex:step_kernel" --launch-skip 4 --launch-count 1 -o "$OUT/ours-$job" -f \
    "$BIN/hydro_sedov" 256 grid --grid --kernel=cuda --limiter=minmod --ic="$ic" --steps=8 >"$OUT/ours-$job.log" 2>&1; echo "ncu ours exit $?")
dcgmi profile --resume 2>&1 | tail -1
(module purge; module load cuda/13.0
 for who in luna ours; do
     "$NCU" -i "$OUT/$who-$job.ncu-rep" --page details --csv >"$OUT/$who-$job-details.csv" 2>/dev/null
     "$NCU" -i "$OUT/$who-$job.ncu-rep" --page raw --csv >"$OUT/$who-$job-raw.csv" 2>/dev/null
     "$NCU" -i "$OUT/$who-$job.ncu-rep" --page raw --csv --print-metric-instances values --metrics sass__inst_executed_per_opcode \
         >"$OUT/$who-$job-opcodes.csv" 2>/dev/null
     echo "$who: $(wc -l <"$OUT/$who-$job-details.csv") detail rows"
 done)
