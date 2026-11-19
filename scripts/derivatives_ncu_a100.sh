#!/bin/bash
# Nsight Compute on the forward, tangent and adjoint step kernels (hydro_derivatives_test --profile N: double, then grad16; each
# policy one warm-up and one measured round). Per kernel: time, registers, warps active, local (spill) bytes, DRAM bytes, FP64 use.
#   sbatch -o <logs>/ncu-%j.log scripts/derivatives_ncu_a100.sh <build dir> [N = 2097152]
#SBATCH -J deriv_ncu
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00
set -uo pipefail
source /etc/profile.d/modules.sh; module purge; module load gcc/14 cuda/13.0 cmake hdf5-serial nsight_compute/2026
nvidia-smi --query-gpu=name,clocks.sm,clocks.applications.graphics --format=csv,noheader
build="${1:?build dir}"
n="${2:-2097152}"
out=${OUT:-$build/ncu-${SLURM_JOB_ID:-local}}
mkdir -p "$out"
ncu=${NCU:-/mpcdf/soft/SLE_15/packages/x86_64/nsight_compute/2026/ncu}
metrics=gpu__time_duration.sum,launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active
metrics+=,l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum,dram__bytes_read.sum,dram__bytes_write.sum
metrics+=,sm__sass_thread_inst_executed_op_dfma_pred_on.sum,sm__sass_thread_inst_executed_op_dadd_pred_on.sum,sm__sass_thread_inst_executed_op_dmul_pred_on.sum
dcgmi profile --pause 2>/dev/null
trap 'dcgmi profile --resume 2>/dev/null' EXIT
"$ncu" --target-processes all --clock-control none --kernel-name-base demangled \
    --kernel-name "regex:Sweep|FaceFlux|TimeStep|Tangent|Adjoint|apply|primitives" \
    --metrics $metrics --csv --print-units base \
    "$build/hydro_derivatives_test" --profile "$n" > "$out/kernels.csv" 2> "$out/ncu.err"
echo "csv lines: $(wc -l < "$out/kernels.csv")"
tail -3 "$out/ncu.err"
python3 /raven/u/uli/luna-cpp-ad/scripts/derivatives_ncu_summary.py "$n" "$out/kernels.csv"
