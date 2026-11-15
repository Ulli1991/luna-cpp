#!/bin/bash
# LUNA's DRAM bytes and kernel time per step (Sedov 3D, Pallas): nsys counts the kernels of a step, ncu sums DRAM over two steps.
#   sbatch -o <logs>/lunaprof-%j.out scripts/luna_profile_a100.sh <out> [side] [x64]
#SBATCH -J luna_profile
#SBATCH -p gpudev
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/13 cuda/12 nsight_systems/2026
out="$1" side="${2:-128}" x64="${3:-0}"
mkdir -p "$out"
luna_dir="${LUNA_DIR:-/raven/u/uli/LUNA}"
luna_env="${LUNA_ENV:-$HOME/jax-env}"
NCU=/mpcdf/soft/SLE_15/packages/x86_64/nsight_compute/2026/ncu
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm --format=csv,noheader
tag="luna_${side}_x64$x64"
run=("$luna_env/bin/python" -u -m scripts.validation.runs.sedov3d_run nx="$side" ORDER=1 LIMITER=0 riemann_solver=hllc enable_x64=$x64
     hydro_kernel=pallas write_snapshots=0 enable_diagnostics=0 output_cadence=10 output_dir=/tmp/luna_prof)
cd "$luna_dir"
nsys profile -t cuda -o "$out/nsys_$tag" -f true "${run[@]}" max_steps=40 > "$out/nsys_$tag.log" 2>&1
nsys export -t sqlite -o "$out/nsys_$tag.sqlite" -f true "$out/nsys_$tag.nsys-rep" > /dev/null 2>&1
# Kernels per step: launches between step 20 and 30 of 40, over 10 (the last ones, steady state).
per_step=$(python3 - "$out/nsys_$tag.sqlite" <<'EOF'
import sqlite3, sys
c = sqlite3.connect(sys.argv[1])
ks = [s for (s,) in c.execute("select start from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
n = len(ks)
# steady state: the last 20 steps hold the last share of launches; assume equal launches per step after compile
print(max(1, (n - 0) // 40))
EOF
)
total=$(python3 -c "import sqlite3;c=sqlite3.connect('$out/nsys_$tag.sqlite');print(c.execute('select count(*) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0])")
echo "kernels total=$total per_step_estimate=$per_step"
dcgmi profile --pause
timeout 600 "$NCU" --target-processes all --clock-control none --launch-skip $((total - 6 * per_step)) --launch-count $((2 * per_step)) \
    --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active \
    --csv --page raw "${run[@]}" max_steps=40 > "$out/ncu_$tag.csv" 2> "$out/ncu_$tag.err"
echo "ncu exit $?"
dcgmi profile --resume
