#!/bin/bash
# Flux-formulation levers of branch flux-latency on one A100. Variants: l0..l3 are the builds <ROOT>/build-l<n>
# (-DLUNA_FLUX_LEVER=n; l0 is the branch's own code), hll is build-l0 with --riemann=hll. Modes:
#   speed <side>  Sedov moving 40 steps, fenced phases, ROUNDS rounds, variants alternating
#   tend <side>   Sedov moving to 1e4 yr: steps, GPU s, field L1, checksum
#   gates         sound 64 at 1e-3 and 1e-6 (moving and static), vortex 64 and 128 moving (order), KH 2D 512 to t = 1.5
#                 (moving and static), the 1D ladder (hydro1d_cases_test; not for hll)
#   ncu           Nsight Compute of the moving flux passes at 128^3 (time, registers, warps, FP64 pipe, instructions, DRAM)
#   sbatch --export=ALL,ROOT=<tree>,BUILDS=<dir of build-l*>,OUT=<dir on /ptmp>[,VARIANTS="l0 l1 l2 l3 hll"] -o <logs>/ff-%j.out scripts/flux_formulation_a100.sh <mode> [side]
#SBATCH -J flux_formulation
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=01:30:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0 cmake hdf5-serial
mode="${1:-speed}" side="${2:-128}"
here="$ROOT/scripts"
py="${PY:-$HOME/jax-env/bin/python}"
export PYTHONPATH="${LUNA_DIR:-/raven/u/uli/LUNA}/src"
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,clocks.sm,clocks.applications.graphics --format=csv,noheader
variants="${VARIANTS:-l0 l1 l2 l3 hll}"
builds="${BUILDS:-$ROOT}"  # where build-l0..3 are (outside the source tree: /ptmp)
bin() { [[ "$1" == hll ]] && echo "$builds/build-l0" || echo "$builds/build-$1"; }
flags() { [[ "$1" == hll ]] && echo "--riemann=hll" || echo ""; }
case "$mode" in
speed)
    for round in ${ROUNDS:-1 2 3}; do
        for v in $variants; do
            echo "== $v ${side}^3 round $round"
            "$(bin $v)/hydro_sedov" "$side" jittered --moving --steps=40 --phases $(flags $v) | grep -E "^(phases_ms_median|checksum)"
        done
    done
    ;;
tend)
    for v in $variants; do
        echo "== $v ${side}^3 to t_end"
        "$(bin $v)/hydro_sedov" "$side" jittered --moving --snapshots=1 $(flags $v) --out="$OUT/tend${side}_$v" | grep -E "^(run|checksum|fallbacks)"
        "$py" "$here/sedov_field_l1.py" "$OUT/tend${side}_$v" | sed "s/^/$v /"
    done
    ;;
gates)
    for v in $variants; do
        b="$(bin $v)" f="$(flags $v)"
        for m in --moving ""; do
            t=${m:-static}; t=${t#--}
            for amp in 1e-3 1e-6; do
                "$b/hydro_wave3d" sound 64 $m --amp=$amp $f --out="$OUT/sound${amp}_${t}_$v.npy" > /dev/null
                "$py" "$here/sound3d_l1.py" "$OUT/sound${amp}_${t}_$v.npy" $amp | sed "s/^/$v $t amp=$amp /"
            done
            echo "$v $t kh: $("$b/hydro_kh2d" 512 $m $f --t-end=1.5 2>&1 | grep -oE 'steps=[0-9]+ wall_s=[0-9.]+|M_final=[^ ]+|energy=[^ ]+' | tr '\n' ' ')"
        done
        for nx in 64 128; do
            "$b/hydro_wave3d" vortex $nx --moving $f --out="$OUT/vortex${nx}_$v.npy" > /dev/null
            "$py" "$here/norm3d.py" vortex "$OUT/vortex${nx}_$v.npy" $nx | tail -2 | sed "s/^/$v vortex $nx: /"
        done
        if [[ $v != hll ]]; then
            echo "$v ladder: $("$b/hydro1d_cases_test" 2>&1 | grep -cE '^\s*FAIL') FAIL lines; $("$b/hydro1d_cases_test" > /dev/null 2>&1; echo exit $?)"
        fi
    done
    ;;
ncu)
    module load nsight_compute/2026
    ncu=/mpcdf/soft/SLE_15/packages/x86_64/nsight_compute/2026/ncu
    m=gpu__time_duration.sum,launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active
    m+=,sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.sum,dram__bytes_read.sum,dram__bytes_write.sum
    m+=,l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum
    dcgmi profile --pause 2>/dev/null
    trap 'dcgmi profile --resume 2>/dev/null' EXIT
    for v in $variants; do
        echo "== ncu $v"
        "$ncu" --target-processes all --clock-control none --kernel-name-base demangled --kernel-name "regex:MovingFlux" \
            --launch-skip 40 --launch-count 4 --metrics $m --csv --print-units base "$(bin $v)/hydro_sedov" 128 jittered --moving --steps=14 $(flags $v) 2>/dev/null |
            grep -E '^"' | awk -F'","' 'NR>1 {gsub(/"/,"",$15); k=($5~/lambda\(int, long long/)?"fluxes":"update"; print $1"\" "k" "$13"="$15}' | tr -d '"' | grep fluxes
    done
    ;;
esac
