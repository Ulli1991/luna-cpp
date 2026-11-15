#!/bin/bash
# Each face once by its owner (--update=owner), R14 and cheaper fluxes against main in one A100 job (docs/measurements.md,
# "Faces once by the owner"). Modes: speed, bits, ncu, tend, waves, kh. RUNS: "name=dir[@flag+flag][%ENV=v,ENV=v]" ...
#   sbatch --export=ALL,RUNS="main=<dir> once=<dir>@--update=owner",OUT=<dir> -o <logs>/once-%j.out scripts/hydro_once_a100.sh <mode> [side]
#SBATCH -J hydro_once
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0 cmake hdf5-serial
mode="${1:-speed}" side="${2:-128}"
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm --format=csv,noheader
here="${HERE:-/raven/u/uli/luna-cpp-once/scripts}"
py="${PY:-$HOME/jax-env/bin/python}"
export PYTHONPATH="${LUNA_DIR:-/raven/u/uli/LUNA}/src"
parse() {  # name=dir[@flag+flag][%A=1,B=2]: sets name, dir, flags, envs
    local r="$1" e=""
    [[ "$r" == *%* ]] && e="${r#*%}" r="${r%%%*}"
    name="${r%%=*}" rest="${r#*=}" dir="${rest%%@*}" flags="" envs="$(echo "$e" | tr ',' ' ')"
    [[ "$rest" == *@* ]] && flags="$(echo "${rest#*@}" | tr '+' ' ')"
}
case "$mode" in
speed)
    for round in ${ROUNDS:-1}; do
        for p in ${PRECISIONS:-grad32 double}; do
            for mesh in ${MESHES:---moving static}; do
                for b in $RUNS; do
                    parse "$b"
                    echo "== $name ${side}^3 $mesh $p round $round $flags $envs"
                    env $envs "$dir/hydro_sedov" "$side" jittered --steps=40 --memory --precision=$p ${mesh/static/} $flags | grep -E "^(mesh:|run|phases|throughput|memory: phase=steps)"
                done
            done
        done
    done
    ;;
bits)
    for b in $RUNS; do
        parse "$b"
        for run in "" "--moving" "--order=0" "--riemann=hll --moving" "--precision=double --moving" "--precision=grad16 --moving" "--precision=double" "--precision=grad16"; do
            tag="$(echo "$run" | tr -d ' -=')"
            env $envs "$dir/hydro_sedov" 128 jittered $run $flags --snapshots=1 --steps=50 --out="$OUT/bits_${tag:-static}_$name" | grep -E "^checksum" | sed "s/^/$name ${tag:-static} /"
        done
    done
    ;;
ncu)
    # One moving step (two after SKIP launches), every FiniteVolume kernel: time, DRAM bytes, registers, occupancy, FP64 pipe.
    NCU=/mpcdf/soft/SLE_15/packages/x86_64/nsight_compute/2026/ncu
    M=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,launch__registers_per_thread,launch__grid_size
    M+=,sm__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed
    M+=,sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active,sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active
    M+=,smsp__inst_executed.sum,sm__instruction_throughput.avg.pct_of_peak_sustained_active,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
    M+=,smsp__thread_inst_executed_per_inst_executed.ratio,lts__t_bytes.sum
    dcgmi profile --pause
    for b in $RUNS; do
        parse "$b"
        env $envs timeout 400 "$NCU" --target-processes all --clock-control none --kernel-name-base demangled --kernel-name "regex:FiniteVolume" \
            --launch-skip "${SKIP:-60}" --launch-count "${COUNT:-12}" --metrics "$M" --csv --page raw \
            "$dir/hydro_sedov" 128 jittered --moving --steps=8 ${PREC:+--precision=$PREC} $flags >"$OUT/ncu_$name.csv" 2>"$OUT/ncu_$name.err"
        echo "ncu $name exit $?"
    done
    dcgmi profile --resume
    ;;
tend)
    for b in $RUNS; do
        parse "$b"
        echo "== $name ${side}^3 moving to t_end $flags $envs"
        env $envs "$dir/hydro_sedov" "$side" jittered --moving --snapshots=1 $flags --out="$OUT/tend_$name" | grep -E "^(run|fallbacks|checksum)"
        "$py" "$here/sedov_field_l1.py" "$OUT/tend_$name" | sed "s/^/$name /"
    done
    ;;
waves)
    # The 3D vortex 64 and sound waves at 1e-3 and 1e-6 (64 x 16 x 16), moving and static, per precision.
    for p in ${PRECISIONS:-double grad32 grad16}; do
        for b in $RUNS; do
            parse "$b"
            for m in --moving ""; do
                t=${m:-static}; t=${t#--}
                env $envs "$dir/hydro_wave3d" vortex 64 $m --precision=$p $flags --out="$OUT/vortex_${t}_${p}_$name.npy" | grep "^ours3d" >/dev/null
                "$py" "$here/norm3d.py" vortex "$OUT/vortex_${t}_${p}_$name.npy" 64 | tail -1 | sed "s/^/$name vortex $t $p: /"
                for amp in 1e-3 1e-6; do
                    env $envs "$dir/hydro_wave3d" sound 64 $m --amp=$amp --precision=$p $flags --out="$OUT/sound${amp}_${t}_${p}_$name.npy" | grep "^ours3d" >/dev/null
                    "$py" "$here/sound3d_l1.py" "$OUT/sound${amp}_${t}_${p}_$name.npy" $amp | tail -1 | sed "s/^/$name sound $amp $t $p: /"
                done
            done
        done
    done
    ;;
kh)
    for b in $RUNS; do
        parse "$b"
        for m in "" --moving; do
            t=${m:-static}; t=${t#--}
            echo "kh $t $name: $(env $envs "$dir/hydro_kh2d" 128 $m $flags | grep -E '^(run|totals|ours)' | tr '\n' ' ')"
        done
    done
    ;;
esac
