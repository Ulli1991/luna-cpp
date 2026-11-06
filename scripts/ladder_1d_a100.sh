#!/bin/bash
# The 1D convergence ladder on A100s, a queue of runs (scripts/ladder_1d.py plan), one worker per GPU. From the repository root:
#   sbatch --export=ALL,REPO=$PWD,BIN=<cuda build copy>,OUT=<runs dir> -o <logs>/ladder-%j.out scripts/ladder_1d_a100.sh <series,..> <problems,..|all> <b_lo> <b_hi>
# Runs already written are skipped, so the same line continues where a job stopped. Above the first rung that does not reach
# t_end a series is not run for that problem. LUNA's long runs go on from their state (resume=).
#SBATCH -J ladder1d
#SBATCH -p gpu1,gpudev
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00
set -uo pipefail
source /etc/profile.d/modules.sh
repo="${REPO:-$PWD}"
bin="$BIN"
out="$OUT"
limit="${LIMIT:-$((13 * 60))}"
luna_dir="${LUNA_DIR:-/raven/u/uli/LUNA}"
luna_env="${LUNA_ENV:-$HOME/jax-env}"
mkdir -p "$out"
hostname
nvidia-smi --query-gpu=index,name,clocks.max.sm,power.limit --format=csv,noheader

module purge
module load gcc/13 cuda/12
luna_ld="$LD_LIBRARY_PATH"
luna_path="$PATH"
module purge
module load gcc/14 cuda/13.0 cmake hdf5-serial
ours_ld="$LD_LIBRARY_PATH"
export PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false LUNA_REFERENCE_SOLUTIONS=/raven/u/uli/luna-reference-solutions
export PYTHONPATH="$luna_dir/src"

tasks=$(mktemp -p "$out" tasks.XXXX)
LD_LIBRARY_PATH="$luna_ld" PATH="$luna_path" "$luna_env/bin/python" "$repo/scripts/ladder_1d.py" plan "$1" "$2" "$3" "$4" >"$tasks"
gpus=$(nvidia-smi -L | wc -l)
echo "tasks $(wc -l <"$tasks") gpus $gpus"

flags_of() {  # the hydro_1d flags of a series
    case "$1" in
    static) ;;
    static-*) echo "--limiter=${1#static-}" ;;
    moving) echo "--moving" ;;
    moving-*) echo "--moving --limiter=${1#moving-}" ;;
    esac
}
limiter_of() {
    case "$1" in
    luna-minmod) echo 0 ;;
    luna-mc) echo 1 ;;
    luna-vanleer) echo 2 ;;
    esac
}

worker() {
    local id="$1" s p n sign tag amp file left
    export CUDA_VISIBLE_DEVICES="$id"
    awk -v i="$id" -v g="$gpus" 'NR % g == i' "$tasks" | while read -r s p n sign; do
        left=$((limit - SECONDS))
        ((left > 20)) || break
        tag=""
        amp=1e-6
        if [[ "$sign" == "-" ]]; then
            tag=_m
            amp=-1e-6
        fi
        if [[ "$s" == luna-* ]]; then file="$out/${s}_${p}_${n}${tag}.npz"; else file="$out/${s}_${p}_${n}${tag}.npy"; fi
        [[ -f "$file.json" ]] && continue
        [[ -f "$out/stop_${s}_${p}" ]] && continue
        local eps=()
        [[ " sound-right sound-left entropy sound-advected " == *" $p "* ]] && eps=("$amp")
        if [[ "$s" == luna-* ]]; then
            (cd "$out" && LD_LIBRARY_PATH="$luna_ld" PATH="$luna_path" timeout $((left + 15)) "$luna_env/bin/python" -u "$repo/scripts/luna_1d_run.py" "$p" "$n" hllc "$file" \
                series="$s" LIMITER="$(limiter_of "$s")" repeats=1 max_s="$((left - 15))" resume="$out/state_${s}_${p}_${n}${tag}.npz" ${eps:+eps=$amp} 2>&1 | grep -E "^luna|Error|error" | cut -c1-200)
        else
            local amp_flag=()
            [[ ${#eps[@]} -gt 0 ]] && amp_flag=(--amp="$amp")
            LD_LIBRARY_PATH="$ours_ld" timeout $((left + 15)) "$bin/hydro_1d" "$p" "$n" --repeats=1 --max-wall="$left" --series="$s" --out="$file" "${amp_flag[@]}" $(flags_of "$s") | grep -E "^ours|rror" | cut -c1-200
            # A step that is not positive: that rung is the answer. The wall limit: not a result, the file goes.
            if [[ -f "$file.json" ]] && grep -q '"reached": 0' "$file.json"; then
                if (($(awk "BEGIN{print ($(sed -E 's/.*"wall_s": ([^,]*),.*/\1/' "$file.json") >= $left - 1)}"))); then
                    rm -f "$file" "$file.json"
                    echo "timeout $s $p $n"
                else
                    touch "$out/stop_${s}_${p}"
                fi
            fi
        fi
        if [[ -f "$file.json" ]] && [[ "$s" == luna-* ]] && grep -q '"reached": 0' "$file.json"; then touch "$out/stop_${s}_${p}"; fi
    done
}

for ((g = 0; g < gpus; ++g)); do worker "$g" & done
wait
rm -f "$tasks"
echo "seconds $SECONDS"
