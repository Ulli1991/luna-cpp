#!/bin/bash
# Why the chunked flux/update schedule is slower: Nsight Compute, a light metric set, of every FiniteVolume kernel of the
# third moving step at 128^3 (LUNA_UPDATE_CHUNK from the environment), one line per launch.
#   sbatch --export=ALL,BIN=<build dir>,OUT=<dir>,LUNA_UPDATE_CHUNK=19 -o <logs>/chunk-%j.out scripts/moving_sol_chunk_ncu.sh
#SBATCH -J moving_sol_chunk
#SBATCH -p gpudev
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=00:15:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0 hdf5-serial nsight_compute/2026
prof=/mpcdf/soft/SLE_15/packages/x86_64/nsight_compute/2026/ncu
mkdir -p "$OUT"
m=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sector_hit_rate.pct,launch__grid_size,launch__block_size,sm__warps_active.avg.pct_of_peak_sustained_active
dcgmi profile --pause 2>/dev/null
trap 'dcgmi profile --resume 2>/dev/null' EXIT
"$prof" --target-processes all --clock-control none --kernel-name-base demangled --kernel-name "regex:FiniteVolume" \
    --launch-skip 150 --launch-count 160 --metrics $m --csv --print-units base "$BIN/hydro_sedov" 128 jittered --moving --steps=4 2>/dev/null |
    grep -E '^"' > "$OUT/chunk_${LUNA_UPDATE_CHUNK:-0}.csv"
awk -F'","' 'NR > 1 { k = ($5 ~ /long long/) ? "fluxes" : ($5 ~ /MovingFlux/ ? "update" : "other"); key = $1 FS k; v[$1][$13] = $15; kind[$1] = k }
    END { for (i in kind) printf "%s %s t=%s grid=%s block=%s dram_r=%s dram_w=%s l2hit=%s warps=%s\n", i, kind[i], v[i]["gpu__time_duration.sum"],
          v[i]["launch__grid_size"], v[i]["launch__block_size"], v[i]["dram__bytes_read.sum"], v[i]["dram__bytes_write.sum"],
          v[i]["lts__t_sector_hit_rate.pct"], v[i]["sm__warps_active.avg.pct_of_peak_sustained_active"] }' "$OUT/chunk_${LUNA_UPDATE_CHUNK:-0}.csv" | sort -n
