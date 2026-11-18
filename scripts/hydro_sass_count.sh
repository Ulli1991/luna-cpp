#!/bin/bash
# SASS of the default-precision HLLC flux kernels of hydro_sedov (static per-face fluxes, moving pass 1 and 2): their
# instruction count and the opcodes of square roots, reciprocals, divisions and local memory (docs/tried.md).
#   sbatch -o <logs>/sass-%j.out scripts/hydro_sass_count.sh <label=hydro_sedov> ...
#SBATCH -J hydro_sass
#SBATCH -p gpudev
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
set -uo pipefail
source /etc/profile.d/modules.sh
module purge
module load gcc/14 cuda/13.0
out=${OUT:-sass-${SLURM_JOB_ID:-local}}
mkdir -p "$out"
for item in "$@"; do
    label="${item%%=*}" bin="${item#*=}"
    cuobjdump -sass "$bin" > "$out/$label.sass"
    # One file per function; keep the HLLC Grad32 flux kernels.
    awk -v dir="$out" -v label="$label" '
        /Function : / { name = $3; n++; file = sprintf("%s/%s.fn%05d", dir, label, n); print name > (file ".name") }
        n > 0 { print > file }' "$out/$label.sass"
    for nf in "$out/$label".fn*.name; do
        name=$(c++filt < "$nf")
        case "$name" in
        *HLLC*"Precision<double, float, double, double>"*FaceFlux\<1,\ false\>*"lambda(int, long long&)#2"*) tag="static-fluxes" ;;
        *HLLC*"Precision<double, float, double, double>"*MovingFlux\<false\>*"lambda(int, long long&)#2"*) tag="moving-pass1" ;;
        *HLLC*"Precision<double, float, double, double>"*MovingFlux\<true\>*"lambda(int, long long&)#2"*) tag="moving-pass2" ;;
        *) continue ;;
        esac
        body="${nf%.name}"
        total=$(grep -cE '^\s+/\*[0-9a-f]{4}\*/' "$body")
        ops=$(grep -oE '^\s+/\*[0-9a-f]{4}\*/\s+(@!?U?P[0-9T] )?[A-Z0-9_.]+' "$body" | awk '{print $NF}' | sed 's/\..*//' | sort | uniq -c |
              grep -E ' (MUFU|DSQRT|DMUL|DFMA|DADD|LDL|STL|CALL|BSSY|DSETP|LDG|STG|RED|ATOM)$' | awk '{printf "%s=%s ", $2, $1}')
        mufu=$(grep -oE 'MUFU\.[A-Z0-9]+' "$body" | sort | uniq -c | awk '{printf "%s=%s ", $2, $1}')
        echo "$label $tag: instructions=$total $ops $mufu"
    done
done
