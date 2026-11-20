#!/bin/bash
# A voronoi-cpp pin move on one A100: CUDA builds of OLD (old pin) and NEW (new pin) worktrees, A100 ctest of NEW, then bits
# old against new: Sedov 128^3 (five paths, 50 steps; moving to t_end), 1D Sod / Shu-Osher / Woodward-Colella, 2D KH.
#   sbatch --export=ALL,OLD=<worktree>,NEW=<worktree> -o <logs>/pin_bits-%j.out scripts/pin_bits_a100.sh   (TESTING.md)
#SBATCH -J pin_bits
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=01:30:00
set -uo pipefail
source /etc/profile.d/modules.sh; module purge; module load gcc/14 cuda/13.0 cmake hdf5-serial
nvidia-smi --query-gpu=name,clocks.sm,clocks.applications.graphics,clocks.max.sm --format=csv
new="${NEW:?NEW: the worktree with the new pin}"
old="${OLD:?OLD: the worktree with the old pin}"
p="${OUT:-$new/build-pin-proof}"; mkdir -p "$p"
for t in $old $new; do echo "$t: $(git -C $t log --oneline -1 | cut -c1-60) voronoi-cpp $(git -C $t/extern/voronoi-cpp log --oneline -1 | cut -c1-8)"; done
NEWB=${NEW_BUILD:-$new/build-cuda}; OLDB=${OLD_BUILD:-$old/build-cuda}  # build dirs (e.g. on /ptmp)
declare -A B=([new]=$NEWB [old]=$OLDB)
cfg="-DCMAKE_BUILD_TYPE=Release -DKokkos_DIR=/u/uli/kokkos-cuda/lib64/cmake/Kokkos -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper"

cmake -S $new -B $NEWB $cfg > $p/cuda-new.cfg.log 2>&1; echo "configure new exit $?"
cmake -S $old -B $OLDB $cfg > $p/cuda-old.cfg.log 2>&1; echo "configure old exit $?"
cmake --build $NEWB --parallel 12 > $p/cuda-new.build.log 2>&1 &
cmake --build $OLDB --parallel 6 --target luna_hydro_sedov luna_hydro_1d luna_hydro_kh2d > $p/cuda-old.build.log 2>&1 &
wait -n; e1=$?; wait -n; e2=$?
echo "builds exit $e1 $e2"
grep -E " error|warning:" $p/cuda-new.build.log | grep -v extern/ | head -10
ls -la $NEWB/hydro_sedov $OLDB/hydro_sedov

ctest --test-dir $NEWB --output-on-failure > $p/cuda-new.ctest.log 2>&1; echo "ctest exit $?"
grep -E "Test +#|tests passed|FAIL|Total Test" $p/cuda-new.ctest.log
grep -c "^ok\|  ok" $p/cuda-new.ctest.log

echo "== Sedov 128^3 jittered, 50 steps (checksum: Q, x, volume)"
for flags in "" "--moving" "--order=0" "--update=cell --moving" "--riemann=hll --moving"; do
    for v in old new; do
        d=${!v}
        r=$(${B[$v]}/hydro_sedov 128 jittered $flags --snapshots=1 --steps=50 2>/dev/null | grep -E "^(checksum|mesh_updates)" | tr '\n' ' ')
        echo "$v [${flags:-static}] $r"
    done
done
echo "== Sedov 128^3 moving, to t_end (checksum)"
for v in old new; do
    d=${!v}
    echo "$v $(${B[$v]}/hydro_sedov 128 jittered --moving --snapshots=1 2>/dev/null | grep -E "^(checksum|mesh_updates|run)" | tr '\n' ' ')"
done
echo "== 1D (hash of state and positions)"
for prob in sod shu-osher woodward-colella; do
    for flags in "" "--moving"; do
        for v in old new; do
            d=${!v}
            echo "$v $prob 1024 [${flags:-static}] $(${B[$v]}/hydro_1d $prob 1024 $flags --repeats=1 2>/dev/null | grep -oE "steps=[0-9]+|hash=[0-9a-f]+" | tr '\n' ' ')"
        done
    done
done
echo "== 2D KH 256^2, 300 steps"
for flags in "" "--moving"; do
    for v in old new; do
        d=${!v}
        echo "$v kh [${flags:-static}] $(${B[$v]}/hydro_kh2d 256 $flags --steps=300 2>/dev/null | grep -E "^run:" )"
    done
done
