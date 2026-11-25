#!/bin/bash
# When voronoi-cpp changes (TESTING.md), on one A100: builds OLD and NEW (configured CUDA build dirs; configure where there is
# network, or with -DLUNA_VORONOI_SOURCE), A100 ctest of NEW, then bits old against new: Sedov 128^3 (six paths, 50 steps;
# moving to t_end), 1D Sod / Shu-Osher / Woodward-Colella, 2D KH, 3D vortex and sound. luna_build_info.txt names each build's commits.
#   sbatch --export=ALL,OLD=<build dir>,NEW=<build dir>,OUT=<dir> -o <logs>/voronoi_bits-%j.out scripts/voronoi_bits_a100.sh
#SBATCH -J voronoi_bits
#SBATCH -p gpu1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=01:30:00
set -uo pipefail
source /etc/profile.d/modules.sh; module purge; module load gcc/14 cuda/13.0 cmake hdf5-serial
nvidia-smi --query-gpu=name,clocks.sm,clocks.applications.graphics,clocks.max.sm --format=csv
new="${NEW:?NEW: the build dir with the new voronoi-cpp}"
old="${OLD:?OLD: the build dir with the old voronoi-cpp}"
p="${OUT:-$new/voronoi-proof}"; mkdir -p "$p"

cmake --build $new --parallel 12 > $p/new.build.log 2>&1 &
cmake --build $old --parallel 6 --target luna_hydro_sedov luna_hydro_1d luna_hydro_kh2d luna_hydro_wave3d > $p/old.build.log 2>&1 &
wait -n; e1=$?; wait -n; e2=$?
echo "builds exit $e1 $e2"
grep -E " error|warning:" $p/new.build.log | grep -v "_deps/\|extern/" | head -10
for v in old new; do d=${!v}; echo "$v $(cat $d/luna_build_info.txt 2>/dev/null || echo "build: no luna_build_info.txt (a submodule tree)")"; done

ctest --test-dir $new --output-on-failure > $p/new.ctest.log 2>&1; echo "ctest exit $?"
grep -E "Test +#|tests passed|FAIL|Total Test" $p/new.ctest.log
grep -m1 "^build:" $new/Testing/Temporary/LastTest.log

echo "== Sedov 128^3 jittered, 50 steps (checksum: Q, x, volume)"
for flags in "" "--moving" "--order=0" "--update=cell --moving" "--riemann=hll --moving" "--precision=double --moving"; do
    for v in old new; do
        d=${!v}
        r=$($d/hydro_sedov 128 jittered $flags --snapshots=1 --steps=50 2>/dev/null | grep -E "^(checksum|mesh_updates)" | tr '\n' ' ')
        echo "$v [${flags:-static}] $r"
    done
done
echo "== Sedov 128^3 moving, to t_end (checksum)"
for v in old new; do
    d=${!v}
    echo "$v $($d/hydro_sedov 128 jittered --moving --snapshots=1 2>/dev/null | grep -E "^(checksum|mesh_updates|run)" | tr '\n' ' ')"
done
echo "== 1D (hash of state and positions)"
for prob in sod shu-osher woodward-colella; do
    for flags in "" "--moving"; do
        for v in old new; do
            d=${!v}
            echo "$v $prob 1024 [${flags:-static}] $($d/hydro_1d $prob 1024 $flags --repeats=1 2>/dev/null | grep -oE "steps=[0-9]+|hash=[0-9a-f]+" | tr '\n' ' ')"
        done
    done
done
echo "== 2D KH 256^2, 300 steps"
for flags in "" "--moving"; do
    for v in old new; do
        d=${!v}
        echo "$v kh [${flags:-static}] $($d/hydro_kh2d 256 $flags --steps=300 2>/dev/null | grep -E "^run:" )"
    done
done
echo "== 3D vortex and sound wave, 64 (field L1; the vortex state's sha256)"
for flags in "" "--moving"; do
    t=${flags:-static}; t=${t#--}
    for v in old new; do
        d=${!v}
        $d/hydro_wave3d vortex 64 $flags --out=$p/vortex_${t}_$v.npy 2>/dev/null | grep -E "^ours3d" | sed "s/^/$v vortex [$t] /"
        $d/hydro_wave3d sound 64 $flags 2>/dev/null | grep -E "^ours3d" | sed "s/^/$v sound [$t] /"
        echo "$v vortex [$t] sha256 $(sha256sum < $p/vortex_${t}_$v.npy | cut -c1-16)"
    done
done
