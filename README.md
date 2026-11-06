# luna-cpp

Moving-mesh finite-volume hydro in Kokkos C++ on GPUs (CUDA, validated on an A100 40 GB; OpenMP and Serial builds run
the same code for checks), on the exact Voronoi meshes of
[voronoi-cpp](https://github.com/Ulli1991/voronoi-cpp), which it includes as a git submodule.

MUSCL-Hancock with HLLC or HLL on static meshes, AREPO next generation's two flux passes as the default on moving ones, AREPO-style least-squares
gradients and mesh regularisation, on a static or moving mesh, bit-reproducible run to run, in 3D and 2D (periodic boxes) and 1D (with walls). On Sedov 3D at 256³ the moving mesh takes
925 steps where the static mesh takes 5834 (see "The hydro").

BSD 3-Clause, see [LICENSE](LICENSE).

## Install and build

```sh
git clone --recursive https://github.com/Ulli1991/luna-cpp.git
```

(In an existing clone: `git submodule update --init --recursive`.)

A build stops if `extern/voronoi-cpp` is behind voronoi-cpp `main`: update the pin and commit it. `-DLUNA_ALLOW_OLD_PIN=ON` only warns, to reproduce an old result.

Dependencies:

- CMake 3.16 or later, a C++17 compiler (gcc 14/15 tested).
- Kokkos 5.0 (5.0.2 in CI, 5.0.99 on Raven) with the backend you want: CUDA (A100: `Kokkos_ARCH_AMPERE80`, compiled
  through Kokkos' `nvcc_wrapper`; CUDA 13.0 tested), OpenMP or Serial.
- HDF5 (C) for `hydro_sedov`; without it the other targets still build.

CUDA (A100), on Raven:

```sh
module load gcc/14 cuda/13.0 cmake hdf5-serial
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/u/uli/kokkos-cuda \
  -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper -DCMAKE_CXX_EXTENSIONS=OFF
cmake --build build-cuda -j4
```

OpenMP (CPU), on Raven:

```sh
module load gcc/15 cmake hdf5-serial/1.12.2
cmake -S . -B build-omp -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/u/uli/kokkos-gcc15 -DCMAKE_CXX_FLAGS=-march=icelake-server
cmake --build build-omp -j4
```

Elsewhere, point `CMAKE_PREFIX_PATH` at a Kokkos installation with the desired backend. On Raven, build at most with
`-j4` on the login node, or inside a Slurm job; run GPU work through Slurm.

`cmake --build` with no target gives `hydro_sedov`, `hydro_1d`, `hydro_wave3d` and the tests. The pinned voronoi-cpp still holds its
own copy of the hydro, so the CMake targets here are `luna_hydro_library`, `luna_hydro_sedov` and `luna_hydro_1d`; to
build one program use those names (`--target hydro_1d` would build the submodule's copy).

### Compiler flags in use

- Release: `-O3 -DNDEBUG` (CMake's default), C++17 required (Kokkos sets `-std=c++20`), warnings
  `-Wall -Wextra -Wpedantic` on the programs.
- CUDA: `nvcc_wrapper` with `-extended-lambda -arch=sm_80` (from Kokkos' AMPERE80). Kokkos forces
  `CMAKE_CXX_EXTENSIONS=OFF` for this install and warns unless it is given. Do not set it for the OpenMP build: there
  the code compiles with `-std=gnu++20`, and `-std=c++20` changes gcc's FP contraction, so the bits.
- OpenMP: `-fopenmp` from Kokkos; no `-march` is set by the project (`-DCMAKE_CXX_FLAGS=-march=icelake-server` on
  Raven's CPUs).
- `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` (always on in `CMakeLists.txt`) writes the compilation database clang-tidy
  needs: `bash scripts/lint_cpp.sh build-omp` (Raven: `module load clang/20`).
- `-DVORONOI_SANITIZERS=ON` adds address and undefined-behaviour sanitizers to a host build.

## Tests

Details and run times are in [TESTING.md](TESTING.md). The mesh's own tests are in voronoi-cpp.

| Test | Specifies |
|---|---|
| `hydro` | HLLC and HLL against reference fluxes and under rotation; uniform states stay uniform, conservation to rounding, bit-identical runs, the two update loops agree to rounding on jittered and uniform meshes; moving mesh: uniform state with a bulk velocity, conservation, Galilean invariance of a moving blast |
| `hydro_wave` | the moving 3D step on smooth flow, both integrators: Yee's vortex (jitter 0.1) and a sound wave (jitter 0.25), L1 of rho, v, P and the rate of rho against measured bounds; two passes keep second order where one pass loses it |
| `hydro2d` | the 2D hydro, static and moving, both integrators: uniform states, conservation, bit-identical runs, sound-wave order in cell averages over the actual polygons, Sod, Noh, a Galilean blast |
| `hydro1d` | the 1D hydro: uniform states, conservation, bit-identical runs, inflow, cells from a wall with the budget through it; Sod against the exact solution, static and moving; LUNA's final states of Sod, Shu-Osher and Woodward-Colella at N = 16 to 1e-12 |

```sh
ctest --test-dir build-cuda --output-on-failure
OMP_NUM_THREADS=1 ctest --test-dir build-omp
```

CI (`.github/workflows/checks.yml`) runs clang-tidy and the CPU tests in ordinary and sanitizer builds against a
pinned Serial Kokkos 5.0.2, with the submodule checked out; it does not run GPU jobs.

## The hydro

`hydro::FiniteVolume` (`src/hydro/`) runs finite volumes on a static or
moving `voronoi::Mesh` (moving: the points follow the flow with a drift
towards the cell centroids, the mesh is repaired every step, fluxes go
through the moving faces): extensive conserved state only between steps, least-squares
gradients of the primitives limited at the face centroids, MUSCL-Hancock
half step, HLLC (default) or HLL along each face normal, one global time
step (Courant factor 0.3 on each cell's radius and signal speeds); order 0
is first-order Godunov. Every cell sums over its face list in a fixed
order, so runs are bit-reproducible. Two update loops: per face (a face
flux array, then a per-cell gather; default at second order) or per cell
(each face flux computed from both sides; default at first order).

On a moving mesh at order 1 the default integrator is AREPO next generation's two passes: fluxes on the mesh of the
start of the step, half applied, the mesh update, fluxes on the new mesh from the same primitives and gradients advanced
a whole step, half applied. `--integrator=one-pass` selects the MUSCL-Hancock step of the static mesh there, with the
bits it had before (hash in research_notes/two_pass_2026-11-05.md). The numbers below that a moving run gives are for the
integrator named in each; costs and accuracy of the two side by side are in that note.

```text
hydro_sedov <side> <jittered|uniform> [--order=0|1] [--update=cell|face] [--limiter=face|cell|none]
            [--riemann=hllc|hll] [--precision=default|mixed|low|...] [--moving] [--bulk=km/s] [--out=dir] [--snapshots=10] [--gradients] [--steps=max] [--input-order] [--memory]
            [--integrator=one-pass|two-pass]
hydro_wave3d <sound|simple|vortex|tube> <nx> [--moving] [--integrator=one-pass|two-pass] [--jitter=spacings] [--out=file.npy]
sbatch -o <logs>/hydro-%j.out scripts/hydro_a100.sh build-cuda <out> <mode> (see the script)
```

`hydro_sedov` (needs HDF5; on Raven `module load hdf5-serial`) runs a
Sedov blast (1e51 erg, n = 1 cm⁻³, 50 pc box, 1e4 yr) and writes HDF5
snapshots. Measured on one A100, double, jittered points in Morton order,
median of 40 steps. The 2.1M column is current (static job 31879405, moving jobs 31925156 and
31925157, HLLC rows; the same code varies by about 10 % between GPU jobs). In the 16.8M
column the moving rows are current (job 31925157, 1305 MHz), and so are static HLLC and LUNA
(job 31953468, 1305 MHz); the HLL and first-order rows and the static memory row are the earlier
code (jobs 31699864, 31699865, 31700852, kept until they are re-measured):

| G cell updates per second | 2,097,152 cells | 16,777,216 cells |
|---|---|---|
| moving mesh, two passes (default), HLLC, per-face loop (mesh update and face lists included) | **0.066** | 0.068 (whole run) |
| moving mesh, one pass, same | 0.084 | 0.088 (whole run) |
| static mesh, MUSCL-Hancock HLLC, per-face loop | **0.152** | 0.149 |
| static mesh, MUSCL-Hancock HLL, per-face loop | 0.116 | 0.114 |
| static mesh, first order HLLC, per-cell loop | 0.288 | 0.290 |
| LUNA, MC, HLLC, Pallas, FP64 | 0.76 | 0.954 |
| LUNA, same, FP32 | 1.37 | 2.133 |
| device memory, mesh + hydro (static, per-face, second order) | 2434 B/cell | 2371 B/cell (41.2 GB in use) |
| device memory, moving mesh + hydro, two passes (per-face) | 2319 B/cell | 2319 B/cell (40.5 GB in use, whole run); `--precision=mixed` 2445 B/cell, does not fit at 256³, see Limitations |
| device memory, moving mesh + hydro, one pass (per-face) | 2298 B/cell | 2298 B/cell (40.1 GB in use, whole run); `--precision=mixed` 2334 B/cell (40.7 GB, job 31925336) |
| device memory, LUNA, FP64 | 0.22 GB | 1.25 GB, 0.63 GB in FP32 |

Sedov at 128³ (job 31699863; the moving numbers of this paragraph are one pass), HLLC: shock radius 0.978 of the
Sedov-Taylor value at 1e4 yr, L1 density error 0.075 rho0 (first order:
0.104); mass and energy conserved to 2e-16. On the moving mesh (jobs
31700850-51): 354 steps instead of 1642, L1 density error 0.058 rho0, the
same numbers with the whole box moving at 500 km/s; at 256³ L1 0.034 in
288 s of GPU time (job 31702077; now 176 s, job 31925157, two passes 230 s). These L1 values are sampled at
the mesh generators, which flatters a moving mesh; the field L1 (the integral
of |rho_h - rho| dV) is the fair norm and is what the test suite will report
(research_notes/accuracy_score_exploration_2026-11-04.md). Details:
research_notes/hydro_stage1_2026-10-31.md, hydro_stage3_2026-10-31.md. Field L1 of density, r < 1.5 R_s (one pass, two
passes): 128³ 0.0629, 0.0648; 256³ 0.0369, 0.0377 (research_notes/two_pass_2026-11-05.md).

### 2D (`hydro_kh2d`, `hydro::FiniteVolume<2>`)

![Kelvin-Helmholtz, moving against LUNA](docs/figures/kh_2d_evolution.png)

Kelvin-Helmholtz of McNally, Lyra & Passy (2012), t = 0, 1, 2.5, 4, 5, moving (two passes) against LUNA MC, both N = 2048 and FP64, magma with LUNA's 0.9 to 2.1. Moving: nearest cell plus the run's limited gradient from the centroid, linear reconstruction in cells as in AREPO; LUNA: its cell values. HLLC, LUNA Pallas, CFL 0.8; ours face limiter, per-face, LUNA's rule 0.8. Moving to t = 5 in two legs, the restart not bit-identical: 69535 steps, 0 mesh rebuilds, 0 stuck; mass, x momentum and energy kept to 2e-12 or better, relative. Past t = 2.5 secondary instabilities on the moving billows set in earlier than in LUNA; this is not a converged solution in either code. Jobs 31953688, 31953871; LUNA 31953667, 31954283.

![Kelvin-Helmholtz Voronoi mesh](docs/figures/kh_2d_mesh.png)

The moving mesh in Kelvin-Helmholtz, N = 512, two passes, t = 4, the whole box and three nested zooms into a billow, each box marked on the panel before; magma with LUNA's 0.9 to 2.1. The map is the scheme's own field, linear reconstruction in cells as in AREPO: each pixel takes its nearest cell plus the run's limited gradient from the centroid. Over it the exact Voronoi edges (`scipy.spatial.Voronoi`, periodic images), thin and fading where a cell is shorter than a few pixels. Job 31953904; 0 mesh rebuilds, 0 stuck.

![Kelvin-Helmholtz density](docs/figures/kh_2d_density.png)

Density at t = 1.0, 1.5, 2.5, 3.0, one colour scale, LUNA MC, static, moving (two passes), all N = 2048, with a zoom on one roll-up at t = 1.5 and the differences from LUNA, sampled at LUNA's cell centres, symmetric log scale. Static and moving shown as linear reconstruction in cells as in AREPO. Mean |Δρ| against LUNA, t = 1.0, 1.5, 2.5, 3.0: static 4e-8, 5e-7, 5e-5, 6e-4 (its panel at t = 1.5 is white: largest difference 2e-4); moving 5e-5, 1e-4, 0.012, 0.073, the billows no longer match past t = 1.5. Static job 31953690: 62390 steps.

`hydro_kh2d <N> [--moving] [--integrator=one-pass|two-pass] [--bulk=<u>] [--out-times=t,...] [--stop-after=<s>] [--resume] ...` runs this test; the last two split a long run across jobs.
G cell updates/s at N = 2048 (job 31952221): LUNA 1.78, static 0.756, moving one pass 0.414, two passes 0.322; peak 693, 735 and
749 B/cell for ours. GPU seconds to max |e| = 0.001: LUNA 0.55, static 1.1, moving 15. Details and the comparison with LUNA at the
same N: research_notes/stage2_2d_2026-11-05.md; `scripts/dim2d_a100.sh` runs the ladders, `scripts/compare_kh.py` makes the tables.

### 1D (`hydro_1d`, `finite_volume_1d.hpp`)

`hydro::FiniteVolume<1, Exec, Riemann>` on `voronoi::Mesh<1>`, static or
moving, with walls (reflective, zero gradient, inflow); a wall that gas enters
supplies cells, split conservatively from the end cell. The whole step is one
kernel. Defaults: a static run uses Courant x V / (|v| + c) at 0.8, one launch
per step that also finds the next time step (`step_and_time_step`); a moving
run uses AREPO next generation's rule with its pair term at 0.4 (r = V / 2 in
1D), three launches (time step, step, order check). 3D stays at 0.3. On a
uniform static line the scheme is second-order MUSCL-Hancock with the MC
limiter. Points are regularised as
AREPO next generation does without a face angle: a drift towards the
density-displaced centroid when it is more than 0.75 cell radii away.

G cell updates/s, one A100 40 GB at 1305 MHz (ravg1002, job 31917211, Sod,
HLLC, 2000 steps, 200 at 2²⁴; hydro only is a static step, whole step a
moving one at its defaults with the order check):

| cells | 4,096 | 65,536 | 1,048,576 | 16,777,216 |
|---|---|---|---|---|
| hydro only | 0.081 | 0.948 | 2.810 | 3.226 |
| whole moving step | 0.048 | 0.636 | 1.872 | 2.297 |

Below ~1e5 cells a step is bound by launch latency (~60 us), not by the
kernel. Peak device memory: 144 B/cell static, 152 B/cell moving. Runs are
bit-reproducible (the hash of the final state and positions). Field accuracy on Sod, Shu-Osher and Woodward-Colella:
research_notes/stage1_1d_2026-11-04.md.

## Figures

From `scripts/make_figures.py`, in LUNA's figure style (`scripts/figstyle.py`).

![Sod density](docs/figures/sod.png)

Sod, t = 0.25, N = 512, cell means at centroids against the exact solution; the boxes mark the zooms, filled markers are cells, a regular subset where they would touch (static lies on LUNA, drawn dashed). Moving has the sharper contact and 2.5 times LUNA's error at the tail of the rarefaction (x = 0.48); the strip clips the contact and shock spikes. HLLC, MUSCL-Hancock, FP64; LUNA: MC, Pallas, CFL 0.8; static: uniform points, face limiter, LUNA's time step rule at 0.8; moving: face limiter, the default rule at 0.4, regularised. Job 31917206, voronoi-cpp 947946a.

![Shu-Osher density](docs/figures/shu_osher.png)

Shu-Osher, t = 1.8, N = 1024 on [-5, 5]. LUNA and static, job 31917208, 947946a, runs as for Sod. Moving, default rule at 0.4, job 31953172: gas enters through the left wall, which supplies cells at the initial spacing, 1508 at the end ([research_notes/inflow_2026-11-06.md](research_notes/inflow_2026-11-06.md)). Zooms of the regions marked above, open markers a regular subset of the cells where they would touch. Curves: luna-reference-solutions 99a52d0, `luna/shu_osher/shu_osher_mc.npz` (LUNA b7b0d35, MC HLLC, 65536 on 8192 cells), `athenak/shu_osher/so_ppmx_rk4_16384.npz` (AthenaK c52f88f).

![Woodward-Colella density](docs/figures/woodward_colella.png)

Woodward-Colella, t = 0.038, N = 2048, reflecting walls, shown from x = 0.4 (flat to its left); zooms on each contact and shock, filled markers a regular subset of the cells where they would touch. Moving holds them in far fewer cells than LUNA, with an overshoot at x = 0.765. Job 31917210, 947946a, runs as for Sod. Curves: luna-reference-solutions 99a52d0, `luna/woodward_colella/woodward_colella_mc.npz` (LUNA b7b0d35, MC HLLC, 65536 on 8192 cells), `athenak/woodward_colella/a9_ppmx_rk4_16384.npz` (AthenaK c52f88f).

![Field L1 by N](docs/figures/convergence_1d.png)

Field L1 of the reconstructed field (`scripts/compare_1d.py`), against the exact solution on Sod, LUNA MC 2²¹ cells on Shu-Osher, LUNA MC 2²⁰ cells on Woodward-Colella. Sod: moving ahead of LUNA in rho, 2-13 % over it in P. Shu-Osher: moving at or below LUNA in rho, v and P at every N to 2¹⁷ (ratios 0.39-0.81, 0.79-0.84, 0.83-0.99), N being the cells at the start; rho falls off order 1 above 2¹⁵, a second stall not chased; job 31953172, cell updates in the notes. Woodward-Colella: ahead of LUNA, with a stall near 3.4e-4 in rho. Jobs 31917206/8/10, 947946a.

![GPU time against field L1](docs/figures/time_to_target_1d.png)

GPU seconds of the stepping loop, one A100, one point per N, dotted lines the targets of the notes; LUNA's compile not counted, below ~1e5 cells a step is launch latency. LUNA wins Sod and Shu-Osher, there by 7-16x in GPU time to the same error: moving takes 7.5x the steps and ends with 1.47x the cells. On Woodward-Colella it is close in rho and P, and only moving reaches rho 1e-3 in the ladder. One job per test: Sod 31917206, Shu-Osher 31953172 at 1095 MHz, Woodward-Colella 31917210 at 1305 MHz, 947946a.

![3D Yee vortex](docs/figures/vortex_3d.png)

Yee vortex in a 3D slab, 10 x 10 x 2.5, nx x nx x nx/4 cells, t = 10. L1 of cell values against the exact average over each cell, v the vector. Moving one pass is first order; two passes (now the default) reach order 2.2, then 1.5, and stay above static. O(h) floor: exact averages as a piecewise-constant field against the exact field; it binds a field norm, not this measure. Ours: HLLC, face limiter, Courant 0.3, jitter 0.1, start from the point value at the centroid, jobs 31920446, 31922276. LUNA: MC, HLLC, Pallas, CFL 0.8, luna-reference-solutions 99a52d0 `luna/yee_vortex/yee_vortex_convergence.npz` (LUNA debcdae + uncommitted, bitwise at 60cf841).

![2D Kelvin-Helmholtz](docs/figures/kh_2d.png)

Kelvin-Helmholtz after McNally, Lyra & Passy (2012), N² cells, one A100 at 1305 MHz; e(t) = M / M_ref - 1 against their published reference curve, max over t <= 1.5, dotted lines the targets. LUNA and static lead at equal N, LUNA in GPU time; moving with one pass stalls near 4e-4, two passes recover. All HLLC, FP64; LUNA MC, Pallas, CFL 0.8; ours face limiter, per-face, LUNA's rule 0.8. Jobs 31952216, 31952221.

![Sedov 3D](docs/figures/sedov_3d.png)

Sedov 3D, 1e51 erg, n = 1 cm⁻³, 50 pc box, 1e4 yr: a shock test that cannot rank schemes, order below 1 for every code (0.66 LUNA minmod, 0.78 moving, n = 128 to 256). Left: volume-weighted shells at 128³ with zooms on the shell and the shock front, markers are shells, not cells, branch dim-limiters, jobs 31918879 (moving), 31918880 (static), 31918732 (LUNA). Right: field L1 over r < 1.5 R_s of the piecewise-constant field, not sampled at generators; older code, jobs 31879294, 31879807 (LUNA minmod 128-448, moving 64-256, 256 with the per-cell loop, static 64-192), LUNA MC 128 and 256 from dim-limiters. Ours: HLLC, jittered, Courant 0.3, face limiter; moving is one pass in all these runs, older than the two-pass default.

![Sedov 3D at 256, shock](docs/figures/sedov3d_shock.png)

The same Sedov at 256³, as in LUNA's own 3D figures, snapshots every 1000 yr, two passes, against LUNA MC (HLLC, Pallas, CFL 0.8) and Sedov-Taylor. Top: shock radius as the location of the density peak in the shells, and n(r) at five times, numbers in 10³ yr above each shock. Below: column density at 1e4 yr, one scale, and the difference. Moving: 512² rays along z with 1024 midpoint samples, each in its nearest cell with the run's limited gradient from the centroid, linear reconstruction in cells as in AREPO. LUNA: cell values summed along z. Difference up to 9 % of the peak at the shock, mean 0.2 %; the cross is LUNA's grid. LUNA's published figure is at 1024³: all runs here are 256³, four times coarser, so the peak falls below Sedov-Taylor by resolution, and the two are not to be read as equal. Job 31953353 (moving 925 steps, 1.49 M fallbacks; LUNA 4035 steps).

![Sedov 3D at 256, radial profiles](docs/figures/sedov3d_radial.png)

Volume-weighted shells at 1e4 yr, n_x / 2 bins, density, radial velocity, pressure and temperature against Sedov-Taylor; static job 31953728, 5834 steps, the others as above. The moving curves start at 1.2 pc: inside it the expanded cells are so few that the shells hold no generator, and the curve is left open. The moving mesh follows Sedov-Taylor down to n of 10⁻³, where the fixed grids hold a floor near 10⁻².

![Sedov 3D at 256, mesh at the shell](docs/figures/sedov3d_mesh.png)

Slice z = 25 pc at 1e4 yr, exact plane section of the 3D Voronoi cells, n in each cell, thin edges, generators near the plane in black: (a) moving, the cells gather in the shell and open behind it; zooms into the same window, (b) moving, (c) static.

Throughput against resolution on one A100, 32³ to 256³, Sedov 3D, 1095 MHz, double, HLLC, 40 steps, with LUNA (MC, Pallas, FP64; median of the perf lines of three rounds) and peak memory. Hydro is flat from 96³ to 256³; below 64³ the fixed cost per step dominates. Job 31928911, mesh pin 0b433f0.

![Throughput against resolution](docs/figures/throughput_scaling.png)

Precision options against resolution on one A100, 32³ to 256³, the same Sedov, HLLC, 40 steps: hydro static, whole two-pass moving step, GPU memory below. Job 31953468, e90a913, 1305 MHz. LUNA in the same job (MC, Pallas, 200 steps, median of the perf lines), FP64 to compare with double, FP32 throughout; none of our options is FP32 throughout. At 128³ `grad16` gives +15 % hydro, +10 % whole step, `grad32` +11 % and +7 %, `mixed` +10 % and +6 %, `flux32` nothing. `mixed` two-pass does not fit at 256³.

![Throughput by precision](docs/figures/throughput_precision.png)

| 256³, job 31953468 | double | `grad32` | `grad16` | LUNA FP64 | LUNA FP32 |
|---|---|---|---|---|---|
| hydro static, G cell updates/s | 0.149 | 0.166 | 0.170 | 0.954 | 2.133 |
| whole two-pass step, G cell updates/s | 0.070 | 0.076 | 0.077 | | |
| GPU GB, two passes | 40.5 | 39.4 | 38.9 | 1.25 | 0.63 |

Throughput against resolution in 1D, 2D and 3D on one A100, HLLC, FP64, with LUNA (MC, Pallas) and peak memory. A blast in all three: LUNA's Sedov without units in 1D (gamma 1.4) and 2D (gamma 5/3), the Sedov above in 3D. 1D has no two-pass integrator, so its whole step is one pass. At the largest size the static hydro runs 2.66, 0.67 and 0.15 G cell updates/s, LUNA 2.96, 1.46 and 0.95: level in 1D, 2.2 times ahead in 2D, 6.4 times in 3D, in line with the faces per cell, not separated from other causes. Our 1D leads LUNA up to 4M cells, where LUNA's fixed cost per step rules. Two passes take the whole step to 0.29 in 2D and 0.07 in 3D. Jobs 31953276 and 31953278 at 1305 MHz, 31953467 at 1095 MHz; the 1D and 2D blast is not on main yet, runs at voronoi-cpp d077c41.

![Throughput in 1D, 2D and 3D](docs/figures/throughput_dimensions.png)

## Limitations and open items

- The hydro step is not yet at its target (0.3 G cell updates/s static, 0.15 G the whole two-pass step): on the A100 at 128³
  double precision runs at 0.152 G cell updates/s static, its hydro part at 0.140 G cell updates/s on a moving mesh (one pass), the
  whole moving step at 0.084 G cell updates/s with one pass, 0.066 with two. Roofline at 1095 MHz (luna-cpp 28a647d, voronoi-cpp
  0b433f0, older than this tree): the step sits at about 1.9 flop/B, at 45 % of the sustained 1.40 TB/s and 16 % of the FP64 peak;
  only `update` is at the bandwidth roof, the gradient sweep is latency-bound. Bytes alone would allow 0.34 G static and 0.14 G for
  the whole two-pass step, which achieved 0.059 there. Below kernel level (research_notes/profile_2026-11-06.md, 128³, 1095 MHz) the
  hydro kernels spend 61-98 % of their warp slots waiting for memory, not on flops: gradient sweep 0.268, fluxes 0.571, `update`
  0.805 G cell updates/s static. The sweep is the slowest: 148 registers keep 17 % of its warps active. Compile-time precision options (`--precision=`, default double): mixed +10 %, low +19 % on the
  hydro static, +4 % / +7 % on the whole moving step with one pass (research_notes/hydro_speed_2026-11-04.md).
- Moving 256³ on a 40 GB A100: two passes in double fit (2319 B/cell, 40484 of 42405 MB in use); with `--precision=mixed`
  they do not (2445 B/cell, about 42.6 GB needed): use `--integrator=one-pass` there, or `--precision=grad32`
  (2255 B/cell). The double area and centroid arrays stay beside mixed's float face rows.
- On the CPU (gcc's default FP contraction) some hydro modes differ from the GPU in the last bits; the A100 runs are
  bit-identical run to run.
- Periodic boxes only in 3D and 2D; only 1D has walls.
- 1D moving mesh: a wall supplies cells where gas enters, but nothing removes points where gas leaves: no test has points reach an open wall, and that path is not tested. A subsonic inflow keeps LUNA's zero-gradient wall, whose state drifts by about 1e-7 at N = 2¹⁷ (research_notes/inflow_2026-11-06.md).
- 1D moving mesh: the error of Woodward-Colella stops falling near N = 2¹⁵, at every Courant number tried
  (research_notes/stage1_1d_2026-11-04.md).

## Roadmap

Done: 3D, 2D and 1D hydro on the exact mesh, two passes as the default on moving meshes, a first round of hydro speed
(0.107 -> 0.152 G cell updates/s static).

1. `grad32` as the default.
2. Inflow that adds cells ([#3](https://github.com/Ulli1991/luna-cpp/issues/3)), the 1D soft launch of refinement.
3. Hydro speed: target 0.3 G cell updates/s static, 0.15 G the whole two-pass step. Next: an owner bit in the list entries (estimate +8 % static, +5 % whole step), faces numbered by owner in the mesh (+8 % hydro measured with a copy, +5 % whole step estimated). Sorting cells by list length and reusing the half step between passes were measured and made it slower.
4. A standing 3D test suite with LUNA coverage ([#7](https://github.com/Ulli1991/luna-cpp/issues/7)), including a test that doesn't favour lattices.
5. Time integration study ([#1](https://github.com/Ulli1991/luna-cpp/issues/1)), accuracy scorecard.
6. Local time stepping from AREPO-ng ([#9](https://github.com/Ulli1991/luna-cpp/issues/9)), a CPU path for few active cells ([#10](https://github.com/Ulli1991/luna-cpp/issues/10)).
7. Multi-GPU.
8. Refinement.
9. Multigrid self-gravity.
10. Autodiff in C++ with Enzyme ([#12](https://github.com/Ulli1991/luna-cpp/issues/12)).

Open bugs: a forced rebuild beside two passes runs out of memory at 256³ ([#11](https://github.com/Ulli1991/luna-cpp/issues/11)); stuck flips at 256³ ([#8](https://github.com/Ulli1991/luna-cpp/issues/8)); NaN from zero-area faces on an exact lattice ([#6](https://github.com/Ulli1991/luna-cpp/issues/6)). Mesh bugs that reach the hydro are listed in the [voronoi-cpp README](https://github.com/Ulli1991/voronoi-cpp#limitations-and-open-items).

## Repository layout

- `src/hydro/`: the hydro (header-only). `src/bench/`: `hydro_sedov`, `hydro_1d`, `hydro_wave3d`, `hydro_kh2d`. `src/test/`: the tests,
  `exact_riemann.hpp`, `luna_1d_values.hpp`.
- `extern/voronoi-cpp/`: the mesh library, pinned submodule.
- `scripts/`: `hydro_a100.sh`, `hydro_speed_a100.sh`, `dim1d_a100.sh`, `dim2d_a100.sh`, `dims_a100.sh`, `two_pass_a100.sh`, `snapshot_hash.sh`, `compare_1d.py`, `compare_kh.py`,
  `hydro_sedov_compare.py`, `sedov_field_l1.py`, `norm3d.py`, `luna_*.py`, `make_figures.py`, `figstyle.py`, `lint_cpp.sh`.
- `docs/figures/`: the figures.
- `research_notes/`, `reports/`: status notes.
