# luna-cpp

Moving-mesh finite-volume hydro in Kokkos C++ on GPUs (CUDA, validated on an A100 40 GB; OpenMP and Serial builds run
the same code for checks), on the exact Voronoi meshes of
[voronoi-cpp](https://github.com/Ulli1991/voronoi-cpp), which it includes as a git submodule.

MUSCL-Hancock with HLLC or HLL, AREPO-style least-squares gradients and mesh regularisation, on a static or moving
mesh, bit-reproducible run to run, in 3D (periodic boxes) and 1D (with walls). On Sedov 3D the moving mesh is more
accurate than the static mesh at the same resolution in 4.6x fewer steps (see "The hydro").

BSD 3-Clause, see [LICENSE](LICENSE).

## Install and build

```sh
git clone --recursive https://github.com/Ulli1991/luna-cpp.git
```

(In an existing clone: `git submodule update --init --recursive`.)

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

`cmake --build` with no target gives `hydro_sedov`, `hydro_1d` and the tests. The pinned voronoi-cpp still holds its
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
| `hydro1d` | the 1D hydro: uniform states, conservation, bit-identical runs, inflow; Sod against the exact solution, static and moving; LUNA's final states of Sod, Shu-Osher and Woodward-Colella at N = 16 to 1e-12 |

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

```text
hydro_sedov <side> <jittered|uniform> [--order=0|1] [--update=cell|face] [--limiter=face|cell|none]
            [--riemann=hllc|hll] [--precision=default|mixed|low|...] [--moving] [--bulk=km/s] [--out=dir] [--snapshots=10] [--steps=max] [--input-order] [--memory]
sbatch -o <logs>/hydro-%j.out scripts/hydro_a100.sh build-cuda <out> <mode> (see the script)
```

`hydro_sedov` (needs HDF5; on Raven `module load hdf5-serial`) runs a
Sedov blast (1e51 erg, n = 1 cm^-3, 50 pc box, 1e4 yr) and writes HDF5
snapshots. Measured on one A100, double, jittered points in Morton order,
median of 40 steps. The 2.1M column is current (static job 31879405, moving job 31886231,
HLLC rows; the same code varies by about 10 % between GPU jobs). In the 16.8M
column the moving rows are current (job 31885794); the static rows and the
HLL and first-order rows are the earlier code (jobs 31699864, 31699865,
31700852, kept until they are re-measured):

| Million cell updates per second | 2,097,152 cells | 16,777,216 cells |
|---|---|---|
| moving mesh, MUSCL-Hancock HLLC, per-face loop (mesh update and face lists included) | **84** | 88 (whole run, job 31885794) |
| static mesh, MUSCL-Hancock HLLC, per-face loop | **152** | 105 |
| static mesh, MUSCL-Hancock HLL, per-face loop | 116 | 114 |
| static mesh, first order HLLC, per-cell loop | 288 | 290 |
| device memory, mesh + hydro (static, per-face, second order) | 2434 B/cell | 2371 B/cell (41.2 GB in use) |
| device memory, moving mesh + hydro (per-face) | 2298 B/cell | 2298 B/cell (40.1 GB in use, whole run, job 31885794); `--precision=mixed` 2334 B/cell (40.7 GB, job 31885795) |

Sedov at 128^3 (job 31699863), HLLC: shock radius 0.978 of the
Sedov-Taylor value at 1e4 yr, L1 density error 0.075 rho0 (first order:
0.104); mass and energy conserved to 2e-16. On the moving mesh (jobs
31700850-51): 354 steps instead of 1642, L1 density error 0.058 rho0, the
same numbers with the whole box moving at 500 km/s; at 256^3 L1 0.034 in
288 s of GPU time (job 31702077; now 177 s, job 31885794). These L1 values are sampled at
the mesh generators, which flatters a moving mesh; the field L1 (the integral
of |rho_h - rho| dV) is the fair norm and is what the test suite will report
(research_notes/accuracy_score_exploration_2026-11-04.md). Details:
research_notes/hydro_stage1_2026-10-31.md, hydro_stage3_2026-10-31.md.

### 1D (`hydro_1d`, `finite_volume_1d.hpp`)

`hydro::FiniteVolume<1, Exec, Riemann>` on `voronoi::Mesh<1>`, static or
moving, with walls (reflective, zero gradient, inflow). The whole step is one
kernel. Defaults: a static run uses Courant x V / (|v| + c) at 0.8, one launch
per step that also finds the next time step (`step_and_time_step`); a moving
run uses AREPO next generation's rule with its pair term at 0.4 (r = V / 2 in
1D), three launches (time step, step, order check). 3D stays at 0.3. On a
uniform static line the scheme is second-order MUSCL-Hancock with the MC
limiter. Points are regularised as
AREPO next generation does without a face angle: a drift towards the
density-displaced centroid when it is more than 0.75 cell radii away.

M cell updates/s, one A100 40 GB at 1305 MHz (ravg1002, job 31917211, Sod,
HLLC, 2000 steps, 200 at 2^24; hydro only is a static step, whole step a
moving one at its defaults with the order check):

| cells | 4,096 | 65,536 | 1,048,576 | 16,777,216 |
|---|---|---|---|---|
| hydro only | 81 | 948 | 2810 | 3226 |
| whole moving step | 48 | 636 | 1872 | 2297 |

Below ~1e5 cells a step is bound by launch latency (~60 us), not by the
kernel. Peak device memory: 144 B/cell static, 152 B/cell moving. Runs are
bit-reproducible (the hash of the final state and positions). Field accuracy on Sod, Shu-Osher and Woodward-Colella:
research_notes/stage1_1d_2026-11-04.md.

## Figures

From `scripts/make_figures.py`; each figure states its runs, jobs and commit in its footer.

<picture><source media="(prefers-color-scheme: dark)" srcset="docs/figures/sod-dark.png"><img src="docs/figures/sod.png" alt="Sod density"></picture>

Sod density, N = 512: moving against LUNA and static (same values) and the exact solution; moving has the sharper contact and twice LUNA's error at the rarefaction.

<picture><source media="(prefers-color-scheme: dark)" srcset="docs/figures/shu_osher-dark.png"><img src="docs/figures/shu_osher.png" alt="Shu-Osher density"></picture>

Shu-Osher density, N = 1024: on LUNA's domain moving fails, the grown inflow cell flattens the waves; on the padded domain it follows the reference, at rule 0.3 on OpenMP (older branch, the default 0.4 was not run padded). References: luna-reference-solutions 99a52d0.

<picture><source media="(prefers-color-scheme: dark)" srcset="docs/figures/woodward_colella-dark.png"><img src="docs/figures/woodward_colella.png" alt="Woodward-Colella density"></picture>

Woodward-Colella density, N = 2048: moving holds the contacts in far fewer cells than LUNA, with an overshoot at x = 0.765. References: luna-reference-solutions 99a52d0.

<picture><source media="(prefers-color-scheme: dark)" srcset="docs/figures/convergence_1d-dark.png"><img src="docs/figures/convergence_1d.png" alt="Field L1 by N"></picture>

Field L1 by N, 1D: Sod, moving ahead of LUNA in rho, 2-13 % over it in P; Shu-Osher, a floor at 0.037 on LUNA's domain, padded runs at LUNA's level; Woodward-Colella, ahead of LUNA with a stall near 3.4e-4 in rho.

<picture><source media="(prefers-color-scheme: dark)" srcset="docs/figures/time_to_target_1d-dark.png"><img src="docs/figures/time_to_target_1d.png" alt="GPU seconds to a target error"></picture>

GPU seconds to a target error, 1D: LUNA wins Sod and Shu-Osher; on Woodward-Colella it is close in rho and P, LUNA leads in v, and only moving reaches rho 1e-3 in the ladder.

<picture><source media="(prefers-color-scheme: dark)" srcset="docs/figures/vortex_3d-dark.png"><img src="docs/figures/vortex_3d.png" alt="3D Yee vortex"></picture>

3D Yee vortex, L1 against exact cell averages: moving one pass is first order, two passes (branch dim-1d-diag, not on main) reach order 2.2, then 1.5, and stay above static. O(h) floor marked.

<picture><source media="(prefers-color-scheme: dark)" srcset="docs/figures/kh_2d-dark.png"><img src="docs/figures/kh_2d.png" alt="2D Kelvin-Helmholtz"></picture>

2D Kelvin-Helmholtz: LUNA and static lead at equal N and in GPU time, one-flux moving stalls near 4e-4, two fluxes recover. Hydro on voronoi-cpp branch dim-2d, not yet on luna-cpp main.

<picture><source media="(prefers-color-scheme: dark)" srcset="docs/figures/sedov_3d-dark.png"><img src="docs/figures/sedov_3d.png" alt="Sedov 3D"></picture>

Sedov 3D, a shock test that cannot rank schemes: density profile (dim-limiters) and field L1 ladder (older code, LUNA ladder minmod, MC at 128 and 256).

## Limitations and open items

- The hydro step is not yet at its target (~300 M cell updates/s): on the A100 at 128^3 double precision runs at
  152 M cell updates/s static, its hydro part at 140 M cells/s on a moving mesh, the whole moving step at
  73 M cell updates/s. Compile-time precision options (`--precision=`, default double): mixed +10 %, low +19 % on the
  hydro static, +4 % / +7 % on the whole moving step (research_notes/hydro_speed_2026-11-04.md).
- On the CPU (gcc's default FP contraction) some hydro modes differ from the GPU in the last bits; the A100 runs are
  bit-identical run to run.
- Periodic boxes only in 3D; only 1D has walls.
- 1D moving mesh, inflow: no refinement, so the gas that enters from an inflow wall fills a few grown cells. That gas
  is not uniform in general (Shu-Osher: acoustic waves cross it), so the error there does not converge.
- 1D moving mesh: the error of Woodward-Colella stops falling near N = 2^15, at every Courant number tried
  (research_notes/stage1_1d_2026-11-04.md).

## Roadmap

1. Hydro speed, first round done: 107 -> 152 M cell updates/s static double, 56 -> 73 for the whole moving step. The
   remaining limit is memory waits in the gradient sweep; sliced lists, 16-bit offsets, tiles and a face-major update
   did not help (research_notes/hydro_speed_2026-11-04.md).
2. 2D hydro on voronoi-cpp's 2D mesh; Kelvin-Helmholtz after McNally, Lyra & Passy (2012).
3. Multi-GPU hydro, on the mesh's domain decomposition and ghost layers (voronoi-cpp).
4. Refinement and derefinement: conservative remap from the volumes, on the mesh's split and merge.
5. Self-gravity by multigrid, after the hydro suite passes at speed, starting from LUNA's gravity solvers (same author;
   multigrid with coarse-grid gather): V-cycles warm-started from the previous step's potential, aiming at 2-4
   V-cycles per step, so the solve runs at full GPU speed. Options: an auxiliary Cartesian grid (deposit, solve,
   interpolate) or multigrid on the Voronoi mesh itself with gravity as face fluxes. Coarse levels are inefficient on
   GPUs (too little work per launch), so they can run on the host CPUs: fine levels on the GPU, levels below a size
   threshold on the CPU, one small transfer each way per V-cycle
   (research_notes/multigrid_moving_mesh_literature_2026-11-02.md).

Every test in the suite reports accuracy per cost: GPU-seconds to a target error at 2-3 resolutions.

## Repository layout

- `src/hydro/`: the hydro (header-only). `src/bench/`: `hydro_sedov`, `hydro_1d`. `src/test/`: the tests,
  `exact_riemann.hpp`, `luna_1d_values.hpp`.
- `extern/voronoi-cpp/`: the mesh library, pinned submodule.
- `scripts/`: `hydro_a100.sh`, `hydro_speed_a100.sh`, `dim1d_a100.sh`, `snapshot_hash.sh`, `compare_1d.py`,
  `hydro_sedov_compare.py`, `luna_*.py`, `make_figures.py`, `lint_cpp.sh`.
- `docs/figures/`: the figures, light and dark.
- `research_notes/`, `reports/`: status notes.
