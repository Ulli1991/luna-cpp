# Measurements: provenance

Where each number and figure in the [README](../README.md) comes from: jobs, commits, clocks and the setup details left out of the README. Sections follow the README headings. Unless stated, runs are on one A100 40 GB, double precision, HLLC. The same code varies by about 10 % between GPU jobs.

The records below keep the names of their time: "two passes" (`--integrator=two-pass`) is today's moving step, MUSCL-Hancock on a moving mesh, the only one since 2026-11-10; "one pass" was a single flux at the half step, an option until then, now removed.

## The hydro

### Sedov speed table

README table since 2026-11-15: jobs 32371742 (128³, two rounds) and 32371743 (256³), 1410 MHz, branch `face-once-read` (the cheaper flux and the kept face magnitudes); memory from the same jobs. Before, since 2026-11-14: ours from job 32319468 (128³, two rounds) and 32319469 (256³), 1410 MHz, branch `hydro-speed`; LUNA from job 32271240 (Pallas kernel, minmod, HLLC, 80 steps, snapshots and diagnostics off; details in "Hydro step profile and speed levers"). The LUNA numbers below, 0.47 to 0.76 FP64 at 128³, match its XLA kernel (0.48 in job 32271240), LUNA's default, which `hydro_a100.sh` runs since it passes no `hydro_kernel`; the job-to-job spread up to 0.76 is not explained. LUNA's Pallas kernel is the fast path: 0.82 FP64 and 1.46 FP32 at 128³, 1.04 and 2.35 at 256³. Older paragraphs follow.

Sedov blast of 1e51 erg, n = 1 cm⁻³, 50 pc box, 1e4 yr; jittered points in Morton order; 40 steps. Job 31954615, ravg1020, 1305 MHz, bef77ed: medians of three rounds at 2.1M cells, one round at 16.8M, `--memory`. LUNA in the same job, 200 steps, median of the 20 perf lines. The same code varies by about 10 % between GPU jobs. LUNA FP64 at 2.1M cells moved between jobs: 0.47 here, 0.67 in job 31953960, 0.76 in job 31953468; at 16.8M 0.87, 0.90, 0.95; FP32 0.68, 1.01, 1.37 and 1.74, 1.91, 2.13.

README speed table, `grad32`, job 31954615 at 1305 MHz: static HLLC 0.170 at 2.1M cells and 0.166 at 16.8M; whole moving step with the mesh update 0.072 and 0.076. GPU memory moving 2255 B/cell at both sizes. LUNA in the same job: FP64 0.47 to 0.76 (2.1M, varies by job) and 0.87 (16.8M), FP32 0.68 and 1.74.

voronoi-cpp 2c8c509 to 6ead543 (circumcentres in grid steps #45, edge representatives #48, step kernels, incremental face numbering, ring re-walk), job 32269986, both trees in the same A100 job (SM clock not recorded), two rounds: whole moving step 0.0704 to 0.0729 at 128³ and 0.0737 to 0.0762 at 256³, the mesh update in it 0.279 to 0.323 G; static 0.170 to 0.169. Memory moving 2255 to 2236 B/cell at both sizes (39410 to 39076 MB in use at 256³), static 2185 to 2166. The README's moving row and our memory are these after values.

voronoi-cpp 6ead543 to abd3d39, 2026-11-18. The changes: faster exact start and close-gaps, V1 dropped and R14 opt-in, #60's filtered warp kernels, the cpu/gpu/hybrid modes (`voronoi::Engine`), and S2's local repair with the default `Fallback::cheapest`. Old tree: luna-cpp main 2d08e1e; new tree: branch `pin-abd3d39`. They differ only in the pin and in the `stuck_steps`, `local_repairs` and `local_cells` counters of `hydro_sedov` (the old tree got `stuck_steps` for the comparison). Scripts: `scripts/pin_bits_a100.sh`, `pin_speed_a100.sh` and `pin_ci_host.sh`. The API needed no change.
- Builds and tests: A100 ctest 5 of 5 (job 32520636). CI's host steps (job 32520637: gcc 15, host Kokkos with OpenMP, 2 threads): ctest 5 of 5 with sanitizers off and on, and lint (check_comments.py, clang-tidy 18) clean. `hydro1d_cases` holds the ladder gates.
- Bits (job 32520636, 1305 MHz): the hashes of Q, x and volume are equal on static, moving, order 0, per-cell moving and HLL moving Sedov 128³ (50 steps), and on moving Sedov 128³ to 1e4 yr (349 steps). `hydro_1d` hashes are equal for Sod, Shu-Osher and Woodward-Colella at N = 1024, static and moving, and `hydro_kh2d` hashes for 256² over 300 steps, static and moving. The only difference is the device flip count, a few in 10⁶ (17044 against 17066 over 50 steps); it varies the same way between two runs of one binary.
- Stuck steps (job 32520651), moving Sedov to 1e4 yr, two runs per pin: 128³ (349 steps) had none on either pin. 256³ (919 steps) had none on the old pin. On abd3d39, one run had one step with 6 stuck entries, the count of #8; it was repaired locally (49 cells, no rebuild), and the run ended with Q, x and volume equal to the other three runs. No rebuild and no host finish in any run.
- Speed (job 32520651, ravg1060, SM clock 1305 MHz in all 934 samples under load, `grad32`, 40 steps, three rounds, old and new alternating). G cell updates/s, medians; mesh update ms per step in brackets:

| | 128³ old | 128³ new | 256³ old | 256³ new |
|---|---|---|---|---|
| static | 0.2137 | 0.2134 | 0.2194 | 0.2191 |
| moving, whole step | 0.0800 (6.50) | 0.0807 (6.49) | 0.0867 (40.30) | 0.0876 (39.79) |
| peak B/cell, moving / static | 2300.5 / 2365.6 | 2300.6 / 2365.7 | 2300.6 / 2365.7 | 2300.6 / 2365.7 |

The moving step is 0.9 % and 1.0 % faster: the hydro by 0.5 % (16.52 to 16.44 ms at 128³) and the mesh update by 1.3 % at 256³. Static is 0.1 % slower at both sizes: hydro 9.815 to 9.827 ms at 128³, a little more than the spread of the rounds. The static step runs no mesh code, so the cause is not explained; it may be code placement in the new binary. Moving Sedov to 1e4 yr took 9.68 instead of 9.77 GPU seconds at 128³ and 188.5 instead of 190.4 at 256³. These 1305 MHz rates equal the README's earlier ones, which were recorded as 1410 MHz, to three digits.

Hashes in job 31954615 at 128³, 50 steps, CUDA: `--precision=double` equals main 3449600 on static, moving, moving one pass, order 0, per-cell moving and HLL moving; `default` equals `grad32` static and moving.

Rows left out of the README, same job: `--precision=double` static HLL 0.157 at 2.1M cells; first order 0.312 in both precisions. Peak device memory at 16.8M cells, GB in use: `grad32` static 38.2, two passes 39.4, one pass 39.0; double 39.3, 40.5, 40.1. `--precision=mixed`, two passes, 2445 B/cell, does not fit at 256³; one pass 2334 B/cell (40.7 GB, job 31925336). LUNA memory: job 31953468.

LUNA rows: MC, HLLC, Pallas, FP64 (FP32 where stated).

### Sedov accuracy

Sedov at 128³, job 31699863, HLLC (the moving numbers of this paragraph are one pass): shock radius 0.978 of the Sedov-Taylor value at 1e4 yr, L1 density error 0.075 rho0 (first order: 0.104), mass and energy conserved to 2e-16.

Moving mesh, jobs 31700850 and 31700851: 354 steps instead of 1642, L1 density error 0.058 rho0, the same numbers with the whole box moving at 500 km/s. At 256³ L1 0.034 in 288 s of GPU time (job 31702077); now 176 s (job 31925157), two passes 230 s.

These L1 values are sampled at the mesh generators, which flatters a moving mesh. The field L1 (the integral of |rho_h - rho| dV) is the fair norm (research_notes/accuracy_score_exploration_2026-11-04.md). Details: research_notes/hydro_stage1_2026-10-31.md, hydro_stage3_2026-10-31.md.

Field L1 of density, r < 1.5 R_s (one pass, two passes): 128³ 0.0629, 0.0648; 256³ 0.0369, 0.0377 (research_notes/two_pass_2026-11-05.md).

The one-pass step of the moving mesh keeps the bits it had before the two-pass default; hash in research_notes/two_pass_2026-11-05.md, which also compares cost and accuracy of the two.

### Hydro step profile and speed levers (2026-11-13)

Moving Sedov 128³ jittered, `grad32`, HLLC, one A100; luna-cpp ac1d2b3 (voronoi-cpp 6ead543) against branch `hydro-speed`. Scripts: `scripts/hydro_profile_a100.sh` (nsys, ncu), `scripts/hydro_levers_a100.sh` (speed, bits, t_end), `scripts/hydro_precision_a100.sh`, `scripts/luna_speed_a100.sh`, `scripts/luna_profile_a100.sh`. The clock is the job's; nvidia-smi read 1410 MHz unless stated.

**Profile before** (ncu job 32271238, one moving step; nsys job 32271234, 20 steps). DRAM is read plus write per cell.

| kernel | ms | DRAM B/cell | registers | warps active | DRAM % of peak |
|---|---|---|---|---|---|
| primitives | 0.30 | 97 | 36 | 64 % | 44 % |
| gradient sweep (moving) | 6.12 | 999 | 112 | 22 % | 22 % |
| cell_step + face_step (time step) | 0.73 | 286 | 29, 38 | 94, 58 % | 83, 46 % |
| keep_centroids | 0.27 | 136 | 38 | 68 % | 67 % |
| fluxes, pass 1 (mesh of t^n) | 3.04 | 1388 | 80 | 35 % | 62 % |
| update, pass 1 | 2.67 | 1481 | 40 | 70 % | 75 % |
| drift | 0.14 | 65 | 34 | 64 % | 61 % |
| fluxes, pass 2 (mesh of t^n+1) | 3.12 | 1215 | 96 | 25 % | 53 % |
| update, pass 2 | 2.67 | 1481 | 40 | 70 % | 75 % |
| hydro, sum | 19.1 | 7147 | | | 49 % on average |

The sweep is latency-bound (long-scoreboard stalls 70 % of warp slots at 22 % of DRAM peak); `update` is at the bandwidth roof and reads 3 times its unique bytes (the face rows are in tetrahedron order); the fluxes sit between. Run twice, once per mesh: the flux and update kernels (gathers of both sides' primitive and gradient rows, face rows, the flux rows). Read twice with the same values: the primitive and gradient rows of t^n; the face geometry differs between the passes. nsys per step: 72 kernels (10 of them the hydro's), 0.64 ms with the GPU idle in 115 gaps, 35 fences and blocking copies, mostly the mesh's. 3D vortex 128 (n = 524288, job 32271236): hydro kernels 4.99 ms of a 9.3 ms step, the same pattern.

**Floor.** Each byte read once at 80 % of 1555 GB/s, with the face rows and fluxes stored as now: primitives 104, sweep 706 (face rows 382 of it), old centroids and snapping 144, each pass's fluxes 956 to 980 and update 466, drift 72: about 3900 B/cell, 6.6 ms at 128³, 3.1 ns per cell. The target "hydro faster than the mesh" (3.1 ns per cell, both passes) is that floor: no kernel layout of today's data flow reaches it. It needs fewer bytes: fluxes not stored (summed where computed: -1270 B/cell, 4.4 ms) and the sweep's latency fixed; face rows computed in place (R14) would take off at most another 1150 B/cell of reads but costs ring walks of about the same bytes (plan below).

**Levers, each against main in one job.**

| lever | effect at 128³ moving | static | bits | job |
|---|---|---|---|---|
| owner bit in the lists (default, 3D) | sweep 6.78 -> 6.34 ms, update 5.35 -> 4.32 (both passes, with the next item) | sweep 6.43 -> 6.03, update 2.61 -> 2.09 | equal | 32319468, 32272653 |
| faces numbered by owner | hydro -0.7 ms but the mesh +2.6 ms (number_by_owner, voronoi-cpp #53), +168 B/cell: off on moving meshes | sweep 6.03 -> 5.69, update 2.09 -> 1.91: on for static meshes, +135 B/cell | equal | 32319468, 32272653 |
| next step's primitives written by pass 2's update; fallbacks counted on the device | primitives 0.31 -> 0 ms, two fences fewer | — | equal | 32272653 |
| sweep: launch bounds to 80 or 64 registers | sweep +19 % and +117 % slower (spills) | | equal | 32272227 (1095 MHz) |
| sweep: face loop unrolled by 2 | no change | | equal | 32272227 |
| sweep: time-step terms folded in | sweep +22 % slower (registers) | | equal | 32271602 |
| fluxes: launch bounds, 85 registers | no change | | equal | 32272227 |
| per-cell fluxes and update (`--update=cell`) | fluxes 5.9 times slower | 5.7 times | equal | 32272653 |
| reuse between passes | refuted earlier (research_notes/profile_2026-11-06.md, -2.5 %) | | | |

The kept levers change no bits (job 32319470): hashes of Q, x and volume after 50 steps at 128³ equal main's for static, moving, order 0, per-cell moving, HLL moving and `--precision=double` moving; the 3D vortex 64 files (static, moving), the 1D Sod hashes and the KH 2D 128 hashes (static ef63946b023e78a7, moving 4ec3a18a6b578ba6) are equal. The arithmetic is unchanged, so the 1D ladder, Sedov L1, vortex and KH results are main's. Moving Sedov 128³ to 1e4 yr (job 32319471): 349 steps, GPU 10.67 -> 9.97 s (-6.6 %), field L1 rho 0.06479 in both.

**Before and after** (jobs 32319468 at 128³, two rounds, and 32319469 at 256³, 1410 MHz). G cell updates/s; hydro ns per cell in brackets; memory peak B/cell:

| | 128³ main | 128³ now | 256³ main | 256³ now |
|---|---|---|---|---|
| moving, whole step, `grad32` | 0.0729 | 0.0782 | 0.0763 | 0.0826 |
| moving, hydro part | 0.110 (9.09 ns) | 0.122 (8.17 ns) | 0.108 | 0.121 |
| static, `grad32` | 0.170 | 0.197 | 0.165 | 0.194 |
| moving, whole step, double | 0.0677 | 0.0721 | 0.0707 | 0.0758 |
| static, double | 0.151 | 0.171 | 0.149 | 0.170 |
| memory moving / static | 2236 / 2166 | 2236 / 2301 | 2236 / 2166 | 2236 / 2301 |

Other options now, 128³ moving step and static, then 256³: `grad16` 0.0797, 0.203; 0.0843, 0.196. `single` 0.0885, 0.233; 0.0963, 0.239. `mixed` 0.0829, 0.211. `low` 0.0865, 0.231. In `single`, `mixed` and `low` the float flux rows halve `update` (4.3 -> 2.0 ms).

**Profile after** (ncu job 32319473, one moving step): sweep 5.58 ms (995 B/cell), cell and face step 0.73, keep_centroids 0.28, fluxes 3.00 and 3.12 (unchanged), update 2.09 and 2.22 (1140 and 1193 B/cell, 44 registers), drift 0.14, no primitives kernel: 17.2 ms and 6415 B/cell against 19.1 ms and 7147, at 50 % of DRAM peak. The hydro is now 8.2 ns per cell against the mesh's 3.1 (update) to 4.6 (with update_cells): 2.6 times the target.

**R14, a plan.** Face geometry inside the flux kernel needs from voronoi-cpp a device view of each face's ring (the tetrahedra around its Delaunay edge, in order) and the tetrahedra's circumcentres, plus a function that turns a ring into area vector and centroid exactly as `tet_mesh::faces` does (same summation order, so the same bits). Bytes: the ring of a face (about 5 tetrahedra, 4 B each) and their circumcentres (24 B, shared by about 6 faces each) cost about 50-60 B per face from L2-friendly arrays, against 48 B of stored rows read now; the gain is the mesh's write of 382 B/cell of face rows (and the float rows), and about 380 B/cell of memory, not read traffic. It costs a dependent load chain per face in the sweep and both flux passes, the kernels that already wait on latency. Order of work: (1) voronoi-cpp exposes rings and circumcentres read-only (it has them, `FaceRings`); (2) a luna kernel computes the rows from them and checks the bits against `face_rows`; (3) only then fold it into the sweep and pass 2, measured against the stored rows. Before R14, the larger lever by bytes is not storing the fluxes.

**LUNA on the same A100** (job 32271240, LUNA 60cf841, Sedov 3D as `sedov3d_run`, minmod, HLLC, snapshots and diagnostics off, 80 steps; "perf" = cells x steps / wall seconds over 10-step windows, whole MUSCL-Hancock steps; the first window holds the compile and is left out). G cell updates/s:

| | 128³ | 256³ |
|---|---|---|
| XLA (LUNA's default kernel), FP64 | 0.48 | 0.63 |
| XLA, FP32 | 0.83 | 1.10 |
| Pallas, FP64 | 0.82 | 1.04 |
| Pallas, FP32 | 1.46 | 2.35 |

The README's 0.47 to 0.76 FP64 at 128³ were the XLA kernel (`hydro_a100.sh` passes no `hydro_kernel`); Pallas is the fast path and the fair yardstick: about 1 G FP64 and 2.3 G FP32 at 256³. DRAM per step from ncu on the Pallas FP64 kernels (job 32271533): about 430 B/cell, 143 B per face (3 faces per cell); at 2.56 ms a step that is about 350 GB/s, so LUNA's FP64 step is not bandwidth-bound.

**Per face.** A LUNA cell has 3 faces of its own, ours 7.95. Face-flux updates per second: LUNA Pallas FP64 2.46 G at 128³ and 3.13 G at 256³, FP32 4.37 and 7.04 G. Ours `grad32` static 0.170 G cells -> 1.35 G faces (main), 0.197 G -> 1.57 G faces now; the moving hydro (two flux passes, 15.9 face fluxes per cell) 0.110 -> 1.75 G and 0.122 -> 1.94 G. Matching LUNA FP64 per face means 0.31 G cell updates/s static at 128³ (0.39 at 256³) and about 0.15 G for the moving hydro part, 6.5 ms a step: the same as the target of a hydro faster than the mesh. We are at 63 % of LUNA's per-face rate static; the gap is bytes and latency: about 510 B per face against LUNA's 143 (stored face rows read three times, fluxes stored and read back, gradients and states gathered through an index), and the least-squares sweep at 22 % of DRAM peak, where LUNA's slopes come from contiguous neighbours inside one fused kernel.

**Precision options with the levers** (jobs 32272271 Sedov at 1095 MHz, 32272272 waves at 1410 MHz; build `hydro-speed` before the last cleanup, same arithmetic). `single` is new: double geometry, float gradients, float stored fluxes and float Riemann solve in the face's frame. Sedov 128³ to 1e4 yr, GPU seconds (moving, static), field L1 of rho/rho0 (moving, static), peak B/cell moving; the 3D vortex 64 L1 of rho to 4 digits; the 3D sound wave 64 x 16 x 16 after one period, L1(rho)/amplitude (moving, static):

| precision | Sedov GPU s | Sedov L1 | B/cell | vortex | sound 1e-3 | sound 1e-6 |
|---|---|---|---|---|---|---|
| double | 12.2, 23.4 | 0.0648, 0.0776 | 2300 | as double | 4.32e-3, 4.32e-3 | 3.68e-3, 3.68e-3 |
| grad32 (default) | 11.1, 20.3 | 0.0648, 0.0776 | 2236 | equal to 4 digits | 4.32e-3, 4.32e-3 | 3.68e-3, 3.68e-3 |
| grad16 | 10.9, 19.4 | 0.0648, 0.0776 | 2208 | equal | 4.32e-3, 4.32e-3 | 3.68e-3, 3.68e-3 |
| single | 10.0, 17.2 | 0.0648, 0.0776 | 2163 | equal | 4.31e-3, 4.31e-3 | 1.25, 1.60 |
| mixed (float face rows too, Riemann double) | 10.8, 19.2 | 0.0648, 0.0776 | 2426 | equal | 4.32e-3, 4.32e-3 | 7.2e-3, 2.4e-2 |
| low (mixed with FP16 gradients and float Riemann) | 10.4, 17.8 | 0.0648, 0.0776 | 2398 | equal | 4.30e-3, 4.32e-3 | 1.24, 1.56 |

Every option keeps Sedov, the vortex and the 1e-3 sound wave to the digits printed, so on those GPU time to the same error is GPU time: `single` is 10 % (moving) and 15 % (static) ahead of `grad32`. But a float Riemann solve or float stored fluxes lose a 1e-6 wave: with `single` and `low` the error exceeds the wave itself, `mixed` doubles it moving and is 6.5 times worse static. `grad16` matches double on every test, is 2 to 4 % faster than `grad32` and saves 28 B/cell; it does not make a larger run fit: 384³ (56.6M cells) needs 707 B/cell on 40 GB, three times less than any option, so 256³ stays the largest power-of-two size. 2D has no presets; KH 2D runs double.

### Faces once by the owner, R14 and cheaper fluxes (2026-11-15)

Three levers for the moving step, each against main e350dd7 in one A100 job (gpu1, 1410 MHz), Sedov jittered, HLLC, 40 steps, `grad32` unless stated; branch `hydro-once`, script `scripts/hydro_once_a100.sh` (modes speed, bits, ncu, tend, waves, kh; RUNS as in `hydro_bytes_a100.sh` of branch `hydro-bytes`). The variants are the branch's WIP commits; the branch head keeps lever 3, the script and the `riemann32` preset. Times in ms per step at 128³.

**1. Each face once by its owner** (`--update=owner`, commits 952771e to b9f402e). Each cell computes the faces it owns, adds them and stores each flux once; a second kernel adds the stored fluxes it does not own (sums: owned faces first, then the rest, in list order; T lanes per cell take owned faces of rank l, l + T, ... and combine in a fixed shuffle tree). The update gains as estimated, the flux pass loses three to eight times that:

| 128³, `grad32` | fluxes (both passes) | update | hydro | job |
|---|---|---|---|---|
| main, moving | 6.49 | 4.31 | 17.2 (0.122 G cell updates/s) | 32368509 |
| owner, 1 lane per cell, moving | 14.36 | 3.14 | 23.9 | 32368509 |
| owner, 4 lanes, moving | 13.87 | 3.14 | 23.4 (0.090 G) | 32368509 |
| owner, 8 and 16 lanes (as a reduce), moving | 21.1, 27.6 | 3.14 | 30.7, 37.2 | 32367499 |
| main, static | 2.72 | 1.91 | 10.7 | 32368509 |
| owner, 4 lanes, static | 5.69 | 1.44 | 13.2 | 32368509 |

`double` and 256³ (jobs 32367870, 32367871, 32367873, the reduce version, 4 lanes): the same picture, 256³ moving fluxes 56.1 -> 144 (`grad32`) and 64.3 -> 172 (`double`), update 33.0 -> 23.4. Memory +41 B/cell (the owned sums). ncu (jobs 32367151, 32367874): the update reads 883 instead of 1172 B/cell (1.50 instead of 2.16 ms a pass); the owner pass with one lane per cell moves 4914 B/cell (scattered 40 B stores of each flux in face order, neighbours gathered in cell order) at 127 registers and 19 of 32 threads active per instruction; with 4 lanes 2441 B/cell in pass 1 and, at 158 registers and 12.5 % warps active, 9.75 ms in pass 2. Face evaluations grouped by cell wait on each other and diverge; one thread per face (main) hides that latency. GPU time to t_end, moving Sedov 128³: 10.08 s main, 11.98 s owner (4 lanes), 12.16 s (1 lane), job 32368508.

Bits: with `LUNA_OWNER_CHECK=1` (the second kernel adds every face in list order, as main) `--precision=double --moving` equals main's hash after 50 steps, so the owner pass computes main's fluxes bit for bit in double; `grad32`, order 0 and HLL differ by contraction (job 32366698). The owner sums change bits at rounding level: Sedov 128³ field L1 rho 0.06479 against 0.06479, 349 steps in both; 3D vortex 64 field1 L1 and the sound wave 64 x 16 x 16 at 1e-3 and 1e-6 equal main's to the 4 to 5 digits printed for `double`, `grad32` and `grad16`, moving and static (sound 1e-6 moving 3.6837e-3 in both); KH 2D 128 to t = 1.5, mode amplitude M 0.14321249 static and 0.13071275 moving, equal to main's printed digits, totals equal to 1e-15 (jobs 32368508, 32368510 to 32368512, 32368513). A100 hydro_test with the owner loop in every update test: 233 checks pass; the new check that owner and per face agree to 1e-12 after 10 steps of a rough state fails at 4e-9 to 2e-6 (`low`): float gradients and limiter branches carry the rounding difference (job 32368507). Refuted: the update's saving (1.2 ms of 17.2 moving) is smaller than any grouping of the face evaluations by cell costs. A first 4-lane version ran as a reduce, whose CUDA ranges do not start on a lane group: wrong sums at random (Sedov L1 0.495, job 32367863); the parallel_for in b9f402e fixes it.

**2. R14: face rows computed in the kernels** (voronoi-cpp `step-v1-r14` adef168, copied unchanged; luna 25159e3: `Geometry::face_of` reads the stored rows or calls `face_geometry().face(f, a, c)`, the flux pass computes it once per face, the sweep once per list entry in each of its two loops; `LUNA_R14=1` sets `Layout::stored_faces = false`). Jobs 32367455, 32367456, 32367874; stored against computed in the same build:

| 128³ | sweep | fluxes | update | cells | mesh | hydro | step | memory B/cell |
|---|---|---|---|---|---|---|---|---|
| main (pin 6ead543), moving | 6.37 | 6.50 | 4.33 | 3.04 | 6.75 | 17.2 | 27.2 | 2236 |
| R14 build, stored rows, moving | 7.51 | 6.25 | 5.26 | 3.04 | 23.8 | 19.0 | 46.1 | 2244 |
| R14 build, computed rows, moving | 26.06 | 9.85 | 5.25 | 8.09 | 22.0 | 41.2 | 71.4 | 1850 |
| R14 build, stored, static | 6.58 | 2.58 | 2.46 | | | 11.9 | | 2309 |
| R14 build, computed, static | 28.27 | 5.42 | 2.46 | | | 36.5 | | 1915 |

`double` the same (moving hydro 22.6 -> 45.5). Bits: computed rows give the stored rows' hashes on all eight cases (static, moving, order 0, HLL, `double` and `grad16` static and moving). Memory -394 B/cell, as expected. Bytes (ncu): a flux pass 2.78 -> 4.56 ms and 1161 -> 1415 B/cell (the ring and its circumcentres cost more than the 48 B row), FP64 pipe 42 -> 57 %; the sweep 6.74 -> 25.4 ms, 879 -> 1956 B/cell at 10 % of DRAM peak, 154 registers, four ring walks per face. The mesh saves 1.8 ms (no rows written); update_cells, which sums from the rings, costs 5 ms more. The R14 build's mesh (23.8 ms) is that branch's work in progress, not R14. Even with one walk per face in the sweep the walks cost more than the stored rows save: R14 buys memory, not time.

**3. A cheaper face flux (kept).** ncu of main (job 32367151): the flux passes run the FP64 pipe at 39 to 41 % and issue 40 % of slots, 60 to 68 % of warps waiting on loads, 80 and 96 registers; arithmetic is a large share but not the limit. Exact-arithmetic rewrites with fewer divisions (commit f86f704): 1 / rho once per side in HLLC, shared by c² and E* (two divisions fewer), 1 / (gamma - 1) and 1 / |d| as products (one each). The rounding changes; the formulas do not.

Main against these rewrites in one job each (32368968 at 128³, two rounds, equal to 0.01 ms; 32368969 at 256³), G cell updates/s of the whole step, ms of both flux passes:

| | 128³ main | 128³ rewrites | 256³ main | 256³ rewrites |
|---|---|---|---|---|
| moving, `grad32` | 0.0776 (fluxes 6.49) | 0.0782 (6.26) | 0.0824 (56.2) | 0.0852 (49.4) |
| static, `grad32` | 0.197 (2.72) | 0.199 (2.59) | 0.194 (24.9) | 0.204 (20.3) |
| moving, `double` | 0.0716 (7.54) | 0.0715 (7.55) | 0.0757 (64.3) | 0.0773 (60.1) |
| static, `double` | 0.171 (3.30) | 0.174 (3.09) | 0.170 (28.8) | not run (time limit) |

Memory unchanged (2236 B/cell moving, 2301 static, `grad32`). ncu (jobs 32368970, 32368971, one moving step at 128³): flux instructions 0.44 -> 0.34 G (pass 1) and 0.51 -> 0.42 G (pass 2), but registers 80 -> 86 and warps active 35 -> 25 % in pass 1, so time moves little: the hydro's ten kernels 17.51 -> 17.29 ms and 6632 -> 6379 B/cell (`grad32`), 19.80 -> 19.81 ms (`double`). GPU time to t_end, moving Sedov 128³: 10.07 -> 9.98 s, 349 steps both, field L1 rho 0.06479 both (job 32368972). Accuracy (jobs 32368974 to 32368979): the vortex, the sound waves at 1e-3 and 1e-6 (`double`, `grad32`, `grad16`, moving and static) and KH 2D 128 (M 0.14321249 static, 0.13071275 moving) equal main's printed digits; the 1D ladder, static and moving, every problem at N = 256 to 2048: 12 of 138 rows differ, in the fourth digit of the 1e-6 sound waves and in the entropy wave at its 1e-10 floor; orders unchanged. Every hash differs (rounding). Cost: the HLLC rotation check of hydro_test (rotated states, normal and face velocity give the rotated flux) rises from 3.7e-14 to 1.06e-13 and fails its 1e-13 bound (A100 ctest, job 32369101); the other four tests pass. Kept (owner, 2026-11-15): +1 % at 128³ and +3 to +6 % at 256³, 20 % fewer flux instructions, accuracy unchanged; the rotation bound is now 2e-13.

A float Riemann solve with double fluxes (`--precision=riemann32`, kept as an option): moving fluxes 6.45 -> 5.77 ms at 128³ (job 32367870), Sedov t_end 10.12 s and L1 0.06479 (job 32367863), the vortex and the 1e-3 wave unchanged, but the 1e-6 sound wave is lost: L1/amp 1.25 moving and 1.63 static against 3.7e-3 (job 32367861), as `single`.

**Where this leaves the target.** The whole moving step is 27.0 ms at 128³ (12.9 ns per cell, 0.078 G cell updates/s): mesh and cells 9.8 ms (4.7 ns), hydro 17.2 ms (8.2 ns, 6632 B/cell at 50 % of peak). 3.3 to 5 ns per cell is 2.6 to 3.9 times away; the mesh alone already takes 4.7 ns. None of the three levers changes the data flow enough: the owner pass trades 290 B/cell of update reads for slower face evaluations, R14 trades 394 B/cell of memory for ring walks that read more than the rows, and the flux arithmetic is not what the passes wait on. The bytes floor of "Hydro step profile" (3900 B/cell with today's data flow) is unchanged.

### Face reads once per step (2026-11-15)

Can the step read each face's geometry fewer times, one thread per face kept? Branch `face-once-read`, against main (the cheaper flux, f86f704) in one A100 job each (gpu1, 1410 MHz), Sedov jittered, HLLC, 40 steps; script `scripts/hydro_once_a100.sh`. Where the reads are (ncu, one moving step at 128³, `grad32`): the sweep reads each face's area and centroid from both sides (1007 B/cell, latency-bound at 24 % of DRAM peak); each flux pass reads its mesh's rows once, contiguously; each update gathers every face's stored flux twice, 40 B rows that cost two 32 B sectors each, about 16 x 64 + list + Q = 1170 B/cell, which is what it moves (1150 and 1200). The two flux passes read different meshes (t^n and t^n+1); the mesh of t^n+1 is read by pass 2, then by the next step's sweep, face_step and pass 1, with a global barrier (new primitives, gradients, dt) between each. So a face's geometry can be carried only across those barriers, as a stored per-face value.

| lever (commit) | 128³ moving `grad32` | static `grad32` | bits | job |
|---|---|---|---|---|
| \|area\| of each face kept by the last flux pass on the mesh (pass 2 on a moving mesh, the first pass on a static one) in the flux scratch, so the sweep reads 8 B instead of the 24 B area row per list entry | sweep 6.36 -> 5.75 ms, pass 2 +0.19 ms; step 0.0780 -> 0.0799 G cell updates/s | sweep 5.70 -> 5.01; 0.199 -> 0.213 | equal to main | 32371742, 32371745 |
| the drift folded into pass 1's update (one kernel fewer) | update +0.13 ms, drift -0.16: no change | | equal | 32371268 |

ncu (job 32371744): the sweep 5.60 -> 4.99 ms and 1007 -> 764 B/cell, pass 2 3.11 -> 3.31 ms (+80 B/cell, the magnitudes written); the ten kernels 17.44 -> 17.23 ms and 6430 -> 6358 B/cell. Memory +65 B/cell (8 B a face in the scratch, which sets the peak): 2236 -> 2300 moving, 2301 -> 2366 static. `double` 128³: moving 0.0713 -> 0.0732, static 0.173 -> 0.184. 256³ (job 32371743): moving `grad32` 0.0852 -> 0.0865, static 0.204 -> 0.219; `double` moving 0.0771 -> 0.0783, static 0.178 -> 0.189.

Bits: equal to main for all eight Sedov cases, the 3D vortex and both sound waves (`double`, `grad32`, `grad16`, moving and static, files equal byte for byte) and KH 2D 128 (hashes equal); the 1D code is not touched. So Sedov L1, the vortex, the sound waves, KH and the ladder are main's. Moving Sedov 128³ to t_end: GPU 10.00 -> 9.77 s, 349 steps both, field L1 rho 0.06479 both (job 32371746).

Not tried, and why: a compact per-face struct shared by both halves (the halves read different meshes, and float geometry is the `mixed` preset, which loses the 1e-6 sound wave); fusing the update with anything (each neighbour kernel needs all faces' fluxes first; the update is at its gather floor for 40 B rows; storing fluxes by owner was lever 1 above, refuted).

### Hydro levers of 2026-11-18

The levers of the profile of 2026-11-18 (branch `profile-2026-11-18`, `research_notes/profile_2026-11-18.md`), on branch `hydro-speed-2` from main `73f763a` (voronoi-cpp `abd3d39`). One A100 40 GB, application clock 1305 MHz (logged by every job), Sedov 3D jittered, HLLC, `grad32`, 40 steps, phase medians with `--phases` (fenced) unless a row says wall. Each row in [tried.md](tried.md). Experiments ran from one binary with the variant chosen by an environment variable (not kept), against main's binary in the same job.

**Experiments, 128³, two rounds each (ms per step, medians; the rounds agree within 0.5 %).**

| job | variant | static: sweep phase, fluxes, update | moving: sweep phase, fluxes, update | bits (Q hash, 40 steps) |
|---|---|---|---|---|
| 32522925 | main | 5.02, 2.59, 1.91 | 5.75, 6.46, 4.24 | static 16fd7ed2, moving 20ca7f2a |
| 32522925 | sweep capped at 80 / 64 / 96 registers | 6.65 / 13.0 / 5.42, same, same | 7.5 / 14.7 / 6.04 | equal |
| 32522925 | sweep walk one entry ahead | 4.84 | 5.53 | equal |
| 32522925 | per-cell fluxes and update (`--update=cell`) | 5.97, 14.9 (both) | 6.74, 38.1 (both) | equal |
| 32523403 | sweep: the neighbours' range in a walk of its own (10 sums fewer in the fit loop) | 5.19 (ahead: 5.02) | 6.08 (ahead: 5.86) | equal |
| 32523558 | HLLC: the upwind side's state copied entry by entry, not by a reference to one of two arrays | fluxes 2.50 | fluxes 6.12 | differ: static e4bd3aa6, moving 3c16d52b |

### 2D

| Figure | Setup | Jobs |
|---|---|---|
| `kh_2d_evolution.png` | t = 0, 1, 2.5, 4, 5; moving (two passes) against LUNA MC, N = 2048, FP64, magma colour map 0.9 to 2.1. HLLC, LUNA Pallas, CFL 0.8; ours face limiter, per-face, LUNA's rule 0.8. Moving to t = 5 in two legs, the restart not bit-identical: 69535 steps, 0 mesh rebuilds, 0 stuck; mass, x momentum and energy kept to 2e-12 or better, relative | moving 31953688, 31953871; LUNA 31953667, 31954283 |
| `kh_2d_mesh.png` | N = 512, two passes, t = 4; exact Voronoi edges from `scipy.spatial.Voronoi` with periodic images, fading where a cell is shorter than a few pixels; 0 mesh rebuilds, 0 stuck | 31953904 |
| `kh_2d_density.png` | t = 1.0, 1.5, 2.5, 3.0; LUNA MC, static, moving (two passes), all N = 2048; differences sampled at LUNA's cell centres, symmetric log scale; static run 62390 steps | static 31953690 |

Mean |Δρ| against LUNA at t = 1.0, 1.5, 2.5, 3.0: static 4e-8, 5e-7, 5e-5, 6e-4 (its panel at t = 1.5 is white: largest difference 2e-4); moving 5e-5, 1e-4, 0.012, 0.073.

Rates at N = 2048 (job 31952221), G cell updates/s: LUNA 1.78, static 0.756, moving one pass 0.414, two passes 0.322. Peak 693, 735 and 749 B/cell for ours. GPU seconds to max |e| = 0.001: LUNA 0.55, static 1.1, moving 15. Details and the comparison with LUNA at the same N: research_notes/stage2_2d_2026-11-05.md; `scripts/dim2d_a100.sh` runs the ladders, `scripts/compare_kh.py` makes the tables.

### 1D

Job 31917211 (ravg1002), A100 at 1305 MHz, Sod, HLLC, 2000 steps (200 at 2²⁴). "Hydro only" is a static step; "whole moving step" is a moving step at its defaults with the order check.

Defaults: a static run uses Courant x V / (|v| + c) at 0.8, one launch per step that also finds the next time step (`step_and_time_step`). A moving run uses AREPO next generation's rule with its pair term at 0.4 (r = V / 2 in 1D), three launches (time step, step, order check). 3D stays at 0.3. Points are regularised as AREPO next generation does without a face angle: a drift towards the density-displaced centroid when it is more than 0.75 cell radii away.

Below about 1e5 cells a step is bound by launch latency (about 60 us). Peak device memory: 144 B/cell static, 152 B/cell moving. Field accuracy on Sod, Shu-Osher and Woodward-Colella: research_notes/stage1_1d_2026-11-04.md.

## Figures

Made by `scripts/make_figures.py` in LUNA's figure style (`scripts/figstyle.py`). Common setup of the 1D figures: HLLC, MUSCL-Hancock, FP64; LUNA: MC, Pallas, CFL 0.8; static: uniform points, face limiter, LUNA's time step rule at 0.8; moving: face limiter, the default rule at 0.4, regularised.

| Figure | Provenance |
|---|---|
| `sod.png` | N = 512, t = 0.25. Job 31917206, voronoi-cpp 947946a |
| `shu_osher.png` | N = 1024, t = 1.8. LUNA and static on [-5, 5]: job 31917208, 947946a, setup as Sod. Moving, default rule at 0.4, job 31953172, cells from the inflow wall. Reference curves: luna-reference-solutions 99a52d0, `luna/shu_osher/shu_osher_mc.npz` (LUNA b7b0d35, MC HLLC, 65536 on 8192 cells), `athenak/shu_osher/so_ppmx_rk4_16384.npz` (AthenaK c52f88f) |
| `woodward_colella.png` | N = 2048, t = 0.038, reflecting walls, from x = 0.4. Job 31917210, 947946a, setup as Sod. Curves: luna-reference-solutions 99a52d0, `luna/woodward_colella/woodward_colella_mc.npz` (LUNA b7b0d35, MC HLLC, 65536 on 8192 cells), `athenak/woodward_colella/a9_ppmx_rk4_16384.npz` (AthenaK c52f88f) |
| `convergence_1d.png` | `scripts/compare_1d.py`; references: exact solution on Sod, LUNA MC 2²¹ cells on Shu-Osher, LUNA MC 2²⁰ cells on Woodward-Colella. Sod: moving ahead of LUNA in rho, 2 to 13 % over it in P. Jobs 31917206, 31917210, 947946a; Shu-Osher 31953172, N the initial cells, notes in research_notes/inflow_2026-11-06.md |
| `time_to_target_1d.png` | GPU seconds of the stepping loop, one point per N, dotted lines the targets of the notes; LUNA's compile not counted. Sod 31917206, Shu-Osher 31953172 at 1095 MHz, Woodward-Colella 31917210 at 1305 MHz, 947946a |
| `vortex_3d.png` | slab 10 x 10 x 2.5, nx x nx x nx/4 cells, t = 10. L1 of cell values against the exact average over each cell, v the vector. Ours: HLLC, face limiter, Courant 0.3, jitter 0.1, start from the point value at the centroid, jobs 31920446, 31922276. LUNA: MC, HLLC, Pallas, CFL 0.8, luna-reference-solutions 99a52d0 `luna/yee_vortex/yee_vortex_convergence.npz` (LUNA debcdae + uncommitted, bitwise at 60cf841). The O(h) floor is the exact averages as a piecewise-constant field against the exact field; it binds a field norm, not this measure. Moving one pass is first order; two passes reach order 2.2, then 1.5, and stay above static |
| `kh_2d.png` | N² cells, 1305 MHz; e(t) = M / M_ref - 1 against the published reference curve of McNally, Lyra & Passy (2012), max over t <= 1.5. HLLC, FP64; LUNA MC, Pallas, CFL 0.8; ours face limiter, per-face, LUNA's rule 0.8. Jobs 31952216, 31952221 |
| `sedov_3d.png` | 1e51 erg, n = 1 cm⁻³, 50 pc box, 1e4 yr. Order below 1 for every code (0.66 LUNA minmod, 0.78 moving, n = 128 to 256). Left: volume-weighted shells at 128³, branch dim-limiters, jobs 31918879 (moving), 31918880 (static), 31918732 (LUNA). Right: field L1 over r < 1.5 R_s of the piecewise-constant field, older code, jobs 31879294, 31879807 (LUNA minmod 128 to 448, moving 64 to 256, 256 with the per-cell loop, static 64 to 192), LUNA MC 128 and 256 from dim-limiters. Ours: HLLC, jittered, Courant 0.3, face limiter; moving is one pass in all these runs, older than the two-pass default |
| `sedov3d_shock.png` | 256³, snapshots every 1000 yr, two passes, against LUNA MC (HLLC, Pallas, CFL 0.8). Shock radius is the density peak in the shells. Column density at 1e4 yr: 512² rays along z with 1024 midpoint samples, each in its nearest cell with the run's limited gradient from the centroid. Difference up to 9 % of the peak at the shock, mean 0.2 %. Job 31953353 (moving 925 steps, 1.49 M fallbacks; LUNA 4035 steps) |
| `sedov3d_radial.png` | n_x / 2 bins at 1e4 yr; static job 31953728, 5834 steps, the others as for `sedov3d_shock.png` |
| `sedov3d_mesh.png` | slice z = 25 pc at 1e4 yr, exact plane section of the 3D Voronoi cells |
| `throughput_scaling.png` | 32³ to 256³, 1095 MHz, 40 steps, LUNA median of the perf lines of three rounds. Job 31928911, mesh pin 0b433f0 |
| `throughput_precision.png` | 32³ to 256³, 40 steps, 1305 MHz, job 31953468, e90a913. LUNA in the same job (200 steps, median of the perf lines). None of our options is FP32 throughout. At 128³ `grad16` gives +15 % hydro, +10 % whole step; `grad32` +12 % and +8 %; `mixed` +10 % and +6 %. `mixed` two-pass does not fit at 256³ |
| `throughput_dimensions.png` | HLLC, FP64. A blast in all three: LUNA's Sedov without units in 1D (gamma 1.4) and 2D (gamma 5/3), the Sedov above in 3D. 1D has no two-pass integrator. Jobs 31953276 and 31953278 at 1305 MHz, 31953467 at 1095 MHz; the 1D and 2D blast is not on main yet, runs at voronoi-cpp d077c41. At the largest size static hydro runs 2.66, 0.67 and 0.15 G cell updates/s against LUNA 2.96, 1.46 and 0.95. Two passes take the whole step to 0.29 in 2D and 0.07 in 3D |

The precision table in the README is from job 31953468 at 256³, whose `default` is today's `--precision=double`. `grad32` against double on the final tree, job 31954615, 128³ and 256³: +12 % static, +8 % the whole step with two passes, +8 to +9 % with one, 64 B/cell less moving with two passes, 65 otherwise. An earlier job, 31928265, also 1305 MHz: +12 % static, +9 % the whole moving step, the same 64 B/cell.

`grad32` against double, accuracy: jobs 31928264 to 31929256, post-processing 31929253 to 31930149, gradient-precision scripts of branch precision-check (4b540dd, fbe7fcc), which stayed out of main. Moving two passes, static one pass. Vortex 64 to 256, sound (amplitude 1e-3) 64 to 256, simple wave 128 to 512, also at jitter 0.25, tube 256, Sedov 128³ and 256³. L1 against the exact averages or the Sedov field: 4-digit output, so the largest change seen, 1e-3 in P on the moving vortex, is near that output's resolution at 128 (1.014e-4 against 1.013e-4) and 9e-4 at 256. Pressure contrast (`--cold`) at 128³, moving, 200 steps: energy floor met in 5 cell-steps at 1e-8 in `grad32`, 35 in double; one each at 1e-4; density floor never; nothing non-finite.

Radial profile of the Galilean test (`hydro`): double 6e-14; `grad32` 1.2e-10 on the A100, 2.5e-10 on the CPU. The cell-by-cell measure stays at 7e-8 in both.

## Limitations and open items

Hydro speed target: the moving hydro faster than the mesh, 3.1 ns per cell (0.32 G/s); matching LUNA's Pallas FP64 per face gives the same, 0.31 G static. At 128³, job 32319468, `grad32`: 0.197 static, the moving hydro 0.122 (8.2 ns per cell), the whole moving step 0.078. Earlier (job 31954615): 0.170 static, whole moving step 0.072.

Roofline at 1095 MHz (luna-cpp 28a647d, voronoi-cpp 0b433f0, older than this tree): the step sits at about 1.9 flop/B, at 45 % of the sustained 1.40 TB/s and 16 % of the FP64 peak; only `update` is at the bandwidth roof, the gradient sweep is latency-bound. Bytes alone would allow 0.34 G static and 0.14 G for the whole two-pass step, which achieved 0.059 there.

Kernel level (research_notes/profile_2026-11-06.md, 128³, 1095 MHz): the hydro kernels spend 61 to 98 % of their warp slots waiting for memory. Gradient sweep 0.268, fluxes 0.571, `update` 0.805 G cell updates/s static. The sweep is the slowest: 148 registers keep 17 % of its warps active.

Compile-time precision options (`--precision=`, default `grad32`), against double: mixed +10 %, low +19 % on the hydro static, +4 % / +7 % on the whole moving step with one pass (research_notes/hydro_speed_2026-11-04.md).

Moving 256³ on a 40 GB A100: two passes fit, in `grad32` (2255 B/cell, 39410 of 42405 MB in use) and in double (2319 B/cell, 40484 MB); with `--precision=mixed` they do not (2445 B/cell, about 42.6 GB needed). The double area and centroid arrays stay beside mixed's float face rows.

Woodward-Colella stall: the error stops falling near N = 2¹⁵ at every Courant number tried (research_notes/stage1_1d_2026-11-04.md).

## Roadmap

Hydro speed so far: 0.107 to 0.152 G cell updates/s static. Estimates for the next steps: an owner bit in the list entries (+8 % static, +5 % whole step); faces numbered by owner in the mesh (+8 % hydro measured with a copy, +5 % whole step estimated). Sorting cells by list length and reusing the half step between passes were measured and made it slower.

## Removed, and where to find it

Scripts and modes nothing maintained uses; each line names the last commit that had it (`git show <commit>:<path>`).

- `scripts/hydro_ncu_table.py` and the `ncu` mode of `scripts/hydro_speed_a100.sh`, Nsight Compute tables for research_notes/hydro_speed_2026-11-04.md (jobs 31802044-46); no number here or in the README comes from them: c12980c.
- Modes `builds`, `accuracy`, `stream` and `repro` of `scripts/hydro_speed_a100.sh`, experiments of research_notes/hydro_speed_2026-11-04.md: c12980c.
- Modes `library_check` (voronoi-cpp's own `scripts/bench_a100.sh` checks the library), `memory_compare`, `moving256face` and `ours128` of `scripts/hydro_a100.sh`, which no note or number names: c12980c.
