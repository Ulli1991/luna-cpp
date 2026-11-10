# Measurements: provenance

Where each number and figure in the [README](../README.md) comes from: jobs, commits, clocks and the setup details left out of the README. Sections follow the README headings. Unless stated, runs are on one A100 40 GB, double precision, HLLC. The same code varies by about 10 % between GPU jobs.

The records below keep the names of their time: "two passes" (`--integrator=two-pass`) is today's moving step, MUSCL-Hancock on a moving mesh, the only one since 2026-11-10; "one pass" was a single flux at the half step, an option until then, now removed.

## The hydro

### Sedov speed table

Sedov blast of 1e51 erg, n = 1 cm⁻³, 50 pc box, 1e4 yr; jittered points in Morton order; 40 steps. Job 31954615, ravg1020, 1305 MHz, bef77ed: medians of three rounds at 2.1M cells, one round at 16.8M, `--memory`. LUNA in the same job, 200 steps, median of the 20 perf lines. The same code varies by about 10 % between GPU jobs. LUNA FP64 at 2.1M cells moved between jobs: 0.47 here, 0.67 in job 31953960, 0.76 in job 31953468; at 16.8M 0.87, 0.90, 0.95; FP32 0.68, 1.01, 1.37 and 1.74, 1.91, 2.13.

README speed table, `grad32`, job 31954615 at 1305 MHz: static HLLC 0.170 at 2.1M cells and 0.166 at 16.8M; whole moving step with the mesh update 0.072 and 0.076. GPU memory moving 2255 B/cell at both sizes. LUNA in the same job: FP64 0.47 to 0.76 (2.1M, varies by job) and 0.87 (16.8M), FP32 0.68 and 1.74.

Hashes in job 31954615 at 128³, 50 steps, CUDA: `--precision=double` equals main 3449600 on static, moving, moving one pass, order 0, per-cell moving and HLL moving; `default` equals `grad32` static and moving.

Rows left out of the README, same job: `--precision=double` static HLL 0.157 at 2.1M cells; first order 0.312 in both precisions. Peak device memory at 16.8M cells, GB in use: `grad32` static 38.2, two passes 39.4, one pass 39.0; double 39.3, 40.5, 40.1. `--precision=mixed`, two passes, 2445 B/cell, does not fit at 256³; one pass 2334 B/cell (40.7 GB, job 31925336). LUNA memory: job 31953468.

LUNA rows: MC, HLLC, Pallas, FP64 (FP32 where stated).

### Sedov accuracy

Sedov at 128³, job 31699863, HLLC (the moving numbers of this paragraph are one pass): shock radius 0.978 of the Sedov-Taylor value at 1e4 yr, L1 density error 0.075 rho0 (first order: 0.104), mass and energy conserved to 2e-16.

Moving mesh, jobs 31700850 and 31700851: 354 steps instead of 1642, L1 density error 0.058 rho0, the same numbers with the whole box moving at 500 km/s. At 256³ L1 0.034 in 288 s of GPU time (job 31702077); now 176 s (job 31925157), two passes 230 s.

These L1 values are sampled at the mesh generators, which flatters a moving mesh. The field L1 (the integral of |rho_h - rho| dV) is the fair norm (research_notes/accuracy_score_exploration_2026-11-04.md). Details: research_notes/hydro_stage1_2026-10-31.md, hydro_stage3_2026-10-31.md.

Field L1 of density, r < 1.5 R_s (one pass, two passes): 128³ 0.0629, 0.0648; 256³ 0.0369, 0.0377 (research_notes/two_pass_2026-11-05.md).

The one-pass step of the moving mesh keeps the bits it had before the two-pass default; hash in research_notes/two_pass_2026-11-05.md, which also compares cost and accuracy of the two.

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
| `throughput_precision.png` | 32³ to 256³, 40 steps, 1305 MHz, job 31953468, e90a913. LUNA in the same job (200 steps, median of the perf lines). None of our options is FP32 throughout. At 128³ `grad16` gives +15 % hydro, +10 % whole step; `grad32` +12 % and +8 %; `mixed` +10 % and +6 %; `flux32` nothing. `mixed` two-pass does not fit at 256³ |
| `throughput_dimensions.png` | HLLC, FP64. A blast in all three: LUNA's Sedov without units in 1D (gamma 1.4) and 2D (gamma 5/3), the Sedov above in 3D. 1D has no two-pass integrator. Jobs 31953276 and 31953278 at 1305 MHz, 31953467 at 1095 MHz; the 1D and 2D blast is not on main yet, runs at voronoi-cpp d077c41. At the largest size static hydro runs 2.66, 0.67 and 0.15 G cell updates/s against LUNA 2.96, 1.46 and 0.95. Two passes take the whole step to 0.29 in 2D and 0.07 in 3D |

The precision table in the README is from job 31953468 at 256³, whose `default` is today's `--precision=double`. `grad32` against double on the final tree, job 31954615, 128³ and 256³: +12 % static, +8 % the whole step with two passes, +8 to +9 % with one, 64 B/cell less moving with two passes, 65 otherwise. An earlier job, 31928265, also 1305 MHz: +12 % static, +9 % the whole moving step, the same 64 B/cell.

`grad32` against double, accuracy: jobs 31928264 to 31929256, post-processing 31929253 to 31930149, gradient-precision scripts of branch precision-check (4b540dd, fbe7fcc), which stayed out of main. Moving two passes, static one pass. Vortex 64 to 256, sound (amplitude 1e-3) 64 to 256, simple wave 128 to 512, also at jitter 0.25, tube 256, Sedov 128³ and 256³. L1 against the exact averages or the Sedov field: 4-digit output, so the largest change seen, 1e-3 in P on the moving vortex, is near that output's resolution at 128 (1.014e-4 against 1.013e-4) and 9e-4 at 256. Pressure contrast (`--cold`) at 128³, moving, 200 steps: energy floor met in 5 cell-steps at 1e-8 in `grad32`, 35 in double; one each at 1e-4; density floor never; nothing non-finite.

Radial profile of the Galilean test (`hydro`): double 6e-14; `grad32` 1.2e-10 on the A100, 2.5e-10 on the CPU. The cell-by-cell measure stays at 7e-8 in both.

## Limitations and open items

Hydro speed target: 0.3 G cell updates/s static, 0.15 G the whole two-pass step. At 128³, job 31954615, `grad32`: 0.170 static, hydro part 0.164 on a moving mesh (one pass), whole moving step 0.091 with one pass and 0.072 with two. Double: 0.152, 0.142, 0.084 and 0.067.

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
