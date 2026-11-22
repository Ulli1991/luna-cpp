# Testing luna-cpp

Four CTest programs in `src/test/` specify the hydro; `hydro_sedov`, `hydro_1d`, `hydro_wave3d` and `hydro_kh2d` (`src/bench/`) add the runs that are
too slow for CTest. Every test prints one `ok` or `FAIL` line per expectation and exits non-zero on any failure. All
tests run on the default Kokkos execution space, so a CUDA build checks the device code and an OpenMP or Serial build
the host code.

The tests of the mesh, including `voronoi_mesh`, the contract the hydro relies on, are in voronoi-cpp
(`extern/voronoi-cpp/TESTING.md`). They do not run here; `check.hpp` and `support.hpp` come from that submodule.

## Running them

```sh
ctest --test-dir <build> --output-on-failure            # all
OMP_NUM_THREADS=1 ctest --test-dir <build-omp>          # CPU, one thread
```

## The tests

CPU times: OpenMP build (gcc 15), one thread of a Raven login node (Xeon Platinum 8360Y).

| Test | n | CPU, 1 thread | What it specifies |
|---|---|---|---|
| `hydro` | 4096 (jittered, uniform) | 80 s, Serial build | the finite-volume hydro (`src/hydro/`), on the default `grad16` and on `double` (static, moving, lattice; the moving tests also on `grad32`): HLLC and HLL equal LUNA's `hllc_flux_primitive` / `hll_flux_primitive` on 15 / 12 sample faces (values printed by LUNA, float64; within 1e-14), equal states give the exact flux, also through a moving face (F(U) - (w.n) U), swapping states and normal flips the flux, rotating states, normal and face velocity rotates the flux (1000 random rotations, 1e-13); HLLC on jittered and uniform random meshes, HLL on jittered, HLLC with everything reduced (`low`: float face rows, FP16 gradients, float fluxes and Riemann; uniform within 1e-6, loops within 1e-5, float rows), first and second order, per-cell and per-face loops: a uniform moving state stays uniform to rounding (3 steps, within 1e-13), mass, momentum and energy of a rough state (a blast plus 30 % random perturbations) are conserved over 10 steps within 1e-14 (long double sums); two runs give the same bits; the two loops agree within 1e-12 (measured 4e-16 with HLLC, the same bits with HLL); a static run after `release_mesh()` with its mesh destroyed gives the same bits over 10 steps; moving mesh (points on the mesh's quantisation grid, steps whose displacement is a multiple of a grid step): a uniform state with a bulk velocity stays uniform over 5 steps (1e-13; measured 5e-16), mass, momentum and energy are conserved over 10 steps with moving cells (1e-14), a blast moving with the box at Mach 12 equals the blast at rest cell by cell after 40 steps (1e-6; measured 7e-8, rounding grown through the blast) and in its radial density profile (double 1e-10, measured 6e-14; `grad32` 1e-9, measured 2.5e-10 on the CPU, 1.2e-10 on the A100: float gradients carry the frame's rounding into the profile), while on a static mesh the profiles differ by more than 1000 times that (measured 7e-2); two identical moving runs (a blast, 15 steps) give the same bits in time steps, point velocities, positions, volumes, centroids, state and face count, and every stored circumcentre equals a fresh `tet_mesh::circumcentres` computation bit for bit (the repair's flip order does not reach the geometry; the price, accepted by the owner: mesh step 2.4/2.5/4.7 % slower at spacings 0.03/0.1/0.3, research_notes/hydro_stage3_2026-10-31.md); in a (2, 1, 1) box a uniform state with bulk velocity (1, 0.5, -0.25) stays uniform on the moving mesh (1e-13; measured 5e-16), is conserved over 10 moving steps and the mesh equals an exact rebuild; the moving step, `grad32`, double, mixed and low, per-cell and per-face loops: uniform with a bulk velocity (1e-13; 1e-6 in float), conserved (1e-14), the same bits run to run; a static mesh or order 0 refuse them |
| `hydro_wave` | 864-27,648 | 82 s, Serial build | the moving 3D step on smooth flow, in the default `grad16`, against bounds set from measured values on the CPU and the A100, not from a nominal order: L1 of rho, v and P (V weighted, against the exact state at the centroids) and the rate of rho between nx = 24 and 48: Yee's vortex at t = 2, jitter 0.1 (a steady state), and a sound wave of amplitude 1e-3 after one period, jitter 0.25; `grad32` against double on the same two problems, moving and static, nx = 24: the change of each L1 and the L1 of the two end states by cell id, within 3 to 4 times the values measured on the CPU and the A100. Bounds and why: research_notes/two_pass_2026-11-05.md |
| `hydro2d` | 1024-65536 | see below | the 2D hydro (`hydro::FiniteVolume<2>`), static and moving: a uniform state, at rest or with a bulk velocity, stays uniform on lattice and jittered meshes; mass, momentum and energy conserved to rounding, per-cell and per-face; two runs give the same bits; a lattice's zero-length faces give no NaN; sound-wave order in cell averages over the actual polygons; Sod and Noh against the exact solutions, default integrator; a blast carried at Mach ~2.3 equals the blast at rest cell by cell |
| `hydro1d_cases` | 256-4096 | 60 s (2 threads) | every 1D test of LUNA's suite plus two it lacks, 30 in all (`problems_1d.hpp`, gamma per problem): Toro 1-5, Einfeldt strong and shear, LA-14379 Peak, RCVCR, stationary contact, slow shock, stream collision, vacuum (rho = 0) and near vacuum, LeBlanc (gamma 5/3), planar Noh, Shu-Osher, Woodward-Colella, Titarev-Toro, sound right and left, entropy, advected sound, the exact simple wave, pure shocks at Mach 1.2-10. In LUNA's norm (sum of V \|W - exact cell average\| per variable; linear waves the part odd in the amplitude): **static L1 equals LUNA's MC, minmod and van Leer runs within 1e-6** (linear waves 1e-2, rounding at amplitude 1e-6), at N = 256, 512, 1024 (Titarev-Toro 1024-4096 against AthenaK PPMX on 1024 bins); moving L1 within 1.3x measured at the two smallest N; orders within +-0.2 of measured, per variable; known moving failures (#14, #17) must still fail; LUNA's own checks with LUNA's bounds: Toro fronts at 512 (minmod), budgets to t (F(W_L) - F(W_R)), positivity, LeBlanc overshoot, Noh wall heating, shock offset, inflow and impulse, RCVCR and stream mirror symmetry (1e-13, three limiters), Peak height, RCVCR centre, sound left = right, a run stopped at 0.9 t misses its bound; rho = 0 runs static; stage E of the super-hard tube alone (`super-tube-e`, 370 cells, the spacing of N = 4096) moving to t = 0.2 within 100000 steps: the smallest cell over all steps stays above 1e-3 h (measured 5.4e-3 h in 12776 steps; at the cells' mean velocity 3.5e-7 h and a stall at t = 0.132, #25); Noh moving at AREPO's rule, N = 256: more than 1000 steps and L1 rho within 1.3 x 4.91e-3 (it took one step of 0.6 without the wall's pair term, #27); stage L of the tube v2 alone (`stage-l-6`, density ratio 1e6, pressure ratio 1e12, one Riemann problem) moving at N = 1024 to t = 4: L1 rho within 1.3 x 3.02e-3 (measured in 11154 steps). Gates: `luna_1d_gates.hpp`, generated by `scripts/ladder_1d.py gates` |
| `hydro1d_issue30` | 256 | 2 s (4 threads) | luna-cpp #30's reproducer (`hydro1d_cases_test issue30`): stage L's window of the tube v2 (`super-v2-l`, the 1e-6 slab between the high state and the ambient gas) moving with derefinement (cells narrower than 1e-2 h removed) to t = 0.35 within 100000 steps, smallest cell above 1e-3 h: measured 59995 steps, smallest 1e-2 h. Without derefinement: t = 0.2 after 100000 steps, smallest 3.3e-4 h (research_notes/stage_l_2026-11-19.md, research_notes/derefine_1d_2026-11-20.md) |
| `hydro1d` | 16-1024 | 5 s | the 1D hydro (`finite_volume_1d.hpp`): uniform states stay uniform (periodic with bulk velocity, reflective walls at rest; static and moving, order 0 and 1; 1e-13), mass, momentum and energy of a rough state conserved over 40 steps (1e-14; momentum only when periodic), two runs give the same bits (state and positions), inflow keeps its state; Sod (Toro 1) against the exact cell averages within LUNA's bounds at N = 128, 256, static (LUNA's rule, 0.8) and moving (AREPO's rule, 0.4), uniform and jittered; the fused step agrees with the separate kernels (static to rounding; moving within 1.5e-3, measured 5.6e-4: its next dt takes the point's velocity relative to the gas from t^n); **LUNA's own final states of Sod, Shu-Osher and Woodward-Colella at N = 16, walls included, equal ours to 1e-12** (values printed by LUNA, `luna_1d_values.hpp`; measured 7e-16 to 3e-14) in LUNA's number of steps; Shu-Osher moving at N = 1024 with no cells from the wall: no points cross, regularised or not (unregularised ones did at the cells' mean velocity, before #25); cells from a wall: one split conserves to 1e-15 and keeps the state to 1e-14, inflow from either wall comes out as a lattice of the set spacing, Shu-Osher's totals follow the budget through its walls (fixed inflow 1e-12, zero gradient 1e-11) at N = 512, the same bits twice, on both step paths, the mirrored run; Sod and Woodward-Colella add no cell; derefinement: on a jittered periodic line with cells removed a uniform state stays uniform (1e-13) and the totals are kept (1e-14), and Sod with a threshold no cell crosses gives the same bits as without |
| `hydro_derivatives` (opt-in: `-DLUNA_DERIVATIVE_TESTS=ON`) | 1000 (jittered) | under 1 s (8 threads) | the hand-written tangents and adjoints of the 3D hydro in double (`*_tangent`, `*_adjoint` in euler.hpp and finite_volume.hpp; research_notes/autodiff_l1_2026-11-19.md). Per piece, 20000 random samples in a kernel (primitive, face side with half step and fallback, HLL, HLLC, the limited least-squares gradient with 8-16 neighbours): the tangent equals central differences at h = 1e-6 in 999 of 1000 samples to 1e-6 (measured max 2.6e-8), keeping only samples whose branch pattern is the same at x +- h; the adjoint passes the dot test <w, J v> = <J^T w, v> to 1e-13 (measured 3e-15); a tangent and an adjoint off by 1 % in one entry fail both. Whole static steps (`step_tangent`, `step_adjoint`; HLLC 1 and 3 steps, HLL 1 step, smooth state): Q equals `step()`'s (same bits on the host), J dq equals central differences to 1e-6 of the largest (measured 5e-10), the dot test to 1e-13 over 1 and 3 steps (adjoint backwards from checkpoints of Q), and Q unchanged by `step_adjoint`; a rough state (30 % noise), 3 steps: where a switch lies within +- h of a cell (a clip or a fallback), the tangent equals one of the two one-sided differences to 1e-4 of the cell's largest entry, and the dot test; two reverse passes give the same bits. `--scan`: the rough step against h = 1e-5 .. 1e-9 (research_notes/autodiff_l1_2026-11-19.md). `--cost N`: times of the forward step, `step_tangent` and `step_adjoint` on N cells |

## Bit hashes

`scripts/hydro_speed_a100.sh` mode `bits` (MAIN and FIX: two CUDA build directories) runs five paths at 128³, 50
steps: static, moving, order 0, per-cell moving, HLL moving. `scripts/snapshot_hash.sh` prints hashes of the final U, x
and volume. Two builds of the same hydro and mesh must give equal hashes. A tree from before `grad32` became the default matches
`--precision=double`: in `hydro_speed_a100.sh` give FIX_FLAGS=--precision=double. `hydro_1d` prints a hash of its final state
and positions, the same in every run. When a change moves the last bits (as the HLLC copy of 2026-11-18 did),
`scripts/snapshot_diff.py <run A> <run B>` gives the largest relative difference per conserved variable and the L1 of
the density difference between the two runs' last snapshots, cell by cell.

## Speed levers

`scripts/hydro_lean_a100.sh` measures a hydro change against main in one A100 job (docs/tried.md): mode `speed` runs
Sedov 40 steps static and moving at 128³ and 256³, the binaries alternating, each once with `--phases` (fenced phase
times) and once without (`step_wall_ms_median`: the wall time of each step, no fences inside it); mode `bits` runs the hashes of `hydro_levers_a100.sh`; mode
`ctest` the CTest programs of each build. Each change keeps a row in `docs/tried.md`. `scripts/hydro_quant_a100.sh`
measures how far a change moves the bits and the accuracy (`snapshot_diff.py` after 50 steps, the field L1 to 1e4 yr),
`scripts/hydro_ncu_local_a100.sh` the local memory, registers and warps of each hydro kernel (Nsight Compute, summary by
`hydro_ncu_summary.py`), `scripts/hydro_sass_count.sh` the instructions of the flux kernels, and
`scripts/hydro_wholerun_a100.sh` whole Sedov runs to 1e4 yr (cell updates per GPU second over the run).

## Moving the pin

A new voronoi-cpp pin is checked against the old one with both trees built in the same A100 job: the CTest
programs on the A100 and CI's host steps, the bit hashes above, and `hydro_sedov` to t_end for stuck steps and speed.
`scripts/pin_bits_a100.sh` (builds, A100 ctest, bits), `scripts/pin_speed_a100.sh` (stuck steps, speed) and
`scripts/pin_ci_host.sh` (CI's host steps) take the two worktrees as OLD and NEW. The last move, abd3d39 to 1741a1a
(2026-11-18), is in docs/measurements.md ("voronoi-cpp abd3d39 to 1741a1a"):
- Builds and tests (jobs 32540741, 32540744): A100 ctest 5 of 5; CI's host steps ctest 5 of 5 with sanitizers off
  and on, lint clean.
- Bits (job 32540741): equal hashes of Q, x and volume on static, order 0, moving, and per-cell moving Sedov 128³
  (50 steps); 1D Sod, Shu-Osher and Woodward-Colella at N = 1024, static and moving; 2D KH 256², 300 steps, static
  and moving. HLL moving (50 steps) differs in Q only. Moving Sedov to 1e4 yr differs in Q at 128³ (x and volume
  equal) and in Q, x and volume at 256³. The field L1 of rho, v_r and P is equal to all printed digits (128³ static
  and moving, 192³ and 256³ moving), and so are the 3D vortex and the sound waves at 1e-3 and 1e-6 (job 32540743).
- Stuck steps (job 32540742): none in two runs per pin at 128³ and 256³ to 1e4 yr; no rebuild, no local repair.
- Re-sorting stays off in luna-cpp (`FiniteVolume::no_resort`), although 1741a1a's `voronoi::Layout` re-sorts every
  25 updates by default.

The move before, 6ead543 to abd3d39 (2026-11-18), is in docs/measurements.md ("voronoi-cpp 6ead543 to abd3d39"):
- Bits (job 32520636): equal hashes of Q, x and volume on all five Sedov 128³ paths (50 steps) and on moving Sedov
  128³ to 1e4 yr (349 steps); 1D Sod, Shu-Osher and Woodward-Colella at N = 1024, static and moving; 2D KH 256²,
  300 steps, static and moving. Only the device flip counts differ, by a few in 10⁶, as between two runs of one binary
  (the flips follow atomics; the geometry does not).
- Stuck steps, moving Sedov to 1e4 yr, two runs per pin (job 32520651): none at 128³ (349 steps). At 256³
  (919 steps), one of abd3d39's runs had one step with 6 stuck entries (#8). `Fallback::cheapest` repaired it
  locally (49 cells, no rebuild), and the run's final Q, x and volume equal those of the three runs that did not stick.
  No run rebuilt or took the host finish.
- The ladder gates of `hydro1d_cases` (`luna_1d_gates.hpp`) pass on the A100 and in both host builds.

## The 2D tests (`hydro2d` in CTest; KH on the A100)

`hydro2d`, static and on moving meshes: (in the default `grad32` since 2026-11-19; double before) uniform states, conservation, two runs with the same bits,
sound-wave order in cell averages over the actual polygons, a Galilean blast; Sod and Noh.

KH: `scripts/dim2d_a100.sh kh <N...>` and `khbulk <u> <N...>`, one job per ladder, then
`scripts/compare_kh.py <runs> <logs...>`: M(t) against the Pencil 4096² curve, field L1 on block averages of the actual
cells, GPU time to a target error. Moving-mesh M is summed at cell centroids, where a cell's state belongs.

## The 1D ladder (not in CTest)

`scripts/ladder_1d_a100.sh <series> <problems|all> <b_lo> <b_hi>` runs LUNA and ours on N = 2^b (256 to 2^17) on A100s, one worker per GPU,
resuming where a job stopped; `scripts/ladder_1d.py table` gives L1 per variable and the measured orders, `gates` the CTest header.
Series: LUNA with MC, minmod, van Leer; ours static with the same three and no limiter; moving with the same; moving on a wide periodic
line (`--pad`, for the tubes whose walls #14 stops) and at a fixed step (Noh, #15). Results: research_notes/tests_1d_2026-11-09.md;
figures: `ladder_1d`, `grid_1d`, the per-test figures.

## The 1D tests against LUNA (not in CTest)

`scripts/dim1d_a100.sh <mode>` (see its head) runs both codes in one job on one node; its log holds the node and
`clock-<job>.csv` the clock. `scripts/compare_1d.py <runs> <out> <refs>` prints, per test and series, field L1 of rho,
v, P (the integral of |f_h - f_ref| dx / L of the code's reconstructed field, by Gauss-Legendre on the union of the
cell edges of both fields; never sampled at generators), the ratio to LUNA at the same N, the largest difference of
static runs from LUNA, and GPU-seconds and cell updates to a target error, interpolated between a code's own runs.
Sod's reference is the exact solution; Shu-Osher's and Woodward-Colella's a fine LUNA MC run,
`compare_1d.py ref <problem> <run> <refs>`, checked against AthenaK PPMX 16384. Results and pass bars:
research_notes/stage1_1d_2026-11-04.md.

## The super-hard tube v2 (not in CTest)

`super-tube-v2` (problems_1d.hpp and scripts/problems_1d.py; layout from `scripts/super_tube_layout.py v2`) and its new stages alone
(`super-v2-m`, `-l`, `-so`, `-n`, `-v`, t = 0.1). `scripts/super_tube_v2_run.sh <out> <max_s> "<series> <problem> <N> <t>" ...` runs LUNA,
static and moving on CPU cores, and `scripts/super_tube_v2_gpu.sh` runs them on an A100 (LUNA chunked and resumable).
`scripts/super_tube_v2.py single|score|figures|npz|ic` gives the L1 per stage against the exact solution at t = 0.1 and against LUNA
2^20 at t = 4, positivity, budgets and stops, a figure per stage, the final states for luna-reference-solutions, and initial cells for
other codes. LUNA runs with its positivity floors on (`FLOOR=1`, `luna_1d_run.py floor=1`); without them it goes NaN in the Noh stage
after t = 0.32 at N = 4096 and 32768. The moving mesh cannot run the tube (#14, #29): `super-tube-v2-m0`, the tube without stage M's
stream, stands in for it; it stalls in stage L at t = 0.81 (N = 4096) and 0.65 (N = 32768), #30. With derefinement
(`hydro_1d --derefine=f`, cells narrower than f h removed) it reaches t = 4 at both N. Results:
research_notes/super_tube_v2_2026-11-18.md, research_notes/derefine_1d_2026-11-20.md (`scripts/deref_sweep.py` for the ladder,
`DEREF=f scripts/stage_l_run.sh` for stage L alone).
