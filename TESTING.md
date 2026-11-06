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
| `hydro` | 4096 (jittered, uniform) | 30 s | the finite-volume hydro (`src/hydro/`): HLLC and HLL equal LUNA's `hllc_flux_primitive` / `hll_flux_primitive` on 15 / 12 sample faces (values printed by LUNA, float64; within 1e-14), equal states give the exact flux, also through a moving face (F(U) - (w.n) U), swapping states and normal flips the flux, rotating states, normal and face velocity rotates the flux (1000 random rotations, 1e-13); HLLC on jittered and uniform random meshes, HLL on jittered, HLLC with stored float gradient weights (`weight32`) and with everything reduced (`lowweights`: float face rows and weights, FP16 gradients, float fluxes and Riemann; uniform within 1e-6, loops within 1e-5, float rows), first and second order, per-cell and per-face loops: a uniform moving state stays uniform to rounding (3 steps, within 1e-13), mass, momentum and energy of a rough state (a blast plus 30 % random perturbations) are conserved over 10 steps within 1e-14 (long double sums); two runs give the same bits; the two loops agree within 1e-12 (measured 4e-16 with HLLC, the same bits with HLL); moving mesh, with both integrators (two passes are the default; points on the snapping grid, steps whose displacement is a multiple of a grid step): a uniform state with a bulk velocity stays uniform over 5 steps (1e-13; measured 5e-16), mass, momentum and energy are conserved over 10 steps with moving cells (1e-14), a blast moving with the box at Mach 12 equals the blast at rest cell by cell after 40 steps (1e-6; measured 7e-8, rounding grown through the blast) and in its radial density profile (1e-10; measured 1e-13), while on a static mesh the profiles differ by more than 1000 times that (measured 7e-2); two identical moving runs (a blast, 15 steps) give the same bits in time steps, point velocities, positions, volumes, centroids, state and face count, and every stored circumcentre equals a fresh `tet_mesh::circumcentres` computation bit for bit (the repair's flip order does not reach the geometry; the price, accepted by the owner: mesh step 2.4/2.5/4.7 % slower at spacings 0.03/0.1/0.3, research_notes/hydro_stage3_2026-10-31.md); in a (2, 1, 1) box a uniform state with bulk velocity (1, 0.5, -0.25) stays uniform on the moving mesh (1e-13; measured 5e-16), is conserved over 10 moving steps and the mesh equals an exact rebuild; two passes, double, mixed and low, per-cell and per-face loops: uniform with a bulk velocity (1e-13; 1e-6 in float), conserved (1e-14), the same bits run to run; a static mesh or order 0 refuse them |
| `hydro_wave` | 864-27,648 | 51 s | the moving 3D step on smooth flow with both integrators, against bounds set from measured values on the CPU and the A100, not from a nominal order: L1 of rho, v and P (V weighted, against the exact state at the centroids) and the rate of rho between nx = 24 and 48: Yee's vortex at t = 2, jitter 0.1 (a steady state), and a sound wave of amplitude 1e-3 after one period, jitter 0.25 (one pass stalls at the floor of its regularised mesh motion, two passes do not). Bounds and why: research_notes/two_pass_2026-11-05.md |
| `hydro2d` | 1024-65536 | see below | the 2D hydro (`hydro::FiniteVolume<2>`), static and moving with both integrators: a uniform state, at rest or with a bulk velocity, stays uniform on lattice and jittered meshes; mass, momentum and energy conserved to rounding, per-cell and per-face; two runs give the same bits; a lattice's zero-length faces give no NaN; sound-wave order in cell averages over the actual polygons; Sod and Noh against the exact solutions, default integrator; a blast carried at Mach ~2.3 equals the blast at rest cell by cell |
| `hydro1d` | 16-1024 | 5 s | the 1D hydro (`finite_volume_1d.hpp`): uniform states stay uniform (periodic with bulk velocity, reflective walls at rest; static and moving, order 0 and 1; 1e-13), mass, momentum and energy of a rough state conserved over 40 steps (1e-14; momentum only when periodic), two runs give the same bits (state and positions), inflow keeps its state; Sod (Toro 1) against the exact cell averages within LUNA's bounds at N = 128, 256, static (LUNA's rule, 0.8) and moving (AREPO's rule, 0.4), uniform and jittered; the fused step agrees with the separate kernels; **LUNA's own final states of Sod, Shu-Osher and Woodward-Colella at N = 16, walls included, equal ours to 1e-12** (values printed by LUNA, `luna_1d_values.hpp`; measured 7e-16 to 3e-14) in LUNA's number of steps; Shu-Osher moving at N = 1024 with no cells from the wall: unregularised points cross, regularised ones do not; cells from a wall: one split conserves to 1e-15 and keeps the state to 1e-14, inflow from either wall comes out as a lattice of the set spacing, Shu-Osher's totals follow the budget through its walls (fixed inflow 1e-12, zero gradient 1e-11) at N = 512, the same bits twice, on both step paths, the mirrored run; Sod and Woodward-Colella add no cell |

## Bit hashes

`scripts/hydro_speed_a100.sh` mode `bits` (MAIN and FIX: two CUDA build directories) runs five paths at 128³, 50
steps: static, moving, order 0, per-cell moving, HLL moving. `scripts/two_pass_a100.sh` mode `bits` does the same with the
one-pass runs of a tree from before two passes against `--integrator=one-pass`, and runs two passes twice. `scripts/snapshot_hash.sh` prints hashes of the final U, x
and volume. Two builds of the same hydro and mesh must give equal hashes. `hydro_1d` prints a hash of its final state
and positions, the same in every run.

## The 2D tests (`hydro2d` in CTest; KH on the A100)

`hydro2d`, static and on moving meshes with both integrators: uniform states, conservation, two runs with the same bits,
sound-wave order in cell averages over the actual polygons, a Galilean blast; Sod and Noh with the default integrator.

KH: `scripts/dim2d_a100.sh kh <N...>` and `khbulk <u> <N...>`, one job per ladder, then
`scripts/compare_kh.py <runs> <logs...>`: M(t) against the Pencil 4096² curve, field L1 on block averages of the actual
cells, GPU time to a target error. Moving-mesh M is summed at cell centroids, where a cell's state belongs.

## The 1D tests against LUNA (not in CTest)

`scripts/dim1d_a100.sh <mode>` (see its head) runs both codes in one job on one node; its log holds the node and
`clock-<job>.csv` the clock. `scripts/compare_1d.py <runs> <out> <refs>` prints, per test and series, field L1 of rho,
v, P (the integral of |f_h - f_ref| dx / L of the code's reconstructed field, by Gauss-Legendre on the union of the
cell edges of both fields; never sampled at generators), the ratio to LUNA at the same N, the largest difference of
static runs from LUNA, and GPU-seconds and cell updates to a target error, interpolated between a code's own runs.
Sod's reference is the exact solution; Shu-Osher's and Woodward-Colella's a fine LUNA MC run,
`compare_1d.py ref <problem> <run> <refs>`, checked against AthenaK PPMX 16384. Results and pass bars:
research_notes/stage1_1d_2026-11-04.md.
