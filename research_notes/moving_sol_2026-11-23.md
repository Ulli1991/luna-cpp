# Speed of light of the moving step's hydro kernels, 2026-11-23

Branch `moving-sol` from luna-cpp main 5ff2011 (voronoi-cpp 1741a1a; defaults: `grad16`, float face centroids, int32
steps, HLLC). One A100-SXM4-40GB at 1305 MHz (logged). Sedov 3D jittered, moving, Courant 0.3.

**Profile.** Nsight Compute, full set, of the eight `FiniteVolume` kernels of one moving step at about step 10:
- `scripts/moving_sol_a100.sh`, `scripts/moving_sol_table.py`;
- jobs 32902044 (128³) and 32904580 (256³);
- reports in `/ptmp/uli/luna-cpp-moving-sol/ncu/`;
- the method of `research_notes/fixed_grid_2026-11-19.md` (branch fixed-grid), section 9.

The mesh kernels (voronoi-cpp) are not part of this note.

## Per kernel, 128³ (256³ within 1-6 % in every column)

ns/cell at 256³ in brackets. B/cell are read + write. "Floor DRAM" is the measured bytes at 1.37 TB/s; "floor FP64" is the
time the FP64 pipe is busy.

| kernel | ns/cell | regs | block | occupancy theor./achieved % | eligible warps/scheduler | issue % | top stalls (% of stall cycles) | branch inst/cell | divergent targets/cell | threads/inst | local (spill) | DRAM B/cell | L2 B/cell | FP64 pipe % | floor DRAM ns | floor FP64 ns |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sweep (gradients, limiter, w) | 2.233 (2.228) | 120 | 128 | 25/23 | 0.30 | 25 | long scoreboard 62, wait 19, selected 7 | 1389 | 3.7 | 19.9 | 0 | 514+60 | 3167 | 15 | 0.419 | 0.341 |
| time step (face and cell terms) | 0.330 (0.320) | 38 | 256 | 75/61 | 0.36 | 26 | long scoreboard 81, wait 9 | 134 | 0.0 | 31.8 | 0 | 287+1 | 623 | 38 | 0.210 | 0.127 |
| keep_centroids | 0.125 (0.121) | 38 | 128 | 75/66 | 0.23 | 16 | long scoreboard 79, lg throttle 7 | 26 | 0.0 | 32.0 | 0 | 76+57 | 469 | 12 | 0.097 | 0.015 |
| fluxes, pass 1 | 1.109 (1.085) | 88 | 256 | 25/24 | 0.41 | 31 | long scoreboard 59, wait 17, selected 8 | 331 | 0.0 | 31.9 | 0 | 649+315 | 2044 | 41 | 0.704 | 0.452 |
| update, pass 1 | 0.979 (0.937) | 44 | 128 | 62/59 | 0.05 | 4 | long scoreboard 90, lg throttle 7 | 27 | 0.2 | 24.4 | 0 | 1075+39 | 2866 | 3 | 0.813 | 0.030 |
| drift | 0.056 (0.057) | 34 | 128 | 75/64 | 0.33 | 24 | long scoreboard 53, lg throttle 10 | 20 | 0.0 | 32.0 | 0 | 36+17 | 176 | 4 | 0.038 | 0.002 |
| fluxes, pass 2 | 1.265 (1.254) | 96 | 256 | 25/24 | 0.45 | 34 | long scoreboard 56, wait 18, selected 9 | 378 | 0.0 | 31.9 | 0 | 708+378 | 2330 | 41 | 0.793 | 0.515 |
| update, pass 2 (+ next primitives) | 1.038 (0.995) | 44 | 128 | 62/59 | 0.05 | 5 | long scoreboard 91, lg throttle 7 | 35 | 0.2 | 25.6 | 0 | 1074+93 | 3124 | 4 | 0.852 | 0.043 |
| **hydro, all eight** | **7.136 (6.996)** | | | | | | | | | | **0** | **4321+1000** | | | **3.93** | **1.52** |

What the table says:
- **No kernel spills.** Local memory is 0 everywhere, so spill removal has nothing to remove.
- **No flux kernel diverges.** 31.9 threads per instruction, 0 divergent targets. HLLC's branches are uniform across
  a warp, so selects can save at most their issue slots.
- **The sweep is the largest gap.**
  - 2.23 ns against floors of 0.42 (DRAM) and 0.34 (FP64).
  - It runs at 120 registers, 23 % of warps and 0.30 eligible warps per scheduler, waiting on loads (62 %).
  - It is the only kernel with divergence: 19.9 threads per instruction, because the cells' lists differ in length,
    and 3.7 divergent targets per cell in the limiter.
- **The updates** run at 59 % of warps but 0.05 eligible warps: they wait on the flux rows. They move 1114-1167 B/cell
  against about 470-530 needed (each 40-B row read twice, two sectors each time), at 83-85 % of DRAM peak.
- **The flux passes** are at 63 % of their measured-bytes DRAM floor and keep the FP64 pipe 41 % busy, at 25 % of
  warps (88-96 registers).

## Speed of light and memory of light (128³)

"Need" is the least DRAM traffic of each kernel's job, each row read once and each result written once (estimate from the
array sizes at the defaults):
- primitive row 56, FP16 gradient row 32 + 4, int32 steps 12, positions 32, centroids 24, w 24, list 68;
- face rows 44 a face (owner and neighbour 8, area 24, float centroid 12): 350;
- \|A\| 8 a face: 64;
- flux rows 40 a face: 318;
- Q 40.

SoL is the larger of need at 1.37 TB/s and the FP64 floor.

| kernel | need B/cell | SoL ns/cell | measured | gap ns/cell |
|---|---|---|---|---|
| sweep | about 460 | 0.34 | 2.23 | 1.89 |
| fluxes, pass 1 | about 830 | 0.60 | 1.11 | 0.51 |
| fluxes, pass 2 | about 915 | 0.67 | 1.27 | 0.60 |
| update, pass 1 | about 466 | 0.34 | 0.98 | 0.64 |
| update, pass 2 | about 530 | 0.39 | 1.04 | 0.65 |
| time step | about 180 | 0.13 | 0.33 | 0.20 |
| keep_centroids, drift | about 190 | 0.14 | 0.18 | 0.04 |
| **hydro** | **about 3570** | **2.61** | **7.14** | **4.53** |

The hydro of a moving step is 2.7× from its speed of light: 15.0 ms against about 5.5 ms at 128³.

## Levers, ranked by the ns/cell they could save (128³)

| rank | lever | kernel | could save | evidence | cost and risk |
|---|---|---|---|---|---|
| 1 | Limiter by selects (branch-free; a factor of exactly 1 where nothing is limited, so the same bits) | sweep | part of the 3.7 divergent targets and 1389 branches per cell: about 0.1-0.3 | 14 % of the sweep's branch targets diverge; wait 19 % | cheap; same bits expected |
| 2 | Block size of the sweep, update and time step (64 / 256; no register cap) | sweep, updates, time step | 0-0.2 | the sweep is register-bound (120 registers: 16 warps at 128 or 256 threads); the updates wait on memory at 59 % of warps; the flux passes were measured on 2026-11-20 (smaller blocks slower) | cheap; same bits |
| 3 | Early loads: both sides' primitive, gradient and step rows loaded at the top of the face flux, before the face velocity and geometry arithmetic | fluxes | up to about 0.2 a pass if the loads now wait behind arithmetic; 0 if the compiler already hoists them | long scoreboard 56-60 %, wait 17-18 % | cheap; registers may rise (96 → over 128 halves the warps); same bits |
| 4 | HLLC by selects | fluxes | at most the issue slots of about 40 uniform branches a face: under 0.05 | 0 divergent targets, 31.9 threads per instruction | cheap; same bits |
| 5 | Spill removal | all | 0 | no local memory anywhere | — |
| 6 | CUDA rewrite of the flux kernel (estimate only) | fluxes | 0.1-0.25 a pass | fixed-grid measured 1.22-1.34× for CUDA over Kokkos at matched shapes, but there Kokkos spilled and inlined differently; ours spills nothing and diverges nowhere | high: a second code path |
| — | The structural gaps: the updates' double row reads (1.3), the sweep's list walk (most of 1.9) | updates, sweep | 1.3 + 1.5 | fusion and per-cell lanes were measured slower (docs/tried.md) | not cheap |
