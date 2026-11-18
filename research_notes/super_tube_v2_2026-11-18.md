# The super-hard 1D tube, v2 (2026-11-18)

**Preliminary**: the moving mesh's runs to t = 4 (m0 tube, N = 4096 and 32768, SLURM 32522444 and 32522445 on gpu1) were still running when
this went in; their rows follow in a later commit. Everything else here is measured and final.

Branch super-tube-v2 (from main 2d08e1e, merged with main 73f763a and 68d1e65). [M] measured, [I] inference.
v1: research_notes/super_tube_2026-11-10.md. v2 keeps v1's stages A to H unchanged and adds five of the hardest known 1D problems.
Code:
- `scripts/super_tube_layout.py v2`: layout, fronts, timing;
- `problems_1d.hpp` `super_tube_v2` and `scripts/problems_1d.py` `super2_*`: the same initial state as exact cell averages, and the exact
  solution;
- `scripts/super_tube_v2.py`: scores, initial cells for other codes, figures, npz;
- `scripts/super_tube_v2_{run,gpu,build}.sh`: the runs.

LUNA runs with its positivity floors on, `floor=1` in `scripts/luna_1d_run.py`. That is `preserve_face_states=True`, the RAMSES floors
with `hydro_smallr` = `hydro_smallc` = 1e-10, off by default in LUNA; `allow_first_order_faces=True` as before. Section "LUNA and its
floors" has what changes without them.

## Layout

Gas with γ = 1.4 throughout (the tube holds one γ, so Noh is its γ = 1.4 version) on [0, 40.25]; left wall open (zero gradient), right wall
reflecting; buffers of the ambient state (1, 0, 1) between stages. Buffers are sized from exact Riemann fronts so that no two stages meet
before t = 0.15 (each at least 1 wide). The checkpoint is t₁ = 0.1 as in v1, and the end is t = 4.

| key | stage | where | scored on | state (ρ, v, P) | tests |
|---|---|---|---|---|---|
| M | Mach 1e4 advection | 0 to 2 | 0 to 2.56 | (1, -11.832, 1e-6), plus ρ += 0.5 sin²(π (x - 1.3) / 0.5) on [1.3, 1.8] | cold gas at Mach 1e4 (kinetic/thermal 2.8e7), out through the open wall |
| A | quiet | 3 to 4 | 2.56 to 4.5 | (1, 0, 1) | noise from mesh motion alone |
| B | smooth waves | 5 to 6 | 4.5 to 6.5 | v1's sound and entropy waves, 1e-4 | order on smooth flow |
| C | stationary peak | 7 to 7.05 | 6.5 to 7.49 | (10, 0, 1) | contact sharpness |
| C2 | moving peak | 8.05 to 9.6 | 7.49 to 10.12 | (10, 1, 1) in (1, 1, 1) | moving contact |
| D | sonic rarefaction | 10.6 to 11.6 | 10.12 to 12.1 | (1, 0.75, 1) / (0.125, 0, 0.1) | entropy glitch at the sonic point |
| E | extreme jump | 12.6 to 13.6 | 12.1 to 14.1 | (1, 0, 0.1) / (1e-3, 0, 1e-10) | v1's, kept for comparability |
| L | LeBlanc-type jump | 14.6 to 16.1 | 14.1 to 16.53 | (1, 0, 0.1) / (1e-6, 0, 1e-13) | density ratio 1e6, pressure ratio 1e12 |
| F | slow shock | 17.1 to 18.1 | 16.53 to 18.60 | (3.857143, -0.810631, 10.33333) / (1, -3.44, 1) | ripples behind a slow shock |
| SO | Shu-Osher | 19.1 to 23.9 | 18.60 to 24.34 | (3.857143, 2.629369, 10.33333) on [19.1, 20.1], then ρ = 1 + 0.2 sin(5 (x - 20.1)) for 3 wavelengths | Mach 3 shock into a density sine |
| N | Noh streams | 24.9 to 26.4 | 24.34 to 26.91 | (1, 1, 1e-6) / (1, -1, 1e-6) | cold colliding streams: wall heating |
| S | colliding streams | 28.4 to 29 | 26.91 to 31.09 | Toro 5 | strongest shocks from velocity |
| G | near vacuum | 31.75 to 32.75 | 31.09 to 33.20 | (1, -2, 0.4) / (1, 2, 0.4) | positivity, ρ down to 0.02 |
| V | true vacuum | 33.75 to 35.25 | 33.20 to 36.21 | (1, -4, 0.4) / (1, 4, 0.4) | vacuum: Δv = 8 > 2 (c_L + c_R) / (γ - 1) = 7.48 |
| H | double blast | 39.25 to 40.25 | 36.21 to 40.25 | P = 1000, 0.01, 100, against the right wall | collisions and reflections |

Choices against the brief:
- M: the brief's v = 1e4 at P = 1e-6 is Mach 8.5e6, and the stream would cross the tube in 4e-3. Mach 1e4 at P = 1e-6 is v = 1e4 c = 11.83. The
  pressure's share of the energy, the hard part, depends on the Mach number alone. The stream runs left and is out of the tube by t = 0.17. At
  t₁ the bump is at [0.12, 0.62], whole and away from the wall. Behind the stream the ambient gas expands into vacuum (Δv = 11.8 > 5.92).
- L: next to E, at γ = 1.4 (classic LeBlanc is γ = 5/3 with ratios 1e3 and 1e9). Its low side is 1 wide, so its fronts meet at 0.137 > t₁.
- N: Noh at γ = 1.4. In the limit P → 0: ρ = (γ + 1)/(γ - 1) = 6, P = 1.2, shock speed 0.2. At P = 1e-6 the exact Riemann solution gives
  ρ = 5.99997, P = 1.2000022, speed ±0.2000012; that is what is scored. The streams' outer ends drive shocks into the cold gas from behind
  (speed 1.44), which meet the centre's shocks at t = 0.458.
- SO: the post-shock slab is 1 wide. Its rarefaction catches the shock at about t = 1, after it has crossed the 3 wavelengths (3.77).

Timing (exact fronts) [M]: inside each exact stage, single Riemann problems until t = 0.1056 (F; E 0.111, D 0.128, L 0.137, G 0.201,
V 0.215, SO's slab 0.258, N 0.458, C2 0.493); across buffers, first meetings at t = 0.152 (V with H), 0.155 (S's measured spread with G),
0.164 (S's with N). S and H interact within themselves at once, as in v1.

S sends waves out faster than its edge fronts say. v1 measured 1.49 to the right at t = 0.1. To the left, S alone (`super-v2-s-probe`,
LUNA, spacing of 2^17) is disturbed 0.21 past its edge at t = 0.05 and 0.84 at t = 0.1, a wave at about 12.6 out of its inner collision
(its edge front alone says 0.34). The first v2 layout had a 1.0 buffer there, and S reached N's region by t = 0.1 (ρ = 4.4 at x = 26.6
in every code). The layout now bounds S by 16 t to the right and 11 t to the left, and the N | S buffer is 2.

Scoring regions: from the middle of the quiet line at t₁ between two stages to the next (super_tube_layout.v2, `regions`). Exact at t₁:
M (translation), A, B (linear), C, C2, D, E, L, F, N, G and V by the exact superposition of each interface's Riemann solution.
The Riemann solver is Toro's, with the star state by bisection down to P = 1e-30 and vacuum generation (Toro 4.6); it agrees with LUNA's
exact_riemann to 2e-15 on Toro 1-5, LA slow, LA streams, Einfeldt and on L's states. SO, S and H are scored against LUNA at 2^20.

Initial cells for other codes (AREPO, GIZMO): `python scripts/super_tube_v2.py ic <N> <stem> [m0]` writes `<stem>.npz` and `<stem>.txt`
with x_left, x_right, x, ρ, v, P (primitives of the exact cell averages) and the cell totals of mass, momentum and energy, plus γ, walls,
segments and the smooth parts as metadata. Totals at N = 4096 and 32768: mass 46.6212135, energy 1328.95118.

## Each new stage alone [M]

On its window, from the middle of the buffer on its left (0 for M) to the middle of the one on its right, on the cells of the tube at the
spacing of N = 4096 (M 255, L 256, SO 592, N 306, V 408 cells) and 8 times as many (the spacing of 32768), open walls, to t = 0.1.
Runs: LUNA (60cf841, HLLC, MC, CFL 0.8, floors on, one core), luna-cpp static (LUNA's rule, 0.8) and moving (AREPO's rule, 0.4), OpenMP,
one core. LUNA at 64 times the cells is the check reference. L1 of ρ against the exact solution (SO: against LUNA 64×):

| stage | LUNA 4096 | LUNA 32768 | LUNA 262144 | static 4096 | static 32768 | moving 4096 | moving 32768 |
|---|---|---|---|---|---|---|---|
| M | 6.19e-2 | 2.52e-2 | 1.60e-2 | 6.19e-2 | 2.52e-2 | stops, t = 0.0028 | stops, t = 0.0002 |
| L | 3.53e-2 | 5.38e-3 | 1.16e-3 | 3.53e-2 | 5.38e-3 | 1.55e-2 | 1.96e-3 |
| SO (vs LUNA 64×) | 3.22e-2 | 3.46e-3 | - | 3.22e-2 | 3.46e-3 | 3.93e-2 | 4.44e-3 |
| N | 1.23e-1 | 3.84e-2 | 6.88e-3 | 1.23e-1 | 3.84e-2 | 5.22e-2 | 1.10e-2 |
| V | 7.43e-2 | 1.14e-2 | 1.78e-3 | 7.43e-2 | 1.14e-2 | 4.45e-2 | 5.76e-3 |

- All five are well posed: LUNA converges on each against the exact solution (SO against itself at 64×), with ρ and P positive.
  Static luna-cpp equals LUNA to the digits shown. M converges slowly (first order at the vacuum edge behind the stream).
- M, Mach 1e4: in the bump the static codes hold |P/P₀ - 1| below 3e-7 at every N. The stream's rear edge, where vacuum opens, heats
  to P = 8.4e4 P₀ (4096), 1.2e4 P₀ (32768) and 7.7e3 P₀ (262144). **Moving fails at once**: the end generator reaches the open wall
  in the first step (the gas moves 0.8 h a step) and is clamped there with V = 0. The run stops after 3 steps (255 cells) or 2 steps
  (2040 cells): luna-cpp #14, comment of 2026-11-18.
- L, N, V: moving is 2.3 to 3.5 times better than static at 4096 and 2.0 to 3.5 times at 32768; its smallest cells stay at 0.04 h (L),
  0.05 h (N) and 0.13 h (V).
- SO: moving is 1.2 to 1.3 times worse than static at both N.
- N at 4096: static and LUNA overshoot where the streams meet, to ρ = 7.4 alone and 9.2 in the tube (exact 6.0; Courant number below);
  moving holds 6 with an oscillation in the centre cells.

Figures: docs/figures/super_tube_v2_alone_{M,L,SO,N,V}.png.

## The tube [M]

Runs:
- LUNA and luna-cpp static at N = 4096 and 32768 (CPU, one core each).
- luna-cpp moving: 4096 on one core; 32768 and every run to t = 4 on one A100.
- LUNA at 2^20 on one A100 as the reference.
- luna-cpp static at 2^19 and 2^20 on one A100, to check that reference.
- `super-tube-v2-m0`, the same tube with stage M's stream replaced by the ambient gas. It lets the moving mesh run the other stages (below).

Tables: reports/super_tube_v2_tube.md and reports/super_tube_v2_single.md (`scripts/super_tube_v2.py score`, `single`).

### What breaks

| code | N | what happens |
|---|---|---|
| luna-cpp moving | 4096 (A100) | stops at t = 0.00126 after 32 steps: stage M's end cell collapses at the open wall (#14) |
| luna-cpp moving | 4096 (one core), 32768 | the same collapse, but the run carries on: NaN spreads from the wall cell by 2 cells a step; every cell is NaN at step 16428 (t = 0.0437, 32768); the next step jumps to t_end, and the run reports reached = 1 at t = 0.1 and at t = 4 (#29, filed) |
| LUNA, floors on | 4096, 32768, 2^20 | reaches t = 4: 12953 and 106188 steps (luna-cpp static's counts), 5828783 steps in 3695 s on one A100 for 2^20; ρ and P positive |
| luna-cpp static | 4096 to 2^20 | reaches t = 4 at every N, ρ and P positive (min ρ 0.032, min P 0.15 at t = 4) |

So stage M stops the moving mesh, and no stage stops LUNA or luna-cpp static when floors are on. Stage N needs them; see "LUNA and its
floors".

### t₁ = 0.1: L1(ρ) per stage

Against the exact solution, except SO, S and H against LUNA 2^20. LUNA and luna-cpp static agree to the digits shown, so one column
serves for both. Moving is the m0 tube; at t₁ its stages A and on are those of v2 (static, v2 against m0, equal to rounding there).

| stage | LUNA = static 4096 | LUNA = static 32768 | moving 4096 | moving 32768 |
|---|---|---|---|---|
| M | 6.99e-2 | 3.24e-2 | fails (#14, #29) | fails |
| A | 0 | 0 | 0 | 0 |
| B | 1.84e-6 | 4.90e-8 | 1.90e-6 | 5.08e-8 |
| C | 1.46e-4 | 1.59e-5 | 1.46e-4 | 1.59e-5 |
| C2 | 1.43e-1 | 3.27e-2 | 9.77e-3 | 1.35e-3 |
| D | 1.29e-2 | 2.00e-3 | 1.34e-2 | 1.75e-3 |
| E | 2.52e-2 | 5.47e-3 | 1.37e-2 | 2.42e-3 |
| L | 3.45e-2 | 5.27e-3 | 1.58e-2 | 1.99e-3 |
| F | 5.11e-2 | 1.44e-2 | 4.22e-2 | 6.26e-3 |
| SO | 4.54e-2 | 5.25e-3 | 4.11e-2 | 5.03e-3 |
| N | 1.65e-1 | 4.91e-2 | 5.39e-2 | 1.13e-2 |
| S | 5.10e-1 | 1.35e-1 | 2.05e-1 | 3.87e-2 |
| G | 4.74e-2 | 7.18e-3 | 2.55e-2 | 4.19e-3 |
| V | 9.15e-2 | 1.91e-2 | 4.64e-2 | 6.16e-3 |
| H | 3.12e-1 | 7.39e-2 | 1.48e-1 | 1.47e-2 |

- A stays exactly ambient in every run that reaches t₁.
- At 32768, moving is level with static on B, C and SO (within 4 %) and 1.1 to 24 times better on the rest: C2 24×, H 5.0×, N 4.3×,
  S 3.5×, V 3.1×, L 2.6×, E 2.3×, F 2.3×, G 1.7×, D 1.1×.
- M at t₁, static: in the bump |P/P₀ - 1| ≤ 3e-7 at both N (P₀ = 1e-6). The stream's rear edge heats to 9.7e4 P₀ (4096) and
  1.8e4 P₀ (32768).
- The new stages score worse in the tube than alone: N 1.65e-1 against 1.23e-1 alone (static 4096), on identical cells. The tube's time
  step is set by S and H (423 steps to t₁, against 28 for N alone), so N runs at a Courant number of about 0.05 instead of 0.8, and the
  error in the Noh centre bin is 7.7e-2 against 4.3e-2. Single-stage numbers are therefore not the tube's.
- Budgets: by t₁ the open wall has let out mass 1.18 and energy 82.8, mostly stage M's stream. The m0 tube keeps mass and energy to 1e-13
  on the moving mesh and to 1e-12 static.

### t = 4: L1(ρ) per stage against LUNA 2^20

| stage | LUNA 4096 | LUNA 32768 | static 4096 | static 32768 | moving 4096 (m0) | moving 32768 (m0) | static 2^19 (check) | static 2^20 (check) |
|---|---|---|---|---|---|---|---|---|
| M | 8.47e-5 | 7.39e-5 | 8.47e-5 | 7.39e-5 | - | - | 3.13e-6 | 4.0e-12 |
| A | 6.77e-4 | 1.34e-4 | 6.77e-4 | 1.34e-4 | - | - | 2.69e-6 | 4.6e-12 |
| B | 7.71e-4 | 8.52e-5 | 7.71e-4 | 8.52e-5 | - | - | 3.80e-6 | 9.2e-12 |
| C | 5.89e-3 | 1.20e-3 | 5.89e-3 | 1.20e-3 | - | - | 5.14e-5 | 1.2e-7 |
| C2 | 1.23 | 2.92e-1 | 1.23 | 2.92e-1 | pending | pending | 1.94e-2 | 1.7e-4 |
| D | 1.04 | 2.57e-1 | 1.04 | 2.57e-1 | pending | pending | 2.41e-2 | 9.4e-4 |
| E | 2.49e-1 | 7.22e-2 | 2.49e-1 | 7.22e-2 | pending | pending | 5.81e-3 | 2.0e-4 |
| L | 1.47e-1 | 5.52e-2 | 1.47e-1 | 5.53e-2 | pending | pending | 4.95e-3 | 2.0e-4 |
| F | 9.54e-1 | 4.29e-1 | 9.54e-1 | 4.29e-1 | pending | pending | 3.45e-2 | 3.5e-4 |
| SO | 3.07e-1 | 7.92e-2 | 3.07e-1 | 7.92e-2 | pending | pending | 5.12e-3 | 6.7e-5 |
| N | 3.27e-2 | 1.26e-2 | 3.27e-2 | 1.26e-2 | pending | pending | 1.03e-3 | 1.2e-5 |
| S | 4.37e-1 | 1.31e-1 | 4.37e-1 | 1.31e-1 | pending | pending | 1.30e-2 | 1.8e-4 |
| G | 5.76e-1 | 1.53e-1 | 5.76e-1 | 1.53e-1 | pending | pending | 1.28e-2 | 2.9e-4 |
| V | 7.98e-2 | 2.21e-2 | 7.98e-2 | 2.21e-2 | pending | pending | 1.61e-3 | 1.4e-5 |
| H | 2.20e-1 | 5.42e-2 | 2.20e-1 | 5.42e-2 | pending | pending | 3.21e-3 | 7.4e-6 |

The moving columns are **pending**: jobs 32522444 (4096) and 32522445 (32768). They are scored on C2 to H only: by t = 4 stage M changes
the regions of M, A, B and C (static 32768, max |Δρ| between v2 and m0: 0.74, 0.55, 0.32, 0.08) and nothing after them (≤ 1e-12). LUNA
with its floors equals luna-cpp static to 7e-12 at 4096 and to L1(ρ) 1.5e-3 at 32768 over the tube (max |Δρ| 9e-3).

### Is t = 4 resolved?

Yes, by the brief's criterion.
- Static 2^19 lies within 3.5e-2 of LUNA 2^20 in every stage. That is 8 to 15 times below the 32768 errors in the stages hit hardest
  (C2, D, F, G, S); with first-order convergence at shocks [I], 2^20's own error is about that difference.
- Static 2^20 equals LUNA 2^20 to ≤ 1e-3 per stage (floors and rounding).
- LUNA's mirrored halves differ by at most 2 % of ρ in a cell and 1e-3 in any stage's L1.

So t = 4 stays. LUNA at 2^19 was not rerun with its floors (its run without them went NaN, below); static 2^19 stands in for it.

### LUNA and its floors

LUNA's floors (`preserve_face_states`) are off by default. Our driver did not turn them on for any run before 2026-11-18: not v1's tube,
not the 30-test ladder, not v2's first runs. What they change, measured:
- **v2, N = 4096 and 32768, to t = 4**: without the floors the pressure turns negative at step 1270 (4096) and 10538 (32768), t = 0.32 to
  0.33, in stage N at x = 25.40 and 25.37. That is the shock the ambient gas drives into the cold stream from behind. The run goes NaN, and
  so did LUNA at 2^19 (first layout, step 128760). **With the floors on, LUNA survives stage N** and reaches t = 4 at both N, with the same
  step counts as luna-cpp static.
- **Everything else is bit-identical** with and without the floors: v2 at t = 0.1 at 4096 and 32768, the 2^20 reference at t = 0.1 and
  t = 4, and all 15 single-stage runs.
- **v1**: LUNA's v1 runs (super-tube, 4096 and 32768, to t = 2) are bit-identical with the floors on, so no v1 number changes. v1's 2^20
  run was not repeated.

An earlier draft of this note said stage N breaks LUNA because LUNA has no pressure floor. That was wrong: the floor exists and was off.

### Which stage breaks which code

| stage | LUNA (floors on) | luna-cpp static | luna-cpp moving |
|---|---|---|---|
| M Mach 1e4 out through the open wall | runs; rear edge heats to 1e4-1e5 P₀ | the same | **fails at once** (#14); in the tube the NaN state is reported as reached (#29) |
| L 1e6 / 1e12 jump | runs, positive | runs | runs; 2.6× better than static at t₁ (32768) |
| SO Shu-Osher | runs | runs | runs; level with static (alone 1.3× worse) |
| N Noh streams | runs; without its floors NaN at t ≈ 0.32 (negative P at the cold-stream shock). At t₁, 4096, it overshoots to ρ = 9.2 where the streams meet (exact 6) | runs (floors) | runs; 4.3× better at t₁ (32768) |
| V true vacuum | runs; ρ down to 1.8e-3 at 32768, alone (exact 0) | runs | runs; 3.1× better at t₁ (32768) |

## Open

- Moving to t = 4 (m0) at 4096 and 32768: jobs 32522444 and 32522445 still running; their columns go into the t = 4 table.
- #14: a wall that gas leaves through must remove cells. Until then, the moving mesh cannot run v2 as defined.
- #29: a non-finite state must stop the run.
- An independent reference at 2^20 (AthenaK) for t = 4. AREPO and GIZMO runs of v2 (the next step) from the initial cells in
  luna-reference-solutions `luna-cpp/super_tube_v2/ic/`.

Figures: docs/figures/super_tube_v2_<stage>.png (ρ in the stage's region; t = 0.1 above, t = 4 below; N = 4096 left, 32768 right) and
docs/figures/super_tube_v2_alone_<stage>.png. Data: luna-reference-solutions branch super-tube-v2, `luna/super_tube_v2/`,
`luna-cpp/super_tube_v2/`.
