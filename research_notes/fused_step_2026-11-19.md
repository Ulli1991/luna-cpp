# A fused moving step within the two-pass scheme: design and prototype, 2026-11-19

Goal (the owner): the whole moving step (voronoi-cpp mesh update plus luna-cpp hydro), averaged over a run, at
0.2-0.3 G cell updates/s in double on one A100. At 128³ that is 10.5 ms a step (0.2 G) to 7.0 ms (0.3 G). The scheme
stays as it is: MUSCL-Hancock with two flux passes, on the mesh of t^n and on the mesh of t^n+1, same accuracy. Static
meshes are ignored. Precision levers are another branch's (`precision-levers`).

The prototype was built on branch `fused-step` from luna-cpp main `68d1e65` and measured after merging main (see section 6);
the owner removed it, and this note came to main on branch `fused-docs`. It supersedes the design note `fused-passes`
(d029454) for the update and the sweep.

Numbers without a job id are from the profile of 2026-11-18 (`research_notes/profile_2026-11-18.md` on branch
`profile-2026-11-18`, Nsight Compute step 10, 128³, caches flushed per kernel) and from the precision levers' tip
(`docs/measurements.md` on branch `precision-levers`, job 32545389, fenced phases, 128³: step 23.28 ms = sweep phase 5.51,
fluxes 5.03, update 4.23, mesh + cells 8.37, rest 0.14). One A100 at 1305 MHz. At 128³ 1 ms of a step is 0.477 ns/cell,
and 1 B/cell of DRAM traffic takes 1.53 µs at 1.37 TB/s.

**Answer first.**
- **The design (sections 1-5).** Only two chains of gathered passes can share a kernel: each pass's fluxes with its
  update, and pass 2 with the next sweep. dt and the mesh update are global barriers between the rest. Without halos
  or recomputation they can be ordered by face blocks, buckets and flags.
- **The prototype (section 6).** The most valuable of them, the flux pass and its update in one kernel
  (`--update=fused`, since removed), gave the same bits as today's path everywhere and cut those passes' DRAM traffic by 36 %. But it
  was **14 % slower** on the whole step at 128³ and 256³ (whole run 0.0845 → 0.0759 G at 128³). The update phase
  inherited the flux code's occupancy (27 % of warps) and was no faster from L2 there than alone from DRAM; waits and the
  bucket build come on top.
- **0.2 G (section 7)** is not reachable by fusion in this code. It needs the mesh at its speed of light (about 4.3 ms,
  1.9× beyond step-levers, measured here at 8.1 ms in Sedov) and the hydro at about 6.2 ms (2.4× beyond today). The
  hydro side is register and latency work on the flux and sweep code, not fewer passes.

## 1. Every pass over gathered data in a moving step today

"Gathered" means rows of other cells (or faces) read through an index. DRAM bytes per cell are read + write from
Nsight Compute; for the two passes of `fluxes` and `update` the per-launch values of the same profile
(`ncu_luna_moving_128-32521642.csv`). Rows of the main branch; the precision levers shrink some rows (in brackets).

| # | kernel (loop) | ms | DRAM B/cell r + w | gathers, per item | streams | writes |
|---|---|---|---|---|---|---|
| 1 | `hydro::sweep` (cell, its list twice: fit, limiter) | 4.96 | 667 + 87 | per list entry: the other cell's primitive row (56 B [48]), position (32), centroid (24); the face's \|A\| (8, kept by pass 2), face centroid (24 [12]) | list (68), own rows | gradient row (64 [32]), w (24) |
| 2 | `hydro::time_step` (face terms, then cell terms, one reduction) | 0.77 | 286 + 19 | per face: both sides' primitive rows (v, c, r) and positions | owner/neighbour | dt to the host |
| 3 | `keep_centroids`, `drift`, mesh quantise (2) | 0.71 | 221 + 157 | none | x, w, centroids | x, w, old centroid |
| 4 | `hydro::fluxes`, pass 1, mesh of t^n (face) | 2.85 | 822 + 338 | both sides' primitive rows, gradient rows, w, positions, centroids | owner/neighbour (8), area (24), face centroid (24 [12]) | flux row (40 a face: 318 B/cell) |
| 5 | `hydro::update`, pass 1 (cell, its list) | 2.1 | 1109 + 47 | per list entry: the face's flux row, 40 B in two 32-B sectors, in face order | list, Q | Q |
| 6 | mesh `screen` (tetrahedra) | 1.93 | 271 + 253 | four corners' positions, the opposite vertices | tetrahedra | circumcentres, flags |
| 6' | mesh repair (about 80 flips in Sedov; 30 launches) | 1.22 | 7 | — | — | — |
| 6'' | mesh `faces` (face: its ring) | 2.21 | 579 + 394 | circumcentre per ring entry | rings | area, face centroid, \|A·d\| |
| 6''' | mesh `cell_sums` (cell, its list), lists, scans | 2.12 + 1.11 | 625 + 104 | per list entry: face rows, the other position | lists | volume, centroid |
| 7 | `hydro::fluxes`, pass 2, mesh of t^n+1 (face) | 3.31 | 893 + 473 | as pass 1, plus old centroids | as pass 1 | flux row; \|A\| per face for the next sweep (64 B/cell) |
| 8 | `hydro::update`, pass 2 (cell, its list) | 2.3 | 1109 + 141 | as pass 1 | list, Q, volume | Q, the next primitive rows |

Six of these are the hydro's (1, 2, 4, 5, 7, 8) and three the mesh's (6, 6'', 6'''). Each reads what the one before it
wrote for the whole mesh (500 MB to 2 GB at 128³), so every pass meets its inputs in DRAM, not in the 40 MB of L2.

**What the passes have to wait for.** A fusion can only cross a dependency that is local, i.e. a cell or face waiting for
its neighbours. Three dependencies of the step are global:

1. **dt** (a minimum over all cells) between the sweep and pass 1: pass 1 uses the quantised w (whole grid steps over dt)
   and dt / 2.
2. **The mesh update** between the passes (voronoi-cpp's whole-mesh kernels).
3. **dt of the next step**, after the next sweep.

Everything else is local: the update of a cell waits for its faces' fluxes; a face's flux waits for both cells'
gradients and w; the sweep of a cell waits for its neighbours' new primitives.

So the gathered chains that can run as one kernel are:
- pass 1: fluxes → update;
- pass 2: fluxes → update → **the next step's sweep** (and its time-step terms).

The sweep cannot be fused *forwards* into pass 1 (dt lies between them). It can be fused *backwards* into pass 2 of the
step before, where its inputs (the new primitives, the mesh of t^n+1) are produced. Then a step is two hydro kernels and
the mesh update.

## 2. How a fused kernel orders its work: face blocks, buckets, flags

The measured history rules out every fusion that groups the face evaluations by cell:
- per-cell fluxes and update (each face twice): 2.4-3.6× slower;
- owner loops (each face once, by its owner's thread or lanes): flux passes 2-3× slower;
- atomics: run-to-run bits (rejected by the owner).

The flux evaluation is fast only as one thread per face over faces in face order.

What made the update a separate kernel was the global wait: "each neighbour kernel needs all faces' fluxes first"
(measurements.md, "Face reads once per step"). That wait is only needed per cell, and it can be ordered inside one kernel.

**The scheme** (the same idea as CUB's decoupled look-back and stream-K):

- **Face blocks.** The faces are cut into blocks of FB consecutive face numbers (FB = 1024). Block b computes faces
  [b·FB, (b+1)·FB) one thread per face, exactly as `hydro::fluxes` does now, and stores the flux rows.
- **Buckets.** Each cell is assigned to the block holding the highest-numbered of its faces, mb(i). Bucket b is the list
  of cells with mb(i) = b. Every face of such a cell lies in a block ≤ b. The buckets are built once per mesh, i.e.
  once a step, after `update_cells()`; pass 1 of the next step reuses them.
- **Tickets and flags.**
  - A team takes its block number b from a global ticket counter, so blocks start in increasing order.
  - After its flux phase it publishes `flag[b] = epoch` (fence, barrier, release).
  - It then waits for the flags of the blocks in [low(b), b), where low(b) is the lowest face block of any cell in
    bucket b. It reads its cells' flux rows through L2 (`ld.global.cg`, so no stale L1 line).
  - It updates bucket b's cells in list order, exactly as `hydro::update`.
- **No deadlock.** A team waits only for lower tickets, which have already started and wait only for lower ones still.
- **Faces between blocks.** No halo, no overlap and no recomputation: every face is computed once, by its block; every
  cell is updated once, by its bucket's team, after every block that holds one of its faces. A face between two cells
  of different buckets is read by two teams, each after the flag of the face's block.
- **Determinism.** Every value is computed by one thread from the same inputs as now, and the sums run in list order.
  The order in which blocks run changes only who waits and when.
  - Integer atomics are used for scheduling only (the ticket, the flags) and for the fallback count. They are never
    used for data.
  - **Expected bits: the same as `--update=face`**, unless the compiler contracts the inlined flux differently in the
    new kernel (checked by the Q hash).
- **Why it should be faster.** Today's update reads each 40-B row twice from DRAM, two sectors each time: 1109 B/cell
  against a need of about 500, at 83 % of peak. In the fused kernel a cell is updated moments after its faces'
  blocks: its own block's rows were written by the same team, and most of its other rows by the teams just before it.
  Those reads come from L2, whose bandwidth is about 3× DRAM's.
  - The writes of the rows remain: 318 B/cell, written back when L2 evicts them.
  - The gain therefore depends on how close a cell's faces are in face number, i.e. on the spread mb(i) − f/FB. The
    prototype measures that distribution.

## 3. The candidates, costed

128³ ms are against the precision-levers tip (23.28 ms). 256³ ms are scaled by the measured ns/cell, which is the same
at both sizes to 1-2 % (profile note): 1 ms at 128³ ≈ 8.1 ms at 256³. "Regs" is the fused kernel's register count
against today's kernels (fluxes 96, sweep 118, update 44).

| candidate | bytes saved per cell | expected ms saved at 128³ / 256³ | registers, occupancy risk | faces between blocks | determinism, bits | memory B/cell |
|---|---|---|---|---|---|---|
| **F1. Flux + update in one kernel per pass** (face blocks, buckets, flags) | the update's row reads from DRAM: up to about 640 of 1109 per pass (the rest are list, Q, prims); the rows' writes stay | 1.5-2.7 / 12-22 (update 4.23 → about 1.5-2.7 for both passes; bucket build +0.1-0.2) | the flux's 96 dominate; the update phase runs at the flux's occupancy (25-31 %), enough for L2-latency loads; low risk | none: face blocks, flags | same bits as `--update=face` expected; deterministic | +4 (bucket cell lists) + 12 B a block |
| **F2. Pass 2 + the next sweep + the time-step terms in one kernel** (sweep buckets: a cell after all its neighbours' updates) | the sweep's gathers of new primitive rows and the mesh rows pass 2 just read, from L2: about 300-450 of 754; the time step's own 305 | 1.4-2.2 / 11-18 (sweep 4.9 → about 3.2-4.0; `time_step` 0.77 → about 0.1) | **high**: 118 registers for the whole kernel. Pass 2's flux phase at 118 instead of 96 keeps 16 warps per SM at 256-thread teams, but loses 4 at 128-thread teams; a sweep above 128 halves its warps (measured: deeper pipelining 122-135 registers, slower) | as F1, plus a second flag per block (updates done) | sweep the same code; dt the same minimum of the same terms (a face's term from each side, d and vsig exact negations), so the same bits expected | +4 (sweep buckets) |
| F3. `keep_centroids` and `drift` in pass 1's update phase (the flux phase quantises w per side on the fly, same arithmetic) | 378 streamed | 0.3 / 2.4 | small | — | same bits | 0 |
| Gradient sweep + flux pass 1 (as the brief lists it) | — | not possible: dt, a global minimum, lies between them (quantised w and dt / 2). F2 fuses the sweep backwards instead | — | — | — | — |
| Fluxes summed per cell, no stored rows, no atomics | 318 written + 1109 read | measured: slower. Per cell, each face twice: 2.7× (moving 10.38 → 28.3 ms); owner loops 2-3× | — | recomputation of every face | deterministic | −318 |
| Flux rows kept in shared memory only (no rows in DRAM) | +318 writes per pass | a team that holds 128-256 cells sees 30-60 % of its faces cross to other teams: they must be recomputed (+30-60 % flux work) or stored anyway. Smem caps a team at about 4000 rows per SM | 1-2 teams per SM at the flux's registers: sync bubbles | redundancy or stored crossing rows | deterministic | −318 only with a compact array for crossing rows (−about 200) |
| Face geometry computed where it is read (R14) | −382 stored, but ring + circumcentres read instead | measured: flux pass 2.78 → 4.56 ms, sweep 4× slower, R14 buys memory not time | — | — | same bits | −394 |
| Shared-memory or warp staging of a block's primitives and gradients | L1 already serves 78-89 % of the flux and sweep gathers | small; the measured team-scratch sweep was slower (sums in shared memory, 4.78 → 5.77 ms) | team launch, barriers | halo staging | same | 0 |
| L2 chunking by launches (the `fused-passes` note, D) | as F1 | as F1 minus 2 launches per chunk (about 0.6 ms at 64 chunks) and the tails of small launches | — | dependencies far apart in Morton order break a fixed lag | same | 0 |

F1 is the most valuable first prototype:
- the largest expected saving at the lowest register risk;
- the same bits as the default path;
- the scheduling it builds (face blocks, buckets, flags) is what F2 needs.

F2 is the second lever. F3 is small: the drift alone was measured folded into pass 1's update before, with no change.

## 4. ms budget at 128³

Three columns:
- today (the precision levers' tip, fenced);
- the fused design at the estimates above;
- what 0.2 G (10.5 ms) needs.

The mesh at its step-levers speed: voronoi-cpp `step-levers` 242d218 is 1.17-1.21× faster than step-2x (1741a1a) on
500-step benches. In Sedov the repair is small, and `screen` (−18 to −21 %), `small_rounds` (the repair's 30 launches
into one) and `scan_starts` act. Estimate: 8.37 → about 7.0-7.5 ms (measured in a Sedov run below).

| phase | today | fused (F1 + F2 + F3), est. | needed for 10.5 ms | what the "needed" column assumes |
|---|---|---|---|---|
| mesh: screen, repair, faces, cell sums, lists | 8.37 | 7.0-7.5 (step-levers) | 4.3 | every mesh kernel at about 80 % of DRAM peak on its least bytes (profile SoL 3.6 ms in Sedov), the repair and scans at no overhead |
| pass 1: fluxes + update (+ keep_centroids, drift) | about 2.8 + 2.1 + 0.4 = 5.3 | about 3.6 | 2.3 | the flux arithmetic at about 45 % of FP64 peak (today 23-28 %), the update phase free under it |
| pass 2: fluxes + update + next prims | about 2.3 + 2.1 = 4.4 | about 3.0 | 2.0 | as pass 1 |
| sweep + time step | 5.51 | about 3.4 | 1.8 | the sweep at about 2× its present occupancy-limited speed |
| rest (primitives, read-backs) | 0.14 | 0.1 | 0.1 | |
| **step** | **23.3 (0.090 G)** | **17.1-17.6 (0.119-0.123 G)** | **10.5 (0.200 G)** | |

**Is 0.2 G reachable?** Not with these fusions and the mesh at its step-levers speed. They are worth about 5-6 ms of
the hydro's 15, which gives about 0.12 G.

0.2 G needs all three of:
- **the mesh at about 4.3 ms**, about 1.7× faster than step-levers: every mesh kernel at its bytes, about
  2.6 ns/cell of the profile's mesh levers;
- **the two flux passes at about 4.3 ms of arithmetic-bound work**, i.e. 6031 flop/cell at about 45 % of FP64 peak.
  The best flux kernel here reaches 23-28 %; its warps wait on gathers (long scoreboard 70 %) at 96 registers;
- **the sweep at about 1.5-1.8 ms.** The static tables showed 3.5 ms is possible when the sweep gathers one row per
  entry (72 registers). On a moving mesh they would have to be rebuilt each step.

The hydro at 6.2 ms and the mesh at 4.3 ms are each about at their speed of light. 0.2 G is a speed-of-light target for
the step, not a structural one: no restructuring within the two-pass scheme removes enough work to leave slack.

## 5. Requests to voronoi-cpp (not made; for the owner)

1. **Face numbers in cell order.** F1's L2 hits depend on a cell's faces having nearby numbers. Faces are numbered in
   tetrahedron-slot order and kept from step to step. A re-sort that numbers faces by their lower cell (in the cells'
   curve order) would maximise the hits; the prototype measures the spread first.
2. **Bucket lists from `update_cells()`.** The per-cell lists are built there. The highest face number of each cell's
   list (mb) and the cells grouped by it could be written in the same kernels instead of a hydro pass (about
   0.1-0.2 ms a step).
3. **`faces` → `cell_sums` with the same flag scheme** inside the mesh: cell sums read each face's rows right after
   `faces` wrote them, from L2 (the mesh's own F1).

## 6. Prototype

### F1: the flux pass and its update in one kernel (removed; the code is on branch `fused-step`, 9825471)

**Code, as it was.**
- `FusedPass<Flux>` in `src/hydro/finite_volume.hpp`, selected by `--update=fused` (`Update::fused`): one Kokkos team
  per face block, teams of 128 threads and blocks of 2^10 = 1024 faces, so a bucket held about 130 cells, about one per
  thread.
- The flux phase was `hydro::fluxes`'s body; the update phase was `hydro::update`'s body, with each row value read by
  `ld.global.cg`.
- `build_buckets()` ran once per mesh (three small kernels: counts with the lowest face block per bucket, a scan, a
  fill). The scheduling state was a ticket counter, one flag per block, and an epoch per launch, so the flags were never
  cleared.
- Memory: +4 B/cell (the bucket lists) and 16 B a block.
- `LUNA_FUSED_STATS=1` printed how far below its cell's bucket each list entry's face block lay, and the bucket sizes.
- The scripts `scripts/fused_step_a100.sh` (modes `bits`, `speed`, `tend`, `mesh`) and `scripts/fused_ncu_a100.sh` are on
  `fused-step` too.

**Measured** on the merged tree: main aac2bb1 (main 5b3f706, merged after, changes only the 2D default precision and docs, so the 3D numbers hold for it), voronoi-cpp 1741a1a, the precision levers' defaults (`grad16`, float face
centroids, int32 steps). One A100 at 1305 MHz. Logs are in `/raven/u/uli/voronoi-cpp-joblogs/fused-step/`.

| | `--update=face` | `--update=fused` | job |
|---|---|---|---|
| 128³, 40 steps, fenced medians: fluxes + update (both passes) | 5.05 + 4.25 = 9.30 ms | 12.47 ms | 32580042 (3 rounds, within 0.03 ms) |
| 128³ step | 23.12 ms (0.0907 G) | 26.33 ms (0.0796 G), +13.9 % | same |
| 256³ fluxes + update | 39.6 + 32.4 = 72.1 ms | 96.5 ms | same (2 rounds) |
| 256³ step | 169.5 ms (0.0990 G) | 193.8 ms (0.0866 G), +14.3 % | same |
| whole run to 1e4 yr, 128³: GPU s, G cell updates/s | 8.66 s, 0.0845 | 9.64 s, 0.0759 | 32580044 |
| whole run, 256³ | 168.3 s, 0.0916 | 187.0 s, 0.0825 | same |
| peak B/cell (steps) | 2251.8 | 2255.9 | 32580042 |
| bucket build per step (128³) | — | 0.75 ms | 32580042 |

**Bits.** Identical everywhere:
- Sedov 128³ Q, positions and volumes after 40 and 50 steps, moving and static, and at 256³ after 40 steps.
- Whole runs to 1e4 yr at 128³ and 256³: the same Q hash, 349 / 919 steps, 175,033 / 1,591,036 fallbacks, and field L1
  0.06481, 0.02102, 0.01702 / 0.03774, 0.01075, 0.00990.
- The 3D sound wave at 1e-3 and 1e-6 (L1/amp 4.3193e-3, 3.6839e-3) and the vortex 64 (L1 4.425e-4, 1.246e-3, 6.196e-4):
  the same npy files.
- The hydro unit test on the A100 (ctest 5/5, job 32580047) and on the host (job 32582313): the fused and the per-face
  loop agree bit for bit in all four precisions, and two fused runs give the same bits.

So the loop is deterministic and exactly the scheme's arithmetic: the same values, the same order of sums.

**Nsight Compute** (job 32580046, Sedov 128³ step about 10, caches flushed per kernel; DRAM B/cell read + write):

| | pass 1 | pass 2 |
|---|---|---|
| face: fluxes + update | 2.21 + 2.05 ms; (649 + 315) + (1078 + 48) = 2090 B/cell | 2.55 + 2.18 ms; 2274 B/cell |
| fused | 5.28 ms; 977 + 356 = 1333 B/cell | 6.15 ms; 1063 + 485 = 1548 B/cell |

- Registers: fluxes 88/92, update 44, fused 90/94. Warps active: fluxes 24 %, update 60 %, fused 27 %.
- L2 traffic: 10.3 GB (fluxes + update) against 16.2 GB (fused), pass 1.

**Where the time goes** (fused at 128³, ms for pass 1 / pass 2; each part switched off in turn by a measurement-only
switch, since removed; jobs 32579591, 32579022):

| fused kernel with | pass 1 | pass 2 |
|---|---|---|
| the flux phase only | 3.08 | 3.26 |
| flux phase + waits, no update | 3.14 | 3.33 |
| flux + update phases, no waits | 5.00 | 5.57 |
| everything | 6.37 | 7.24 |
| `--update=face`: fluxes, update | 2.85, 2.1 | 3.3, 2.2 |

1. **The flux phase** costs what `hydro::fluxes` costs (3.1 against 2.9-3.3 ms).
2. **The update phase** costs 1.9-2.3 ms, the same as the update kernel. Its DRAM bytes halve (above), but it runs at the
   flux code's occupancy (27 % of warps instead of 60 %), and there it is latency-bound on L2. The DRAM bandwidth it no
   longer needs was not what limited it at that occupancy.
3. **The waits** cost 1.4-1.7 ms. 42 % of list entries lie in lower face blocks than their cell's bucket (6 % at 128 or
   more blocks below), so 54 % of the buckets wait cell by cell, and every acquire load costs the SM its L1 lines.
4. **The bucket build** costs 0.75 ms a step.

Even with 3 and 4 at zero, F1 would only match today's 9.3 ms.

**What the prototype taught on the way** (rows in docs/tried.md):
- Kokkos's desul atomics on the 64-bit ticket made a do-nothing team cost 8.3 ms a launch (one contended counter).
  libcu++'s `cuda::atomic_ref` removed that.
- desul's `atomic_load`, a compare-and-swap per spin, together with 64-bit `atomic_inc` on a few counters in the
  bucket statistics, made 128³ appear to hang. Spins are now acquire loads with `__nanosleep`; the statistics are
  reductions.

**Verdict on F1.** Deterministic, the same bits, 4 B/cell, and 14 % slower on the whole step at both sizes. Removed by
the owner (2026-11-19); branch `fused-step` keeps it as history.

### F2 and the rest: not built

F2 (pass 2 with the next sweep) rests on the premise F1 has just measured false: that a gathered phase gains when its
inputs sit in L2.
- The sweep already runs at 23 % of warps with 80 % L2 hits.
- Fusing it into pass 2 would add F1's waits and a second bucket build per step.
- It would run pass 2's flux phase at the sweep's 118 registers.

F3 (`keep_centroids` and `drift` folded) is worth at most 0.3 ms, and the drift alone was measured with no change
before. Neither is worth a commit.

## 7. Conclusions, with the budget measured

**The mesh at step-levers speed** (job 32580044):
- Sedov 128³ to 1e4 yr, the same luna-cpp (`pin-1741a1a` 0eaa641), medians of fenced phases.
- voronoi-cpp 1741a1a: mesh + cells 6.84 + 2.43 = 9.27 ms. Step-levers 242d218: 5.85 + 2.25 = 8.10 ms (−12.6 %). The
  hydro and the Q hash are unchanged.
- In the 40-step medians of main (the order still fresh), mesh + cells is 8.28 ms with 1741a1a; step-levers would make
  it about 7.2.

**Budget at 128³, measured where possible.**
- Today, main aac2bb1: 23.12 ms (0.091 G over 40 steps, 0.0845 over a whole run): sweep phase 5.40, fluxes 5.05,
  update 4.25, mesh + cells 8.28, rest 0.13.
- With step-levers: about 22 ms.
- 0.2 G needs 10.5 ms:
  - **mesh about 4.3 ms**: 1.7-1.9× beyond step-levers, at its speed of light, which is voronoi-cpp's work;
  - **hydro about 6.2 ms**: 2.4× beyond today's 14.8.

**Is 0.2 G reachable?** Not by fusion within the two-pass scheme in this code.
- The one fusion with the largest saving on paper (F1) removes 36 % of its passes' DRAM bytes and is still 14 % slower.
  These gathered kernels are bound by the occupancy and latency of the flux code (88-94 registers, a quarter of the
  warps), not by DRAM bytes.
- A phase fused into a kernel inherits that occupancy.
- What the hydro would need for 6.2 ms:
  - the flux evaluation at about 45 % of FP64 peak, against 23-28 % today: more warps or more independent work per
    warp at the same registers;
  - a sweep at about 1.5-1.8 ms;
  - an update at its bytes.
  
  None of these is a fusion. They are register and latency work on the flux and sweep code. Earlier attempts at it
  (launch bounds, deeper pipelining, shared-memory sums) were measured slower; see docs/tried.md.

**Requests to voronoi-cpp** (section 5) are withdrawn as far as F1 needs them:
- face numbering by cell would cut F1's waits, not its update phase;
- buckets from `update_cells()` would save 0.75 ms of a loop that is 3.2 ms behind.

The mesh's own speed of light (about 4.3 ms of its 8.1) remains the larger lever, and that is voronoi-cpp's.
