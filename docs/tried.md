# What we tried for the hydro, what it did, and why we kept or dropped it

A running list of every hydro change measured for speed, memory or accuracy, from the profile of 2026-11-18
(`research_notes/profile_2026-11-18.md` on branch `profile-2026-11-18`) on. One A100 40 GB at 1305 MHz, Sedov 3D
jittered, HLLC, the default precision (`grad32`), 40 steps, 128³ (2,097,152 cells) unless a row says otherwise. The
effect is measured against the code just before the change, in the same job; the reference is luna-cpp main `73f763a`
(voronoi-cpp `abd3d39`): static 0.2134, moving 0.0807 G cell updates/s at 128³. Job ids and full tables are in
[measurements.md](measurements.md), "Hydro levers of 2026-11-18". Earlier hydro levers are in [ideas.md](ideas.md) and
measurements.md, "Hydro speed levers".

Each new measurement adds its row here, in the same commit as its result. The reason gives the structural cause.

"Sweep" below is the sweep phase: the gradient sweep plus the time-step reductions.

| Change | Effect | Bits | Kept? | Reason |
|---|---|---|---|---|
| Sweep: launch bounds capping registers at 80, 64 and 96 (`LaunchBounds<256,3>`, `<128,8>`, `<128,5>`) | sweep phase static 5.02 -> 6.65, 13.0 and 5.42 ms; moving 5.75 -> 7.5, 14.7 and 6.04 ms | same | dropped | the caps spill the list loop's sums to local memory (48-184 B of stack a thread) and the loop waits on its gathers, not on occupancy alone; as measured on 2026-11-13 |
| Per-cell fluxes and update in one kernel (`--update=cell`, no stored fluxes) | fluxes and update static 4.50 -> 14.9 ms (3.3 times), moving 10.7 -> 38.1 ms (3.6 times) | same as the per-face loop (Q hash equal after 40 steps) | dropped | every face is computed twice, once from each side, and the lists of a warp's cells differ in length; as measured on 2026-11-13 (5.9 times then) |
| Sweep: the list walk one entry ahead (the next entry's face and neighbour load while this entry's rows are summed) | sweep phase static 5.02 -> 4.84 ms (-3.6 %), moving 5.75 -> 5.53 ms (-3.9 %) | same | kept | each entry waited on three dependent loads (list, face sides, rows); now on one. The sweep stays latency-bound: 114 registers, a quarter of the warps |
| Sweep: the neighbours' range (lo, hi) in a walk of its own after the fit, 10 sums fewer in the fit loop | sweep phase static 4.84 -> 5.03 ms, moving 5.53 -> 5.86 ms (with the walk ahead) | same (min and max of the same values) | dropped | the third walk gathers every neighbour's row again, and the register count did not fall below the next occupancy step |
| HLLC: the upwind state copied entry by entry instead of taken by a reference to one of the two states; 1 / (gamma - 1) stored on the gas once instead of divided per side; HLLC's side without the sound speed (c^2, a division and a square root it never reads) and with the normal velocity it has already | entry-by-entry copy alone: fluxes static 2.59 -> 2.50 ms (-3.4 %), moving 6.46 -> 6.12 ms (-5.2 %); all three: in measurements.md, "Hydro levers of 2026-11-18" | the copy moves the last bits (multiply-add contraction; quantified in measurements.md); the stored 1 / (gamma - 1) and the lean side keep them | kept (the owner's choice for the spills) | the reference to one of two arrays put both states in local memory: 80 B of stack a thread, the flux kernels' 1250 (static) and 2531 (moving) B/cell of local traffic in the profile |
| `--update=atomic`: each face adds its flux to both cells as it is made, no stored flux rows and no update kernel | static: fluxes and update 4.50 -> 5.59 ms (+24 %), 2366 -> 2106 B/cell; moving: 10.70 -> 10.77 ms, 2301 -> 2163 B/cell | differ, and differ run to run | an option, off by default (the owner decides) | it saves the 318 B/cell of flux rows and the update's re-read of them, but on a static mesh numbered by owner a warp's faces mostly share their owner, so its atomics queue on the same cell; the order of the additions is not fixed |
