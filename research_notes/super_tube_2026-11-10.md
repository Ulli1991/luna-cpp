# A super-hard 1D tube, draft 1 (2026-11-10)

One test that holds every way the 30 1D tests fail, as a quick check after any hydro change; the 30 stay the full run. Build after #23 lands, since it would fail today on the smooth stage.

Gas with γ = 1.4 on the line [0, 33.2]. Left wall open (outflow), right wall reflecting. Between stages, quiet buffers of the ambient state ρ = 1, v = 0, P = 1, wide enough that stages meet late. Positions and wave timing: `scripts/super_tube_layout.py` (exact Riemann fronts of every interface).

| Stage | Where | States (ρ, v, P) | Tests |
|---|---|---|---|
| A quiet | 0 to 1 | ambient | noise from mesh motion alone (#23) |
| B smooth waves | 3 to 4 | sound wave (amplitude 1e-4, wavelength 0.25) and entropy wave on ambient | order on smooth flow |
| C stationary peak | 6.00 to 6.05 | ρ = 10 at P = 1 | contact sharpness, spurious pressure |
| C2 moving peak | 8.05 to 9.60 | ρ = 10 peak carried at v = 1 in a region moving at v = 1 | moving contact |
| D sonic rarefaction | 11.60 to 12.60 | (1, 0.75, 1) next to (0.125, 0, 0.1), Toro 1 | entropy glitch at the sonic point |
| E extreme jump | 14.60 to 15.60 | (1, 0, 0.1) next to (1e-3, 0, 1e-10), ratio 1e9 | shock position, heating at the contact |
| F slow shock | 17.60 to 18.60 | (3.86, -0.81, 10.33) next to (1, -3.44, 1), LA slow | ripples stuck behind a slow shock |
| S colliding streams | 22.60 to 23.20 | (5.99924, 19.5975, 460.894) next to (5.99242, -6.19633, 46.0950), Toro 4 | strongest shocks from velocity |
| G near vacuum | 27.20 to 28.20 | (1, -2, 0.4) next to (1, 2, 0.4) | positivity, density down to about 0.02 |
| H double blast | 32.20 to 33.20, against the right wall | P = 1000, 0.01, 100, Woodward-Colella | collisions and reflections |

Timing (exact fronts): stages A to G stay single Riemann problems until t = 0.129 (F's edge waves meet its centre); the blast reaches G at t = 0.158. S and H interact within themselves almost at once, by design.

Checkpoints:
1. t = 0.1: A to G against the exact superposition of their Riemann solutions, each stage scored on its own; S and H against a high-resolution reference.
2. t = 2 (end): everything against LUNA and AthenaK at very high resolution, stored as `.npz` in luna-reference-solutions.

Pass: each stage's L1 falls with N (about 2 on B, about 1 elsewhere); A stays uniform to rounding until a wave arrives; density and pressure stay positive; totals match the budget through the open wall.

Open: the Galilean twin (with walls a bulk velocity changes the problem, so a periodic copy of A to G only); the name.

## luna-cpp #25: points collapsing at stage E (fixed 2026-11-10, branch fix-25) [M]

Reduced problem: `super-tube-e`, stage E alone on [13.60, 16.60] with ambient buffers and outflow walls, on the cells of the tube (370 cells for the spacing of N = 4096, 2960 for 32768); it stalls at the same place (x = 15.19) as the tube, in seconds on one core. Per-step trace (driver patched locally, not committed): positions, gaps, mass, primitives, point velocity w, w - v, c / r around the smallest cell.

Mechanism (370 cells, t = 0.128): the 1e-3 gas is swept into a layer about 1.4 h thick between stage E's rarefaction (moving right) and the expansion of the ambient gas from x = 15.6 (moving left); the layer's cells are small and light by physics (m about 1e-3 h). The rarefaction's leading cell (2.3 h) had its generator at the layer (c / r = -1.000) and moved with its mean velocity +0.39, gas 1.15 h behind it; the drift, capped at 0.5 c = 0.30, brought it to +0.08, while the layer moved at -0.28. Each layer cell's drift saturated as its left gap closed (c / r 0.96), so its faces ran ahead of its gas and its mass left (m from 1e-3 h to 2e-4 h and below): nothing pushed back. Smallest cell 3.5e-7 h, dt 3e-10. Tube at N = 4096 on one core: the same cell at 15.1901 (2.2e-5 h at t = 0.1356 after 27209 steps).

Checks: without regularisation the reduced run goes through (smallest 2.7e-3 h), so the drift drives it, not the time step (AREPO's pair term is used: courant (d + r) / vsig, vsig from the fluid velocities; the collapse takes 1e5 steps, no crossing in one). With the point at the gas velocity at its generator from the limited gradient, N = 4096 went through (2.46e-3 h) but the spacing of 32768 stalled at t = 0.118 (the layer, at -0.78, ran into the neighbour's generator held at -0.48; the layer's first cell was a velocity minimum, where the MC limiter sets the gradient to zero). Fix: the velocity at the generator interpolated linearly between the centroids of the cell and its neighbour on the generator's side (monotone, no limiter); stage E alone: smallest cell 8.4e-3 h (370 cells, to t = 0.2), 1.85e-2 h (2960 cells, to t = 0.125, 9781 steps).

AREPO next generation takes the cell's mean (Arepo2-nextgeneration, src/voronoi/voronoi_vertex_velocities.cc:28, `p->VelVertex = p->Vel`) plus the drift towards the centroid (:49-132: fraction from :87; speed c with _USE_SOUNDSPEED :106, else min(d / dt, max(c, |v|)) :124-129; applied :132); its rule's |v - w| (src/time_integration/timestep.cc:133) and pair term are this code's. So the fix departs from AREPO-ng: there cells stay near centroidal, where the two agree.
