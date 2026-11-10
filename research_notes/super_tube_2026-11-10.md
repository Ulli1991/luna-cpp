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
