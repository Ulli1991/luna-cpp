# Flux-formulation levers and HLL against HLLC in Sedov, 2026-11-21

Branch `flux-latency` (merged with main a1c1f3f). One A100 at 1305 MHz. Builds and outputs are in
`/ptmp/uli/luna-cpp-flux/`, logs in `/raven/u/uli/voronoi-cpp-joblogs/fused-step/`.

The levers are compile-time variants (`-DLUNA_FLUX_LEVER`, 1f7c44a); l0 is main's code. HLL is the existing
`--riemann=hll`. The rows with every number are in docs/tried.md.

## Verdict per lever

Against l0 in the same jobs: fluxes and step, fenced medians at 128³ and 256³; whole runs to 1e4 yr.

| lever | fluxes 128³ / 256³ | time to 1e4 yr 128³ / 256³ | accuracy against main | verdict |
|---|---|---|---|---|
| 1: float wave speeds | −3.2 % / +2.3 % | −0.4 % / +1.0 % | the 1D ladder fails: advected entropy and sound waves at order 0, budgets 3e-9, symmetry broken | dropped |
| 2: float estimate plus one Newton step | +4.7 % / +4.8 % | +1.0 % / +0.8 % | equal on every gate | dropped |
| 3: cell sound speeds for the wave bounds | −2.8 % / +3.6 % | +0.9 % / +3.9 % (1-3 % more steps) | Sedov L1 −2 %, the rest equal | dropped |
| HLL (an existing option) | −1.5 % / −1.3 % | −35 % / −49 % (226 / 473 steps instead of 349 / 919) | smooth flow worse (sound +4 %, vortex 1.6-2.7×, KH M −1.4 %); Sedov: see below | stays an option; not recommended for Sedov |

None makes the step faster at main's accuracy at both sizes, so the current speed stands.

## Why HLL takes half the steps in Sedov, and what its lower L1 hides

**The comparison was apples to apples:**
- the same binary (build-l0) and flags, only `--riemann`;
- the same printed set-up (ρ, P, E, t_end, 8 deposit cells, p_inject 7.14666e6);
- the same time-step rule and Courant factor;
- the same mesh motion and regularisation;
- step 1's dt identical (1.00445e-9), set by the same blast cell.

**The time step does not depend on the solver.**
- `CellStep` takes radius / (c + |v − w|) per cell; `FaceStep` takes AREPO's pair term from the two cells' primitive
  rows. No solver output enters either.
- The step differs because the states differ (`LUNA_DT_TRACE`, job 32813939):

| | HLLC | HLL |
|---|---|---|
| step 2, the blast cell | ρ 0.86 of its start, c 6.7e4 km/s | ρ unchanged, P lower, c 5.8e4 km/s |
| limiting cell from step 16 (HLLC) and step 8 (HLL) on | the hot centre (r 0.07-0.18 R_s, ρ about 2e-3 ρ0, c 8000-14,000 km/s, large cells) | at the shock (r about 0.9 R_s, c about 360 km/s) |
| dt at step 200 | 3.4e-8 | 8.9e-8 |

**The interior** (shell means against Sedov-Taylor at t_end, 128³; job 32814798):

| r / R_s | ρ/ρ0 HLLC | ρ/ρ0 HLL | exact | c [km/s] HLLC | c HLL | exact |
|---|---|---|---|---|---|---|
| 0.20 | 0.0016 | 0.0226 | 0.0010 | 7956 | 2128 | 10,289 |
| 0.30 | 0.0043 | 0.0243 | 0.0063 | 4889 | 2055 | 4076 |
| 0.35 | 0.0080 | 0.0266 | 0.0128 | 3588 | 1965 | 2871 |
| 0.50 | 0.0482 | 0.0587 | 0.0609 | 1467 | 1331 | 1321 |

At 256³: HLL 0.0089 ρ0 at 0.15 R_s (exact 0.0003), c 3408 against 19,390 km/s.

HLL diffuses mass into the hot, nearly empty bubble. Its interior is too dense and 3-5× too cold, so its sound speeds,
and with them its time step, are too large. HLLC keeps the centre hot, as the exact solution is, and its time step is
limited there. **The larger step is a symptom of HLL's smeared interior, not an advantage.**

**At the shock** (shell means):

| | peak ρ/ρ0 (exact 4.00) | largest cell | half-height radius / R_s | 10-90 % width |
|---|---|---|---|---|
| 128³ HLLC | 3.275 | 3.955 | 1.0106 | 0.98 h |
| 128³ HLL | 3.001 | 3.194 | 1.0107 | 0.95 h |
| 256³ HLLC | 3.586 | 4.287 | 1.0058 | 0.93 h |
| 256³ HLL | 3.385 | 3.537 | 1.0057 | 0.96 h |

Same position, same width; HLLC's peak is higher, as expected.

**Where HLL's lower field L1 comes from** (ρ/ρ0 share by zone, 128³):

| zone (r / R_s) | HLLC | HLL |
|---|---|---|
| [0, 0.9) | 0.0107 | 0.0081 |
| [0.9, 1.1) | 0.0541 | 0.0483 |

Both schemes undershoot the shell behind the peak (0.8-0.98 R_s), where most of the mass is; HLLC undershoots more
there. HLL's broader, lower shell happens to lie closer to the exact profile in this integral norm.

**Verdict.** HLL is worse in the two features that matter for a blast: the post-shock peak and the hot interior. The
field L1 alone does not show this. Sedov's L1 is not evidence for HLL; it should not go into the README as an
advantage.

**Tools:**
- `LUNA_DT_TRACE=1` (`FiniteVolume::trace_step`): the limiting cell and face terms and their rows, one line a step.
- `scripts/sedov_shock_profile.py`: shell profiles at the shock and in the interior, peak, radius, width, and L1 by
  zone.
