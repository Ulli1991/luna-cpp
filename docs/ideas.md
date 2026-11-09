# Ideas: what we will try, and why

A review of research ideas for the hydro, 2026-11-09, checked against how the code works today. Mesh ideas are in voronoi-cpp's `docs/ideas.md`.

## Already done

| Idea | Where it is |
|---|---|
| Uniform state preserved on a moving mesh | `hydro` and `hydro2d` tests: a uniform state with a bulk velocity stays uniform |
| Conservation across faces | tested to rounding; the totals of the inflow wall match the budget through it |
| Comparing schemes by time to a target error | the 1D and Kelvin-Helmholtz time-to-target figures |
| Deterministic sums | every cell sums its faces in a fixed order; runs give the same bits |
| Mixed precision where it is safe | `grad32` is the 3D default, checked against double |

## Planned, in order

1. **A geometric conservation check.** Compare each cell's volume change over a step from the two meshes with the volume swept by its moving faces. It needs no AD and tells us whether the two-pass faces are time-centred correctly, including steps with flips.
2. **Derivative checks of one step (Enzyme).** Forward-mode derivatives of reconstruction, half step, Riemann flux and update against finite differences, on a fixed mesh and branch pattern. Enzyme has no Kokkos support yet, so this runs on the host build with clang (versions 18 to 21 are on Raven) and Enzyme built from source. It is the first step of autodiff (#12), following LUNA's `docs/AUTODIFF.md`.

## Worth trying later

- **Mesh motion that avoids needless flips,** steering the regularisation away from near-degenerate configurations, judged by error per GPU second.
- **Sensitivities of the final state** to initial conditions or parameters, the adjoint part of #12.

## Not now, and why

- **Reusing last step's limiter or wave speeds to skip work.** The saving is small next to the risk of missing a shock.
- **Space-time finite volumes.** A research project of its own.
