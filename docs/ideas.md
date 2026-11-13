# Ideas for a faster and more accurate hydro

Every idea from the research review of 2026-11-08, checked on 2026-11-09 against the code. Nothing is dismissed: each idea has what it could buy, a first test, and a status. Mesh ideas are in voronoi-cpp's `docs/ideas.md`.

Where we stand: one A100, 128³ Sedov, `grad32`: static hydro 0.170 G cell updates/s, whole moving step 0.073; LUNA 0.47 to 0.76 FP64 in the same jobs. Memory 2236 B/cell moving. The profile says the hydro waits on memory, not flops: the gradient sweep at 0.268 G/s with 17 % of warps active, fluxes 0.571, update 0.805.

Status: **done**, **next**, **open**, **research**, as in voronoi-cpp's file.

## Speed and memory of the step

| ID | Idea | Buys | First test | Status |
|---|---|---|---|---|
| — | Owner bit in list entries; faces numbered by owner in the mesh | +8 % static, +5 % whole step (estimate, partly measured) | from the profile notes | next |
| — | Fewer registers in the gradient sweep (148 now, 3 of 16 warps) | sweep +30 to +50 % (estimate) | the fluxes kernel at 80 registers is the model | open |
| R4 | Reuse last step's gradients, limiter state or wave speeds where a conservative test says it is safe | less work per step | how often the test passes, and what a missed change costs | open |
| R3 | Read only the geometry the scheme needs | memory | the array reader list, with voronoi-cpp's R3 | next |
| R6 | Precision by condition | memory, speed | `grad32` is done in 3D; 2D next (#19) | next |
| #9 | Local time steps from AREPO-ng | fewer cell updates | Sedov, as in the issue | open |
| #10 | A CPU path for few active cells | speed with local steps | after #9 | research |
| R14 | Fuse face geometry from the mesh with the flux kernel, so faces are not stored and read back | memory and memory traffic | see voronoi-cpp's R14 | research |
| R15 | Memory order of cells by Morton, Hilbert or the graph, on a long run | locality in the sweep and fluxes | with voronoi-cpp's #34 | next |

## Accuracy and checks

| ID | Idea | Buys | First test | Status |
|---|---|---|---|---|
| O4, L4 | Geometric conservation: each cell's volume change over a step against the volume swept by its moving faces; uniform flow, and steps with flips | tests the moving step's time-centring | endpoint volumes against swept faces, no AD needed | next |
| L1 | Derivatives of reconstruction, half step, Riemann flux and update against finite differences on a fixed mesh and branch pattern | finds wrong derivatives and silent bugs | Enzyme on the host build: clang 18 to 21 on Raven, Enzyme built from source; no Kokkos support in Enzyme yet | next |
| L2 | Differentiable Riemann solver | verification, sensitivity | after L1 | open |
| L7 | Limiter sensitivity, piecewise | understanding the Sod ringing and stalls | after L1 | open |
| L6 | Time-step control sensitivity | stable larger steps | smooth surrogates of the CFL rule | research |
| T1 | Metamorphic tests: uniform state on a moving mesh, conservation, Galilean invariance, frame changes | robustness | the first three exist; add rotations and frames | done, extend |
| T3, O5 | Herbie and stochastic arithmetic on conserved-to-primitive conversion, energies and fluxes | accuracy | one expression first | open |
| O6 | Deterministic sums | reproducible bits | fixed order of every cell's faces | done |
| R9 | Time to solution at fixed error | right decisions | the 1D and Kelvin-Helmholtz figures | done |
| R17 | Local fault finding: per cell and face, area-vector closure, shared-face orientation, volume change against swept volume, uniform-state preservation, flux cancellation, positivity; reported by cell, not as a global total | turns a conservation failure into the failing cell, and a small regression case | inject known face and time-centring errors and check they are found | next |
| T0 | One scheme, MUSCL-Hancock, in every dimension: on a moving mesh, fluxes on the old and new mesh, half each (once called "two-pass"); on a static mesh the usual step. 1D gets the moving step, `--integrator` and the one-pass option go | second order on every moving mesh (one pass is first order there) | the 1D sound wave of #23 before and after | done (fix-23) |
| T1 | RK2 (Heun, SSP) with PLM against MUSCL-Hancock on the moving mesh: gradients and limiters recomputed on the new mesh, no half-step predictor; costs a second gradient sweep | possibly more robust in shear and on the 1D sound wave | GPU time to a target error on the 1D sound wave, 3D vortex and KH; stability on Sedov and Noh (#1) | open |
| T2 | Mesh motion within a step: the start-of-step velocity held fixed (one mesh update) against Heun on the positions (velocity recomputed at stage 2, a second mesh update); not clear in advance which is better | accuracy, flips and mesh quality per GPU second | both under MUSCL-Hancock and RK2 with PLM: error at fixed N, time to target, flips per step, mesh quality (#1) | open |
| T3 | Better results on tests with shocks and contacts, within the first-order limit any scheme has there: contact steepening for the static mesh, a less diffusive limiter in smooth regions, flattening only at shocks, damping of the ripples behind slow shocks; each a switch, off by default, kept only if it matches or beats today on all 30 1D tests and the super-hard tube | contacts to order about 1 on the static mesh; smaller errors at a given N; LA slow and colliding streams above 0.4-0.5 | each switch on the 30-test ladder and the super-hard tube, region by region | after #23 |
| #25 | Point velocity at strong contacts: branch fix-25 (fef76a5) moves each point with the gas velocity interpolated to it. The super-hard tube then reaches t = 2, 2.6-3.5x better than static, but on the ladder Toro 3-5 get 2.3-3.4x worse at 65536-131072, and Shu-Osher and shock-10 fall behind LUNA at 131072. Try a narrower fix first: interpolate only where the point is far from its centroid, or a drift that pushes back on shrinking cells | robustness on strong contacts without losing high-N accuracy | the tube to t = 2 and the 30-test ladder against #23's numbers | postponed until voronoi-cpp's speed work is done |

## Mesh motion and beyond

| ID | Idea | Buys | First test | Status |
|---|---|---|---|---|
| L3, O3, J1 | Mesh velocity optimised for accuracy, cell quality and fewer flips, with a smooth barrier near flips | fewer repairs, better cells | flips per step and error with and without the barrier | research |
| J2, J3 | Regularisation aware of hydro error and of mesh cost | accuracy per GPU second | after L1 | research |
| L8 | Mesh adaptation from error sensitivities | resolution where it matters | with refinement | research |
| L5, L9, T8 | Adjoints over many steps; calibration of EOS and source terms | inverse problems | #12, after L1 | research |
| O2, J4 | Space-time control volumes swept by moving cells; a differentiable moving mesh | exact time-centring, gradients through the mesh | a 2D prototype with one flip | research |
| R11 | Give the mesh the hydro's acceleration so it can predict flips better | fewer surprises in repair | see voronoi-cpp's R11 | research |
| R19 | Recompute only what a step invalidated | less work per step | see voronoi-cpp's R19 | research |
