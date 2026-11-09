# Ideas for a faster and more accurate hydro

Every idea from the research review of 2026-11-08, checked on 2026-11-09 against the code. Nothing is dismissed: each idea has what it could buy, a first test, and a status. Mesh ideas are in voronoi-cpp's `docs/ideas.md`.

Where we stand: one A100, 128³ Sedov, `grad32`: static hydro 0.170 G cell updates/s, whole two-pass moving step 0.072; LUNA 0.47 to 0.76 FP64 in the same jobs. Memory 2255 B/cell moving. The profile says the hydro waits on memory, not flops: the gradient sweep at 0.268 G/s with 17 % of warps active, fluxes 0.571, update 0.805.

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

## Accuracy and checks

| ID | Idea | Buys | First test | Status |
|---|---|---|---|---|
| O4, L4 | Geometric conservation: each cell's volume change over a step against the volume swept by its moving faces; uniform flow, and steps with flips | tests the two-pass time-centring | endpoint volumes against swept faces, no AD needed | next |
| L1 | Derivatives of reconstruction, half step, Riemann flux and update against finite differences on a fixed mesh and branch pattern | finds wrong derivatives and silent bugs | Enzyme on the host build: clang 18 to 21 on Raven, Enzyme built from source; no Kokkos support in Enzyme yet | next |
| L2 | Differentiable Riemann solver | verification, sensitivity | after L1 | open |
| L7 | Limiter sensitivity, piecewise | understanding the Sod ringing and stalls | after L1 | open |
| L6 | Time-step control sensitivity | stable larger steps | smooth surrogates of the CFL rule | research |
| T1 | Metamorphic tests: uniform state on a moving mesh, conservation, Galilean invariance, frame changes | robustness | the first three exist; add rotations and frames | done, extend |
| T3, O5 | Herbie and stochastic arithmetic on conserved-to-primitive conversion, energies and fluxes | accuracy | one expression first | open |
| O6 | Deterministic sums | reproducible bits | fixed order of every cell's faces | done |
| R9 | Time to solution at fixed error | right decisions | the 1D and Kelvin-Helmholtz figures | done |

## Mesh motion and beyond

| ID | Idea | Buys | First test | Status |
|---|---|---|---|---|
| L3, O3, J1 | Mesh velocity optimised for accuracy, cell quality and fewer flips, with a smooth barrier near flips | fewer repairs, better cells | flips per step and error with and without the barrier | research |
| J2, J3 | Regularisation aware of hydro error and of mesh cost | accuracy per GPU second | after L1 | research |
| L8 | Mesh adaptation from error sensitivities | resolution where it matters | with refinement | research |
| L5, L9, T8 | Adjoints over many steps; calibration of EOS and source terms | inverse problems | #12, after L1 | research |
| O2, J4 | Space-time control volumes swept by moving cells; a differentiable moving mesh | exact time-centring, gradients through the mesh | a 2D prototype with one flip | research |
