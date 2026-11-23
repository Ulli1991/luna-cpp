# Derivatives doing work: examples after astronomix (2026-11-23)

Branch `autodiff-examples` off `autodiff-l2` 3dd0288, worktree /raven/u/uli/luna-cpp-adx. [M] measured, [P] published (cited),
[I] inference or plan. Step 1 of 3: the survey of astronomix's published derivative examples and the picks.

## Step 1: astronomix, what it publishes

**What it is** [P]. astronomix (formerly jf1uids), L. Storcks, T. Buck (and N. Thuerey on the 2026 paper): a differentiable
(magneto)hydrodynamics code in JAX, 1D/2D/3D, finite volume (VL2 predictor-corrector, PLM, HLL/HLLC) and a fifth-order
finite-difference WENO-Z scheme with constrained transport, Pallas GPU kernels, multi-GPU.
- Repository: github.com/leo1200/astronomix, MIT licence. Docs: astronomix-mhd.web.app. PyPI `astronomix` (0.6.x).
- Papers: Storcks & Buck, arXiv:2410.23093 (jf1uids, NeurIPS 2024 ML4PS workshop); Storcks, Thuerey & Buck, arXiv:2607.18176
  ("Per Astronomix ad Astra", July 2026).
- Reverse mode throughout (`jax.grad`, adaptive time steps with online checkpointing and checkpointed while loops); forward mode
  (`jacfwd`, tangents) in the jf1uids wind sensitivity and the Gauss-Newton Jacobians of the KH study.
- **Clean-room.** MIT allows reading, but the rule here is stricter: we take problem setups and published numbers only, as
  written in their papers and docs; no code of theirs is copied or run. Nothing was downloaded.

**The published derivative examples** [P]:

| # | source | problem | differentiated w.r.t. | optimised / inferred | resolution, horizon | numbers reported |
|---|---|---|---|---|---|---|
| A | 2410.23093 | 1D radial adiabatic stellar wind (spherical, geometric source terms) | wind parameters (stellar mass M, v∞) | recover (M, v∞) from the final state; gradient descent on a summed absolute misfit | N = 401 | optimisation paths only; forward sensitivity dρ,dP,dv/dv∞ at v∞ = 2000 km/s by `jacfwd`, against central FD with Δv∞ = 0.1 km/s ("remarkably stable", a spike at the shock) |
| B | 2607.18176 §5.1, `sensitivity.py` | linearised acoustics, ρ = 1, c_s = 2, γ = 5/3, amplitude 1e-6, periodic | the whole initial perturbation (ρ, v), reverse mode | none: gradient of J = ½∫(ρ'² + v'²) against the exact Fourier-space functional derivative | 3D Gaussian (σ = L/10) N = 64, L = 10, t = 1.5; 1D sine k = 4π/L, t = 0.15, N = 16–256 | 1D gradient L1 error: FV (2nd order) 6.2e-1 → 7.2e-3 over N = 16–256 (O(N⁻²)); FD-WENO 3.4e-2 → 1.1e-6 over 16–128 (O(N⁻⁵)) |
| C | 2607.18176 §5.2, `shock_tube_sensitivity.py` | Sod, θ = (ρL, PL, uL, ρR, PR, uR) = (1, 1, 0, 0.125, 0.1, 0), γ = 5/3, L = 1, open walls | the 6 parameters θ, reverse mode | none: gradient of J = ½∫(ρ² + v² + P²) dx at t = 0.2 against the smaller one-sided FD | N = 200, 250 fixed steps of dt = 8e-4 | residual falls as O(h); at h = 1e-6: 5.9e-6 (FV VL2+PLM+HLLC), 7.2e-6 (FD-WENO) |
| D | 2607.18176 §5.3 | cost of reverse mode with checkpoints | — | — | 481 adaptive steps, A100 | Pallas FD adjoint 15–17× faster and ~10× less memory than native-JAX backward; no ratio to one forward stated |
| E | 2607.18176 §6.1, `eigen_initialization.py` | 2D KH, 300², periodic x, open y, density 10:1, M = 0.5, γ = 5/3 | the step's Jacobian, one Fourier mode in x, dense eigen-decomposition | the most unstable eigenmode as an IC (λ ≈ 0.99 + 1.5i) | to t = 4.04 ≈ 1.5 τ_KH | eigenmode grows from t = 0; a velocity kick first loses amplitude |
| F | 2607.18176 §6.2, `kh_recon.py`, `kh_shooting_notes.md` | 2D KH twin experiment: ρ = 1, tanh shear ΔV = 1, thickness δ = 0.02, M = 0.3 (P ≈ 6.67), γ = 5/3, periodic x, open y | 10 controls: cos and sin amplitudes of k_x = 2..6 in v_y, Gaussian envelope w = 0.04; truth at 1 % rms of ΔV | recover the 10 from the terminal velocity field (loss ½‖u − u_obs‖²); Adam (cosine decay), single vs multiple shooting; also Gauss-Newton/LM on 10 tangents | 256², T = 20 t_g (~730 steps) and 60 t_g (~2200), t_g = δ/ΔV = 0.02, adaptive CFL | example IC error 0.001 (single shooting, T = 20 t_g); 0.046 (multiple shooting, T = 60); SS-GN success (IC err < 0.1): 16/16, 15/16, 10/16, 4/16 at T = 60, 80, 100, 120; Adam at N = 128 1.8–3.6 s a step; GH200 |
| G | 2607.18176 §6.3, docs notebook | field-level inference: velocity field so the projected column density at t_end matches an image | the whole initial velocity field: 3 × 128³ ≈ 6.3e6 (paper, MHD) or 3 × 32³ = 98 304 (docs, hydro) | Adam, reverse mode through checkpointed steps | paper 128³, ~1100 steps, MSE 5.6e-6; docs 32³, Adam lr 0.05, 150 iterations, 10 checkpoints, loss 6.9e-5 → 3.0e-8 | no runtime or memory per gradient published |
| H | 2607.18176 §6.4, `solver_in_the_loop.py` | correction network trained inside the solver | network weights | training | — | not in the text we read |

**Their hardware and forward speed** [P]: A100 (H100 in brackets) for single-GPU; weak scaling 209–276 M cell/s per GPU at
fp32 (FD-Pallas, 3D sound wave, H100). No published cost of one gradient relative to one forward step, and no B/cell for
gradients; our cost numbers will compare against LUNA's JAX instead (LUNA's AUTODIFF.md: grad/forward 5.3–5.4 at 128³–256³,
JVP/forward 2.8–2.9, MI300A; to be remeasured on the A100 for like for like).

**Not like for like, whatever we pick** [I]:
- Scheme: our MUSCL-Hancock (one stage, HLLC, face-centroid or minmod/MC/van Leer limiter) against their VL2+PLM (two stages) or
  fifth-order WENO-Z with SSPRK4. Their FV numbers are the comparable ones.
- Walls: luna-cpp's 2D/3D runs on a periodic Voronoi box (no open walls). KH in y and the Sod tube in 2D become periodic doubles
  (a second interface far enough away that nothing reaches it by t_end).
- Mesh: our static 2D/3D mesh is a Voronoi mesh of a jittered or regular lattice, not a Cartesian grid; the regular lattice is
  the Cartesian grid as a Voronoi mesh.
- Moving: luna-cpp has moving derivatives in 2D only (1D has none: `FiniteVolume<1>` is static; 3D is in progress).

## Step 2: the picks

1. **Sod θ (example C), 1D static, then an inverse problem on it.** Exactly their setup (N = 200, γ = 5/3, 250 steps of 8e-4,
   outflow walls, J = ½∫(ρ² + v² + P²)), so the FD residual compares directly with their 5.9e-6 (FV). Then:
   - (b) sensitivity: dJ/dθ from 6 tangents and from 1 adjoint (the same number twice), and the adjoint's per-cell map dJ/dQ₀;
   - (a) recover θ from the final density field from a perturbed start, L-BFGS on our gradients, loss and |θ − θ_true| per iteration;
   - LUNA's JAX on the same problem (its 1D hydro, minmod, HLLC, the same fixed dt): gradients side by side and time per gradient.
   Why: the one published example with a hard number we can match setup for setup, and 6 inputs makes forward mode competitive.
2. **KH twin experiment (example F), 2D, static and moving.** 256², the 10 Fourier controls, Adam on the terminal velocity field,
   T = 20 t_g (the horizon where their single shooting recovers to IC error 0.001). Their walls are open in y; ours is a periodic
   double shear layer, the second layer unperturbed (or mirrored), said so in the figure. Why: 2D is where our moving derivatives
   exist, 10 controls give (c) both ways (10 tangents against 1 adjoint, time and B/cell per gradient), and it has a
   published success metric.
3. **Field-level inference (example G), 2D static and moving, and 3D static at 32³.** The initial velocity field (2 × 256² in 2D,
   3 × 32³ = 98 304 in 3D, their docs' size) optimised so the final column density (3D) or density (2D) matches a target image of
   our own. Why: the many-inputs-one-output case where only reverse mode is practical, so (c)'s other side; the docs' 32³ numbers
   (150 Adam iterations, 6.9e-5 → 3.0e-8) are the comparison. Turbulent driving for the start is replaced by a random field
   at the same rms Mach number (luna-cpp has no driving), stated.

Not picked: A (spherical 1D, luna-cpp has no radial geometry), E (dense eigen-decomposition of the step Jacobian: possible with
tangents but a different kind of work; later), H (needs a network inside the C++ step).

**Steps over many steps** [I]. The C++ API differentiates one step: `step_tangent(dt, dq)` advances dq beside Q;
`step_adjoint(dt, qbar)` expects Q at the start of the step. A rollout's adjoint therefore stores the states of each step
(256² × 4 doubles = 2 MB; 730 steps ≈ 1.5 GB, on the device or host) or checkpoints and recomputes. dt is fixed per run
(recorded in the forward and replayed), as in example C; CFL dt's dependence on Q is not differentiated.
