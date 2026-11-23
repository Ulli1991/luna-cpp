# A fixed (Cartesian) grid mode in luna-cpp: design note (2026-11-19)

Design only; no solver code. LUNA (`/raven/u/uli/LUNA`, 60cf841) was read as the source. Its fixed grid is its strongest feature, and its A100 speed is the bar. The grid mode takes LUNA's numerics and its lessons on speed, and the kernels are written fresh for the A100. A fixed grid does not use voronoi-cpp.

Owner decisions so far (2026-11-19):
- the A100 path uses hand-written CUDA kernels, not Kokkos;
- phase 1 builds the first kernel in both CUDA and Kokkos, and that measurement decides how the rest is written;
- no Pallas-style translation: we take LUNA's numerics, not its XLA/Pallas structure.

Owner decisions of 2026-11-20 (the open questions at the end):
- FMA contraction on by default, and a bit-exact build (`LUNA_BIT_EXACT`) for the tests against LUNA;
- the per-axis HLLC and HLL cores in euler.hpp, beside `flux()`;
- the Kokkos form stays after phase 1 regardless, as the CI and MI300 path;
- a `--grid` flag in the existing benches;
- LUNA's references are stored in phase 0, in luna-reference-solutions `luna/`.

Phase 0 is done (section 8); phase 1 is in progress (section 9).

## 1. Purpose, and what of LUNA is needed

**Uses**
- **LUNA parity in one C++ code.** One binary runs the same problems on a grid and on a Voronoi mesh: the same ICs, outputs and diagnostics.
- **Speed.** At least LUNA's cell updates per second on the A100, in double and in single.
- **A clean reference for the moving mesh.** A grid run has no mesh errors, so differences point at the Voronoi path. It also gives the exact-lattice comparison that #6 blocks today.
- **Cross-validation.** The grid against LUNA (bits, or rounding), and against static and moving Voronoi runs on the same problem (#7).
- **Derivatives on a grid.** The tangent and adjoint are simpler than on a Voronoi mesh, with no geometry and a fixed stencil. LUNA's autodiff gives an independent check (`jax.jvp` / `jax.vjp` of its XLA step).
- **Later:** multi-GPU, and hybrid or AMR uses (a grid block beside or inside a Voronoi region). Not designed here.

**LUNA features needed.** For each: LUNA's source, and the phase that ports it.

| Feature | LUNA | Phase |
|---|---|---|
| 1D, 2D, 3D | `muscl_hancock.py`, the Pallas kernels in every dimension | 3D first (phase 1), then 1D and 2D (3) |
| MUSCL-Hancock, unsplit, one predictor per cell | `_predict`; `W_half = W - (0.5 dt) A dW` | 1 |
| Limiters LIMITER 0 minmod, 1 MC (LUNA's default), 2 van Leer | `slopes.slope_limiter` | 0 |
| HLLC (default) and HLL | `fluxes._hllc_core`, `_hll_core` | 0 |
| Boundaries 0 periodic, 1 zero-gradient, 2 reflective, 4 diode | `utils/halo_local.pad_halo_local`, two ghosts (the wide path) | 1 (0 and 1), 3 (2 and 4) |
| Floors (`preserve_face_states`): floored primitives, RAMSES face guard, first-order fallback (audit C8) | `conversions.cons_to_prim_floored`, `slopes.face_rule` / `ramses_face` | 3 |
| Passive scalars: upwind on the sign of the mass flux, HLLC only | `_face` in the Pallas kernel | 3 |
| Precision: FP64, FP32, recon32 (3D: reconstruction in FP32, fluxes and update in FP64) | `recon32`, state dtype | 4 |
| Time step: `CFL / sum_d (abs(v_d) + c) / dx_d` per cell, minimum over cells, NaN becomes -inf | `integrators.cell_dt`, `_make_compute_dt_fn` | 1 |
| Face report (count and location of first-order or floored faces) | `_own_faces`, `integrators._face_report` | 3, optional |
| Face mass fluxes (`gravity_energy='flux'`), smooth-tangent minmod (`limiter_tangent_eps`) | `_store_mass_flux`, `SmoothTangentMinmod` | later, with gravity / derivatives |

## 2. LUNA function to luna-cpp counterpart

The aim is LUNA's bits in FP64. So each piece is either an existing euler.hpp function that has been checked to compute the same expression in the same order, or a line-by-line port. All of them are `KOKKOS_INLINE_FUNCTION`, which expands to `__host__ __device__` under nvcc. CUDA kernels can therefore call them directly (section 4).

| LUNA | luna-cpp | Same bits? |
|---|---|---|
| `conversions.cons_to_prim_floored` | `hydro::primitive()` (euler.hpp:104) | yes by reading: the same floors and order. `cons_to_prim` without floors needs a no-floor variant, because luna-cpp always floors (smallr 1e-10) |
| `prim_to_cons` | `hydro::conserved()` | check E = P / gm1 + ke: LUNA divides, it does not multiply by 1/(gamma-1) |
| `slopes.minmod`, `minmod3`, `moncen`, `vanleer` (with the `tiny` test) | new `grid::limit(dl, dr, L)` on differences, not gradients. The 1D code's minmod and van Leer act on gradients and have no MC | port |
| `jacobian.primitive_jacobian_rows_3d_axis`, summed x, then y, then z | `hydro::half_step_rate` + `half_step` (euler.hpp:186, 208), with `g[v][d] = slope * inv_dx` | yes by reading: the same rows, the same left-to-right sums, `(0.5 * dt) * A` |
| `primitive_jacobian_rows_1d`, `_2d` (use `rho * (gamma P / rho)`, not `gamma P`) | new `grid::rate_1d`, `rate_2d` | port; `half_step_rate` agrees only to rounding in 1D and 2D |
| `fluxes._hllc_core` / `_hll_core` with `rotate_primitive_pair` | new `HLLC::core` / `HLL::core` in the normal frame (vn, vt1, vt2), axis fixed at compile time | port (see below) |
| `slopes.face_rule`, `ramses_face`, `first_order_if_unphysical` | new `grid::face_guard(Wf, Wc, spec)` | port |
| `integrators.cell_dt` + `jnp.min` | fused into the update kernel: a per-block minimum, then one `atomicMin` on the bits of a positive double (a minimum is order-free, so the bits are deterministic) | yes |
| `halo_local.pad_halo_local` | an index map in the kernel's load (wrap, clamp, mirror with the normal velocity negated, diode clamp). There is no padded copy | yes (pure copies) |
| Passive-scalar flux `F_rho * (L if F_rho >= 0 else R)` | the same, in the face function | port |

**Why HLLC is ported, not reused.** `HLLC::flux` in euler.hpp is LUNA's core rewritten for a moving face. With `n = e_d` and `w = 0`, it differs from LUNA by rounding in two places:
- the normal momentum of the star state is `rho* (vn + (S_M - vn))`, where LUNA has `rho* S_M`;
- on the y and z axes, the kinetic energy is summed x, y, z, where LUNA sums normal first.

It also multiplies by the zero components of n and w, which the compiler must keep under IEEE rules. An axis-specialised core is therefore both bitwise LUNA and cheaper. The existing `HLLC::flux` stays the reference for it (agreement to rounding in a unit test). Its hand-written tangent and adjoint serve as the check for the core's derivatives (section 7).

**Bits and their limits.**
- LUNA's XLA and Pallas paths are bitwise equal on the A100. Triton does not contract FMAs, and divisions and square roots are IEEE.
- FP64 bits therefore need nvcc `-fmad=false` on the grid kernels (and `-ffp-contract=off` for the host reference), plus LUNA's operation order everywhere.
- LUNA measured the FMA gain at +2–6 % (CODE_NOTES, A100 tuning). FMA contraction becomes a build switch: off for the bit tests, and on as a speed option whose bits differ from LUNA. The owner decides the default.
- NaN semantics differ (`jnp.minimum` propagates NaN, `fmin` drops it). That matters only on states that are already broken.
- FP32 and recon32 will agree only to rounding: Triton's FP32 division is a reciprocal times the numerator.

## 3. The grid step: data layout and kernels

**State.**
- `U` is a `Kokkos::View<double****, LayoutRight>` of shape (nvar, nx, ny, nz), with z contiguous, as LUNA's (nvar, N...) arrays.
- It is double-buffered: `U` and `U_next`, 80 B/cell in FP64. At 512³ that is 10.7 GB; 768³ needs 36 GB, which is tight on a 40 GB card.
- There are no ghost arrays. Boundaries are an index map in the load.

**Kernel K1: the whole step in one launch.**
This is LUNA's fused organisation, without the steps outside the kernel that cost LUNA 22 % (its `pad` and `add` fusions, CODE_NOTES 2026-11-04). As built (phase 0), the step reads W (the primitives) and U, and writes `U_next` and its primitives `W_next`, so the next step's W costs no kernel of its own. That is 160 B/cell of traffic instead of 80, still far from the bandwidth limit, and cheaper than converting U to W 25 times per cell. For one cell, in this order:
1. Load the stencil of W: every cell within L1 distance 2 (25 points in 3D). The boundary index map supplies the ghosts.
2. Compute the slopes, the predictor for the cell itself, and the predictors for its six neighbours.
3. Compute the six faces (HLLC), with the face guard.
4. Compute `U_next = U + dt * sum_d (F_minus - F_plus) * inv_dx_d`, summed x, then y, then z, as LUNA does.
5. Fold this cell's term of the next time step into a block minimum.

**Launch and tile.** One cell per thread. Blocks of 256 threads, tile 1×4×64 (z fastest), as LUNA's best FP64 tile with `num_warps = cells / 32`. Tile shape is swept in phase 2.

**Memory floor of K1.**
- About 40 B read (more where tiles re-read their edges through L1/L2) and 40 B written per cell.
- At 1.37 TB/s that is about 0.06 ns/cell, a limit near 17 G cell/s.
- LUNA's split step moves about 515 B/cell. Bytes are not the limit; arithmetic and latency are.

**Arithmetic floor (estimates, to be replaced by ncu counts).**
- The A100's FP64 rate at 1305 MHz is 108 SMs × 32 × 1.305 GHz = 4.5·10¹² instructions/s.
- LUNA's split step runs about 2030 FP64 instructions per cell, with its faces kernel at 78–85 % of the FP64 pipe.
- K1, like LUNA's fused kernel, runs 7 predictors and 6 faces per cell: about 2000–2500 instructions, a floor of about 1.8–2.2 G cell/s.
- Each predictor once and each face once is about 1000 instructions, a floor of about 4.5 G cell/s.
- LUNA has already tried this (a CUDA shared-memory kernel with x-marching, each predictor and face once, 80/124 registers, no spills). It ran at 0.95–1.00× the tuned Pallas kernel, because synchronisation and shared-memory traffic ate the halved arithmetic. That is our prior for lever L2 below; we still measure it with FMA on and with our own layout.

**Levers after phase 1.** One per commit, measured alone, bits checked, a row in docs/tried.md each.
- **L1 split:** a predictor kernel writes `W_half` on the interior plus one ring; the faces kernel recomputes slopes (LUNA's NVIDIA production path: 1.2–1.46× its fused kernel in 2D and 3D, 0.6–0.9× in 1D).
- **L2 faces once:** each thread computes its +d face, and the −d face comes from the neighbour through shared memory or a warp shuffle (z is the lane axis).
- **L3 shared-memory W tile:** each U is converted to W once per load, not once per use.
- **L4 FMA contraction:** a build option, with bits off LUNA.
- **L5 the axis-specialised Riemann core without zero multiplies:** already in by design; measured against `HLLC::flux` with `n = e_d` to know what it bought.
- **L6 tile and block-size sweep.** No `__launch_bounds__` register caps by default. Twice measured in luna-cpp: caps spill and slow the kernel. Measure both ways.

## 4. Direct CUDA inside luna-cpp, and what it costs

**Build.**
- luna-cpp builds CUDA with Kokkos's `nvcc_wrapper` as the C++ compiler, so `.cpp` files are already compiled as CUDA. `__global__` kernels and `<<<>>>` launches work in the same CMake, with no new language.
- The kernels sit in `src/hydro/grid_cuda.hpp` behind `#if defined(KOKKOS_ENABLE_CUDA)`.
- They take raw pointers from the Kokkos Views and launch on `exec.cuda_stream()`, so allocation, ICs, outputs, fences and the drivers stay in Kokkos.
- Host CI builds (Serial/OpenMP) do not compile them.

**Sharing euler.hpp.** No macro is needed while Kokkos is in the build: `KOKKOS_INLINE_FUNCTION` is `__host__ __device__ inline` under nvcc, and `Kokkos::Array`, `Kokkos::fmin` and `Kokkos::sqrt` are device-callable. A `LUNA_HD` macro is needed only if the grid kernels must ever build without Kokkos. They need not.

**The per-cell function is the unit of sharing.**
- `grid::cell_step<D, Limiter, Riemann>(load, i, j, k, dt) -> U_next` holds all the numerics.
- The CUDA kernel, the Kokkos kernel and the host reference all call it. Only tile loading, shared memory and the reduction differ.
- This keeps the two paths from drifting.

**CI.**
- The Kokkos form of the step, run on the Serial or OpenMP backend, is the host reference. CI covers the numerics through it: conservation, symmetry, LUNA's 1D values, a few stored LUNA steps.
- CI cannot run the CUDA kernel. A GPU job compares it bit for bit against the Kokkos form on the device and against the host reference.
- So the CUDA kernel's own code (indexing, shared memory, shuffles) is tested only in GPU jobs. That is a real gap: an indexing bug lands only when a GPU job runs.

**HIP / MI300.**
- hipify converts the kernels mechanically.
- MI300's 64-wide wavefronts change every warp-level trick and the tile choice. LUNA's tiles were chosen on MI300A first, and its NVIDIA rule did not carry over.
- So a CUDA path means a second tuning pass on MI300; the Kokkos form runs there unchanged as the fallback. This is a reason to keep the Kokkos form even if CUDA wins.

**Derivatives.**
- The tangent kernel is K1 with dual values: each thread calls the existing `*_tangent` functions next to the forward ones. It is forward mode and has the same stencil and launch.
- The adjoint is three gather kernels, deterministic and without atomics:
  - faces: the bars of L and R per face;
  - cells: the bars of `W_half` and the slopes, gathered from the cell's six faces;
  - stencil: each cell gathers what its neighbours' slopes and predictors hand back.
- They are written in CUDA for speed and in the host reference for CI, both calling the same `*_adjoint` functions.

**Costs, plainly.**
- About 1–1.5 k more lines than a Kokkos-only grid.
- Two orchestration paths to keep equal, and the CUDA one untested in CI.
- A hipify and retune for MI300.
- Derivative kernels written twice (CUDA and host).
- What it buys: launch shape, shared memory, shuffles, `cp.async` and per-kernel flags under our control. Whether that is worth anything is phase 1's question.

## 5. Accuracy and the test plan

**Bits (FP64, `-fmad=false`).** The same IC, dt, limiter, Riemann solver, boundary and N in LUNA and in luna-cpp.
- A script (`scripts/luna_grid_bits.py`, run in a GPU job) runs LUNA's step on the A100 for 1, 10 and 100 steps, with the dt sequence fixed from LUNA's run.
- It stores the states as npz in `luna-reference-solutions/luna/grid_bits/`.
- luna-cpp must match them bit for bit: 3D at 32×48×64 and 128³; 2D at 128×192; 1D at 4096.
- The cases are minmod, MC and van Leer; HLLC and HLL; boundaries 0, 1, 2 and 4; floors on and off; one passive scalar.
- In FP32 and recon32 the bound is rounding, measured and reported.

**Accuracy.** At matched settings, and with what cannot be matched listed:
- the 1D ladder against LUNA's values in `src/test/luna_1d_values.hpp`;
- KH 2D against `luna/mcnally_kh` and the luna-cpp static and moving KH npz;
- Sedov 3D and 3D sound waves.

LUNA has no stored Sedov 3D or sound-wave npz yet. Its runs are made in the same job and stored in `luna/`. The grid's own final states go to `luna-cpp/grid/`.

**Speed against LUNA.**
- One GPU job on one node: LUNA's production step (Pallas split, FP64, HLLC, minmod and MC) and ours at 128³, 256³ and 384³.
- The SM clock is logged, with 1305 MHz as the target. LUNA's numbers vary by node (0.47–1.05 G at 128³).
- Reported: the median of 20 steps after warm-up, in G cell updates/s.

## 6. Phases, effort and risks

Effort is in working days of one agent, with GPU queue time on top.

| Phase | Work | Effort |
|---|---|---|
| 0 | Ports (limiters, the 1D/2D rates, the HLLC and HLL cores, face guard, no-floor primitives); the per-cell `cell_step` for 1D, 2D and 3D; a host reference; the LUNA bits script and its npz; unit tests against `HLLC::flux` and LUNA steps on the host | 2–3 |
| **1** | **K1, 3D, double, in CUDA and in Kokkos** (details below) | 3–4 |
| 2 | Levers L1–L6, one at a time, in the form phase 1 picked | 3–5 |
| 3 | 1D and 2D kernels (LUNA tiles: 64 fused in 1D; 2D split 2×64, fused 4×32); boundaries 2 and 4, including the narrow first-order diode wall in 1D and 2D as LUNA has it; floors, face guard, report; passive scalars | 3–4 |
| 4 | Precision: an FP32 state (luna-cpp's state is always double today, so this is a new policy) and recon32 | 2 |
| 5 | Tangent and adjoint (double), host first, then device; finite differences, adjoint against tangent, and LUNA's `jax.jvp`/`vjp` | 5 |
| 6 | The accuracy suite (ladder, KH, Sedov, sound), npz to luna-reference-solutions, README / TESTING / docs/measurements.md, tried.md rows | 2–3 |
| later | Multi-GPU halo exchange (two ghosts, telescoping seams, as LUNA's wide path); hybrid and AMR | — |

**Phase 1, in detail.** One kernel algorithm (K1), with the same numerics, tile and launch shape in both forms, and the same bits.
- **CUDA form:** a `__global__` kernel, one cell per thread, a 1×4×64 tile, 256 threads.
- **Kokkos form:** a `TeamPolicy` with team size 256, one team per tile, each thread mapping to the same cell as in CUDA. It runs in two variants:
  - (a) no scratch, loads through L1, as K1 itself;
  - (b) a scratch tile of W, matched by a shared-memory variant of the CUDA kernel.
- An `MDRangePolicy` with tile {1, 4, 64} is measured as a third Kokkos variant only if (a) loses, to separate the policy's overhead from Kokkos as such.
- Both forms are built with the same nvcc flags. A test asserts equal bits between the two forms and the host reference.
- **Measurement:** Sedov 3D at 128³, 256³ and 384³ (powers of two and 3×2^k only), FP64, minmod and MC, HLLC, boundary 1, CFL as LUNA's.
- Reported, against LUNA in the same job on the same node, with the clock logged:
  - ms per step and G cell updates/s;
  - registers, spills (local bytes), occupancy and active warps;
  - DRAM B/cell;
  - FP64 pipe %;
  - the top stall reasons (ncu on an exclusive node, `dcgmi profile --pause`).
- The deliverable is a speed-of-light table per kernel.
- **Decision rule for the owner:** if Kokkos is within a few per cent of CUDA, the owner may still keep CUDA (it is his decision); the table shows what the CUDA path costs and buys.

**Risks**
- **Register pressure in FP64.** LUNA's fused FP64 kernel had 255 registers and spilled; its split faces kernel has 128 at 25 % occupancy. K1 may need the split (L1) to beat LUNA at all.
- **Faces once may not pay**, as LUNA's own CUDA prototype showed. If L1 and L2 both fail, the CUDA kernel may only match LUNA (about 1.1–1.3 G FP64), not beat it.
- **Bitwise parity is fragile.** One reordered sum, an FMA, or LUNA's 1D/2D `rho * (gamma P / rho)` breaks it. The ports must be read line by line, and the bits test catches the rest. LUNA may also change its numerics: the bits npz record LUNA's commit.
- **Node-to-node variation** of about 2× for LUNA on Raven. Compare only in one job on one node.
- **Two code paths:** drift is kept out by the shared `cell_step` and the GPU bits test, but the CUDA kernel is untested in CI.
- **Memory at 768³** (36 GB double-buffered) is tight on 40 GB A100s.

## 7. Interaction with existing issues and docs/ideas.md

- **#6 (NaN from zero-area faces on an exact lattice).** The grid gives the exact-lattice comparison with LUNA that #6 blocks, without a jitter. #6 itself stays open: it is a Voronoi-path fix.
- **#7 (standing 3D suite).** The grid supplies each test's fixed-grid counterpart and LUNA's run at the same settings. The 3D suite should list a grid column beside static and moving.
- **ideas.md rows:**
  - **R9 (time to a target error):** the grid joins the figures; GPU time to a target error decides between grid and mesh per problem.
  - **L1 / L2 / L7 (derivatives):** the grid reuses the existing `*_tangent` / `*_adjoint` and checks them against LUNA's autodiff. LUNA's `SmoothTangentMinmod` is the L7 proposal in working form.
  - **R6 (precision by condition):** the grid ports LUNA's FP32 and recon32, the measured counterpart to `grad16` / `grad32`.
  - **R17 (local fault finding):** a grid run is the uniform-state and conservation reference per cell.
  - **R15 (memory order):** not needed on a grid.
  - **"Faces in colours":** not needed on a grid, where each face's owner is fixed.
  - **#9 (local time steps):** out of scope here.
- **docs/ideas.md** gets one new row, "fixed grid mode (this note)", status next, when the owner approves.

## Open questions for the owner (answered 2026-11-20, see the top)

1. **FP64 bits.** Bitwise LUNA (`-fmad=false`, about 2–6 % slower) by default, or FMA by default with a bitwise build for tests?
2. **HLLC.** An axis-specialised HLLC core in euler.hpp (bitwise LUNA, cheaper), beside the existing `HLLC::flux`, rather than reusing `HLLC::flux` with `n = e_d` (rounding-level only)?
3. **Keep the Kokkos form** after phase 1 as the CI and MI300 path even if CUDA wins, or drop it?
4. **Where the grid sits in the drivers.** A `--grid` flag in the existing benches (`hydro_sedov`, `hydro_kh2d`, `hydro_wave3d`, `hydro_1d`), or a separate bench per problem?
5. **The first LUNA reference runs** (Sedov 3D, 3D sound wave and the bits npz): store them in `luna-reference-solutions/luna/` now, in phase 0?

## 8. Phase 0: what was built and what it showed (2026-11-20)

**Code** (branch `fixed-grid`):
- `src/hydro/grid.hpp`: the per-cell numerics, ported from LUNA operation for operation:
  - limiters minmod, MC, van Leer;
  - LUNA's 1D, 2D and 3D Jacobian rows;
  - the predictor;
  - the face rule and floors;
  - the face flux;
  - primitives;
  - the time step;
  - the ghost index map for boundaries 0, 1, 2 and 4;
  - `cell_step()`.
- `euler.hpp`: `HLLC::axis_core` and `HLL::axis_core`, plus NaN-propagating `nan_min` / `nan_max` (as `jnp.minimum` / `jnp.maximum`).
- `src/hydro/grid_finite_volume.hpp`: `GridFiniteVolume<D, Riemann>` with `refresh()`, `time_step()` and `step(dt)`, and the Kokkos kernel (a `TeamPolicy`, one team per tile).
- `src/hydro/grid_cuda.hpp`: the CUDA kernel. It uses the same `cell_step()` and numbering; its block minimum is a shared-memory tree and an `atomicMin` on an order key.
- `hydro_sedov <side> grid --grid`: LUNA's Sedov run.
- `src/test/grid_test.cpp`, with 25 LUNA cases in `src/test/data/luna_grid`:
  - dimensions: 1D 64, 2D 16×24, 3D 8×8×12;
  - limiters minmod, MC and van Leer;
  - HLLC and HLL;
  - boundaries 0, 1 and 2 (and 4 in 3D);
  - floors, with and without first order;
  - 10 steps each.
- `scripts/luna_grid_bits.py`, `luna_grid_refs.py`, `grid_build.sh`, `grid_ctest_a100.sh`, `grid_phase1_a100.sh`.

**Bits against LUNA.** Without FMA contraction luna-cpp's step is LUNA's step bit for bit:
- every one of the 25 cases;
- on the CPU (OpenMP build, gcc 15) and on the A100 (`LUNA_BIT_EXACT`);
- the Kokkos and CUDA kernels alike;
- every time step and the state after 10 steps.

The larger cases are in luna-reference-solutions `luna/grid_bits` (1D 4096, 2D 128×192, 3D 32×48×64 with MC and minmod, 1, 10 and 100 steps): job 32645116, below. Two things had to be learnt from LUNA's compiled HLO, because XLA rewrites LUNA's Python:
- **Division by a constant.** XLA turns a division by a compile-time constant into a product with its reciprocal. `prim_to_cons` computes `P * (1 / (gamma - 1))`, and the dt function computes `(|v| + c) * (1 / dx)`.
- **Folded constants in the time step.** XLA folds `gamma * ((gamma - 1) * e)` into `e * (gamma (gamma - 1))`.

The Pallas (Triton) kernels and `U + dt dU` were not rewritten: their bits came out right on the first try. `grid.hpp` mirrors both rewrites, `to_conserved` and `cell_dt`. Before that, the states matched while 1 to 5 of 10 time steps were 1 ulp off.

**With FMA contraction** (the default build), the grid agrees with LUNA to rounding: at most 8.6e-16 of max |U| after 10 steps. The CUDA and Kokkos kernels then differ from each other in the last bits (266,658 values of 32×32×64×5 after 10 steps), because nvcc contracts the same source differently in the two kernels. `grid_test` asks for bits only where nothing contracts.

**References in luna-reference-solutions** (branch `fixed-grid`, `luna/`):
- `grid_bits`;
- `sedov3d/sedov3d_128.npz`: LUNA's Sedov run to 1e4 yr, 1075 steps, 1.09 G cell updates/s on ravg1196 for the whole jitted run;
- `sound3d`: luna-cpp's sound wave, amplitudes 1e-3 and 1e-6, nx 64, 128 and 256.

All are from SLURM 32610243, LUNA 60cf841. The 3D 128³ bits case (81 MB) stays in `/ptmp/uli/luna-cpp-fixed-grid/luna_refs/grid_bits`, out of luna-reference-solutions, because of the /raven/u quota.

**First speed, K1** (job 32610551, ravg1132, Sedov 128³, minmod, 30 steps, median after warm-up):
- the CUDA kernel 0.41 G cell updates/s (2.46 ns/cell);
- the Kokkos kernel 0.44 G (2.27 ns/cell);
- LUNA's Pallas step on this problem about 1.1 G.

K1 with 7 predictors and 6 faces per cell is about 2.5 times slower than LUNA: phase 1 profiles why (section 9).

## 9. Phase 1: CUDA against Kokkos against LUNA

Jobs 32719012 (128³), 32719016 (256³), 32719019 (384³) and 32719021 (ncu, 256³), 2026-11-21, `scripts/grid_compare_a100.sh`. All four ran on ravg1002, one A100-SXM4-40GB, with the SM clock at 1305 MHz throughout the timed runs (logged every second). Two builds of luna-cpp are compared:
- **nounroll**: K1 as committed in phase 0 (`-DLUNA_GRID_NO_UNROLL`);
- **unroll**: the loops of `grid.hpp` unrolled on the device (`LUNA_GRID_UNROLL`), the first lever.

Each build comes in both forms (CUDA and Kokkos).

**What is matched / what is not** (every table below):

| Matched | Not matched |
|---|---|
| LUNA's Sedov 3D with units, the same initial bits (`--ic` from LUNA's own state), N, limiter, HLLC, FP64, no floors, BOUNDARY 1, CFL 0.8 | the kernels (the point of the comparison) |
| 100 steps, each with its own CFL dt; median step after 10 warm-up steps; three alternating rounds | LUNA's step is two jitted calls (dt function, step); ours is one kernel that also leaves the next dt. Both have one host sync a step |
| The timed region: the whole step with its time step, one host sync, no output, no allocation | Python dispatch in LUNA's loop. It matters only at 128³: LUNA's rounds there spread 0.69–0.88 G |
| Node, GPU, job, clock | |
| CUDA and Kokkos: the same `cell_step()`, tile 1×4×64, 256 threads, tile numbering, nvcc flags and translation unit | Kokkos launches 32768 blocks of two tiles each at 256³ and 384³ (its grid cap); CUDA launches one block per tile. At 128³ the shapes are equal |

**Speed, G cell updates/s** (median of three rounds):

| N | limiter | LUNA (Pallas) | CUDA nounroll | Kokkos nounroll | CUDA unroll | Kokkos unroll |
|---|---|---|---|---|---|---|
| 128³ | minmod | 0.767 | 0.406 | 0.438 | 0.502 | 0.376 |
| 128³ | MC | 0.873 | 0.395 | 0.419 | 0.422 | 0.345 |
| 256³ | minmod | 1.092 | 0.386 | 0.427 | 0.505 | 0.365 |
| 256³ | MC | 0.979 | 0.378 | 0.410 | 0.425 | 0.334 |
| 384³ | minmod | 1.139 | 0.377 | 0.417 | 0.487 | 0.370 |
| 384³ | MC | 1.028 | 0.370 | 0.401 | 0.411 | 0.339 |

**Agreement, on the same runs** (128³, after 100 steps):
- The bit-exact build (`LUNA_BIT_EXACT`, unroll) equals LUNA bit for bit, in both forms and with both limiters.
- The FMA builds are within 2.5e-15 of max|U| of LUNA.
- With unrolling, CUDA and Kokkos give the same bits even with FMA. Without it they did not.

**Why Kokkos was ahead before unrolling.** It was not a launch or flag setting:
- Both forms ran 256 threads per block, the same tile and numbering, the same flags and the same timed region, at 25 % theoretical and achieved occupancy.
- Neither nvcc nor Kokkos set launch bounds or a register cap (110 and 122 registers).
- The difference was local memory. In both kernels nvcc kept the per-thread state arrays (neighbours, slopes, fluxes), indexed by loop counters that were not unrolled, in a stack frame: 816 B a thread in CUDA, 704 B in Kokkos.
- nvcc inlined differently in the two contexts (47 and 76 calls in the SASS), so the Kokkos form spilled about 9 % less:

| Form | local B/cell | DRAM B/cell | DRAM peak | long-scoreboard stalls |
|---|---|---|---|---|
| CUDA | 7008 | 2438 | 61 % | 70 % |
| Kokkos | 6384 | 2165 | 60 % | 59 % |

Both were bound by that spill traffic. The FP64 pipe was only 21 % busy for 1603 and 1622 FP64 instructions per cell.

**After unrolling.** The fix is shared by both forms:
- Both stacks are gone (0 local bytes). DRAM falls to 181 B/cell (CUDA) and 307 B/cell (Kokkos).
- CUDA becomes 1.24–1.31× faster with minmod and 1.07–1.12× with MC.
- Kokkos becomes 0.84–0.87× slower: 242 registers against CUDA's 194, 7 % more instructions, 70 % more DRAM bytes.
- Both now run one block of 8 warps per SM (12.5 %).
- With matched shapes and flags, CUDA is now 1.22–1.38× faster than Kokkos. The Kokkos grid cap is not the cause: at 128³, with equal shapes, CUDA is 1.22–1.34× faster as well.

**Speed of light, 256³ minmod** (ncu at 1305 MHz). The A100 limits are 4.5·10¹² FP64 instructions/s (108 SMs × 32 × 1.305 GHz) and 1.37 TB/s.

| Kernel | ns/cell | FP64 inst/cell | FP64 floor ns/cell | FP64 pipe | DRAM B/cell | DRAM floor ns/cell | regs | warps active | top stalls |
|---|---|---|---|---|---|---|---|---|---|
| ours, CUDA unroll (whole step) | 1.98 | 1576 | 0.35 | 23 % | 181 | 0.13 | 194 | 12.5 % | wait 21 %, long scoreboard 13 % |
| ours, Kokkos unroll | 2.75 | 1541 | 0.34 | 17 % | 307 | 0.22 | 242 | 12.5 % | long scoreboard 22 %, wait 19 % |
| LUNA predictor (`kernel`) | 0.11 | 173 | 0.04 | 52 % | 112 | 0.08 | 78 | 35 % | long scoreboard 63 % |
| LUNA faces (`kernel__1`) | 0.50 | 1390 | 0.31 | 78 % | 133 | 0.10 | 132 | 18.5 % | wait 29 %, math 23 % |
| LUNA pad (cons_to_prim + ghosts) | 0.11 | 15 | — | 3 % | 153 | 0.11 | 32 | 91 % | long scoreboard 67 % |
| LUNA add (`U + dt dU`) | 0.09 | 10 | — | 3 % | 119 | 0.09 | 26 | 86 % | long scoreboard 47 % |
| LUNA dt and small kernels | 0.04 | 27 | — | — | 42 | 0.03 | — | — | — |
| LUNA, all kernels | 0.86 | 1615 | 0.36 | — | 559 | 0.41 | — | — | — |

LUNA's rows come from job 32645795 (the same node class, ncu at 1305 MHz).

**Why K1 is 2.2–2.6× slower than LUNA.** It is not the arithmetic: both execute about 1600 FP64 instructions per cell. It is not memory either: we move 181 B/cell against LUNA's 559. It is how busy the FP64 pipe is, 23 % against 78 % in LUNA's faces kernel, for three measured reasons:
1. **Registers and occupancy.** The fused cell (25 stencil loads, 7 predictors, 6 faces) needs 194 registers, so 8 warps fit per SM. LUNA splits the step: its faces kernel runs at 132 registers with 12 warps, its predictor at 78 registers.
2. **Instructions besides FP64.** We issue 6935 thread instructions per cell against LUNA's about 5380, for the same FP64 work. The extra 1600 or so are mostly integer index work: the ghost map for each of the 25 loads, branches.
3. **Branches against selects.** Our HLLC, limiters and ghost map branch (250 static branches), and the stalls are on fixed-latency dependencies (wait 21 %). LUNA's numerics are branch-free selects (`jnp.where`), which Triton schedules for parallel issue.

**Levers, ranked by the ns/cell they can save** (one per commit, measured in the same job against the build before):
1. **Split predictor and faces kernels** (LUNA's NVIDIA organisation; less register pressure per kernel). Target: LUNA's 0.11 + 0.50 ns, from 1.98.
2. **An interior fast path.** Tiles that touch no boundary skip the ghost map: fewer integer instructions.
3. **Branch-free selects** in HLLC, the limiters and the face rule, as LUNA's. The bits are unchanged, because a select picks the same value.
4. **Block shape and registers** (128-thread blocks, a tile sweep). Launch bounds are measured both ways and never assumed.
5. **Faces once** (shared memory or shuffles), after 1–4.

**Lever 1, the split step (done, 2026-11-22).** Jobs 32814949/50/53 and ncu 32814954, ravg1002 at 1305 MHz, with the same box as above (phase 1 build against the split build against LUNA, three rounds).

| N | limiter | LUNA | CUDA phase 1 | CUDA split | Kokkos phase 1 | Kokkos split |
|---|---|---|---|---|---|---|
| 128³ | minmod | 0.971 | 0.501 | 0.881 | 0.375 | 0.579 |
| 128³ | MC | 0.877 | 0.422 | 0.838 | 0.344 | 0.539 |
| 256³ | minmod | 1.135 | 0.506 | 0.885 | 0.366 | 0.547 |
| 256³ | MC | 1.014 | 0.426 | 0.843 | 0.337 | 0.514 |
| 384³ | minmod | 1.154 | 0.487 | 0.851 | 0.370 | 0.592 |
| 384³ | MC | 1.027 | 0.411 | 0.814 | 0.339 | 0.556 |

Bits: the bit-exact split build equals LUNA after 100 Sedov steps at 128³ (both forms, both limiters); with FMA, split and fused give the same bits.

ncu at 256³ minmod:

| Kernel | ns/cell | regs | warps active | FP64 pipe | FP64 inst/cell | DRAM B/cell |
|---|---|---|---|---|---|---|
| predictor | 0.11 | 76 | 35 % | 38 % | 124 | 81 |
| faces | 1.03 | 156 | 12.5 % | 26 % | 928 | 310 |

The predictor now matches LUNA's (0.11 ns). The faces kernel does 928 FP64 instructions per cell against LUNA's 1390, but takes 1.03 ns against LUNA's 0.50. With 156 registers in 256-thread blocks it still runs one block, 8 warps, per SM. 128-thread blocks would fit three blocks, 12 warps, as LUNA's (lever 4).

**Lever 4, block shape and registers (done, 2026-11-23).** Done before lever 2 because the ncu numbers pointed there. The faces tile sweep at 256³ (job 32819613, one binary, `--tile`) put the 128-thread tiles at 1.38× the 256-thread one, and 64-thread tiles at 1.30×. The default 3D split tile is now 1 × 2 × 64, LUNA's SPLIT_TILE faces tile, in both forms. Jobs 32894404/06 and 32894932 ran lever 1 against lever 4 against LUNA in each job, with ncu in 32894934; ravg1002 at 1305 MHz, the same box as above.

| N | limiter | LUNA | CUDA lever 1 | **CUDA lever 4** | Kokkos lever 1 | Kokkos lever 4 | CUDA cap 4 blocks |
|---|---|---|---|---|---|---|---|
| 128³ | minmod | 0.76–0.92 | 0.885 | **1.220** | 0.580 | 0.770 | 1.279 |
| 128³ | MC | 0.78–0.87 | 0.842 | **1.139** | 0.540 | 0.720 | 1.200 |
| 256³ | minmod | 1.090 | 0.885 | **1.221** | 0.547 | 0.727 | 1.057 |
| 256³ | MC | 0.977 | 0.844 | **1.154** | 0.514 | 0.686 | 1.088 |
| 384³ | minmod | 1.140 | 0.851 | **1.167** | 0.592 | 0.781 | 0.952 |
| 384³ | MC | 1.028 | 0.814 | **1.115** | 0.556 | 0.747 | 0.927 |

The CUDA step is now ahead of LUNA at 256³ and 384³: 1.12× and 1.02× with minmod, 1.18× and 1.08× with MC. Bits: the bit-exact build equals LUNA after 100 Sedov steps at 128³, in both forms and with both limiters.

ncu at 256³ minmod, faces kernel:

| Variant | ns/cell | regs | warps active | FP64 pipe | local B/cell | DRAM B/cell |
|---|---|---|---|---|---|---|
| 256 threads (lever 1) | 1.03 | 156 | 12.5 % | 26 % | 0 | 310 |
| 128 threads (lever 4) | 0.73 | 156 | 18.75 % | 37 % | 0 | 304 |
| 128 threads, cap 4 blocks | 0.84 | 128 | 25 % | 32 % | 488 | 369 |

The cap spills, and its extra warps do not pay for that at 256³ and up. It is off, and the macro stays for sweeps. Our whole step now takes 0.11 + 0.73 = 0.84 ns/cell of kernel time, against LUNA's 0.86. LUNA's faces kernel is still busier (0.50 ns, 78 % FP64 pipe) but does 1390 FP64 instructions per cell against our 928, and LUNA carries 0.20 ns/cell of padding and update that we do not have.

**Why our faces kernel is slower than LUNA's** (job 32896266, 2026-11-23). ncu's full set ran on one launch of each kernel in the same job on ravg1002 at 1300 MHz: ours is lever 4's `step_kernel<3, HLLC, split>`, LUNA's is Pallas `kernel__1`; Sedov 256³ minmod, from the same initial bits. Reports are in `/ptmp/uli/luna-cpp-fixed-grid/faces-ncu/`.

| | LUNA faces | ours, faces |
|---|---|---|
| time (ms, 16.8 M cells) | 8.11 (0.48 ns/cell) | 12.13 (0.72 ns/cell) |
| thread instructions per cell | 4271 | 4090 |
| of which FP64 (DFMA + DMUL + DADD) | 1396 (354 + 564 + 478; no contraction) | 934 (501 + 230 + 203) |
| selects (FSEL + SEL) | 834 | 465 |
| branches and convergence (BRA + BSSY + BSYNC) | 204 | 660 |
| integer compares (ISETP) | 55 | 302 |
| global loads (LDG) per cell, sectors per request | 100, 8.6 | 105, 8.4 |
| shared memory | none (0 B static or dynamic: Pallas stages nothing; its tile ops load through L1) | 8 KB static (our block minimum of dt); 7 LDS/STS per cell |
| registers, blocks per SM, warps per SM | 128, 4, 16 (25 %) | 156, 3, 12 (18.75 %) |
| eligible warps per scheduler, issue slots busy, IPC | 0.90, 49 %, 1.96 | 0.43, 31 %, 1.26 |
| FP64 pipe | 81 % | 37 % |
| warp cycles per issued instruction | 8.06 | 9.46 |
| L1 hit, L2 hit, DRAM throughput | 38 %, 79 %, 17.6 % | 33 %, 64 %, 27 % |
| branch efficiency, divergent branches (avg.) | 100 %, 1.75 | 99.45 %, 2134 |

Stall cycles per issued instruction (ncu `smsp__average_warps_issue_stalled_*_per_issue_active`):

| stall | LUNA | ours |
|---|---|---|
| long scoreboard (waiting on a global load) | 0.21 | **2.89** |
| wait (fixed-latency dependency) | 1.78 | 2.27 |
| branch resolving | 0.23 | 0.63 |
| barrier | 0 | 0.14 |
| math pipe throttle (the FP64 pipe is full: a good stall) | **2.01** | 0.65 |
| no instruction | 1.81 | 1.32 |
| not selected (another warp issued) | 0.84 | 0.36 |
| selected (issuing) | 1.00 | 1.00 |

**What explains the 0.24 ns/cell.**
- The two kernels issue the same number of instructions (4090 against 4271 per cell), so the gap is the issue rate. Ours issues every 9.46 / 3 = 3.15 cycles per scheduler, LUNA every 8.06 / 4 = 2.02. That predicts 1.51× and 1.50× is measured.
- **Occupancy (about 0.18 of the 0.24 ns).** Our 156 registers leave 3 warps per scheduler, LUNA's 128 leave 4. At our own cycles per instruction, 4 warps would give 0.72 × 3/4 = 0.54 ns.
- **Exposed load latency (most of the rest, about 0.06 ns).** Our warps wait 2.89 cycles per instruction on global loads, against LUNA's 0.21. The loads are the same count, size and coalescing (105 against 100 per cell, 8.4 sectors per request). Triton issues a tile's loads together ahead of their use. Ours sit inside the per-axis loop, between that axis's flux computations, so few are in flight when needed. LUNA's warps instead stall on a full FP64 pipe (math throttle 2.01): it is compute-bound, we are latency-bound.
- **Control flow (inside the above).** We execute about 460 more branch and convergence instructions and 250 more integer compares per cell than LUNA. Of these, 176 BSSY/BSYNC pairs come from divergent branches (2134 divergent branches against LUNA's 1.75). LUNA computes every branch and selects (834 selects per cell), and spends its extra issue slots on FP64: 1396 FP64 instructions against our 934, uncontracted.
- **Not the cause.** Memory traffic is not it: L2 serves both, our DRAM 27 % is no limit. Shared memory is not it either: LUNA uses none.

**Levers this points to**, in order:
1. **Fewer registers, to 128 or less without spills:** 16 warps. The launch-bounds cap spilled (lever 4c). The registers must come from the code: branch-free selects shorten live ranges, and the faces kernel need not hold a whole State per neighbour.
2. **Loads issued early:** all of a cell's W and half-step loads at the start of the faces kernel, as Triton does. This must be weighed against registers. `__ldg` / `const __restrict__` let the compiler move them.
3. **Branch-free selects (lever 3):** removes about 700 control instructions per cell and the branch-resolving stall (0.63 cycles per issue).
4. **The interior fast path (lever 2):** removes integer index work, about 300 ISETP per cell.


## 10. Speed LUNA leaves unused

LUNA's own profile, CODE_NOTES 2026-11-04 (Sedov 256³, FP64, minmod, HLLC, split Pallas path, at ncu's base clock of 1095 MHz), and what each item would be worth:
- **The padding and update kernels.** `loop_pad_fusion_1` (cons_to_prim and the ghost pad, 152 B/cell, 86 % of DRAM peak) is 12 % of the step. `loop_add_fusion_1` (`U + dt dU`, 119 B/cell) is 9 %. Both are pure bandwidth: 22 % of the step and about 270 of its 515 B/cell. K1 needs neither: the ghosts are an index map, and the update and next W are written by the step kernel.
- **Repeated predictors and faces.** The split path computes each predictor once, but each face twice, once from each side: 6 faces per cell. Its faces kernel is 65 % of the step at 78 % of the FP64 pipe. The fused path also computes 7 predictors per cell. Computing each face once would halve the faces' arithmetic. LUNA's own CUDA attempt at this did not pay (0.95–1.00×).
- **Spills.** LUNA's fused FP64 Pallas kernel: 255 registers and 10.5 M spill requests. Its split faces kernel: 128–132 registers, 18.5–25 % of warps active, no spills.
- **No FMA.** Triton-Pallas contracts nothing: ncu counts 0 fused FMAs, and estimates up to +39 % (FP64) and +50 % (FP32) from contraction. LUNA's inline-PTX prototype measured +2–6 % on top of its reciprocals.
- **Node-to-node spread.** The same LUNA code runs at 833 M cell/s on ravg1175 and 428–562 on ravg1002 at 128³, with the cause not found. Every phase 1 job logs the node name and the SM clock every second, so the spread can be probed in our kernels and LUNA's on the same node.

**Measured in phase 1** (ncu, 256³ minmod, job 32645795; LUNA's whole step 0.86 ns/cell of kernel time, 0.88–0.92 ns/cell of wall time):
- **Padding and update:** 0.11 + 0.09 = 0.20 ns/cell, 23 % of LUNA's kernel time, and 272 of its 559 B/cell. K1 has no such kernels.
- **Repeated work:** LUNA's faces kernel executes 1390 FP64 instructions per cell, 86 % of the step's FP64 work. It computes each face twice and recomputes the slopes. That is the largest single block of arithmetic.
- **Spills:** none in the split path (132 and 78 registers). Our own spills before unrolling cost us 7 KB/cell of local traffic: that, not LUNA, was the spill story of phase 1.
- **FMA:** LUNA's FP64 instruction mix shows no contraction. Ours, with FMA on, executes 1576 FP64 instructions per cell against LUNA's 1615 for the same numerics.
- **Node:** every phase 1 job landed on ravg1002, LUNA's "slow" node of CODE_NOTES. There LUNA ran at 1.09–1.14 G at 256³–384³ minmod (CODE_NOTES' best elsewhere: 1.19–1.29 G), and at 0.69–0.88 G at 128³, with 1305 MHz throughout. A job pinned to another node is still to run.

**Lever 3, branch-free selects (measured, dropped, 2026-11-23).** Job 32901695 (256³, against lever 4, with nsys kernel sums) and ncu 32901696; ravg1002 at 1305 MHz, the same box.

| 256³ | LUNA | CUDA lever 4 | CUDA selects | Kokkos lever 4 | Kokkos selects |
|---|---|---|---|---|---|
| whole step, minmod (G) | 1.091 | 1.219 | 0.988 | 0.727 | 0.806 |
| whole step, MC (G) | 0.979 | 1.154 | 0.909 | 0.686 | 0.745 |
| kernel time only, minmod (nsys, 40 steps; G and ns/cell) | 1.135 (0.881) | 1.152 (0.868) | 0.980 (1.020) | 0.728 (1.373) | 0.793 (1.262) |

LUNA's kernels per step (nsys, CUDA graph nodes traced): faces 8.17 ms, padding 1.91, predictor 1.91, update 1.46, the dt function's 0.90; ours: faces 12.70, predictor 1.83. On kernel time alone ours and LUNA's are level (0.868 against 0.881 ns/cell); the whole-step lead (0.82 against 0.92 ns/cell) is LUNA's Python dispatch and separate dt call. (nsys's sums are means over 40 steps, warm-up included, under tracing; the whole-step numbers are medians after 10 steps.)

The selects cost more arithmetic than they save: every case is computed (FP64 instructions 928 -> 1017 per cell), the registers fall only to 146 (still 3 blocks of 128), and nvcc had already predicated the short branches. Divergence comes from elsewhere: of 0.055 divergent branch targets per cell, 0.039 are the block minimum's shared-memory tree (5 divergent levels per 128-thread block) and 0.016 the ghost map at the boundary; the limiters and Riemann solvers contribute none. The variants stay behind `-DLUNA_GRID_SELECT_LIMITER`, `_RIEMANN` and `_REST`.

Next, from the side-by-side: registers to 128 or fewer (4 blocks of 128) by removing run-time switches (limiter, boundary and floors as compile-time parameters, as LUNA's are static), the block minimum by warp shuffles instead of the shared-memory tree, and early loads.
