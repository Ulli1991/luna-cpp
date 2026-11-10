# Design notes for the hydro sources

This file holds the design text, measured numbers and file:line references that used to sit in long comments in `src/`. The sources keep one-line comments; the reasoning is here. Notes referenced below are in `research_notes/`.

## FiniteVolume<D> (src/hydro/finite_volume.hpp), D = 2, 3

### Scheme

Finite volumes on a `voronoi::Mesh<D>`, static or moving with the flow. MUSCL-Hancock (order 1) or first-order Godunov (order 0). A Riemann solver acts on every face along its normal, in the frame of the moving face: HLLC by default (as LUNA's riemann_solver), HLL, or any type with their static `flux()`. One global time step.

The state between steps is the extensive conserved Q = (mass, momentum, energy) of each cell, as AREPO next generation keeps it (mesh_fluxes.cc:162-164; density = mass / volume in mesh_update_primitives.cc:129-131). Primitives are computed from Q / V where needed. Line references for the stage 1 and 3 work are in `hydro_stage1_2026-10-31.md` and `hydro_stage3_2026-10-31.md`.

### One step (hydro_speed_2026-11-04.md)

`time_step()` computes W = primitive(Q / V), the sound speed and the cell radius once per cell, then makes one sweep over each cell's face list:

- (moving) the mesh-generating point velocity: the fluid velocity plus AREPO-ng's regularisation;
- (order 1) the least-squares gradients of W over the face neighbours, limited at the face centroids, stored for `step()`.

Then the global time step, a reduction over the cells and over the faces. It is two small kernels: fusing them into the sweep costs warps and fills its registers (hydro_speed_2026-11-04.md).

`step(dt)`:

1. Per face: the face velocity from the two point velocities (AREPO-ng). Each side's W is advanced half a step with the primitive Jacobian and extrapolated to where the face centroid is at the half step; if that leaves rho or P not positive, the cell's W is used instead (AREPO next generation). The flux through the moving face along n = A / |A|, times |A|, on the geometry at the start of the step.
2. Q -= dt * sum over the cell's faces of the outgoing flux.
3. (moving) the points drift by dt * w and the mesh is updated (`voronoi::Mesh::update`, `update_cells`).

That is the step of a static mesh, and of order 0. On a moving mesh at order 1 there is one step, MUSCL-Hancock on a moving mesh (below): the fluxes on the old mesh, half applied, then the mesh moves, then the fluxes on the new mesh, half applied. Everything a cell sums is summed over the mesh's per-cell face list in its fixed order, without atomics: the same bits every run.

### Two update loops

- per_cell: one thread per cell computes the flux of each of its faces (each face twice, once from each side, with the same arguments, so both get the same bits) and updates its state, in place at order 1 (the fluxes read only the primitive and gradient rows). No face array.
- per_face: one thread per face writes its flux into a face array, then one thread per cell gathers them and updates in place.

Per-cell and per-face data are rows (LayoutRight), so a random access reads one or a few 32-byte sectors. The primitives and gradients (one row per cell) and the face fluxes (one row per face) live in the mesh's scratch memory, requested in `time_step()`. Scratch is valid until the mesh's next `update()`, `rebuild()` or `update_cells()`; nothing else may use the mesh's scratch between `time_step()` and `step()`. The moving step reads the primitive and gradient rows after the mesh update, so there they live in an array of their own (`held_`), allocated only then, and the scratch holds the fluxes.

### Precision (src/hydro/precision.hpp)

The default, `DefaultPrecision<D>`, is `grad32` in 3D (float gradients, limiter and face-state change, everything else double) and double in 2D, where `grad32` has not been measured; 1D has no policies. `Precision<>`, `--precision=double`, is double throughout, with the least-squares fit solved in the sweep (the bits from before `grad32` became the default). Reduced precision and stored weights are compile-time options of the weights, gradients, stored face fluxes, Riemann solve and face geometry rows. Q, volumes, centroids, W and the sums of the face fluxes stay double in every policy. With a float gradient type, the differences W_j - W_i and the separations are formed in double and rounded once, and a face state is W + (its change in float), so a uniform state stays exactly uniform. FP16 gradients use one power-of-two exponent per variable group (rho, v, P): the exponent k of a group whose largest magnitude is m satisfies m 2ᵏ in [2¹⁴, 2¹⁵) (k = 0 for m = 0), within [-114, 126], so there is no overflow and 11 bits relative to the group's largest component. The FP16 storage is the precision study's (branch precision-study 59ff9d1, src/hydro/precision.hpp). Weights are computed in double by the mesh at every update (`tet_mesh::gradient_weights`); the measured effect is in hydro_speed_2026-11-04.md.

### Gradient limiter

The limiter keeps the extrapolated values within the neighbours' range (AREPO next generation, mesh_gradients.cc:84-94): at the face centroids (AREPO's default), at the neighbours' centroids (AREPO's USE_MINMOD_SLOPE_LIMITING), or nowhere.

### One moving step

A single flux at the half step on a moving mesh is first order on smooth flow (two_pass_2026-11-05.md; 1D: luna-cpp #23); it was an option until 2026-11-10 and is gone in every dimension.

### Time step rule

`StepRule::arepo`: the smallest of AREPO next generation's cell time steps (FORCE_EQUAL_TIMESTEPS, time_integration/timestep.cc:39-63) without the rounding to a power-of-two time bin. Cell i: courant * min(r_i / (c_i + |v_i - w_i|), min_j (d_ij + r_i) / vsig_ij), with w_i the point velocity (0 on a static mesh; timestep.cc:156-165, sph_particle_data.h:305-334), r_i the radius of a sphere of the cell's volume (sph_particle_data.h:280-287), and for every face neighbour j with vsig_ij = c_i + c_j - (v_j - v_i).d_ij / d_ij > 0, d_ij between the generators (timestep_treebased.cc:72-91). AREPO walks every particle a signal can reach; here only the face neighbours. The pair terms vsig_ij and d_ij are the same from both sides, so each face gives both cells' terms at once.

`StepRule::luna` (2D): LUNA's (RAMSES) rule, the smallest over the cells of courant / sum_d (|v_d - w_d| + c) / dx with dx = sqrt(V); on a square lattice that is LUNA's cell_dt (integrators.py:11-13).

### The sweep, in detail

Point velocity: AREPO-ng's set_vertex_velocities (src/voronoi/voronoi_vertex_velocities.cc:26-139 with REGULARIZE_MESH_CM_DRIFT, _USE_SOUNDSPEED, _FACE_ANGLE). It is the fluid velocity (:31), plus, for a cell whose largest face angle sqrt(A / π) / (d_ij / 2) (src/voronoi/cell.cc:563, 576-578; in 2D A / 2 / (d_ij / 2), A the face's length, AREPO voronoi_2d.c:1309) exceeds 0.75 cell_max_angle_factor, a drift towards the cell's centroid at fraction * c. The fraction rises linearly from 0 to cell_shaping_speed at cell_max_angle_factor (:100-106, :109-137; no gravity). The constants are those of AREPO's Sedov examples (examples/sedov_3d/sedov_globaltstp_3d/Config.sh:11-13: REGULARIZE_MESH_CM_DRIFT, _USE_SOUNDSPEED, _FACE_ANGLE; param.txt:93-94). The courant factor 0.3 is that of param.txt:52 there. The point velocities use the arithmetic of main 3fd11aa's kernel (its bits).

Gradients: as AREPO's next generation (Arepo2-nextgeneration-SoA, src/mesh/mesh_gradients.cc:16-106). The area-weighted least-squares fit W_j - W_i = a + g.d_ij with an intercept, d_ij between the centroids, is summed in the loop and solved after it. With the weights option, g = sum over the face neighbours j of weight_k (W_j - W_i), the mesh's weights of the list entry (:59). Then, for each variable, the gradient is scaled down face by face (faces above 1e-6 of the cell's surface, in the cell's face order) until the value extrapolated to every face centroid stays within the range of the neighbours' values (:62-97, limit_gradient :120-148).

### Face state and flux

The state of a side follows AREPO's prepare_state (mesh_fluxes.cc:451-479): the time and space extrapolations are added only if rho and P stay positive, otherwise the cell's own state is used. The space extrapolation goes to where the face centroid is at the half step, r + dt/2 w_face, the half step in the frame of the moving cell (AREPO-ng shifts its extrapolation the same way, mesh_fluxes.cc:441). At order 0, W goes straight into the Riemann solve, computed from Q and not read from the primitive rows: the compilers contract its products into the solver's sums (euler.hpp), and with loads the bits are not main's (hydro_speed_2026-11-04.md).

The face velocity is AREPO-ng's prepare_face_velocity (mesh_fluxes.cc:294-331): the mean of the two point velocities plus, along the generator direction n, (c.(w_L - w_R)) / |d| with c the face centroid relative to the generators' midpoint, |c| limited to 0.9 |d|.

### MUSCL-Hancock on a moving mesh

AREPO next generation's step (Arepo2-nextgeneration, src/main/run.cc, src/mesh/mesh_fluxes.cc). The fluxes at the start of the step (run.cc:284, dtExtrapolation 0 at mesh_fluxes.cc:620), half applied (fac, mesh_fluxes.cc:144); the points drift with w^n and the mesh is updated; then the fluxes on the new mesh, with its areas, normals and face centroids and the face velocities of w^n (run.cc:165, prepare_face_velocity at mesh_fluxes.cc:574), from the same W^n and g advanced a whole step (mesh_fluxes.cc:640-712; AREPO's d -= dt v with the convective derivative is -dt A here), half applied. The primitives from the new volumes and the gradients follow at the next `time_step()` (run.cc:210, 219). One mesh update per step, two flux sweeps (`step_moving()`, `MovingFlux`). Extra memory: `held_` and the centroids of t^n; the scratch then needs only the fluxes, so the cell rows are not paid for twice.

On the mesh of t^n: W^n + g (x_f^n - c^n), no half step. On the mesh of t^n+1: W^n - dt A(W^n) g + g (x_f^n+1 - c^n), which is `FaceFlux::side` with twice the step as its dt and d = x_f^n+1 - c^n: the generator moved by dt w^n, c holds the centroids relative to their generators.

## Finite volumes in 1D (src/hydro/finite_volume_1d.hpp)

The scheme of FiniteVolume<D> in one dimension, with walls, in one fused kernel per step (owner's decision 2, dimension_design_2026-11-02.md).

On the sorted line a cell's neighbours are the slots s - 1 and s + 1, so one thread per slot reads the five cells s - 2 .. s + 2 and computes, in registers, the limited gradients of s - 1, s, s + 1, the face states and the fluxes through its two faces, and writes its new state (moving: and drifts its point). Each face is computed by both of its cells from the same data in the same frame (every distance is an integer of the line, `sorted_mesh::Line`), so both get the same bits and the update conserves to rounding. A static step is two launches, `time_step()` and `step()`; a moving one three, with the mesh's order check; a sort runs only when points crossed. `step_and_time_step()` fuses them into one for LUNA's rule, the static default. AREPO's rule, the moving default, cannot be fused: its pair term reads the neighbours' next state. Fusing only the order check into `step()` was measured and is 2-6 % slower (stage1_1d_2026-11-04.md).

Scheme, as finite_volume.hpp unless said:

- Gradients: AREPO-ng's least-squares fit with an intercept over the face neighbours, weighted by the face area (mesh_gradients.cc; lines in the stage 1 notes). In 1D the two neighbours with weight 1 determine the fit exactly: g = (dW_R - dW_L) / (d_R - d_L), d the separations of the cell centroids. The limiter is AREPO-ng's (limit_gradient, mesh_gradients.cc:120-148): scaled face by face (left, then right) until the values extrapolated to the face centroids (or the neighbours' centroids) stay within the neighbours' range. On a uniform line this is the MC limiter of LUNA (slopes.py:22-24) on the central difference.
- The MUSCL-Hancock half step and HLLC/HLL in 1D (euler.hpp); each side's face state is the cell's if rho or P would not stay positive.
- Moving: the point velocity is the fluid velocity plus AREPO-ng's regularisation. A face in 1D has no extent, so the face angle of the 2D/3D code does not exist; the criterion is AREPO-ng's other branch (Arepo2-nextgeneration, set_vertex_velocities, !REGULARIZE_MESH_FACE_ANGLE, with CellShapingFactor 1 and CellShapingSpeed 0.5 of its examples): a drift at fraction * c towards the cell's centroid, displaced towards higher density, fraction rising from 0 at an offset of 0.75 r to 0.5 at r, r = V / 2. It keeps generators from crossing, which would swap the two cells' volumes under their masses (stage1_1d_2026-11-04.md). Below 0.5 r the displaced centre stays below the threshold of 0.75 r. The face velocity is the mean of the two point velocities: AREPO-ng's prepare_face_velocity (mesh_fluxes.cc:294-331) with the face centroid at the generators' midpoint.
- Moving step: MUSCL-Hancock with the fluxes on the meshes of both ends of the step, half each, as finite_volume.hpp's `step_moving()` (AREPO next generation's): the first on the mesh of t^n with W^n + g (x_f^n - c^n), the second on the mesh of t^n+1 with W^n - dt A(W^n) g + g (x_f^n+1 - c^n), the same W^n, g and face velocity. One thread computes both: its stencil gives the new positions of s - 1, s, s + 1. At order 0 the two are one flux. A single flux at the half step is first order on a moving mesh and is gone (luna-cpp #23).
- Face motion on the grid: the kernel snaps each moved point to its key first, and a face's velocity is its move between the keys of t^n and t^n+1 divided by dt (half the sum of its two generators' moves), not the mean point velocity. Then the volume change of every cell is dt (w_R - w_L) to rounding, the discrete geometric conservation law, and a uniform state stays uniform whatever the points do. With the mean point velocity, the volumes from the rounded positions (x + dt w rounds to ulp(x)) and the fluxes disagree by a random walk of about sqrt(steps) ulp per cell; at amplitude 1e-6 that noise was the moving linear waves' error from N = 4096 on, growing with N (luna-cpp #23).
- Walls (a mesh with `Ends::walls`): a wall face is static and takes a ghost state for the outside, LUNA's ghost cells on a uniform line (halo_local.py:16-31, the wide step of muscl_hancock.py:246-247). Reflective: the mirror of the cell's face state (the ghost's stencil is the mirror of the cell's). Zero gradient: the cell's W (two equal ghosts, a flat stencil, no half step; LUNA's BOUNDARY 1, mode "edge", halo_local.py:16-17). Inflow: a fixed state, which LUNA does not have. The ghost is also the cell's outer neighbour in its gradient, at the mirrored centroid.

Gas entering (moving, through a wall of either open kind): points leave a fixed wall, so without new ones the end cell grows without bound, by v t whatever N is, and holds whatever passes through it in one cell (luna-cpp #3). So the wall supplies cells. With d the distance from the wall to the end point and V the end cell's volume, a cell is added when V > 2 delta and d >= 1.5 delta, delta = `inflow_spacing()` (default the initial spacing L / N): AREPO-ng splits above twice the target volume (voronoi_criteria_refine_and_derefine.cc:158-163). The new point goes at d - delta from the wall, so a steady inflow gets a lattice of spacing delta, with cells centred on their points and no regularisation; AREPO-ng's most distant vertex (voronoi_execute_refinements.cc:775-783) would be the wall itself here. The transfer is AREPO-ng's (same file, :504-516): the new cell takes the conserved quantities in the fraction of the old cell's volume it overlaps and the old one loses them, so totals hold to rounding, both keep the old cell's primitives and no extremum appears. Nothing is stored for a gradient: the next step computes them from the states. AREPO-ng has no such wall: its special boundary cells do not take part in refinement, so this is this code's reading. A reflective wall adds none: gas leaving it is a rarefaction, not inflow. An open wall costs a look at the end cells every 64 steps until gas enters; from then on the time step's kernel watches them, which costs 6-11 % of a step (a second reducer), only in such runs. research_notes/inflow_2026-11-06.md.

Time step rules. arepo: AREPO next generation's, as in `time_step()` of the 3D code: courant * min_i min(r_i / (c_i + |v_i - w_i|), min_j (d_ij + r_i) / vsig_ij), with |v_i| on a static mesh (timestep.cc:156-165, sph_particle_data.h:305-334), vsig_ij = c_i + c_j - (v_j - v_i).d_ij / d_ij > 0 in the fluid velocities (timestep_treebased.cc:81-88). AREPO-ng has no 1D mesh, so three points are this code's reading in 1D: r_i = V_i / 2, the radius of a 1D ball of the cell's volume, as its 2D and 3D branches are of theirs (sph_particle_data.h:280-287 falls through to the cube root for ONEDIMS); j runs over the two neighbours, not every point a signal reaches; and a wall gives no pair term, there being no point behind it. stage1_1d_2026-11-04.md has what each costs. luna: courant * min_i V_i / (|v_i - w_i| + c_i), the RAMSES rule of LUNA's integrators.py:11-13 with dx the cell's length, in the frame of the point.

Defaults of a 1D run (owner, 2026-11-05): static, LUNA's rule at 0.8, which reproduces LUNA; moving, AREPO's rule at 0.4. Not 3D's 0.3.

Ghost at a wall: zero gradient and inflow take the ghost's own W, its two-cell stencil being flat, so its slope and half step vanish (LUNA's wide step, muscl_hancock.py:246-247 on halo_local.py:16-17, BOUNDARY 1: mode "edge").

## Euler equations (src/hydro/euler.hpp)

Conventions are LUNA's (src/luna/hydro/conversions.py:1): primitives W = (rho, v_1 .. v_D, P), conserved U = (rho, rho v_1 .. rho v_D, E) per unit volume, D + 2 entries (LUNA's layout, ic.py:141). Line references (LUNA file:line, AREPO file:line) are in hydro_stage1_2026-10-31.md. LUNA sources: floors template_config.py:41-42; `cons_to_prim_floored` conversions.py:28-36; W to U conversions.py:39-48; Jacobians jacobian.py:84-97, summed over the axes in order as muscl_hancock.py:105-116; HLL `_hll_core` fluxes.py:135-167; HLLC `_hllc_core` fluxes.py:51-106 (Toro 2009, ch. 10) with guards c² < 0 on either side takes F_R (fluxes.py:62, 88-94) and divisions kept off zero by 1e-20 (fluxes.py:71-72); branch order F_L, F*_L, F*_R, F_R (fluxes.py:88-94). The lab-frame conversion is AREPO next generation's flux_convert_to_lab_frame (mesh_fluxes.cc:492-498): energy + w.F_momentum + |w|²/2 F_rho, momentum + w F_rho.

Bit identity with the 3D-only code. Every expression is written so that D = 3 is the arithmetic, in the same order and the same form, of the 3D-only code before: dot products written out left to right, vectors and states built as brace initializers {f(0), .., f(N - 1)} where that code had them, loops over d < D where it looped over d < 3. Compilers contract a * b + c into FMAs depending on that form, so a loop where an initializer was gives other bits (measured on the A100 and with gcc).

Riemann solvers. Wave speeds after Davis as LUNA (fluxes.py:146-147, 65-66). LUNA rotates the states into (normal, two tangents), solves, and rotates the flux back (fluxes.py:9-48). Here the vectors stay in box axes: every branch of HLL and HLLC is chosen by scalars (S_L, S_M, S_R) that do not depend on the frame, and within a branch each flux component is the same linear combination of the corresponding components of F, U and U*, so rotating is the identity up to rounding. For HLLC the star momentum is rho* (v_t + S_M n), the tangential velocity v_t = v - (v.n) n carried through, which is LUNA's rho* (S_M, vt1, vt2) in box axes (in 1D, n = ±1 and v_t = 0). The CTest `hydro` checks both against LUNA's values and under random rotations. For a moving face the flux returned is the lab-frame flux (F(U) - (w.n) U for equal states).

## Other sources

- `snapshot.hpp`: HDF5 snapshots in LUNA's layout (write_snapshot_distributed.py, checkpoint_utils.py:_write_units): `<dir>/snapdir_NNN/snapshot_NNN.0.hdf5`, group `cells` with LUNA's attributes and datasets (density, velocity_x/y/z, internal_energy per unit volume, pressure, U as [5][N]), plus for the Voronoi mesh mesh = "voronoi", N_cells, and the datasets x, y, z (generating points) and volume. Group `units` as LUNA writes it.
- `problems_1d.hpp`: LUNA's three 1D tests; `scripts/luna_1d_run.py` has the file:line of each. Sod uses exact cell averages of the jump at t → 0 (LUNA: er.cell_average(..., t=1e-12)); at t = 0 that is the volume fraction. Shu-Osher: shu_osher.py:26-46; Woodward-Colella: woodward_colella.py:14-27.
- `problems_3d.hpp`, gamma 1.4: sound, a right-going linear wave along x (c = 1), [0, 1) × [0, 1/4)², nx · nx/4 · nx/4 cells, t = 1, one period; simple, the exact isentropic simple wave of hydro_1d along x, the same box, t = 0.2; vortex, Yee's isentropic vortex (beta 5) in [-5, 5)² × [0, 2.5), nx · nx · nx/4 cells, t = 10, a steady state; tube, periodic Sod, left state in [0.5, 1.5) of [0, 2) × [0, 1/4)², 2 nx · nx/4 · nx/4 cells, t = 0.25.
- `test/exact_riemann.hpp`: Toro 2009, ch. 4 (the pressure function, Newton's method for the star pressure, the sampling). Used for Sod against the exact cell averages, as LUNA's exact_riemann.cell_average (src/luna/hydro/exact_riemann.py:105-113: conservative averages of sub-samples per cell). Pressure function f_K(p) and derivative: eqs. 4.6, 4.7, 4.37.

## Benchmark programs (src/bench/)

### hydro_sedov

The Sedov-Taylor blast of LUNA's `scripts/validation/runs/sedov3d_run.py` on the Voronoi mesh (constants: sedov3d_run.py:19-26, src/luna/ic_generation/sedov3d.py:9-14).

    hydro_sedov <side> <jittered|uniform> [--order=0|1] [--update=cell|face]
                [--limiter=face|cell|none] [--riemann=hllc|hll] [--precision=<name>]
                [--moving] [--bulk=<km/s>] [--no-blast] [--box=fx,fy,fz] [--out=<dir>]
                [--snapshots=10] [--steps=<max>] [--input-order] [--memory] [--gradients]
                [--cold=f] [--watch]

side³ generating points in a 50 pc periodic box (jittered: a lattice moved by up to 0.1 spacings; uniform: uniform random). AREPO units (kpc, 1e10 Msun, km/s), gamma 5/3, n = 1 cm⁻³, T = 1e4 K, 1e51 erg into the cells nearest the centre whose volumes add up closest to LUNA's 8 central cells at the same resolution (8 (L / side)³), as extra pressure (gamma - 1) E / V_deposit over their actual volume V_deposit; run to 1e4 yr.

- `--moving`: the points move with the flow (AREPO-ng's regularisation), the mesh is updated every step.
- `--bulk`: the whole box moves along x at that speed (a moving blast wave).
- `--box`: the box sides as multiples of 50 pc (default 1,1,1; about side³ cells).
- `--out` writes snapshots 0..snapshots at equal times (LUNA's time_bet_snaps = t_end / n_snaps); `--gradients` also writes `<out>/gradients.bin` at the end (six doubles per cell: centroid offset, limited density gradient).
- `--steps` stops after that many steps (timing runs).
- Particles are numbered in Morton order of their positions (`--input-order`: as generated, lattice rows or random).
- `--update`: the update loop, default face at order 1 and cell at order 0 (the faster on the A100, hydro_stage1_2026-10-31.md).
- `--limiter`: `hydro::Limiter`, default face. `--riemann`: HLLC (default, as LUNA) or HLL. `--precision`: the hydro's precision policy (default `grad32`; `double` is double throughout).
- `--cold`: multiplies the background pressure by f, a larger contrast at the blast. `--watch`: a line per step with fallbacks, cells at the floors, smallest rho and P, cells not finite.
- `--no-blast`: no blast, a uniform stream (with `--bulk`), printing its largest deviation from uniform.
- Prints per step phase medians and cell updates per second and a checksum of the final state, positions and volumes (bits), the face sides whose extrapolation fell back to the cell state (total, steps with any, most in one step) and what the mesh updates did. `--memory` adds device memory (Kokkos allocations) per phase.

### hydro_kh2d

Smooth Kelvin-Helmholtz after McNally, Lyra & Passy 2012.

    hydro_kh2d <N> [--moving] [--order=0|1] [--dt=luna|arepo|fixed] [--courant=<c>] [--fixed-dt=<dt>]
               [--riemann=hllc|hll] [--limiter=face|cell|none] [--update=face|cell] [--jitter=<spacings>]
               [--bulk=<u>] [--t-end=<t>] [--steps=<max>] [--repeats=<r>]
               [--phases] [--memory] [--out=<prefix>] [--out-times=t,...] [--stop-after=<s>] [--resume]

The set-up is LUNA's KH-McNally-2D (ic_generation/kh_mcnally.py): the unit box, rho and V_x in two exponential shear layers (eqs. 1, 3; L = 0.025, rho 1 / 2, V_x 0.5 / -0.5), P = 2.5, gamma 5/3, V_y = 0.01 sin 4π x, point values at the generators: N² points on the cell centres, moved by up to `--jitter` / 2 spacings. `--bulk` adds a velocity along x. Particles are numbered along a Morton curve.

M(t) (McNally eqs. 6-9) at t = 0.02 k: s, c and d summed over the cells at their centroids, weighted by volume (on a square lattice LUNA's sums, test_hydro_hard_mcnally_kh.py:52-61); K = max rho V_y² / 2. Observing is not timed.

- `--dt`: LUNA's rule, default Courant 0.8; AREPO's, default 0.3; or a constant. `--steps` stops after that many steps (throughput; no observations).
- Every run is printed, the fastest gives the summary, in M cell updates/s = N² · steps / stepping time; the final state hash must agree between runs. `--phases` fences the phases and prints their rates.
- `--out` writes `<prefix>.txt` (t M K), `<prefix>.bin` (doubles per cell: generator x y, centroid x y, V, rho, rho_vy) and `<prefix>.grad.bin` (doubles per cell: the limited density gradient of the end state, to reconstruct from the centroid). `--out-times=t1,t2,...` writes the same at those multiples of 0.02, as `<prefix>_t<t>_<N>`.
- `--stop-after=<s>` saves the state in `<prefix>.state` and stops at the next 0.02 past s seconds; `--resume` goes on from it: a long run in several jobs, not bit-identical across the restart, the mesh being rebuilt from the generators.

### hydro_1d

LUNA's 1D tests on the 1D Voronoi mesh, timed.

    hydro_1d <problem of problems_1d.hpp> <N> [--moving] [--riemann=hllc|hll] [--order=0|1]
             [--dt=luna|arepo|fixed] [--courant=<c>] [--fixed-dt=<dt>] [--jitter=<spacings>]
             [--limiter=face|cell|none|minmod|vanleer] [--steps=<max>] [--repeats=<r>] [--phases] [--memory]
             [--no-regularise] [--unfused] [--inflow-spacing=<spacings>] [--amp=<a>] [--pad=<length>] [--max-wall=<s>] [--out=<file.npy>] [--series=<name>]

Set-ups are LUNA's (`scripts/luna_1d_run.py` has the file:line of each), gamma 1.4: Sod (Toro 1: (1, 0, 1) | (0.125, 0, 0.1) at 0.5 on [0, 1], t = 0.25, zero-gradient walls); Shu-Osher (left = (3.857143, 2.629369, 10.33333) for x < -4, (1 + 0.2 sin 5x, 0, 1) beyond, on [-5, 5], t = 1.8, zero-gradient walls); Woodward-Colella (rho = 1, u = 0, p = 1000 | 0.01 | 100 split at 0.1 and 0.9 on [0, 1], t = 0.038, reflective walls). Every cell starts from the exact average of the initial state over its own interval (LUNA's: cell averages over its uniform cells; equal on a uniform line). N points on a uniform lattice, or moved by up to `--jitter` / 2 spacings.

- `--dt`: LUNA's rule, the default of a static run (`--courant` default 0.8), AREPO's, the default of a moving run (default 0.4), or a constant `--fixed-dt`; the last step lands on t_end. `--steps` stops after that many steps (throughput).
- The run is made `--repeats` times (default 2) in one process, from the same initial state; every run is printed, and the fastest gives the summary (M cell updates/s = N · steps / wall time of the stepping loop, time steps included). The state hash must be the same in every run. LUNA's rule runs fused, one launch per step (`--unfused`: `time_step()` and `step()` apart). `--phases`, always unfused, fences the phases of a step and prints their rates. `--memory` adds the peak of the device allocations.
- `--limiter`: also `minmod` and `vanleer`, LUNA's other limiters (slopes.py:8, 26), on the one-sided slopes between centroids; 1D only. On a uniform line the default is LUNA's MC.
- `--amp`: the signed amplitude of the linear waves, or the simple wave's.
- `--pad`: widens the line by at least that length on each side, in cells of the same spacing, and closes it periodically: no walls, for a moving tube whose ends cannot run yet (#14). The seam's waves must stay outside the box; N in the .json stays the box's.
- A run stops early, `reached=0`, on a step that is not positive or after `--max-wall` seconds; what it has is written.
- `--inflow-spacing` (moving): the spacing of the cells a wall supplies when gas enters, in initial spacings L / N (default 1; 0: none, the end cell grows). N grows then: the summary gives the initial and final cells, and cell updates are summed step by step.
- `--out` writes an .npy of one row per cell (generator, centroid, V, rho, u, p) in x order, x in the problem's coordinates; with `--series` also `<file.npy>.json` (series, problem, N, steps, wall_s, peak_MB, file) for `scripts/compare_1d.py`.

### hydro_wave3d

3D problems with an exact solution (problems_3d.hpp, above), on the hydro's default paths.

    hydro_wave3d <sound|simple|vortex|tube> <nx> [--moving] [--courant=c]
                 [--jitter=spacings] [--amp=a] [--limiter=face|none] [--t-end=t] [--max-steps=m] [--precision=default|double|grad32|grad16] [--out=file.npy]

Prints steps, GPU seconds of the steps and cell updates per second. `--out`: rows generator xyz, centroid xyz, V, rho, vx, vy, vz, P in box coordinates plus the origin, id order, for `scripts/norm3d.py`.

## Tests (src/test/)

- `hydro_test`: HLL and HLLC on sample states equal LUNA's hll_flux_primitive / hllc_flux_primitive (src/luna/hydro/fluxes.py; values printed by LUNA in float64); equal states give the exact flux, also through a moving face (F(U) - (w.n) U); swapping the states and the normal flips the flux; rotating states and normal rotates the flux (solving along n in box axes equals LUNA's rotated frame). A uniform state stays uniform to rounding on jittered and uniform random meshes, first and second order, both update loops. Mass, momentum and energy are conserved to rounding in the periodic box (a blast plus random perturbations, second order). Two runs give the same bits; the two update loops agree to rounding. Moving mesh: a uniform state with a bulk velocity stays uniform; conservation while the cells move; a blast moving with the box equals the blast at rest (Galilean invariance) while on a static mesh it smears. With float face rows the areas of a cell close only to float rounding, so a uniform state stays uniform to about 1e-7 only. Moving-mesh tests use points on the snapping grid and steps whose displacement v dt is a multiple of the grid step (2⁻³⁰ of the box). The Galilean comparison differs by rounding that grows through the nonlinear blast and the limiter; measured 7e-8 on the CPU.
- `hydro2d_test`: the 2D hydro. Uniform states (at rest or with a bulk velocity) stay uniform on static and moving, lattice and jittered meshes; mass, momentum and energy are conserved to rounding; two runs give the same bits; the per-cell and per-face loops agree to rounding; a static lattice with zero-length faces (ties) gives no NaN. Sound wave: second order on static lattice and jittered meshes and the moving lattice; on the moving jittered mesh, which relaxes by the regularisation, second order too (a single flux at the half step gave order 1.2 there). Sod along an axis against the exact solution (the tube mirrored in a periodic box); Noh's implosion against the exact solution, whose L1 bound outside is 1.1 times the moving mesh's measured 0.31 (11 % of rho there; static 0.16, 5.5 %), a regression bound, not a claim of accuracy; the Noh run is 65536 cells, 1900 steps, about 11× slower under sanitizers. A blast carried by a bulk velocity equals the blast at rest cell by cell on the moving mesh.
- `hydro1d_test`: a wall that gas enters supplies cells: a split conserves to rounding and keeps the state, inflow comes out as a lattice of the set spacing from either wall, Shu-Osher's totals follow the budget through its walls, the same bits twice, on both step paths, mirrored; without flow through a wall no cell is added. Also a uniform state (with a bulk velocity on a periodic line; at rest between reflective walls) stays uniform, static and moving, to rounding; mass, momentum and energy of a rough state are conserved on a periodic line, and mass and energy between reflective walls; two identical moving runs give the same bits; inflow keeps its state; Sod's tube (Toro test 1) against the exact solution's cell averages within LUNA's bounds at 128 and 256 cells (test_hydro_accuracy_shocktube.py:14), on static and moving meshes, uniform and jittered; LUNA's own final states of Sod, Shu-Osher and Woodward-Colella at N = 16 on a static uniform line (values printed by LUNA, walls included), to 1e-12; Shu-Osher on a moving mesh: the regularised points never cross, the unregularised ones do.
- `hydro_wave_test` (CTest `hydro_wave`): L1 of rho, v and P of the cells against the exact state at their centroids, volume weighted, on jittered lattices: the vortex is steady (0.1 spacings, t = 2); the sound wave has returned after one period (0.25 spacings). The bounds are values measured with this code (the same to 3 digits on the CPU and the A100), 10 % above them; the rates sit just below the measured ones, never a nominal 2. rho at rate 2.2 and 1.7 (a single flux at the half step gave 1.1 and 0).
