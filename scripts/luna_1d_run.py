"""LUNA's 1D tests (scripts/problems_1d.py, every problem of the table) at LUNA's settings, timed.

Run with LUNA's environment, LUNA's src on the path, outside LUNA's tree:
    python luna_1d_run.py <problem> <N> <hllc|hll> <out.npz> [LIMITER=1] [CFL=0.8] [steps=0]
        [kernel=auto] [repeats=2] [series=luna] [max_s=0] [resume=<state.npz>] [eps=1e-6]

Setups as LUNA's tests: Sod = Toro test 1 from the exact cell averages
(tests/test_hydro_accuracy_shocktube.py:96-99, exact_riemann.py:9, 105-113),
BOUNDARY 1, t = 0.25; Shu-Osher = initial_conditions (shu_osher.py:26-46,
exact cell averages), Lx = 10, BOUNDARY 1, t = 1.8 (test_hydro_hard_shu_osher.py:29, 62);
Woodward-Colella = initial_conditions (woodward_colella.py:14-27), BOUNDARY 2,
t = 0.038 (test_hydro_hard_woodward_colella.py:28, 63). gamma 1.4, ORDER 1,
FP64, CFL 0.8 (template_config.py:32), the last step shortened to t_end
(as tests/_helpers.py:186). steps > 0: stop after that many steps (throughput).

The loop is one jitted lax.while_loop around LUNA's compute_dt and step
(create_solver_distributed, setup/hydro.py:140-202, as tests/_helpers.py
evolve): compiled ahead, then timed `repeats` times from the same initial
state; the fastest counts, each is printed. The faces MC and van Leer predict unphysical take the
first-order state (allow_first_order_faces, as LUNA's tests of those limiters). eps: the signed amplitude of the linear waves. kernel: LUNA's hydro_kernel
(auto: Pallas on a GPU, setup/hydro.py hydro_kernel_name). Peak memory:
peak_bytes_in_use of the device. Writes rho, u, p (cell order), steps, wall_s,
and <out>.json for scripts/compare_1d.py. max_s > 0: stop after about that long
(in chunks of steps) and write the state, for a very long reference run;
resume continues one, and the .json is written only when t_end is reached. A run that cannot start or finish (LUNA's true
vacuum: a NaN dt) writes reached = 0 in the .json.
"""
import json
import os
import sys
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from luna.template_config import Config  # noqa: E402
from luna.ic_generation.ic import initial_conditions  # noqa: E402
import problems_1d as pr  # noqa: E402
from luna.setup.hydro import create_solver_distributed, hydro_kernel_name  # noqa: E402
from luna.utils.grid_params import GridParams  # noqa: E402

def main():
    problem, n, solver, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    extra = dict(a.split("=") for a in sys.argv[5:])
    limiter, cfl, max_steps = int(extra.get("LIMITER", 1)), float(extra.get("CFL", 0.8)), int(extra.get("steps", 0))
    repeats, series, kernel = int(extra.get("repeats", 2)), extra.get("series", "luna"), extra.get("kernel", "auto")
    max_s, resume = float(extra.get("max_s", 0)), extra.get("resume", "")
    pb = pr.problem(problem)
    G, t_end = pb["gamma"], pb["t"]
    test_problem = pb.get("luna", "Sod-Shock")
    cfg = Config(SPATIAL_DIMENSION=1, Nx=n, gamma=G, ORDER=1, LIMITER=limiter, riemann_solver=solver, CFL=cfl,
                 enable_x64=True, PLOT_AFTER_SIM=False, enable_checkpointing=False, hydro_kernel=kernel,
                 test_problem=test_problem, Lx=pb["box"][1] - pb["box"][0], BOUNDARY=pb["boundary"], t_end=t_end,
                 allow_first_order_faces=True)
    cfg.validate()
    U0 = pr.initial(problem, n, float(extra.get("eps", pr.EPS)))
    U0 = np.asarray(initial_conditions(cfg)) if U0 is None else U0
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("x",))
    compute_dt, step, sharding = create_solver_distributed(cfg, GridParams.from_config(cfg), mesh)

    def advance(U, t, k, cap):
        def body(s):
            U, t, k = s
            dt = jnp.minimum(compute_dt(U), t_end - t)
            return step(U, dt), jnp.where(dt > 0, t + dt, jnp.nan), k + 1  # a NaN or non-positive dt: t = NaN stops the loop

        def go(s):
            return (s[1] < t_end * (1 - 1e-12)) & (s[2] < cap)

        return jax.lax.while_loop(go, body, (U, t, k))

    t_start, k_start = 0.0, 0
    if resume and os.path.exists(resume):
        d = np.load(resume)
        U0, t_start, k_start = d["U"], float(d["t"]), int(d["k"])
    U_start = jax.device_put(jnp.asarray(U0, dtype=jnp.float64), sharding)
    t_jnp, k_jnp = jnp.float64(t_start), jnp.int32(k_start)
    t0 = time.perf_counter()
    args = (U_start, t_jnp, k_jnp, jnp.int32(max_steps if max_steps > 0 else 2**31 - 1))
    advance = jax.jit(advance).lower(*args).compile()
    print(f"compile_s={time.perf_counter() - t0:.3f}")
    if max_s > 0:  # a very long run: chunks, stop in time, keep the state
        U, t, k = U_start, t_jnp, k_jnp
        begin, size = time.perf_counter(), 1000
        while float(t) < t_end * (1 - 1e-12) and time.perf_counter() - begin < max_s:
            t0 = time.perf_counter()
            U, t, k = jax.block_until_ready(advance(U, t, k, jnp.int32(int(k) + size)))
            size = max(1000, int(size * min(4.0, 30.0 / (time.perf_counter() - t0 + 1e-3))))
        wall = time.perf_counter() - begin
        np.savez(resume, U=np.asarray(U), t=float(t), k=int(k))
    else:
        walls = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            U, t, k = jax.block_until_ready(advance(*args))
            walls.append(time.perf_counter() - t0)
            print(f"run: wall_s={walls[-1]:.4f} M_cell_updates_per_s={n * int(k) / walls[-1] / 1e6:.1f}")
        wall = min(walls)
    U = np.asarray(U)
    rho, u, p = pr.primitive(U, G)
    steps = int(k)
    reached = bool(float(t) >= t_end * (1 - 1e-12) and np.isfinite(U).all())
    stats = jax.devices()[0].memory_stats() or {}
    peak_mb = stats.get("peak_bytes_in_use", 0) / 1e6
    np.savez(out, rho=rho, u=u, p=p, steps=steps, wall_s=wall, t=float(t), N=n, problem=problem, solver=solver,
             limiter=limiter, cfl=cfl, device=str(jax.devices()[0]))
    if reached or max_s == 0 or not np.isfinite(float(t)):
        with open(out + ".json", "w") as meta:
            json.dump(dict(series=series, problem=problem, N=n, steps=steps, wall_s=wall, peak_MB=peak_mb, t=float(t), reached=int(reached),
                           file=os.path.basename(out)), meta)
    print(f"luna: problem={problem} N={n} solver={solver} LIMITER={limiter} CFL={cfl} kernel={hydro_kernel_name(cfg)} "
          f"steps={steps} t={float(t):.6g} wall_s={wall:.4f} cell_updates={n * steps:.4g} "
          f"M_cell_updates_per_s={n * steps / wall / 1e6:.1f} peak_MB={peak_mb:.1f} device={jax.devices()[0]}")


if __name__ == "__main__":
    main()
