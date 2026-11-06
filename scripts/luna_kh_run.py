"""LUNA's KH-McNally-2D at LUNA's test settings, timed; M(t) at t = 0.02 k.

Run with LUNA's environment, LUNA's src on the path, outside LUNA's tree:
    python luna_kh_run.py <N> <out.npz> [LIMITER=1] [CFL=0.8] [ORDER=1] [solver=hllc] [kernel=auto] [bulk=0] [steps=0]
        [repeats=1] [t_end=1.5] [fixed_dt=0] [check=1] [full=0]

bulk adds a velocity along x; steps > 0 stops after that many steps (throughput). Writes t, M, K, rho, my, steps, wall_s.
"""
import sys
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from luna.template_config import Config  # noqa: E402
from luna.ic_generation.ic import initial_conditions  # noqa: E402
from luna.setup.hydro import create_solver_distributed, hydro_kernel_name  # noqa: E402
from luna.utils.grid_params import GridParams  # noqa: E402

G = 5.0 / 3.0


def mode_amplitude(U):
    """M of McNally eqs. 6-9 (test_hydro_hard_mcnally_kh.py:52-61)."""
    U = np.asarray(U, np.float64)
    vy = U[2] / U[0]
    n, m = U.shape[1], U.shape[2]
    x, y = (np.arange(n) + 0.5) / n, (np.arange(m) + 0.5) / m
    w = np.where(y < 0.5, np.exp(-4 * np.pi * np.abs(y - 0.25)), np.exp(-4 * np.pi * np.abs((1.0 - y) - 0.25)))
    s = np.sin(4 * np.pi * x) @ vy @ w
    c = np.cos(4 * np.pi * x) @ vy @ w
    return float(2.0 * np.hypot(s / (n * w.sum()), c / (n * w.sum())))


def coarsen(a, m):
    n = a.shape[0]
    return a.reshape(m, n // m, m, n // m).mean(axis=(1, 3))


def main():
    n, out = int(sys.argv[1]), sys.argv[2]
    extra = dict(a.split("=") for a in sys.argv[3:])
    limiter, cfl, order = int(extra.get("LIMITER", 1)), float(extra.get("CFL", 0.8)), int(extra.get("ORDER", 1))
    solver, kernel, bulk = extra.get("solver", "hllc"), extra.get("kernel", "auto"), float(extra.get("bulk", 0))
    max_steps, repeats, t_end = int(extra.get("steps", 0)), int(extra.get("repeats", 1)), float(extra.get("t_end", 1.5))
    fixed_dt = float(extra.get("fixed_dt", 0))
    cfg = Config(test_problem="KH-McNally-2D", SPATIAL_DIMENSION=2, Nx=n, Ny=n, Lx=1.0, Ly=1.0, BOUNDARY=0, gamma=G,
                 ORDER=order, LIMITER=limiter, riemann_solver=solver, CFL=cfl, t_end=t_end, enable_x64=True,
                 PLOT_AFTER_SIM=False, enable_checkpointing=False, hydro_kernel=kernel)
    cfg.validate()
    U0 = np.array(initial_conditions(cfg), dtype=np.float64)
    if bulk != 0:
        U0[3] += U0[1] * bulk + 0.5 * U0[0] * bulk * bulk
        U0[1] += U0[0] * bulk
    devs = np.asarray(jax.devices()[:1]).reshape(1, 1)
    compute_dt, step, sharding = create_solver_distributed(cfg, GridParams.from_config(cfg), Mesh(devs, ("x", "y")))

    def advance(U, t, t1, k, cap):
        def body(s):
            U, t, k = s
            dt = jnp.minimum(jnp.float64(fixed_dt) if fixed_dt > 0 else compute_dt(U), t1 - t)
            return step(U, dt), t + dt, k + 1

        def go(s):
            return (s[1] < t1 * (1 - 1e-13)) & (s[2] < cap)

        return jax.lax.while_loop(go, body, (U, t, k))

    start = jax.device_put(jnp.asarray(U0, dtype=jnp.float64), sharding)
    cap = jnp.int32(max_steps if max_steps > 0 else 2**31 - 1)
    t0 = time.perf_counter()
    compiled = jax.jit(advance).lower(start, jnp.float64(0), jnp.float64(0.02), jnp.int32(0), cap).compile()
    print(f"compile_s={time.perf_counter() - t0:.3f}")
    walls = []
    for _ in range(repeats):
        U, t, k = start, jnp.float64(0), jnp.int32(0)
        M, K, T, wall = [mode_amplitude(U0)], [float((0.5 * U0[2] ** 2 / U0[0]).max())], [0.0], 0.0
        outputs = 1 if max_steps > 0 else int(round(t_end / 0.02))
        for j in range(1, outputs + 1):
            t1 = jnp.float64(1e300 if max_steps > 0 else j * 0.02)
            t0 = time.perf_counter()
            U, t, k = jax.block_until_ready(compiled(U, t, t1, k, cap))
            wall += time.perf_counter() - t0
            if max_steps == 0:
                Uh = np.asarray(U)
                M.append(mode_amplitude(Uh))
                K.append(float((0.5 * Uh[2] ** 2 / Uh[0]).max()))
                T.append(float(t))
        walls.append(wall)
        print(f"run: wall_s={wall:.4f} steps={int(k)} M_cell_updates_per_s={n * n * int(k) / wall / 1e6:.1f}")
    wall = min(walls)
    steps = int(k)
    Uh = np.asarray(U)
    m = n if int(extra.get("full", 0)) else min(n, 256)
    stats = jax.devices()[0].memory_stats() or {}
    peak = stats.get("peak_bytes_in_use", 0)
    np.savez(out, t=np.array(T), M=np.array(M), K=np.array(K), rho=coarsen(Uh[0], m), my=coarsen(Uh[2], m), steps=steps,
             wall_s=wall, N=n, bulk=bulk, limiter=limiter, cfl=cfl, solver=solver)
    print(f"luna: N={n} order={order} solver={solver} LIMITER={limiter} CFL={cfl} bulk={bulk} kernel={hydro_kernel_name(cfg)} "
          f"steps={steps} wall_s={wall:.4f} M_cell_updates_per_s={n * n * steps / wall / 1e6:.1f} "
          f"peak_B_per_cell={peak / n / n:.1f} M_final={M[-1]:.8e} device={jax.devices()[0]}")


if __name__ == "__main__":
    main()
