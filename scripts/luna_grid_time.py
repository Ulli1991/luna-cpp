"""LUNA's Sedov 3D step timed the way hydro_sedov --grid times luna-cpp's (research_notes/fixed_grid_2026-11-19.md, phase 1).

Run with LUNA's environment on a GPU node (never the login node), outside LUNA's tree:
    PYTHONDONTWRITEBYTECODE=1 python luna_grid_time.py <N> <LIMITER> [steps=100] [warm=10] [out=<file.npy>] [kernel=pallas]

The step is LUNA's production one (setup.hydro.create_solver_distributed on one device: shard_map halo, Pallas divergence,
U + dt dU) on LUNA's Sedov-Blast-3D with units (luna_grid_refs.sedov). Each step: dt = LUNA's dt function, then the step, then
block_until_ready: the same timed region as hydro_sedov --grid (the whole step with its time step and one host sync, no output).
Prints the initial state's three values (for hydro_sedov --grid --ic=...), the median step wall time after `warm` steps, and the
state after `steps` steps (out=, float64 .npy) for the agreement check.
"""
import os
import sys
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from luna_grid_refs import sedov  # noqa: E402
from luna.setup.hydro import create_solver_distributed, hydro_kernel_name  # noqa: E402
from luna.utils.grid_params import GridParams  # noqa: E402


def main():
    n, limiter = int(sys.argv[1]), int(sys.argv[2])
    extra = dict(a.split("=", 1) for a in sys.argv[3:])
    steps, warm, kernel = int(extra.get("steps", 100)), int(extra.get("warm", 10)), extra.get("kernel", "pallas")
    cfg, U0 = sedov(n, limiter, kernel)
    cfg.validate()
    centre = n // 2
    print(f"luna_ic: rho={float(U0[0, 0, 0, 0])!r} p_bg={float(U0[4, 0, 0, 0] * (cfg.gamma - 1.0))!r} E_bg={float(U0[4, 0, 0, 0])!r} "
          f"E_blast={float(U0[4, centre, centre, centre])!r}", flush=True)
    devs = np.asarray(jax.devices()[:1]).reshape(1, 1, 1)
    compute_dt, step, sharding = create_solver_distributed(cfg, GridParams.from_config(cfg), Mesh(devs, ("x", "y", "z")))
    U = jax.device_put(jnp.asarray(U0, dtype=jnp.float64), sharding)
    walls, dts = [], []
    for s in range(steps):
        t0 = time.perf_counter()
        dt = compute_dt(U)
        U = jax.block_until_ready(step(U, dt))
        walls.append(time.perf_counter() - t0)
        dts.append(float(dt))
    ms = 1e3 * float(np.median(walls[warm:]))
    cells = n ** 3
    print(f"luna_run: n={n} LIMITER={limiter} kernel={hydro_kernel_name(cfg)} steps={steps} warm={warm} step_wall_ms_median={ms:.4f} "
          f"G_cell_updates_per_s={cells / (ms * 1e-3) / 1e9:.4f} ns_per_cell={ms * 1e6 / cells:.4f} dt_first={dts[0]!r} dt_last={dts[-1]!r}",
          flush=True)
    if "out" in extra:
        np.save(extra["out"], np.asarray(U))


if __name__ == "__main__":
    main()
