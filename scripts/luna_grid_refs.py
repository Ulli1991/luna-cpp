"""LUNA's fixed-grid final states for luna-cpp's grid accuracy checks (research_notes/fixed_grid_2026-11-19.md): Sedov 3D and the 3D
sound wave, stored in luna-reference-solutions/luna/.

Run with LUNA's environment on a GPU node (never the login node), outside LUNA's tree, which stays read only:
    PYTHONDONTWRITEBYTECODE=1 python luna_grid_refs.py <out dir> sedov <N> [LIMITER=0] [kernel=pallas]
    PYTHONDONTWRITEBYTECODE=1 python luna_grid_refs.py <out dir> sound <nx> <amplitude> [LIMITER=1] [kernel=pallas]

sedov: LUNA's Sedov-Blast-3D with units (scripts/validation/runs/sedov3d_run.py: 1e51 erg in the central 2^3 cells of a 50 pc box,
BOUNDARY 1, gamma 5/3, to 1e4 yr), N^3 cells, LIMITER 0 as that run. sound: luna-cpp's `sound` (src/hydro/problems_3d.hpp): a right-going
linear wave along x, gamma 1.4, [0, 1) x [0, 1/4)^2 with nx x nx/4 x nx/4 cells, periodic, one period (t = 1), cell-centre values; MC.
Both: ORDER 1, HLLC, CFL 0.8, FP64, steps dt = min(LUNA's dt, t_end - t) as LUNA's runner. Writes <out>/<name>.npz: U (conserved, the
final state), t, steps, wall_s, and the provenance keys.
"""
import os
import subprocess
import sys
import time
from dataclasses import replace

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from luna.template_config import Config  # noqa: E402
from luna.units import UnitSystem  # noqa: E402
from luna.ic_generation.ic import initial_conditions  # noqa: E402
from luna.setup.hydro import create_solver_distributed, hydro_kernel_name  # noqa: E402
from luna.utils.grid_params import GridParams  # noqa: E402


def sedov(n, limiter, kernel):
    """The Config of sedov3d_run.config(nx=n) with the step settings here."""
    year, kpc, msun = 3.15576e7, 3.08568e21, 1.98892e33
    units = UnitSystem(UnitLength_in_cm=kpc, UnitMass_in_g=1e10 * msun, UnitVelocity_in_cm_per_s=1e5)
    t_end = 1.0e4 * year / (kpc / 1e5)
    cfg = replace(Config(), test_problem="Sedov-Blast-3D", SPATIAL_DIMENSION=3, BOUNDARY=1, Nx=n, Ny=n, Nz=n, Lx=50.0e-3, Ly=50.0e-3,
                  Lz=50.0e-3, gamma=5.0 / 3.0, ORDER=1, LIMITER=limiter, riemann_solver="hllc", CFL=0.8, t_end=t_end, units=units,
                  enable_x64=True, PLOT_AFTER_SIM=False, enable_checkpointing=False, hydro_kernel=kernel)
    return cfg, np.array(initial_conditions(cfg), dtype=np.float64)


def sound(nx, amp, limiter, kernel):
    """luna-cpp's sound (problems_3d.hpp wave_initial): rho 1 + w, v_x w, P (1 + gamma w) / gamma, w = amp sin(2 pi x)."""
    G = 1.4
    cfg = replace(Config(), test_problem="Soundwave-3D", SPATIAL_DIMENSION=3, BOUNDARY=0, Nx=nx, Ny=nx // 4, Nz=nx // 4, Lx=1.0, Ly=0.25,
                  Lz=0.25, gamma=G, ORDER=1, LIMITER=limiter, riemann_solver="hllc", CFL=0.8, t_end=1.0, enable_x64=True,
                  PLOT_AFTER_SIM=False, enable_checkpointing=False, hydro_kernel=kernel)
    x = (np.arange(nx) + 0.5) / nx
    w = amp * np.sin(6.283185307179586 * x)
    shape = (nx, nx // 4, nx // 4)
    rho = np.broadcast_to((1 + w)[:, None, None], shape)
    vx = np.broadcast_to(w[:, None, None], shape)
    P = np.broadcast_to(((1 + G * w) / G)[:, None, None], shape)
    U = np.stack([rho, rho * vx, np.zeros(shape), np.zeros(shape), P / (G - 1) + 0.5 * rho * vx * vx])
    return cfg, U


def main():
    out, problem = sys.argv[1], sys.argv[2]
    rest = [a for a in sys.argv[3:] if "=" not in a]
    extra = dict(a.split("=") for a in sys.argv[3:] if "=" in a)
    kernel = extra.get("kernel", "pallas")
    if problem == "sedov":
        n = int(rest[0])
        limiter = int(extra.get("LIMITER", 0))
        cfg, U0 = sedov(n, limiter, kernel)
        name = f"sedov3d_{n}"
    else:
        n, amp = int(rest[0]), float(rest[1])
        limiter = int(extra.get("LIMITER", 1))
        cfg, U0 = sound(n, amp, limiter, kernel)
        name = f"sound{rest[1]}_{n}"
    cfg.validate()
    devs = np.asarray(jax.devices()[:1]).reshape(1, 1, 1)
    compute_dt, step, sharding = create_solver_distributed(cfg, GridParams.from_config(cfg), Mesh(devs, ("x", "y", "z")))
    t_end = float(cfg.t_end)

    def advance(U, t, k):
        def body(s):
            U, t, k = s
            dt = jnp.minimum(compute_dt(U), t_end - t)
            return step(U, dt), t + dt, k + 1

        return jax.lax.while_loop(lambda s: s[1] < t_end * (1 - 1e-13), body, (U, t, k))

    start = jax.device_put(jnp.asarray(U0, dtype=jnp.float64), sharding)
    compiled = jax.jit(advance).lower(start, jnp.float64(0), jnp.int32(0)).compile()
    t0 = time.perf_counter()
    U, t, k = jax.block_until_ready(compiled(start, jnp.float64(0), jnp.int32(0)))
    wall = time.perf_counter() - t0
    luna = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.modules["luna"].__file__))))
    commit = subprocess.run(["git", "-C", luna, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", luna, "status", "--porcelain", "src"], capture_output=True, text=True).stdout.strip()
    cells = int(np.prod(U0.shape[1:]))
    provenance = (f"LUNA {commit}{'+dirty-src' if dirty else ''}, jax {jax.__version__}, {jax.devices()[0].device_kind}, host "
                  f"{os.uname().nodename}, SLURM {os.environ.get('SLURM_JOB_ID', '-')}, kernel {hydro_kernel_name(cfg)}; luna-cpp "
                  f"scripts/luna_grid_refs.py {' '.join(sys.argv[2:])}")
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, name + ".npz"), U=np.asarray(U), t=float(t), steps=int(k), wall_s=wall, problem=problem, limiter=limiter,
             cfl=0.8, riemann="hllc", provenance=provenance)
    print(f"{name}: t={float(t)!r} steps={int(k)} wall_s={wall:.3f} G_cell_updates_per_s={cells * int(k) / wall / 1e9:.4f} "
          f"kernel={hydro_kernel_name(cfg)} nan={int(np.isnan(np.asarray(U)).sum())}", flush=True)


if __name__ == "__main__":
    main()
