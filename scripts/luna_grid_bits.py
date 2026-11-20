"""LUNA's fixed-grid step on deterministic states: the references of luna-cpp's grid bits test (research_notes/fixed_grid_2026-11-19.md).

Run with LUNA's environment on a GPU node (never the login node), outside LUNA's tree, which stays read only:
    PYTHONDONTWRITEBYTECODE=1 python luna_grid_bits.py <out dir> <ci|full> [kernel=pallas]

Each case gets a directory <out>/<case>/ with meta.txt (one `key value` per line), dt.npy (the time steps LUNA's dt function gave, one
per step) and U_<s>.npy (the conserved state after s steps, U_0 the initial one, (nvar, Nx[, Ny[, Nz]]), float64), and <out>/<case>.npz with the same arrays
and the provenance. ci: the small cases of src/test/data/luna_grid (10 steps, U_10 only); full: the luna-reference-solutions set.

The initial state is a formula of the integer cell indices whose operations are all exact (sums of multiples of 1/16, and one
correctly rounded product for the floors case), so grid_test rebuilds it bit for bit (initial_primitives in grid_test.cpp).
LUNA's step: integrators.create_muscl_hancock_modular_step (the production step: halo, Pallas or XLA divergence, U + dt dU), its dt
function before every step.
"""
import os
import subprocess
import sys

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from luna.hydro.conversions import prim_to_cons  # noqa: E402
from luna.time_integration.integrators import create_muscl_hancock_modular_step  # noqa: E402

GAMMA = 1.4
CFL = 0.8
LENGTHS = (1.0, 0.75, 1.3)  # different spacings per axis, so a swapped axis shows
SMALLR, SMALLC = 1e-4, 1e-3  # the floors case's (large enough to act)


def initial_primitives(shape, floors):
    """(rho, v.., P) of every cell: noise of period 17 in i 7 + j 13 + k 29 and a jump at i = Nx / 2 (dense, high pressure on the left).
    floors: the right half 1e-6 as dense and 1e-8 as hot, the halves moving apart at 1.5 (a strong rarefaction)."""
    dim = len(shape)
    idx = np.indices(shape)
    base = sum((7, 13, 29)[a] * idx[a] for a in range(dim))

    def h(s):
        return ((base + s) % 17) / 16.0

    left = idx[0] < shape[0] // 2
    rho = 1.0 + h(1) + np.where(left, 1.0, 0.0)
    v = [0.5 * (h(3 + a) - 0.5) for a in range(dim)]
    P = 1.0 + h(7) + np.where(left, 2.0, 0.0)
    if floors:
        rho = np.where(left, rho, 1e-6 * rho)
        P = np.where(left, P, 1e-8 * P)
        v[0] = np.where(left, -1.5, 1.5)
    return np.stack([rho] + v + [P])


def cases(mode):
    """(name, shape, limiter, solver, boundary, floors, allow_first_order, saved steps)."""
    if mode == "ci":
        shapes = {1: (64,), 2: (16, 24), 3: (8, 8, 12)}
        out = []
        for dim, shape in shapes.items():
            variants = [("mc", 1, "hllc", 1, False, False), ("minmod", 0, "hllc", 1, False, False), ("vanleer", 2, "hllc", 1, False, False),
                        ("hll", 1, "hll", 1, False, False), ("periodic", 1, "hllc", 0, False, False),
                        ("reflective", 1, "hllc", 2, False, False), ("floors", 1, "hllc", 1, True, False),
                        ("floors-first-order", 1, "hllc", 1, True, True)]
            if dim == 3:
                variants.append(("diode", 1, "hllc", 4, False, False))
            for name, lim, solver, bc, floors, allow in variants:
                out.append((f"{dim}d-{name}", shape, lim, solver, bc, floors, allow, (10,)))
        return out
    return [("1d-mc-4096", (4096,), 1, "hllc", 1, False, False, (1, 10, 100)),
            ("2d-mc-128x192", (128, 192), 1, "hllc", 1, False, False, (1, 10, 100)),
            ("3d-mc-32x48x64", (32, 48, 64), 1, "hllc", 1, False, False, (1, 10, 100)),
            ("3d-minmod-32x48x64", (32, 48, 64), 0, "hllc", 1, False, False, (1, 10, 100)),
            ("3d-mc-128", (128, 128, 128), 1, "hllc", 1, False, False, (100,))]


def run(case, out, kernel, commit):
    name, shape, limiter, solver, boundary, floors, allow, saved = case
    dim = len(shape)
    dx = [LENGTHS[a] / shape[a] for a in range(dim)] + [1.0] * (3 - dim)
    W = initial_primitives(shape, floors)
    U = prim_to_cons(jnp.asarray(W), GAMMA)
    dt_fn, step = create_muscl_hancock_modular_step(
        None, GAMMA, 1, limiter, dx[0], dx[1], dx[2], 1.0 / dx[0], 1.0 / dx[1], 1.0 / dx[2], boundary, dim, CFL,
        riemann_solver=solver, kernel=kernel, floors=(SMALLR, SMALLC) if floors else None, allow_first_order=allow)
    # U_0 is the formula's state (initial_primitives through prim_to_cons); kept unless the grid has more than 2^20 cells.
    first = (0,) if np.prod(shape) <= 2**20 else ()
    arrays, dts = ({"U_0": np.asarray(U)} if first else {}), []
    for s in range(1, max(saved) + 1):
        dt = dt_fn(U)
        U = step(U, dt)
        dts.append(float(dt))
        if s in saved:
            arrays[f"U_{s}"] = np.asarray(U)
    arrays["dt"] = np.asarray(dts)
    nan = int(np.isnan(arrays[f"U_{max(saved)}"]).sum())
    meta = {"case": name, "dim": dim, "shape": " ".join(map(str, shape)), "lengths": " ".join(repr(x) for x in LENGTHS[:dim]),
            "gamma": repr(GAMMA), "cfl": repr(CFL), "limiter": limiter, "riemann": solver, "boundary": boundary, "floors": int(floors),
            "smallr": repr(SMALLR), "smallc": repr(SMALLC), "allow_first_order": int(allow), "steps": " ".join(map(str, saved)),
            "kernel": kernel, "luna_commit": commit, "jax": jax.__version__, "device": jax.devices()[0].device_kind, "nan_cells": nan}
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "meta.txt"), "w") as f:
        for k, v in meta.items():
            f.write(f"{k} {v}\n")
    np.save(os.path.join(d, "dt.npy"), arrays["dt"])
    for s in first + tuple(saved):
        np.save(os.path.join(d, f"U_{s}.npy"), arrays[f"U_{s}"])
    np.savez(os.path.join(out, name + ".npz"), **arrays, **{"meta_" + k: str(v) for k, v in meta.items()})
    print(f"{name}: steps={max(saved)} dt0={dts[0]!r} dt_last={dts[-1]!r} nan_cells={nan}", flush=True)


def main():
    out, mode = sys.argv[1], sys.argv[2]
    kernel = dict(a.split("=") for a in sys.argv[3:]).get("kernel", "pallas")
    luna = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.modules["luna"].__file__))))
    commit = subprocess.run(["git", "-C", luna, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", luna, "status", "--porcelain", "src"], capture_output=True, text=True).stdout.strip()
    commit += "+dirty-src" if dirty else ""
    os.makedirs(out, exist_ok=True)
    for case in cases(mode):
        run(case, out, kernel, commit)


if __name__ == "__main__":
    main()
