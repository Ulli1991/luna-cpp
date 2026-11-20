"""LUNA's JAX derivatives of `steps` steps of its 1D hydro at a fixed dt, for hydro_derivatives_test (luna-cpp's hand-written ones).

Run with LUNA's environment, LUNA's src importable, outside LUNA's tree:
    python luna_derivatives.py <out.hpp> <problem>:<N>:<LIMITER>:<steps> ...

Per case: the initial state U0 (exact cell averages, scripts/problems_1d.py), dt (LUNA's compute_dt of U0 at CFL 0.8, then fixed),
the direction dU and the weights w (deterministic formulas, printed), LUNA's final state, J dU (jax.jvp) and J^T w (jax.vjp) of
U0 -> U after the steps. HLLC, FP64, unphysical Hancock faces take the cell's state (allow_first_order_faces, as luna_1d_run.py).
LIMITER: 0 minmod, 1 MC, 2 van Leer. Writes a C++ header of the values (columns rho, momentum, energy per volume, as LUNA's U).
"""
import os
import sys

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from luna.template_config import Config  # noqa: E402
from luna.setup.hydro import create_solver_distributed  # noqa: E402
from luna.utils.grid_params import GridParams  # noqa: E402
import problems_1d as pr  # noqa: E402


def case(problem, n, limiter, steps, noise=0.0):
    pb = pr.problem(problem)
    cfg = Config(SPATIAL_DIMENSION=1, Nx=n, gamma=pb["gamma"], ORDER=1, LIMITER=limiter, riemann_solver="hllc", CFL=0.8,
                 enable_x64=True, PLOT_AFTER_SIM=False, enable_checkpointing=False, hydro_kernel="xla",
                 test_problem=pb.get("luna", "Sod-Shock"), Lx=pb["box"][1] - pb["box"][0], BOUNDARY=pb["boundary"], t_end=pb["t"],
                 allow_first_order_faces=True)
    cfg.validate()
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("x",))
    compute_dt, step, sharding = create_solver_distributed(cfg, GridParams.from_config(cfg), mesh)
    U0 = np.asarray(pr.initial(problem, n), dtype=np.float64)
    if noise > 0:  # no cell exactly flat (a flat state is where MC has no derivative and the two codes' conventions differ)
        U0 = U0 + noise * np.sin(2.3 * np.arange(n)[None, :] + np.arange(3)[:, None]) * (np.abs(U0) + 1.0)  # additive: momentum 0 too
    U0 = jnp.asarray(U0)
    dt = float(compute_dt(U0))
    i = np.arange(n)[None, :]
    v = np.arange(3)[:, None]
    dU = jnp.asarray(np.sin(0.7 * i + 1.3 * v + 0.2) * (np.abs(np.asarray(U0)) + 0.1))
    w = jnp.asarray(np.cos(0.9 * i + 0.4 * v + 0.1))

    def run(U):
        for _ in range(steps):
            U = step(U, dt)
        return U

    U, JdU = jax.jit(lambda U, d: jax.jvp(run, (U,), (d,)))(U0, dU)
    _, pullback = jax.vjp(jax.jit(run), U0)
    (JTw,) = pullback(w)
    return dict(U0=np.asarray(U0), dt=dt, dU=np.asarray(dU), w=np.asarray(w), U=np.asarray(U), JdU=np.asarray(JdU), JTw=np.asarray(JTw))


def array(name, a):
    rows = ",\n".join("    {" + ", ".join(repr(float(x)) for x in a[:, k]) + "}" for k in range(a.shape[1]))
    return f"const double {name}[{a.shape[1]}][3] = {{\n{rows}}};\n"


def main(argv):
    out = argv[1]
    lines = ["#pragma once\n",
             "// LUNA's JAX derivatives of its 1D hydro (HLLC, FP64, CPU, XLA kernel), K steps at LUNA's dt of U0 (CFL 0.8), fixed:",
             "// J dU (jax.jvp) and J^T w (jax.vjp) of U0 -> U; U per volume (rho, momentum, energy) per cell. Written by scripts/luna_derivatives.py.",
             "// Reference for hydro_derivatives_test (FiniteVolume<1>::step_tangent, step_adjoint).\n"]
    names = []
    for spec in argv[2:]:
        parts = spec.split(":")
        problem, n, limiter, steps = parts[:4]
        noise = float(parts[4]) if len(parts) > 4 else 0.0
        n, limiter, steps = int(n), int(limiter), int(steps)
        d = case(problem, n, limiter, steps, noise)
        tag = f"luna_d_{problem.replace('-', '_')}_{n}_l{limiter}" + ("_noisy" if noise > 0 else "")
        names.append(tag)
        lines.append(f"// {problem}, N = {n}, LIMITER = {limiter}, {steps} steps" + (f", U0 + {noise} sin(2.3 i + v) (|U0| + 1)" if noise > 0 else ""))
        lines.append(f"namespace {tag} {{")
        lines.append(f"const int n = {n}, limiter = {limiter}, steps = {steps};")
        lines.append(f"const double dt = {d['dt']!r};")
        for key in ("U0", "dU", "w", "U", "JdU", "JTw"):
            lines.append(array(key, d[key]))
        lines.append("}\n")
        print(f"{spec}: dt={d['dt']:.17g} |JdU|={np.abs(d['JdU']).max():.3e} |JTw|={np.abs(d['JTw']).max():.3e}")
    with open(out, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main(sys.argv))
