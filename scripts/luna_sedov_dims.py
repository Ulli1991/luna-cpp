"""LUNA on the centred blast of hydro_1d and hydro_kh2d --problem=sedov, as its Sedov-Blast-3D without units: rho 1, P 1e-2, energy 0.5 in the central 2^D cells, gas at rest.

    cd <LUNA> && python <this file> <dim> <N> <output dir> [gamma=5/3] [max_steps=200] [Config field=value ...]

LUNA has no Sedov in 1D or 2D: the state goes to luna.launch as a user state. ORDER 1, MC, HLLC, Pallas, FP64.
"""
import sys
from dataclasses import fields, replace

import numpy as np

sys.path.insert(0, ".")
from luna.setup.cli import launch  # noqa: E402
from luna.template_config import Config  # noqa: E402


def main():
    dim, n, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    gamma = 5.0 / 3.0 if dim != 1 else 1.4
    cfg = replace(Config(), test_problem="Sod-Shock" if dim == 1 else "Kelvin-Helmholtz", SPATIAL_DIMENSION=dim, BOUNDARY=1, Nx=n, Ny=n if dim > 1 else 1, Nz=1, Lx=1.0, Ly=1.0, Lz=1.0,
                  gamma=gamma, ORDER=1, LIMITER=1, riemann_solver="hllc", enable_x64=True, hydro_kernel="pallas", t_end=1.0, time_bet_snaps=1.0, PLOT_AFTER_SIM=False, output_dir=out,
                  clear_output_dir=True, enable_checkpointing=False, write_snapshots=False, max_steps=200)
    types = {f.name: f.type for f in fields(cfg)}
    over = {}
    for arg in sys.argv[4:]:
        key, value = arg.split("=", 1)
        over[key] = int(value) if types[key] in (int, "int") else float(value) if types[key] in (float, "float") else value
    cfg = replace(cfg, **over)
    h = 1.0 / n
    shape = (n,) if dim == 1 else (n, n)
    U = np.zeros((dim + 2,) + shape)
    U[0] = 1.0
    U[-1] = 1e-2 / (gamma - 1)
    centre = (n // 2 - 1, n // 2)
    if dim == 1:
        U[-1, centre[0]:centre[1] + 1] += 0.5 / (2 * h)
    else:
        U[-1, centre[0]:centre[1] + 1, centre[0]:centre[1] + 1] += 0.5 / (4 * h * h)
    launch(cfg, U0=U)


if __name__ == "__main__":
    main()
