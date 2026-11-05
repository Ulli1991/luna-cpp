"""Sedov 3D: field L1 of a hydro_sedov snapshot against the Sedov-Taylor solution.

    python scripts/sedov_field_l1.py <run dir> [lattice points per R_s, default 200]

The piecewise-constant field of the Voronoi cells (nearest generator), never sampled at generators, on a fixed lattice of
the ball r < 1.5 R_s around the blast: mean |f_h - f| of rho / rho0, v_r / max v_r, P / max P. Needs LUNA on PYTHONPATH.
"""
import glob
import sys

import h5py
import numpy as np
from luna.hydro import exact_sedov
from scipy.spatial import cKDTree

KPC, MSUN = 3.08568e21, 1.98892e33
UNIT_RHO = 1e10 * MSUN / KPC ** 3
UNIT_V = 1e5
RHO0 = 1.6726219e-24 / UNIT_RHO
E0 = 1e51 / (1e10 * MSUN * UNIT_V ** 2)
GAMMA = 5.0 / 3.0

root = sys.argv[1]
per_radius = int(sys.argv[2]) if len(sys.argv) > 2 else 200
with h5py.File(sorted(glob.glob(root + "/snapdir_*/*.hdf5"))[-1], "r") as f:
    c = f["cells"]
    L, t = float(c.attrs["Lx"]), float(c.attrs["time"])
    pos = np.stack([c["x"][...], c["y"][...], c["z"][...]], axis=1)
    rho, P = c["density"][...], c["pressure"][...]
    vel = np.stack([c["velocity_x"][...], c["velocity_y"][...], c["velocity_z"][...]], axis=1)
R = exact_sedov.solution(np.array([0.0]), t, E0, RHO0, GAMMA)[3]
axis = (np.arange(-3 * per_radius // 2, 3 * per_radius // 2) + 0.5) * R / per_radius
tree = cKDTree(pos % L, boxsize=L)
total, err = 0, np.zeros(3)
for x0 in np.array_split(axis, max(1, axis.size // 16)):
    X, Y, Z = np.meshgrid(x0, axis, axis, indexing="ij")
    keep = X ** 2 + Y ** 2 + Z ** 2 < (1.5 * R) ** 2
    d = np.stack([X[keep], Y[keep], Z[keep]], axis=1)
    r = np.sqrt((d ** 2).sum(axis=1))
    ids = tree.query((d + 0.5 * L) % L, workers=-1)[1]
    rho_a, vr_a, P_a, _ = exact_sedov.solution(r, t, E0, RHO0, GAMMA)
    vmax, pmax = exact_sedov.solution(np.array([R * (1 - 1e-9)]), t, E0, RHO0, GAMMA)[1:3]
    vr = (vel[ids] * d).sum(axis=1) / r
    err += [np.abs(rho[ids] - rho_a).sum() / RHO0, np.abs(vr - vr_a).sum() / vmax[0], np.abs(P[ids] - P_a).sum() / pmax[0]]
    total += r.size
print(f"{root} field L1 rho/rho0 {err[0] / total:.5f} v_r/vmax {err[1] / total:.5f} P/Pmax {err[2] / total:.5f}  samples {total}  R_s {R:.6g}")
