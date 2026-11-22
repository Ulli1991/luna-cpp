"""Sedov 3D at the shock: volume-weighted shell profiles of rho, P and v_r against the Sedov-Taylor solution, the shock
radius, the peak density, the shock width in cells, and the field L1 split into the interior, the shock zone and the rest.

    python scripts/sedov_shock_profile.py <run dir> [<run dir> ...]   (needs LUNA on PYTHONPATH)

Shells are h/4 wide (h = L / N^(1/3), the mean cell spacing). Shock radius: the outermost shell where rho is half-way
between its peak and rho0. Width: the distance between 10 % and 90 % of the outer rise (rho0 to the peak), in h.
Field L1: as scripts/sedov_field_l1.py (nearest cell on a lattice of 200 points per R_s), split by r / R_s.
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


def load(root):
    with h5py.File(sorted(glob.glob(root + "/snapdir_*/*.hdf5"))[-1], "r") as f:
        c = f["cells"]
        L, t = float(c.attrs["Lx"]), float(c.attrs["time"])
        pos = np.stack([c["x"][...], c["y"][...], c["z"][...]], axis=1)
        vel = np.stack([c["velocity_x"][...], c["velocity_y"][...], c["velocity_z"][...]], axis=1)
        return L, t, pos, vel, c["density"][...], c["pressure"][...], c["volume"][...]


def outer_crossing(r, y, level, i_peak):
    """Outermost radius beyond the peak where y falls through level (linear between shells)."""
    for k in range(i_peak, len(y) - 1):
        if y[k] >= level > y[k + 1]:
            return r[k] + (y[k] - level) / (y[k] - y[k + 1]) * (r[k + 1] - r[k])
    return np.nan


for root in sys.argv[1:]:
    L, t, pos, vel, rho, P, vol = load(root)
    h = L / round(len(rho) ** (1 / 3))
    d = pos - 0.5 * L
    r = np.sqrt((d ** 2).sum(axis=1))
    vr = (vel * d).sum(axis=1) / np.maximum(r, 1e-300)
    R = exact_sedov.solution(np.array([0.0]), t, E0, RHO0, GAMMA)[3]
    edges = np.arange(0, 1.5 * R + h / 4, h / 4)
    idx = np.digitize(r, edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    w = np.bincount(idx[ok], vol[ok], len(edges) - 1)
    shell = lambda q: np.bincount(idx[ok], (q * vol)[ok], len(edges) - 1) / np.maximum(w, 1e-300)
    rc = 0.5 * (edges[1:] + edges[:-1])
    s_rho, s_P, s_vr = shell(rho), shell(P), shell(vr)
    a_rho, a_vr, a_P, _ = exact_sedov.solution(rc, t, E0, RHO0, GAMMA)
    i_peak = int(np.argmax(s_rho))
    peak = s_rho[i_peak]
    r_half = outer_crossing(rc, s_rho, 0.5 * (peak + RHO0), i_peak)
    r10 = outer_crossing(rc, s_rho, RHO0 + 0.1 * (peak - RHO0), i_peak)
    r90 = outer_crossing(rc, s_rho, RHO0 + 0.9 * (peak - RHO0), i_peak)
    rho_peak_exact = (GAMMA + 1) / (GAMMA - 1) * RHO0
    print(f"{root}: t={t:.6g} R_s(exact)={R:.6g} h={h:.4g} ({R / h:.1f} cells per R_s)")
    print(f"  shells: rho peak {peak / RHO0:.3f} rho0 at r/R_s={rc[i_peak] / R:.4f} (exact {rho_peak_exact / RHO0:.3f} at 1);"
          f" largest cell rho {rho.max() / RHO0:.3f} rho0; half-height radius r/R_s={r_half / R:.4f};"
          f" width 10-90 % {(r10 - r90) / h:.2f} h")
    pk = exact_sedov.solution(np.array([R * (1 - 1e-9)]), t, E0, RHO0, GAMMA)
    print(f"  shells at the peak: P {s_P[i_peak] / pk[2][0]:.3f} P_s, v_r {s_vr[i_peak] / pk[1][0]:.3f} v_s;"
          f" largest shell P {s_P.max() / pk[2][0]:.3f} P_s, v_r {s_vr.max() / pk[1][0]:.3f} v_s")
    # Field L1 as sedov_field_l1.py, split by r / R_s.
    per_radius = 200
    axis = (np.arange(-3 * per_radius // 2, 3 * per_radius // 2) + 0.5) * R / per_radius
    tree = cKDTree(pos % L, boxsize=L)
    zones = [(0.0, 0.9), (0.9, 1.1), (1.1, 1.5)]
    err = np.zeros((3, 3))
    total = 0
    vmax, pmax = pk[1][0], pk[2][0]
    for x0 in np.array_split(axis, max(1, axis.size // 16)):
        X, Y, Z = np.meshgrid(x0, axis, axis, indexing="ij")
        keep = X ** 2 + Y ** 2 + Z ** 2 < (1.5 * R) ** 2
        q = np.stack([X[keep], Y[keep], Z[keep]], axis=1)
        rq = np.sqrt((q ** 2).sum(axis=1))
        ids = tree.query((q + 0.5 * L) % L, workers=-1)[1]
        rho_a, vr_a, P_a, _ = exact_sedov.solution(rq, t, E0, RHO0, GAMMA)
        vrq = (vel[ids] * q).sum(axis=1) / rq
        e = np.stack([np.abs(rho[ids] - rho_a) / RHO0, np.abs(vrq - vr_a) / vmax, np.abs(P[ids] - P_a) / pmax], axis=1)
        for z, (lo, hi) in enumerate(zones):
            m = (rq >= lo * R) & (rq < hi * R)
            err[z] += e[m].sum(axis=0)
        total += rq.size
    for z, (lo, hi) in enumerate(zones):
        print(f"  field L1 share r/R_s in [{lo}, {hi}): rho {err[z, 0] / total:.5f} v_r {err[z, 1] / total:.5f} P {err[z, 2] / total:.5f}")
    print(f"  field L1 total: rho {err[:, 0].sum() / total:.5f} v_r {err[:, 1].sum() / total:.5f} P {err[:, 2].sum() / total:.5f}")
    # The profile near the shock, every second shell from 0.8 to 1.15 R_s.
    near = np.where((rc > 0.8 * R) & (rc < 1.15 * R))[0][::2]
    print("  r/R_s   rho/rho0 (exact)   P/P_s (exact)   v_r/v_s (exact)")
    for k in near:
        print(f"  {rc[k] / R:.3f}   {s_rho[k] / RHO0:7.3f} ({a_rho[k] / RHO0:6.3f})   {s_P[k] / pmax:6.3f} ({a_P[k] / pmax:6.3f})"
              f"   {s_vr[k] / vmax:6.3f} ({a_vr[k] / vmax:6.3f})")
    # The interior, where the exact solution has rho -> 0 and c -> infinity: shells every 0.05 R_s from 0.05 to 0.7.
    print("  interior: r/R_s   rho/rho0 (exact)   P/P_s (exact)   c [km/s] (exact)")
    for x in np.arange(0.05, 0.71, 0.05):
        k = int(np.argmin(np.abs(rc - x * R)))
        c_num = np.sqrt(GAMMA * s_P[k] / max(s_rho[k], 1e-300))
        c_ex = np.sqrt(GAMMA * a_P[k] / max(a_rho[k], 1e-300))
        print(f"  {rc[k] / R:.3f}   {s_rho[k] / RHO0:8.4f} ({a_rho[k] / RHO0:8.4f})   {s_P[k] / pmax:6.3f} ({a_P[k] / pmax:6.3f})"
              f"   {c_num:8.1f} ({c_ex:8.1f})")
