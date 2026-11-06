"""The measures of a 3D end state, side by side (hydro_wave3d --out rows: generator xyz, centroid xyz, V, rho, vx, vy, vz, P).

    python norm3d.py <vortex|simple|tube> <file.npy> <nx> [subdivision] [--t=t] [--amp=a]

A regular grid of `subdivision` points per lattice spacing is assigned to the nearest generator (the Voronoi cell, periodic).
Columns, L1 of rho, |v - v_exact|, P, V or sample weighted:
  gen     cell value against the exact value at the generator (the wrong place; first order on a moving mesh)
  cen     against the exact value at the centroid (what l1_3d.py prints)
  avg     against the exact average over the cell (LUNA's cell-average table): exact(centroid) + mean over the cell's samples of exact
          minus exact at the samples' mean, so that the samples' error in the centroid cancels
  field0  piecewise constant field against the exact field, sample mean (O(h) even for exact averages)
  exact0  the same with the exact averages as the field: the floor of field0
  field1  W_i + g_i (x - c_i), g_i unlimited least squares over the 12 nearest centroids
tube needs LUNA's exact Riemann solver on PYTHONPATH (luna.hydro.exact_riemann).
"""
import sys

import numpy as np
from scipy.spatial import cKDTree

G = 1.4
a = dict(s[2:].split("=") for s in sys.argv[1:] if s.startswith("--") and "=" in s)
pos = [s for s in sys.argv[1:] if not s.startswith("--")]
problem, path, nx = pos[0], pos[1], int(pos[2])
sub = int(pos[3]) if len(pos) > 3 else 4
amp = float(a.get("amp", 0.1))
d = np.load(path)
if problem == "vortex":
    side = np.array([10.0, 10.0, 2.5])
    origin = np.array([-5.0, -5.0, 0.0])
    cells = (nx, nx, nx // 4)
elif problem == "tube":
    side = np.array([2.0, 0.25, 0.25])
    origin = np.zeros(3)
    cells = (2 * nx, nx // 4, nx // 4)
else:
    side = np.array([1.0, 0.25, 0.25])
    origin = np.zeros(3)
    cells = (nx, nx // 4, nx // 4)
gen = (d[:, :3] - origin) % side
cen = (d[:, 3:6] - origin) % side
V = d[:, 6]
W = np.stack([d[:, 7], d[:, 8], d[:, 9], d[:, 10], d[:, 11]])



def simple(x, t, amp):  # the isentropic simple wave of hydro_1d: rho, v, P
    x0 = (x - t) % 1.0
    for _ in range(60):
        c = 1 + amp * np.sin(2 * np.pi * x0)
        f = x0 + (6 * c - 5) * t - x
        f = f - np.round(f)
        x0 = x0 - f / (1 + 6 * t * amp * 2 * np.pi * np.cos(2 * np.pi * x0))
    c = 1 + amp * np.sin(2 * np.pi * x0)
    rho = c ** 5
    return np.stack([rho, 5 * (c - 1), rho ** G / G])


def tube(x, t):  # periodic Sod on [0, 2): the Sod problem at x = 1.5, its mirror image at x = 0.5
    from luna.hydro import exact_riemann as er
    WL, WR = er.TORO[1][:2]
    y = np.where(x >= 1, x, 2 - x)
    S = np.stack(er.sample(WL, WR, G, (y - 1.5) / t)[:3])
    S[1] = np.where(x >= 1, S[1], -S[1])
    return S


if problem in ("simple", "tube"):
    t = float(a.get("t", 0.2 if problem == "simple" else 0.25))
    xs = np.linspace(0, side[0], 1 << 20, endpoint=False)
    T = simple(xs, t, amp) if problem == "simple" else tube(xs, t)


def exact(p):  # p (3, n) in box coordinates
    if problem == "vortex":
        x, y = p[0] + origin[0], p[1] + origin[1]
        r2 = x * x + y * y
        beta = 5.0
        Tv = 1 - (G - 1) * beta ** 2 / (8 * G * np.pi ** 2) * np.exp(1 - r2)
        rho = Tv ** (1 / (G - 1))
        w = beta / (2 * np.pi) * np.exp(0.5 * (1 - r2))
        return np.stack([rho, -w * y, w * x, 0 * x, rho * Tv])
    i = (p[0] % side[0]) / side[0] * xs.size
    j = np.floor(i).astype(np.int64)
    f = i - j
    j0, j1 = j % xs.size, (j + 1) % xs.size
    row = lambda q: T[q, j0] * (1 - f) + T[q, j1] * f
    return np.stack([row(0), row(1), 0 * f, 0 * f, row(2)])


def scalars(e):  # (5, n) differences -> rho, |v|, P
    return np.stack([np.abs(e[0]), np.sqrt(e[1] ** 2 + e[2] ** 2 + e[3] ** 2), np.abs(e[4])])


n = len(V)
tree = cKDTree(gen, boxsize=side)
cen_tree = cKDTree(cen, boxsize=side)
# Unlimited least-squares gradients over the 12 nearest centroids.
dist, ngb = cen_tree.query(cen, k=13, workers=-1)
ngb = ngb[:, 1:]
dx = cen[ngb] - cen[:, None, :]
dx -= side * np.round(dx / side)
M = np.einsum("nki,nkj->nij", dx, dx)
grad = np.empty((5, n, 3))
for v in range(5):
    b = np.einsum("nki,nk->ni", dx, W[v][ngb] - W[v][:, None])
    grad[v] = np.linalg.solve(M, b[..., None])[..., 0]

count = np.zeros(n)
sum_x = np.zeros((3, n))
sum_e = np.zeros((5, n))
f0 = np.zeros(3)
f1 = np.zeros(3)
g_axes = [(np.arange(cells[k] * sub) + 0.5) / (cells[k] * sub) * side[k] for k in range(3)]
for x0 in np.array_split(g_axes[0], max(1, g_axes[0].size // 16)):
    X, Y, Z = np.meshgrid(x0, g_axes[1], g_axes[2], indexing="ij")
    p = np.stack([X.ravel(), Y.ravel(), Z.ravel()])
    _, ids = tree.query(p.T, workers=-1)
    E = exact(p)
    rel = p.T - gen[ids]
    rel -= side * np.round(rel / side)
    count += np.bincount(ids, minlength=n)
    for k in range(3):
        sum_x[k] += np.bincount(ids, rel[:, k], minlength=n)
    for v in range(5):
        sum_e[v] += np.bincount(ids, E[v], minlength=n)
    f0 += scalars(W[:, ids] - E).sum(axis=1)
    rc = p.T - cen[ids]
    rc -= side * np.round(rc / side)
    f1 += scalars(W[:, ids] + (grad[:, ids] * rc).sum(axis=2) - E).sum(axis=1)
total = count.sum()
weight = V / V.sum()
c_samples = gen + (sum_x / np.maximum(count, 1)).T
curv = sum_e / np.maximum(count, 1) - exact(c_samples.T)
hit = count > 0
avg = exact(cen.T) + np.where(hit, curv, 0.0)
line = lambda e: (scalars(e) * weight).sum(axis=1)
# The floor of field0: the exact averages as a piecewise constant field.
f0x = np.zeros(3)
for x0 in np.array_split(g_axes[0], max(1, g_axes[0].size // 16)):
    X, Y, Z = np.meshgrid(x0, g_axes[1], g_axes[2], indexing="ij")
    p = np.stack([X.ravel(), Y.ravel(), Z.ravel()])
    _, ids = tree.query(p.T, workers=-1)
    f0x += scalars(avg[:, ids] - exact(p)).sum(axis=1)
fmt = lambda v: " ".join(f"{e:.3e}" for e in v)
print(f"gen {fmt(line(W - exact(gen.T)))} | cen {fmt(line(W - exact(cen.T)))} | avg {fmt(line(W - avg))} | linf_avg {fmt(scalars(W - avg).max(axis=1))} | field0 {fmt(f0 / total)} | exact0 {fmt(f0x / total)} | field1 {fmt(f1 / total)}"
      f" | offset/h {np.sqrt(((cen - gen + side / 2) % side - side / 2) ** 2).sum(axis=1).mean() / (side[0] / cells[0]):.3f}")
