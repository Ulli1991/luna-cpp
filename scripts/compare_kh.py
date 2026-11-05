"""KH M(t) of LUNA and voronoi-cpp against the reference, field errors, and GPU time to a target error.

    python compare_kh.py <runs dir> <log file> [...] [--eps=0.01,0.001] [--series=static,moving,...]

Reference: the Pencil Code 4096^2 M(t) of McNally, Lyra & Passy 2012, fig. 4 (digitised to the PDF's path coordinates,
luna-reference-solutions/pencil/mcnally_kh/mcnally2012_fig4_Mref.txt; GCI <= 1.9e-6 relative), and as a check LUNA's
own MC 4096^2 run from the same repository (luna/mcnally_kh/mcnally2012_luna_mc_ref.npz). Error e(t) = M / M_ref - 1,
quoted as max over t <= 1.5 and the time mean of |e|, as tests/test_hydro_hard_mcnally_kh.py:m_errors.

Times come from the lines the runs printed: `luna: N=.. wall_s=..` and `ours: N=.. wall_s=..` in the given logs, the
last line per series and N. T(eps): log-log interpolation between the two runs bracketing eps in the same series.
"""
import glob
import math
import os
import re
import sys

import numpy as np

REF = os.environ.get("LUNA_REFERENCE_SOLUTIONS", "/raven/u/uli/luna-reference-solutions")


def reference():
    t, m, _ = np.loadtxt(f"{REF}/pencil/mcnally_kh/mcnally2012_fig4_Mref.txt", unpack=True)
    return t, m


def errors(M, mref, t):
    k = min(len(M), len(mref))
    if k < 2:
        return None
    e = M[:k] / mref[:k] - 1
    return float(np.abs(e).max()), float(np.trapezoid(np.abs(e), t[:k]) / t[k - 1])


def field_error(runs, key, n, blocks=128):
    """L1 of rho and V_y at t = 1.5 against AthenaK PPMX 4096^2 and LUNA MC 4096^2, on `blocks`^2 block averages.

    Each block average integrates the actual cells: sub x sub sample points per block go to their Voronoi cell (nearest
    generator, periodic) and take its value extrapolated by nothing: a block is much larger than a cell for
    N >= 2 blocks, so the piecewise-constant field's O(h) point error averages away and what remains is the state's
    own error. Coarser meshes than 2 blocks are not reported. Static lattices use the same path."""
    from scipy.spatial import cKDTree
    path = f"{runs}/{key.replace('-', '_')}_{n}.bin"
    ref = {}
    with np.load(f"{REF}/athenak/mcnally_kh/a5_ppmx_rk4_4096.npz") as z:
        ref["athenak"] = {"rho": np.asarray(z["N4096_rho"], np.float64), "my": np.asarray(z["N4096_my"], np.float64)}
    with np.load(f"{REF}/luna/mcnally_kh/mcnally2012_luna_mc_ref.npz") as z:
        ref["luna"] = {k: np.asarray(z[k], np.float64) for k in ("rho", "my")}
    if not os.path.exists(path) or n < 2 * blocks:
        return None
    d = np.fromfile(path).reshape(-1, 7)
    sub = max(8, 2 * n // blocks)  # at least 2 x 2 samples per lattice cell: no aliasing with the lattice
    m = blocks * sub
    g = (np.arange(m) + 0.5) / m
    px, py = np.meshgrid(g, g, indexing="ij")
    _, cell = cKDTree(np.mod(d[:, :2], 1.0), boxsize=1.0).query(np.c_[px.ravel(), py.ravel()], workers=8)
    rho = d[cell, 5].reshape(m, m).reshape(blocks, sub, blocks, sub).mean(axis=(1, 3))
    my = d[cell, 6].reshape(m, m).reshape(blocks, sub, blocks, sub).mean(axis=(1, 3))
    out = {}
    for name, r in ref.items():
        c = r["rho"].shape[0] // blocks
        rr = r["rho"].reshape(blocks, c, blocks, c).mean(axis=(1, 3))
        mr = r["my"].reshape(blocks, c, blocks, c).mean(axis=(1, 3))
        out[name] = (float(np.abs(rho - rr).mean()), float(np.abs(my / rho - mr / rr).mean()))
    return out


def timings(logs):
    out = {}
    for log in logs:
        for line in open(log, errors="replace"):
            m = re.match(r"(luna|ours): (.*)", line)
            if not m:
                continue
            d = dict(kv.split("=", 1) for kv in m.group(2).split() if "=" in kv)
            n = int(d["N"])
            if m.group(1) == "luna":
                key = f"luna-bulk{float(d['bulk']):g}" if float(d.get("bulk", 0)) != 0 else "luna"
            else:
                key = d["mesh"] + ("-twoflux" if d.get("integrator") == "two-pass" else "") + ("-arepo" if d["dt"] == "arepo" else "") + (f"-jitter{d['jitter']}" if float(d["jitter"]) else "")
                key += f"-bulk{d['bulk']}" if float(d["bulk"]) else ""
            out[(key, n)] = (float(d["wall_s"]), int(d["steps"]), float(d["M_cell_updates_per_s"]))
    return out


def load(runs, key, n):
    prefix = key.replace("-", "_")
    for path in (f"{runs}/{prefix}_{n}.npz",):
        if os.path.exists(path):
            return np.load(path)["M"]
    path = f"{runs}/{prefix}_{n}.txt"
    if os.path.exists(path):
        return np.loadtxt(path)[:, 1]
    return None


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    runs, logs = argv[0], [p for a in argv[1:] for p in glob.glob(a)]
    eps = [0.05, 0.01, 0.001]
    for a in sys.argv[1:]:
        if a.startswith("--eps="):
            eps = [float(v) for v in a[6:].split(",")]
    t, mref = reference()
    with np.load(f"{REF}/athenak/mcnally_kh/a5_ppmx_rk4_4096.npz") as z:
        print(f"the references against each other, max |M / M_Pencil - 1|: AthenaK PPMX 4096 {np.abs(z['N4096_M'] / mref - 1).max():.3g}", end="")
    with np.load(f"{REF}/luna/mcnally_kh/mcnally2012_luna_mc_ref.npz") as z:
        print(f", LUNA MC 4096 {np.abs(z['M'] / mref - 1).max():.3g}")
    times = timings(logs)
    keys = sorted({k for k, _ in times})
    print("| series | N | steps | T (s) | M cell updates/s | max |e| | mean |e| |")
    print("|---|---|---|---|---|---|---|")
    table = {}
    for key in keys:
        for n in sorted(n for k, n in times if k == key):
            wall, steps, rate = times[(key, n)]
            M = load(runs, key, n)
            if M is None or len(M) < 76:
                print(f"| {key} | {n} | {steps} | {wall:.3g} | {rate:.1f} | no series | |")
                continue
            e = errors(M, mref, t)
            table.setdefault(key, []).append((n, wall, e[0], e[1]))
            print(f"| {key} | {n} | {steps} | {wall:.3g} | {rate:.1f} | {e[0]:.3g} | {e[1]:.3g} |")
    print("\nL1 at t = 1.5 on 128^2 block averages (rho, V_y), against AthenaK PPMX 4096^2 and LUNA MC 4096^2:")
    for key in keys:
        if key.startswith("luna"):
            continue
        for n in sorted(n for k, n in times if k == key):
            f = field_error(runs, key, n)
            if f:
                print(f"| {key} | {n} | " + " | ".join(f"{a:.3g} | {b:.3g}" for a, b in f.values()) + " |")
    print("\nGPU seconds to reach max |e| <= eps (interpolated between the bracketing runs of a series):")
    print("| series | " + " | ".join(f"{v:g}" for v in eps) + " |")
    print("|---|" + "---|" * len(eps))
    for key, rows in table.items():
        cells = []
        for v in eps:
            found = None
            for (n0, t0, e0, _), (n1, t1, e1, _) in zip(rows, rows[1:]):
                if e0 >= v >= e1 and e0 > e1:
                    f = math.log(v / e0) / math.log(e1 / e0)
                    found = t0 * (t1 / t0) ** f
                    break
            if found is None and rows and rows[0][2] <= v:
                found = rows[0][1]
            cells.append(f"{found:.3g}" if found else "not reached")
        print(f"| {key} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
