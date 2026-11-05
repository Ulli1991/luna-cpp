"""The figures of docs/figures/ from run files. Nothing is typed in: every number is read here.

    python scripts/make_figures.py [names ...]     # default: all; names as in FIGURES below

Run with LUNA's environment: LUNA's src on PYTHONPATH (exact Riemann and Sedov-Taylor solutions), numpy, matplotlib, h5py, Pillow.
Data paths, each an environment variable, none of them in this repository:

    VORONOI_RUNS               voronoi-cpp checkout with the build-* run directories
    LUNA_REFERENCE_SOLUTIONS   luna-reference-solutions at 99a52d0 (sha256 checked against its MANIFEST.json)
    LUNA_CPP_FIGURES           output directory, default docs/figures next to this script's parent
"""
import glob
import hashlib
import importlib.util
import json
import os
import re
import sys

import h5py
import matplotlib
import matplotlib.ticker
import numpy as np
from matplotlib.patches import ConnectionPatch, Rectangle
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from figstyle import FIGSIZE, apply_style, decorate  # noqa: E402
from luna.hydro import exact_sedov  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get("VORONOI_RUNS", "/raven/u/uli/voronoi-cpp")
REFS = os.environ.get("LUNA_REFERENCE_SOLUTIONS", "/raven/u/uli/luna-reference-solutions")
OUT = os.environ.get("LUNA_CPP_FIGURES", os.path.join(HERE, "..", "docs", "figures"))
SET = f"{RUNS}/build-1d/runs-set"            # jobs 31917206/8/10: LUNA, static, moving default at each N in one job
REPORT = f"{RUNS}/build-1d/report-set/dim1d_errors.json"
FINE = f"{RUNS}/build-1d/runs/refs"          # the references of the L1 table: LUNA MC 2^21 (Shu-Osher), 2^20 (WC)
MATRIX = f"{RUNS}/build-1ddiag/runs_m/matrix.json"
WIDE_CPU = f"{RUNS}/build-1ddiag/runs_mc"
VORTEX_LOGS = f"{RUNS}/build-1ddiag/logs"
KH = f"{RUNS}/build-2d/runs/kh_table.md"
SEDOV = f"{RUNS}/build-limiters/sedov"
SEDOV_L1 = f"{RUNS}/build-limiters/sedov_l1.json"
SEDOV_LADDER = f"{RUNS}/build-metric"

spec = importlib.util.spec_from_file_location("compare_1d", f"{HERE}/compare_1d.py")
cmp1d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmp1d)

apply_style()
LUNA_C, STATIC_C, MOVING_C = "C0", "C1", "C2"
STYLE = {"luna": dict(color=LUNA_C, ls="-", marker="o"),
         "static": dict(color=STATIC_C, ls="--", marker="s"),
         "moving": dict(color=MOVING_C, ls="-", marker="^"),
         "two": dict(color=MOVING_C, ls="--", marker="D"),
         "padded": dict(color=MOVING_C, ls="--", marker="v"),
         "padded3": dict(color=MOVING_C, ls=":", marker="x")}
REF = dict(color="k")
AK = dict(color="k", ls=":")
MS = 3.5  # as LUNA's figures


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/{name}.png"
    for ax in fig.axes:
        decorate(ax)
        for child in ax.child_axes:
            decorate(child)
    fig.savefig(path)
    plt.close(fig)
    Image.open(path).convert("RGB").quantize(256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE).save(path, optimize=True)
    print(f"{path} {os.path.getsize(path) / 1024:.0f} kB")


def line(ax, x, y, s, **over):
    k = STYLE[s]
    return ax.plot(x, y, color=k["color"], ls=k["ls"], **over)[0]


def cells(ax, x, y, s, xlim, most=32):
    """Actual cells in a window: a thin line and small open markers, every cell while they stay apart."""
    k = STYLE[s]
    keep = (x >= xlim[0]) & (x <= xlim[1])
    ax.plot(x[keep], y[keep], color=k["color"], ls=k["ls"], lw=1.0, marker=k["marker"], ms=MS, mfc="white",
            markevery=max(1, keep.sum() // most), zorder=3)


def corner(ax, text, x=0.03, y=0.95, ha="left"):
    ax.text(x, y, text, transform=ax.transAxes, va="top", ha=ha)


def key(s, label, marker=False):
    k = STYLE[s]
    return plt.Line2D([], [], color=k["color"], ls=k["ls"], marker=k["marker"] if marker else None, ms=MS, mfc="none", label=label)


def rkey(label, kw=REF):
    return plt.Line2D([], [], label=label, **kw)


def loglog(ax, x, y, s, **over):
    k = STYLE[s]
    style = dict(color=k["color"], ls=k["ls"], marker=k["marker"], ms=MS, mfc="none", zorder=3 if s != "static" else 2)
    style.update(over)
    ax.plot(x, y, **style)


def log_axes(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")


def fixed_x(ax, major, minor=()):
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(major))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([str(m) for m in major]))
    ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(minor))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def reference_file(path):
    """A file of luna-reference-solutions, checked against the manifest; a missing or changed file stops here."""
    manifest = json.load(open(f"{REFS}/MANIFEST.json"))
    files = manifest if isinstance(manifest, list) else manifest.get("files", manifest)
    want = {f["path"]: f["sha256"] for f in files} if isinstance(files, list) else {k: v["sha256"] for k, v in files.items()}
    full = f"{REFS}/{path}"
    if hashlib.sha256(open(full, "rb").read()).hexdigest() != want[path]:
        raise SystemExit(f"{full}: sha256 differs from MANIFEST.json")
    return np.load(full, allow_pickle=True)


def ours(series_name, problem, n, root=SET):
    d = np.load(f"{root}/{series_name}_{problem}_{n}.npy")
    return d[:, 1], d[:, 3], d[:, 4], d[:, 5]  # centroid, rho, v, P: cell means


def luna(problem, n):
    d = np.load(f"{SET}/luna_{problem}_{n}.npz")
    a, b = cmp1d.DOMAIN[problem]
    return a + (np.arange(n) + 0.5) * (b - a) / n, d["rho"], d["u"], d["p"]


def centres(a, b, n):
    return a + (np.arange(n) + 0.5) * (b - a) / n


def wide_rule_l1():
    """Field L1 on [-5, 5] of the padded moving runs at the default rule, as matrix.py on dim-1d-diag: N per length 10."""
    ref = cmp1d.Reference(f"{FINE}/ref_shu-osher.npz")
    out = {}
    for n in (512, 2048, 8192):
        cmp1d.DOMAIN["w"], cmp1d.BOUNDARY["w"] = (-15.0, 5.0), "outflow"
        rec, _ = cmp1d.ours(f"{WIDE_CPU}/aro_shu-osher-wide_{n}.npy", "w")
        cmp1d.DOMAIN["w"] = (-5.0, 5.0)
        out[n // 2] = cmp1d.l1(rec, ref, "w", (-5.0, 5.0)).tolist()
    return out


def report():
    rows = json.load(open(REPORT))
    return {(r["series"], r["problem"]): sorted([x for x in rows if (x["series"], x["problem"]) == (r["series"], r["problem"])], key=lambda x: x["N"])
            for r in rows}


def vortex_rows():
    out = {}
    for path, keys in ((f"{VORTEX_LOGS}/n-vortex-31920446.out", ("mov", "sta")), (f"{VORTEX_LOGS}/fin-norm-31922276.out", ("tp",))):
        for line in open(path):
            m = re.search(r"vortex (\S+) nx=(\d+) gen .*\| avg (\S+) (\S+) (\S+) \| field0 \S+ \S+ \S+ \| exact0 (\S+) (\S+) (\S+)", line)
            if m and m.group(1) in keys:
                g = m.groups()
                out[g[0], int(g[1])] = ([float(a) for a in g[2:5]], [float(a) for a in g[5:8]])
    return out


def kh_rows():
    runs, block, target = {}, {}, {}
    part = 0
    for line in open(KH):
        if line.startswith("L1 at t = 1.5"):
            part = 1
        elif line.startswith("GPU seconds to reach"):
            part = 2
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not line.startswith("|") or cells[0] in ("series", "---"):
            continue
        if part == 0:
            runs.setdefault(cells[0], []).append(tuple(float(c) for c in (cells[1], cells[3], cells[5])))
        elif part == 1:
            block.setdefault(cells[0], []).append(tuple(float(c) for c in cells[1:3]))
        else:
            target[cells[0]] = [None if "not" in c else float(c) for c in cells[1:]]
    return runs, block, target


def sedov_state(path):
    if path.endswith(".npz"):
        z = np.load(path)
        rho = np.asarray(z["U"][0], float)
        n = rho.shape[0]
        c = (np.arange(n) + 0.5) * 0.05 / n
        X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
        pos, vol, rho = np.stack([X.ravel(), Y.ravel(), Z.ravel()]), np.full(n ** 3, (0.05 / n) ** 3), rho.ravel()
        t = float(z["t"])
    else:
        with h5py.File(sorted(glob.glob(f"{path}/snapdir_*/*.hdf5"))[-1], "r") as f:
            c = f["cells"]
            pos, vol, rho = np.stack([c["x"][...], c["y"][...], c["z"][...]]), c["volume"][...], c["density"][...]
            t = float(c.attrs["time"])
    return np.sqrt(((pos - 0.025) ** 2).sum(0)), vol, rho, t


def zoom(main, ax, xlim, runs, refs, letter, label, place="ur", ylim=None):
    """A zoom panel: reference, static as a line (it lies on LUNA's cells), cells of the other runs; its region marked on main."""
    ys = [y[(x >= xlim[0]) & (x <= xlim[1])] for x, y, *_ in [*runs, *refs]]
    if ylim is None:
        lo, hi = min(y.min() for y in ys if y.size), max(y.max() for y in ys if y.size)
        ylim = (lo - 0.08 * (hi - lo), hi + 0.22 * (hi - lo))
    for x, y, kw in refs:
        ax.plot(x, y, zorder=2, **kw)
    for x, y, s in runs:
        if s == "static":
            line(ax, x, y, s, lw=1.0, zorder=4)
        else:
            cells(ax, x, y, s, xlim)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    u, v, ha, va = dict(ur=(0.96, 0.95, "right", "top"), ul=(0.04, 0.95, "left", "top"), lr=(0.96, 0.05, "right", "bottom"),
                        ll=(0.04, 0.05, "left", "bottom"))[place]
    ax.text(u, v, f"({letter}) {label}", transform=ax.transAxes, ha=ha, va=va)
    mark(main, xlim, ylim, letter)
    for corner_x, side in ((xlim[0], 0), (xlim[1], 1)):
        main.figure.add_artist(ConnectionPatch(xyA=(corner_x, ylim[0]), coordsA=main.transData, xyB=(side, 1), coordsB=ax.transAxes,
                                               color="0.5", lw=0.6))


def mark(main, xlim, ylim, letter):
    """A thin rectangle round a zoom's region, with the zoom's letter."""
    main.add_patch(Rectangle((xlim[0], ylim[0]), xlim[1] - xlim[0], ylim[1] - ylim[0], fill=False, ec="0.3", lw=0.8, zorder=5))
    main.text(0.5 * (xlim[0] + xlim[1]), ylim[1], letter, ha="center", va="bottom")


def fig_sod():
    n, p = 512, "sod"
    x, r, *_ = ours("moving-a04", p, n)
    xs, rs, *_ = ours("static", p, n)
    xl, rl, *_ = luna(p, n)
    ex = cmp1d.Exact()
    fine = np.linspace(0, 1, 40001)
    exact = ex(fine)[0]
    fig = plt.figure(figsize=(FIGSIZE[0] * 2, FIGSIZE[1] * 2))
    outer = fig.add_gridspec(2, 1, height_ratios=[4, 2.2], hspace=0.2)
    top = outer[0].subgridspec(2, 1, height_ratios=[3.2, 1], hspace=0)
    ax = fig.add_subplot(top[0])
    res = fig.add_subplot(top[1], sharex=ax)
    ax.plot(fine, exact, **REF)
    line(ax, xl, rl, "luna")
    line(ax, xs, rs, "static")
    line(ax, x, r, "moving")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel(r"$\rho$")
    ax.tick_params(labelbottom=False)
    ax.legend(handles=[key("luna", "LUNA MC"), key("static", "static"), key("moving", "moving"), rkey("exact")], loc="upper right")
    res.axhline(0, color="k", lw=0.6)
    for a, b, s in ((xl, rl, "luna"), (xs, rs, "static"), (x, r, "moving")):
        line(res, a, b - ex(a)[0], s)
    res.set_ylim(-0.012, 0.012)
    res.set_ylabel(r"$\rho-\rho_{\rm exact}$")
    res.set_xlabel(r"$x$")
    z = outer[1].subgridspec(1, 4, wspace=0.34).subplots()
    runs, refs = [(xl, rl, "luna"), (xs, rs, "static"), (x, r, "moving")], [(fine, exact, REF)]
    zoom(ax, z[0], (0.19, 0.23), runs, refs, "a", "rarefaction head")
    zoom(ax, z[1], (0.46, 0.52), runs, refs, "b", "rarefaction tail")
    zoom(ax, z[2], (0.70, 0.76), runs, refs, "c", "contact")
    zoom(ax, z[3], (0.92, 0.955), runs, refs, "d", "shock")
    z[0].set_ylabel(r"$\rho$")
    for a in z:
        a.set_xlabel(r"$x$")
    save(fig, "sod")


def fig_shu_osher():
    n, p = 1024, "shu-osher"
    x, r, *_ = ours("moving-a04", p, n)
    xs, rs, *_ = ours("static", p, n)
    xl, rl, *_ = luna(p, n)
    w = np.load(f"{WIDE_CPU}/aro_shu-osher-wide_{2 * n}.npy")
    xw, rw = w[:, 1], w[:, 3]
    lr = reference_file("luna/shu_osher/shu_osher_mc.npz")["rho"]
    ar = reference_file("athenak/shu_osher/so_ppmx_rk4_16384.npz")["ref_rho"]
    refs = [(centres(-5, 5, lr.size), lr, REF), (centres(-5, 5, ar.size), ar, AK)]
    fig = plt.figure(figsize=(FIGSIZE[0] * 2, FIGSIZE[1] * 2.6))
    outer = fig.add_gridspec(2, 1, height_ratios=[2.5, 1.1], hspace=0.16)
    top = outer[0].subgridspec(2, 1, hspace=0)
    a = fig.add_subplot(top[0])
    b = fig.add_subplot(top[1], sharex=a)
    for ax in (a, b):
        for rx, ry, kw in refs:
            ax.plot(rx, ry, **kw)
        ax.set_xlim(-5, 5)
        ax.set_ylim(0.4, 5.4)
        ax.set_ylabel(r"$\rho$")
    line(a, xl, rl, "luna")
    line(a, xs, rs, "static")
    line(a, x, r, "moving")
    line(b, xl, rl, "luna")
    line(b, xw, rw, "padded")
    a.tick_params(labelbottom=False)
    corner(a, "LUNA's domain", 0.02, 0.95)
    corner(b, "padded domain", 0.02, 0.95)
    a.legend(handles=[key("luna", "LUNA MC"), key("static", "static"), key("moving", "moving"), key("padded", "moving, padded"),
                      rkey("LUNA, high resolution"), rkey("AthenaK PPMX", AK)], loc="lower left", ncol=2)
    b.set_xlabel(r"$x$")
    z = outer[1].subgridspec(1, 4, wspace=0.34).subplots()
    runs = [(xl, rl, "luna"), (xs, rs, "static"), (x, r, "moving"), (xw, rw, "padded")]
    views = (((-3.2, 0.2), "a", "left-going waves", "ll", (3.45, 4.35)), ((0.6, 2.3), "b", "wave train", "ll", (2.5, 5.0)),
             ((1.76, 1.92), "c", "post-shock peak", "ul", (4.25, 4.95)), ((2.33, 2.45), "d", "main shock", "ur", None))
    for ax, (xlim, letter, label, place, ylim) in zip(z, views):
        zoom(b, ax, xlim, runs, refs, letter, label, place, ylim)
        mark(a, xlim, ax.get_ylim(), letter)
        ax.set_xlabel(r"$x$")
    z[0].set_ylabel(r"$\rho$")
    save(fig, "shu_osher")


def fig_woodward_colella():
    n, p = 2048, "woodward-colella"
    x, r, *_ = ours("moving-a04", p, n)
    xs, rs, *_ = ours("static", p, n)
    xl, rl, *_ = luna(p, n)
    lr = reference_file("luna/woodward_colella/woodward_colella_mc.npz")["rho"]
    ar = reference_file("athenak/woodward_colella/a9_ppmx_rk4_16384.npz")["ref_rho"]
    refs = [(centres(0, 1, lr.size), lr, REF), (centres(0, 1, ar.size), ar, AK)]
    fig = plt.figure(figsize=(FIGSIZE[0] * 2.5, FIGSIZE[1] * 1.8))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.5, 1], hspace=0.2)
    ax = fig.add_subplot(outer[0])
    for rx, ry, kw in refs:
        ax.plot(rx, ry, **kw)
    line(ax, xl, rl, "luna")
    line(ax, xs, rs, "static")
    line(ax, x, r, "moving")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 8.6)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\rho$")
    ax.legend(handles=[key("luna", "LUNA MC"), key("static", "static"), key("moving", "moving"), rkey("LUNA, high resolution"), rkey("AthenaK PPMX", AK)],
              loc="upper left")
    z = outer[1].subgridspec(1, 5, wspace=0.32).subplots()
    runs = [(xl, rl, "luna"), (xs, rs, "static"), (x, r, "moving")]
    zoom(ax, z[0], (0.585, 0.607), runs, refs, "a", "contact", "ul", (0, 3.2))
    zoom(ax, z[1], (0.640, 0.656), runs, refs, "b", "shock", "ul", (1.6, 6.6))
    zoom(ax, z[2], (0.755, 0.777), runs, refs, "c", "contact", "ul", (2.2, 8.0))
    zoom(ax, z[3], (0.788, 0.806), runs, refs, "d", "shock", "ur", (0, 7.4))
    zoom(ax, z[4], (0.855, 0.875), runs, refs, "e", "shock", "ur", (0, 1.6))
    z[0].set_ylabel(r"$\rho$")
    for a in z:
        a.set_xlabel(r"$x$")
    save(fig, "woodward_colella")


PROBLEMS = (("sod", "Sod"), ("shu-osher", "Shu-Osher"), ("woodward-colella", "Woodward-Colella"))
ROWS = ((0, r"$L_1(\rho)$"), (2, r"$L_1(P)$"))


def fig_convergence():
    rows = report()
    wide = wide_rule_l1()
    matrix = [r for r in json.load(open(MATRIX)) if r["problem"] == "shu-osher-wide" and r["config"] == "c04"]
    fig, axes = plt.subplots(2, 3, figsize=(FIGSIZE[0] * 3, FIGSIZE[1] * 2), sharex=True, gridspec_kw=dict(hspace=0, wspace=0.27))
    for j, (p, title) in enumerate(PROBLEMS):
        for i, (var, ylabel) in enumerate(ROWS):
            ax = axes[i, j]
            log_axes(ax)
            for s, k in (("static", "static"), ("luna", "luna"), ("moving", "moving-a04")):
                r = rows[k, p]
                loglog(ax, [x["N"] for x in r], [x["L1"][var] for x in r], s)
            if p == "shu-osher":
                loglog(ax, [m["N"] for m in matrix], [m["L1"][var] for m in matrix], "padded")
                ks = sorted(wide)
                loglog(ax, ks, [wide[k][var] for k in ks], "padded3")
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == 1:
                ax.set_xlabel(r"$N$")
            ax.set_xlim(150, 5e5)
        corner(axes[0, j], title, *((0.96, 0.95, "right") if p == "woodward-colella" else (0.06, 0.12)))
    axes[1, 0].legend(handles=[key("luna", "LUNA MC", True), key("static", "static", True), key("moving", "moving", True)], loc="lower left")
    axes[1, 1].legend(handles=[key("padded", "padded, LUNA rule", True), key("padded3", "padded, default rule", True)], loc="upper right", bbox_to_anchor=(1, 0.87))
    save(fig, "convergence_1d")


def fig_time():
    rows = report()
    fig, axes = plt.subplots(2, 3, figsize=(FIGSIZE[0] * 3, FIGSIZE[1] * 2), sharex=True, gridspec_kw=dict(hspace=0, wspace=0.27))
    for j, (p, title) in enumerate(PROBLEMS):
        for i, (var, ylabel) in enumerate(ROWS):
            ax = axes[i, j]
            log_axes(ax)
            for eps in cmp1d.TARGETS[p][cmp1d.VARS[var]]:
                ax.axhline(eps, color="0.6", lw=0.8, ls=":", zorder=1)
            for s, k in (("static", "static"), ("luna", "luna"), ("moving", "moving-a04")):
                r = rows[k, p]
                loglog(ax, [x["wall_s"] for x in r], [x["L1"][var] for x in r], s)
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == 1:
                ax.set_xlabel(r"GPU time (s)")
            ax.set_xlim(4e-3, 6e2)
        corner(axes[0, j], title, *((0.96, 0.95, "right") if p == "woodward-colella" else (0.06, 0.12)))
    axes[1, 0].legend(handles=[key("luna", "LUNA MC", True), key("static", "static", True), key("moving", "moving", True)], loc="upper right")
    save(fig, "time_to_target_1d")


def fig_vortex():
    rows = vortex_rows()
    d = reference_file("luna/yee_vortex/yee_vortex_convergence.npz")
    pick = (d["dim"] == 3) & (d["limiter"] == 1) & (d["vx"] == 0) & (d["cfl"] == 0.8)
    fig, axes = plt.subplots(1, 2, figsize=(FIGSIZE[0] * 2, FIGSIZE[1]), gridspec_kw=dict(wspace=0.24))
    nx = (32, 64, 128, 256)
    for i, ax, ylabel, v in ((0, axes[0], r"$L_1(\rho)$", "rho"), (1, axes[1], r"$L_1(v)$", "v")):
        log_axes(ax)
        ax.plot(nx, [rows["sta", n][1][i] for n in nx], color="k", ls=":", zorder=1)
        for q, over in ((8, dict()), (1, dict(ls="--"))):
            s = pick & (d["ic_quad"] == q)
            o = np.argsort(d["nx"][s])
            loglog(ax, d["nx"][s][o], d[f"avg_{v}_L1"][s][o], "luna", **over)
        loglog(ax, nx, [rows["sta", n][0][i] for n in nx], "static")
        loglog(ax, nx, [rows["mov", n][0][i] for n in nx], "moving")
        loglog(ax, nx, [rows["tp", n][0][i] for n in nx], "two")
        fixed_x(ax, nx)
        ax.set_xlim(26, 310)
        ax.set_xlabel(r"$n_x$")
        ax.set_ylabel(ylabel)
    axes[0].legend(handles=[key("luna", "LUNA MC", True), plt.Line2D([], [], color=LUNA_C, ls="--", marker="o", ms=MS, mfc="none", label="LUNA MC, point start"),
                            key("static", "static", True), key("moving", "moving, one pass", True), key("two", "moving, two passes", True),
                            rkey(r"$O(h)$ floor", dict(color="k", ls=":"))], loc="lower left")
    save(fig, "vortex_3d")


def fig_kh():
    runs, _, _ = kh_rows()
    eps = [float(c) for c in next(l for l in open(KH).read().splitlines() if l.startswith("| series | 0.05")).strip("| \n").split("|")[1:]]
    fig, (a, b) = plt.subplots(1, 2, figsize=(FIGSIZE[0] * 2, FIGSIZE[1]), sharey=True, gridspec_kw=dict(wspace=0.06))
    for e in eps:
        b.axhline(e, color="0.6", lw=0.8, ls=":", zorder=1)
    for k, s in (("static", "static"), ("luna", "luna"), ("moving", "moving"), ("moving-twoflux", "two")):
        r = np.array(runs[k])
        loglog(a, r[:, 0], r[:, 2], s)
        loglog(b, r[:, 1], r[:, 2], s)
    log_axes(a)
    log_axes(b)
    fixed_x(a, [64, 256, 1024], [128, 512, 2048])
    a.set_xlim(52, 2500)
    a.set_xlabel(r"$N$")
    a.set_ylabel(r"$\max\,|e|$")
    b.set_xlabel("GPU time (s)")
    a.legend(handles=[key("luna", "LUNA MC", True), key("static", "static", True), key("moving", "moving, one flux", True), key("two", "moving, two fluxes", True)], loc="lower left")
    save(fig, "kh_2d")


def fig_sedov():
    KPC, MSUN = 3.08568e21, 1.98892e33
    rho0 = 1.6726219e-24 / (1e10 * MSUN / KPC ** 3)
    e0 = 1e51 / (1e10 * MSUN * 1e10)
    fig = plt.figure(figsize=(FIGSIZE[0] * 2.5, FIGSIZE[1] * 1.8))
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25], hspace=0.28, wspace=0.26)
    a, b = fig.add_subplot(grid[0, :2]), fig.add_subplot(grid[:, 2])
    z = [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    prof = {}
    for name, path in (("static", f"{SEDOV}/static_face_128"), ("luna", f"{SEDOV}/luna/luna1_128.npz"), ("moving", f"{SEDOV}/moving_face_128")):
        r, vol, rho, t = sedov_state(path)
        prof[name] = (r, vol, rho)
    R = exact_sedov.solution(np.array([0.0]), t, e0, rho0, 5 / 3)[3]
    edges = np.linspace(0, 1.15, 70) * R
    mid = 0.5 * (edges[1:] + edges[:-1]) / R
    fine = np.linspace(1e-4, 1.15, 3000) * R
    exact = exact_sedov.solution(fine, t, e0, rho0, 5 / 3)[0] / rho0
    a.plot(fine / R, exact, **REF)
    shells = []
    for name in ("luna", "static", "moving"):
        r, vol, rho = prof[name]
        y = np.histogram(r, edges, weights=rho * vol)[0] / np.maximum(np.histogram(r, edges, weights=vol)[0], 1e-300) / rho0
        line(a, mid, y, name)
        shells.append((mid, y, name))
    a.set_xlim(0, 1.15)
    a.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    a.set_xlabel(r"$r/R_{\rm s}$")
    a.set_ylabel(r"$\rho/\rho_0$")
    a.set_ylim(0, 4.8)
    refs = [(fine / R, exact, REF)]
    zoom(a, z[0], (0.72, 1.0), shells, refs, "a", "shell", "ul", (0.1, 4.4))
    zoom(a, z[1], (0.95, 1.08), shells, refs, "b", "shock front", "ur", (0.5, 4.4))
    for ax in z:
        ax.set_xlabel(r"$r/R_{\rm s}$")
    z[0].set_ylabel(r"$\rho/\rho_0$")
    lad = {**json.load(open(f"{SEDOV_LADDER}/norms_luna.json")), **json.load(open(f"{SEDOV_LADDER}/norms_vor.json"))}
    mc = json.load(open(SEDOV_L1))
    for prefix, s, over in (("sta", "static", dict()), ("luna", "luna", dict(ls="--")), ("mov", "moving", dict())):
        keys = [k for k in lad if re.fullmatch(prefix + r"\d+", k) or k == ("mov256cell" if prefix == "mov" else "")]
        keys.sort(key=lambda k: lad[k]["n"])
        loglog(b, [lad[k]["n"] for k in keys], [lad[k]["field_rho"] for k in keys], s, **over)
    loglog(b, [128, 256], [mc["luna-1_128"]["rho"], mc["luna-1_256"]["rho"]], "luna")
    log_axes(b)
    fixed_x(b, [64, 128, 256, 448])
    b.set_xlim(56, 520)
    b.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.04, 0.06, 0.1]))
    b.yaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(["0.04", "0.06", "0.1"]))
    b.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    b.set_xlabel(r"$n$")
    b.set_ylabel(r"$L_1(\rho)$")
    b.legend(handles=[key("luna", "LUNA MC", True), plt.Line2D([], [], color=LUNA_C, ls="--", marker="o", ms=MS, mfc="none", label="LUNA minmod"),
                      key("static", "static", True), key("moving", "moving", True), rkey("Sedov-Taylor")], loc="upper right")
    save(fig, "sedov_3d")


FIGURES = {"sod": fig_sod, "shu_osher": fig_shu_osher, "woodward_colella": fig_woodward_colella, "convergence_1d": fig_convergence,
           "time_to_target_1d": fig_time, "vortex_3d": fig_vortex, "kh_2d": fig_kh, "sedov_3d": fig_sedov}


def main():
    for name in sys.argv[1:] or list(FIGURES):
        FIGURES[name]()


if __name__ == "__main__":
    main()
