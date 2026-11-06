"""The figures of docs/figures/ from run files. Nothing is typed in: every number is read here.

    python scripts/make_figures.py [names ...]     # default: all; names as in FIGURES below

Run with LUNA's environment: LUNA's src on PYTHONPATH (exact Riemann and Sedov-Taylor solutions), numpy, matplotlib, h5py, Pillow.
Data paths, each an environment variable, none of them in this repository:

    VORONOI_RUNS               voronoi-cpp checkout with the build-* run directories
    LUNA_REFERENCE_SOLUTIONS   luna-reference-solutions at 99a52d0 (sha256 checked against its MANIFEST.json)
    LUNA_CPP_FIGURES           output directory, default docs/figures next to this script's parent

Every figure is checked before it is written: a legend or panel label on plotted data, or a legend on a zoom box, stops the
script with a non-zero exit after all figures have been tried.
"""
import glob
import hashlib
import importlib.util
import json
import os
import re
import statistics
import sys
from collections import defaultdict

import h5py
import matplotlib
import matplotlib.ticker
import numpy as np
from matplotlib.path import Path
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
INFLOW_SET = f"{RUNS}/build-1dinsert/runs-so"  # Shu-Osher with cells from the inflow wall: LUNA, static, moving in one job
INFLOW_REPORT = f"{RUNS}/build-1dinsert/report-so/dim1d_errors.json"
VORTEX_LOGS = f"{RUNS}/build-1ddiag/logs"
KH = f"{RUNS}/build-2d/runs/kh_table.md"
SEDOV = f"{RUNS}/build-limiters/sedov"
SEDOV_L1 = f"{RUNS}/build-limiters/sedov_l1.json"
SEDOV_LADDER = f"{RUNS}/build-metric"
SCALING = f"{RUNS}/build-roofline/logs/roof-31928911.out"  # scripts/roofline_a100.sh on branch roofline, job 31928911

spec = importlib.util.spec_from_file_location("compare_1d", f"{HERE}/compare_1d.py")
cmp1d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmp1d)

apply_style()
LUNA_C, STATIC_C, MOVING_C = "C0", "C1", "C2"
LW, THIN = 1.2, 0.8  # runs; static over LUNA and the references
STYLE = {"luna": dict(color=LUNA_C, ls="-", marker="o", lw=LW, zorder=2),
         "static": dict(color=STATIC_C, ls="--", marker="s", lw=THIN, zorder=2.5),
         "moving": dict(color=MOVING_C, ls="-", marker="^", lw=LW, zorder=3),
         "two": dict(color=MOVING_C, ls="--", marker="D", lw=LW, zorder=3)}
REF = dict(color="k", lw=THIN, zorder=2.7)
AK = dict(color="k", lw=THIN, ls=":", zorder=2.7)
MS = 2.8      # open markers
LABEL = 10    # letters and labels inside panels, as in LUNA's figures
EDGE = "0.65"  # connectors
CONNECTORS = []  # of the figure being drawn, figure coordinates
TEXTS, BOXES = [], []  # panel labels and box letters, zoom boxes, of the figure being drawn
FAILED = []  # (figure, what) of every overlap
WIDTH = FIGSIZE[0] * 2  # every figure
LEGEND = dict(loc="upper right", fontsize=LABEL)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/{name}.png"
    for ax in fig.axes:
        decorate(ax)
    overlaps(fig, name)
    fig.savefig(path)
    plt.close(fig)
    Image.open(path).convert("RGB").quantize(256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE).save(path, optimize=True)
    print(f"{path} {os.path.getsize(path) / 1024:.0f} kB, connectors crossing: {crossings()}")


def overlaps(fig, name):
    """Legends and labels against the plotted lines (targets and floors included), legends also against the zoom boxes."""
    fig.canvas.draw()
    legends = [("legend", leg.get_window_extent()) for leg in [ax.get_legend() for ax in fig.axes] + fig.legends if leg]
    labels = [(t.get_text(), t.get_window_extent()) for t in TEXTS]
    paths = []
    for ax in fig.axes:
        for ln in ax.get_lines():
            xy = ln.get_transform().transform(np.ma.filled(np.ma.array(ln.get_xydata(), dtype=float), np.nan))
            xy = xy[np.isfinite(xy).all(axis=1)]
            if len(xy) > 1 and ln.get_linestyle() != "None":
                paths.append((ax.get_window_extent(), Path(xy)))
    for what, box in legends + labels:
        if any(area.overlaps(box) and path.intersects_bbox(box, filled=False) for area, path in paths):
            FAILED.append((name, what))
    for what, box in legends:
        if any(r.get_window_extent().overlaps(box) for r in BOXES):
            FAILED.append((name, f"{what} on a zoom box"))
    TEXTS.clear()
    BOXES.clear()


def line(ax, x, y, s, **over):
    k = STYLE[s]
    return ax.plot(x, y, **{"color": k["color"], "ls": k["ls"], "lw": k["lw"], "zorder": k["zorder"], **over})[0]


def cells(ax, x, y, s, xlim, most=32):
    """Actual cells in a window: small open markers, every cell while they stay apart."""
    k = STYLE[s]
    keep = (x >= xlim[0]) & (x <= xlim[1])
    ax.plot(x[keep], y[keep], color=k["color"], ls=k["ls"], lw=THIN, zorder=k["zorder"], marker=k["marker"], ms=MS, mew=0.6, mfc="white",
            markevery=max(1, keep.sum() // most))


def corner(ax, text, x=0.04, y=0.95, ha="left", va="top"):
    TEXTS.append(ax.text(x, y, text, transform=ax.transAxes, va=va, ha=ha, fontsize=LABEL, bbox=dict(fc="white", ec="none", pad=1.0), zorder=6))


def key(s, label, marker=False):
    k = STYLE[s]
    return plt.Line2D([], [], color=k["color"], ls=k["ls"], lw=k["lw"], marker=k["marker"] if marker else None, ms=MS, mfc="none", label=label)


def rkey(label, kw=REF):
    return plt.Line2D([], [], label=label, **kw)


def loglog(ax, x, y, s, **over):
    k = STYLE[s]
    style = dict(color=k["color"], ls=k["ls"], lw=k["lw"], marker=k["marker"], ms=MS, mew=0.6, mfc="none", zorder=k["zorder"])
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


def report():
    rows = [r for r in json.load(open(REPORT)) if r["problem"] != "shu-osher"] + json.load(open(INFLOW_REPORT))
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


def zoom(main, ax, xlim, xticks, runs, refs, letter, label, ylim=None):
    """A zoom panel: reference, static as a thin line (it lies on LUNA's cells), cells of the others; its region boxed on main,
    connectors from the box's bottom corners to the panel's top corners."""
    ys = [y[(x >= xlim[0]) & (x <= xlim[1])] for x, y, *_ in [*runs, *refs]]
    if ylim is None:
        lo, hi = min(y.min() for y in ys if y.size), max(y.max() for y in ys if y.size)
        ylim = (lo - 0.1 * (hi - lo), hi + 0.34 * (hi - lo))
    for x, y, kw in refs:
        ax.plot(x, y, **kw)
    for x, y, s in runs:
        if s == "static":
            line(ax, x, y, s)
        else:
            cells(ax, x, y, s, xlim)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(xticks))
    digits = max(len(f"{t:.4f}".rstrip("0").split(".")[1]) for t in xticks)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(f"%.{digits}f"))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))
    ax.tick_params(labelsize=LABEL)
    corner(ax, f"({letter}) {label}")
    mark(main, xlim, ylim, letter)
    to_fig = main.figure.transFigure.inverted()
    for x, side in ((xlim[0], 0), (xlim[1], 1)):
        main.figure.add_artist(ConnectionPatch(xyA=(x, ylim[0]), coordsA=main.transData, xyB=(side, 1), coordsB=ax.transAxes,
                                               color=EDGE, lw=0.6, clip_on=False, zorder=10))
        CONNECTORS.append((to_fig.transform(main.transData.transform((x, ylim[0]))), to_fig.transform(ax.transAxes.transform((side, 1)))))


def crossings():
    """Pairs of connectors of a figure that cross; the zooms' windows and lower limits are chosen to keep this at 0."""
    def side(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    n = sum(side(p, q, u) * side(p, q, v) < 0 and side(u, v, p) * side(u, v, q) < 0
            for i, (p, q) in enumerate(CONNECTORS) for u, v in CONNECTORS[i + 1:])
    CONNECTORS.clear()
    return n


def mark(main, xlim, ylim, letter):
    """A thin light box round a zoom's region, with the zoom's letter."""
    BOXES.append(main.add_patch(Rectangle((xlim[0], ylim[0]), xlim[1] - xlim[0], ylim[1] - ylim[0], fill=False, ec="0.5", lw=0.5, zorder=5)))
    TEXTS.append(main.text(0.5 * (xlim[0] + xlim[1]), ylim[1], letter, ha="center", va="bottom", fontsize=LABEL))


def fig_sod():
    n, p = 512, "sod"
    x, r, *_ = ours("moving-a04", p, n)
    xs, rs, *_ = ours("static", p, n)
    xl, rl, *_ = luna(p, n)
    ex = cmp1d.Exact()
    fine = np.linspace(0, 1, 40001)
    exact = ex(fine)[0]
    fig = plt.figure(figsize=(WIDTH, FIGSIZE[1] * 2))
    outer = fig.add_gridspec(2, 1, height_ratios=[4, 2.2], hspace=0.16)
    top = outer[0].subgridspec(2, 1, height_ratios=[3.2, 1], hspace=0)
    ax = fig.add_subplot(top[0])
    res = fig.add_subplot(top[1], sharex=ax)
    ax.plot(fine, exact, **REF)
    line(ax, xl, rl, "luna")
    line(ax, xs, rs, "static")
    line(ax, x, r, "moving")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.05, 1.12)
    ax.set_ylabel(r"$\rho$")
    ax.tick_params(labelbottom=False)
    ax.legend(handles=[key("luna", "LUNA MC"), key("static", "static"), key("moving", "moving"), rkey("exact", REF)], **LEGEND)
    res.axhline(0, color="k", lw=0.6)
    for a, b, s in ((xl, rl, "luna"), (xs, rs, "static"), (x, r, "moving")):
        line(res, a, b - ex(a)[0], s)
    res.set_ylim(-0.012, 0.012)
    res.set_ylabel(r"$\rho-\rho_{\rm exact}$")
    z = outer[1].subgridspec(1, 4, wspace=0.3).subplots()
    runs, refs = [(xl, rl, "luna"), (xs, rs, "static"), (x, r, "moving")], [(fine, exact, REF)]
    views = (((0.19, 0.23), [0.20, 0.21, 0.22], "a", "rarefaction head"), ((0.46, 0.52), [0.47, 0.49, 0.51], "b", "rarefaction tail"),
             ((0.70, 0.76), [0.71, 0.73, 0.75], "c", "contact"), ((0.92, 0.955), [0.93, 0.94, 0.95], "d", "shock"))
    for a, (xlim, ticks, letter, label) in zip(z, views):
        zoom(ax, a, xlim, ticks, runs, refs, letter, label)
    z[0].set_ylabel(r"$\rho$")
    for a in z:
        a.set_xlabel(r"$x$")
    save(fig, "sod")


def fig_shu_osher():
    n, p = 1024, "shu-osher"
    x, r, *_ = ours("moving-a04", p, n, INFLOW_SET)
    xs, rs, *_ = ours("static", p, n)
    xl, rl, *_ = luna(p, n)
    lr = reference_file("luna/shu_osher/shu_osher_mc.npz")["rho"]
    ar = reference_file("athenak/shu_osher/so_ppmx_rk4_16384.npz")["ref_rho"]
    refs = [(centres(-5, 5, lr.size), lr, REF), (centres(-5, 5, ar.size), ar, AK)]
    fig = plt.figure(figsize=(WIDTH, FIGSIZE[1] * 1.8))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.5, 1], hspace=0.16)
    ax = fig.add_subplot(outer[0])
    for rx, ry, kw in refs:
        ax.plot(rx, ry, **kw)
    line(ax, xl, rl, "luna")
    line(ax, xs, rs, "static")
    line(ax, x, r, "moving")
    ax.set_xlim(-5, 5)
    ax.set_ylim(0.4, 5.9)
    ax.set_ylabel(r"$\rho$")
    ax.legend(handles=[key("luna", "LUNA MC"), key("static", "static"), key("moving", "moving"),
                       rkey("LUNA, high resolution", REF), rkey("AthenaK PPMX", AK)], **LEGEND)
    z = outer[1].subgridspec(1, 4, wspace=0.3).subplots()
    runs = [(xl, rl, "luna"), (xs, rs, "static"), (x, r, "moving")]
    views = (((-3.2, 0.2), [-3, -2, -1, 0], "a", "left-going waves"), ((0.6, 1.25), [0.7, 0.9, 1.1], "b", "wave train"),
             ((1.76, 1.92), [1.8, 1.85, 1.9], "c", "post-shock peak"), ((2.33, 2.45), [2.36, 2.40, 2.44], "d", "main shock"))
    for a, (xlim, ticks, letter, label) in zip(z, views):
        zoom(ax, a, xlim, ticks, runs, refs, letter, label)
        a.set_xlabel(r"$x$")
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
    fig = plt.figure(figsize=(WIDTH, FIGSIZE[1] * 1.8))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.5, 1], hspace=0.16)
    ax = fig.add_subplot(outer[0])
    for rx, ry, kw in refs:
        ax.plot(rx, ry, **kw)
    line(ax, xl, rl, "luna")
    line(ax, xs, rs, "static")
    line(ax, x, r, "moving")
    ax.set_xlim(0.4, 1)
    ax.set_ylim(0, 9.2)
    ax.set_ylabel(r"$\rho$")
    ax.legend(handles=[key("luna", "LUNA MC"), key("static", "static"), key("moving", "moving"), rkey("LUNA, high resolution", REF), rkey("AthenaK PPMX", AK)],
              **LEGEND)
    z = outer[1].subgridspec(1, 5, wspace=0.3).subplots()
    runs = [(xl, rl, "luna"), (xs, rs, "static"), (x, r, "moving")]
    views = (((0.585, 0.607), [0.588, 0.596, 0.604], "a", "contact", 2.9), ((0.640, 0.656), [0.643, 0.648, 0.653], "b", "shock", 7.2),
             ((0.755, 0.777), [0.758, 0.766, 0.774], "c", "contact", 8.6), ((0.788, 0.806), [0.790, 0.797, 0.804], "d", "shock", 8.2),
             ((0.855, 0.875), [0.858, 0.865, 0.872], "e", "shock", 1.3))
    for a, (xlim, ticks, letter, label, top) in zip(z, views):
        zoom(ax, a, xlim, ticks, runs, refs, letter, label, (0, top))  # one lower limit: the connectors do not cross
    z[0].set_ylabel(r"$\rho$")
    for a in z:
        a.set_xlabel(r"$x$")
    save(fig, "woodward_colella")


PROBLEMS = (("sod", "Sod"), ("shu-osher", "Shu-Osher"), ("woodward-colella", "Woodward-Colella"))
ROWS = ((0, r"$L_1(\rho)$"), (2, r"$L_1(P)$"))


def grid_legend(fig, axes, handles):
    """One legend above a grid of panels."""
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, axes[0, 0].get_position().y1), ncol=len(handles), fontsize=LABEL,
               columnspacing=1.5)


def fig_convergence():
    rows = report()
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, FIGSIZE[1] * 1.7), sharex=True, gridspec_kw=dict(hspace=0, wspace=0.34))
    for j, (p, title) in enumerate(PROBLEMS):
        for i, (var, ylabel) in enumerate(ROWS):
            ax = axes[i, j]
            log_axes(ax)
            for s, k in (("static", "static"), ("luna", "luna"), ("moving", "moving-a04")):
                r = rows[k, p]
                loglog(ax, [x["N"] for x in r], [x["L1"][var] for x in r], s)
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == 1:
                ax.set_xlabel(r"$N$")
            ax.set_xlim(150, 5e5)
        corner(axes[0, j], title, 0.95, 0.95, "right")
    grid_legend(fig, axes, [key("luna", "LUNA MC", True), key("static", "static", True), key("moving", "moving", True)])
    save(fig, "convergence_1d")


def fig_time():
    rows = report()
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, FIGSIZE[1] * 1.7), sharex=True, gridspec_kw=dict(hspace=0, wspace=0.34))
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
        corner(axes[0, j], title, 0.95, 0.95, "right")
    grid_legend(fig, axes, [key("luna", "LUNA MC", True), key("static", "static", True), key("moving", "moving", True)])
    save(fig, "time_to_target_1d")


def fig_vortex():
    rows = vortex_rows()
    d = reference_file("luna/yee_vortex/yee_vortex_convergence.npz")
    pick = (d["dim"] == 3) & (d["limiter"] == 1) & (d["vx"] == 0) & (d["cfl"] == 0.8)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, FIGSIZE[1]), gridspec_kw=dict(wspace=0.24))
    nx = (32, 64, 128, 256)
    for i, ax, ylabel, v in ((0, axes[0], r"$L_1(\rho)$", "rho"), (1, axes[1], r"$L_1(v)$", "v")):
        log_axes(ax)
        ax.plot(nx, [rows["sta", n][1][i] for n in nx], color="k", ls=":", lw=THIN, zorder=1)
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
    axes[0].legend(handles=[key("luna", "LUNA MC", True), plt.Line2D([], [], color=LUNA_C, ls="--", lw=LW, marker="o", ms=MS, mfc="none", label="LUNA MC, point start"),
                            key("static", "static", True), key("moving", "moving, one pass", True), key("two", "moving, two passes", True),
                            rkey(r"$O(h)$ floor", dict(color="k", ls=":", lw=THIN))], **LEGEND, ncol=2)
    axes[0].set_ylim(top=axes[0].get_ylim()[1] * 9)
    save(fig, "vortex_3d")


def fig_kh():
    runs, _, _ = kh_rows()
    eps = [float(c) for c in next(l for l in open(KH).read().splitlines() if l.startswith("| series | 0.05")).strip("| \n").split("|")[1:]]
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, FIGSIZE[1]), sharey=True, gridspec_kw=dict(wspace=0.06))
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
    a.legend(handles=[key("luna", "LUNA MC", True), key("static", "static", True), key("moving", "moving, one flux", True), key("two", "moving, two fluxes", True)], **LEGEND)
    save(fig, "kh_2d")


def fig_sedov():
    KPC, MSUN = 3.08568e21, 1.98892e33
    rho0 = 1.6726219e-24 / (1e10 * MSUN / KPC ** 3)
    e0 = 1e51 / (1e10 * MSUN * 1e10)
    fig = plt.figure(figsize=(WIDTH, FIGSIZE[1] * 1.8))
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], hspace=0.2, wspace=0.3)
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
    a.set_ylabel(r"$\rho/\rho_0$")
    a.set_ylim(0, 5.9)
    refs = [(fine / R, exact, REF)]
    zoom(a, z[0], (0.72, 0.95), [0.75, 0.85, 0.95], shells, refs, "a", "shell", ylim=(0.1, 3.3))
    zoom(a, z[1], (0.95, 1.08), [0.96, 1.0, 1.04], shells, refs, "b", "shock front", ylim=(0.1, 5.2))
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
    b.legend(handles=[key("luna", "LUNA MC", True), plt.Line2D([], [], color=LUNA_C, ls="--", lw=LW, marker="o", ms=MS, mfc="none", label="LUNA minmod"),
                      key("static", "static", True), key("moving", "moving", True), rkey("Sedov-Taylor", REF)], **LEGEND)
    save(fig, "sedov_3d")


def scaling_log():
    """Medians by side of the scaling job: ours {(side, mesh): dict}, LUNA {side: throughput of each round, peak MB}."""
    ours, luna = defaultdict(list), defaultdict(list)
    memory = {}
    for block in re.split(r"^== ", open(SCALING).read(), flags=re.M)[1:]:
        head = block.split("\n")[0].split()
        if head[0] == "ours":
            mesh = "static" if head[2] == "static" else head[3].split("=")[1]
            row = lambda start: next(text for text in block.split("\n") if text.startswith(start))  # noqa: E731
            speed = dict(re.findall(r"(\w+)=([\d.e+-]+)", row("throughput")))
            phase = dict(re.findall(r"(\w+)=([\d.e+-]+)", row("phases")))
            n = int(re.search(r"mesh: n=(\d+)", block)[1])
            mem = re.search(r"peak_MB=([\d.]+).*gpu_used_MB=([\d.]+)", row("memory: phase=steps"))
            ours[int(head[1]), mesh].append(dict(n=n, faces=int(re.search(r"faces=(\d+)", block)[1]), hydro=float(speed["hydro"]), step=float(speed["step"]),
                                                 mesh=n / (float(phase["mesh"]) + float(phase["cells"]) + float(phase["drift"]) or 1) * 1e-3 if mesh != "static" else 0,
                                                 views_MB=float(mem[1]), gpu_MB=float(mem[2])))
        elif head[0] == "luna":
            perf = [float(x) for x in re.findall(r"perf = ([\d.]+) Mcells", block)]
            if perf:
                luna[int(head[1])].append(statistics.median(perf))
                memory[int(head[1])] = 1e3 * float(re.findall(r"mem = ([\d.]+) GB", block)[-1])
    return {k: {f: statistics.median(r[f] for r in rows) for f in rows[0]} for k, rows in ours.items()}, luna, memory


def fig_throughput():
    ours, luna, memory = scaling_log()
    sides = sorted(k for k in {k[0] for k in ours} if k >= 32)
    x = [ours[s, "static"]["n"] for s in sides]
    fig, (top, low) = plt.subplots(2, 1, figsize=(WIDTH, FIGSIZE[1] * 1.45), sharex=True, gridspec_kw=dict(height_ratios=(3, 1.25), hspace=0))
    style = dict(lw=LW, ms=MS + 1, mew=0.8, mfc="none")
    curves = (("static", "hydro", STATIC_C, "-", "s", "hydro, static"), ("two-pass", "hydro", MOVING_C, ":", "^", "hydro, two passes"),
              ("two-pass", "mesh", "0.5", "-", "v", "mesh only"), ("one-pass", "step", MOVING_C, "--", "o", "whole moving step, one pass"),
              ("two-pass", "step", MOVING_C, "-", "D", "whole moving step, two passes"))
    for mesh, field, color, ls, marker, label in curves:
        top.loglog(x, [ours[s, mesh][field] / 1e3 for s in sides], color=color, ls=ls, marker=marker, label=label, **style)
    shown = [s for s in sorted(luna) if s >= 32]
    top.loglog([s ** 3 for s in shown], [statistics.median(luna[s]) / 1e3 for s in shown], color=LUNA_C, marker="o", label="LUNA (MC, Pallas), median", **style)
    top.set_ylabel("G cell updates/s")
    top.set_ylim(top=3)
    corner(top, f"1 cell update = {ours[256, 'static']['faces'] / ours[256, 'static']['n']:.2f} face updates", 0.97, 0.95, "right")
    # From the DRAM bytes per cell by size (research note of branch roofline): all in L2 at 24^3, all in DRAM at 48^3.
    for ax in (top, low):
        ax.axvspan(24 ** 3, 48 ** 3, color="0.90", lw=0, zorder=0)
    TEXTS.append(top.text(32 ** 3, 1.1, "L2 to DRAM", ha="center", fontsize=LABEL))
    top.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, fontsize=LABEL, borderaxespad=0, columnspacing=1.5)
    low.loglog(x, [ours[s, "two-pass"]["gpu_MB"] / 1e3 for s in sides], color=MOVING_C, marker="D", label="ours, two passes, GPU in use", **style)
    low.loglog(x, [ours[s, "two-pass"]["views_MB"] / 1e3 for s in sides], color=MOVING_C, ls="--", marker="D", label="ours, Kokkos views", **style)
    kept = [s for s in shown if memory[s] >= 10]  # LUNA prints two decimals of a GB
    low.loglog([s ** 3 for s in kept], [memory[s] / 1e3 for s in kept], color=LUNA_C, marker="o", label="LUNA, reported", **style)
    low.axhline(42.4, color="0.6", lw=0.8, ls=":")  # the A100's memory
    low.set_ylabel("peak GB")
    low.set_xlabel("cells")
    low.set_xticks([s ** 3 for s in sides], [f"${s}^3$" for s in sides])
    low.set_ylim(0.004, 40000)
    low.legend(loc="upper left", ncol=3, fontsize=LABEL, columnspacing=1.5)
    save(fig, "throughput_scaling")


FIGURES = {"sod": fig_sod, "shu_osher": fig_shu_osher, "woodward_colella": fig_woodward_colella, "convergence_1d": fig_convergence,
           "time_to_target_1d": fig_time, "vortex_3d": fig_vortex, "kh_2d": fig_kh, "sedov_3d": fig_sedov, "throughput_scaling": fig_throughput}


def main():
    for name in sys.argv[1:] or list(FIGURES):
        FIGURES[name]()
    if FAILED:
        raise SystemExit("overlaps: " + "; ".join(f"{name}: {what}" for name, what in FAILED))


if __name__ == "__main__":
    main()
