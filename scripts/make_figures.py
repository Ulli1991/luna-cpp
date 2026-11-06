"""The figures of docs/figures/ from run files. Nothing is typed in: every number is read here.

    python scripts/make_figures.py [names ...]     # default: all; names as in FIGURES below

Run with LUNA's environment: LUNA's src on PYTHONPATH (exact Riemann and Sedov-Taylor solutions), numpy, matplotlib, h5py, Pillow.
Data paths, each an environment variable, none of them in this repository:

    VORONOI_RUNS               voronoi-cpp checkout with the build-* run directories
    KH_TABLE, KH_FIELDS        the KH table and the density fields, default under VORONOI_RUNS
    LUNA_REFERENCE_SOLUTIONS   luna-reference-solutions at 99a52d0 (sha256 checked against its MANIFEST.json)
    LUNA_SEDOV_RUNS            snapshots of the 256^3 Sedov runs; profiles and column densities are cached there
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
import matplotlib.colors
import matplotlib.ticker
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import ConnectionPatch, Rectangle
from PIL import Image
from matplotlib.collections import LineCollection, PolyCollection
from scipy.spatial import Voronoi, cKDTree
from matplotlib.collections import LineCollection, PolyCollection
from scipy.spatial import Voronoi, cKDTree

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
KH = os.environ.get("KH_TABLE", f"{RUNS}/build-2d/runs/kh_table.md")
KH_FIELDS = os.environ.get("KH_FIELDS", f"{RUNS}/build-2d/runs")  # field_* of dim2d_a100.sh khfield
SEDOV = f"{RUNS}/build-limiters/sedov"
SEDOV_L1 = f"{RUNS}/build-limiters/sedov_l1.json"
SEDOV_LADDER = f"{RUNS}/build-metric"
SEDOV_256 = os.environ.get("LUNA_SEDOV_RUNS", f"{RUNS}/sedov_256")  # moving, static, LUNA at 256^3, snapshots every 1000 yr
PRECISION = f"{RUNS}/build-roofline/logs/ladder-31953468.out"  # scripts/profile_a100.sh ladder on branch roofline, e90a913, 1305 MHz
DIMS = f"{RUNS}/build-roofline/logs"  # scripts/dims_a100.sh
SCALING = f"{RUNS}/build-roofline/logs/roof-31928911.out"  # scripts/roofline_a100.sh on branch roofline, job 31928911

LAD_RUNS = f"{RUNS}/build-1dtests/runs"      # scripts/ladder_1d_a100.sh
LADDER = f"{RUNS}/build-1dtests/report/ladder_1d.json"
LADDER_NAMES = {"toro-1": "Toro 1", "toro-2": "Toro 2", "toro-3": "Toro 3", "toro-4": "Toro 4", "toro-5": "Toro 5", "einfeldt-shear": "Einfeldt shear", "la-peak": "LA Peak",
                "la-rcvcr": "LA RCVCR", "la-stationary": "LA stationary", "la-slow": "LA slow", "la-streams": "LA streams", "la-near-vacuum": "LA near vacuum", "leblanc": "LeBlanc",
                "noh": "Noh", "shu-osher": "Shu-Osher", "woodward-colella": "Woodward-Colella", "titarev-toro": "Titarev-Toro", "sound-right": "sound right", "sound-left": "sound left",
                "sound-advected": "sound advected", "simple-wave": "simple wave"}

sys.path.insert(0, HERE)
import ladder_1d as lad  # noqa: E402
import problems_1d as pr1d  # noqa: E402
ldr = lad

spec = importlib.util.spec_from_file_location("compare_1d", f"{HERE}/compare_1d.py")
cmp1d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmp1d)

apply_style()
LUNA_C, STATIC_C, MOVING_C = "C0", "C1", "C2"
LW = 2.2  # every curve; connectors, zoom boxes, cell edges and guide lines stay thin
STYLE = {"luna": dict(color=LUNA_C, ls="-", marker="o", lw=LW, zorder=2),
         "static": dict(color=STATIC_C, ls="--", marker="s", lw=LW, zorder=2.5),
         "moving": dict(color=MOVING_C, ls="-", marker="^", lw=LW, zorder=3),
         "two": dict(color=MOVING_C, ls="--", marker="D", lw=LW, zorder=3),
         "padded": dict(color=MOVING_C, ls="--", marker="v", lw=LW, zorder=3),
         "padded3": dict(color=MOVING_C, ls=":", marker="x", lw=LW, zorder=3)}
REF = dict(color="k", lw=LW, zorder=2.7)
AK = dict(color="k", lw=LW, ls=":", zorder=2.7)
MS = 6.5      # filled markers
LABEL = 10    # letters and labels inside panels, as in LUNA's figures
EDGE = "0.65"  # connectors
CONNECTORS = []  # of the figure being drawn, figure coordinates
TEXTS, BOXES = [], []  # panel labels and box letters, zoom boxes, of the figure being drawn
FAILED = []  # (figure, what) of every overlap
WIDTH = FIGSIZE[0] * 2  # every figure
LEGEND = dict(loc="upper right", fontsize=LABEL)
KH_MESH_WINDOWS = (((0, 1), (0, 1)), ((0.2, 0.5), (0.1, 0.4)), ((0.28, 0.36), (0.2, 0.28)), ((0.328, 0.354), (0.249, 0.275)))  # kh_2d_mesh, nested
KH_VMIN, KH_VMAX = 0.9, 2.1  # LUNA's KH figures, magma (its scripts/validation/plots/kh_grid.py)
KH_WINDOW = ((0.27, 0.57), (0.11, 0.41))  # the roll-up of the zoom panels


def save(fig, name, dpi=None):
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/{name}.png"
    for ax in fig.axes:
        decorate(ax)
        if ax.images:
            ax.grid(False)
    overlaps(fig, name)
    fig.savefig(path, dpi=dpi or plt.rcParams["savefig.dpi"])
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


def cells(ax, x, y, s, xlim, most=14):
    """Actual cells in a window: small filled markers, every cell while they stay apart."""
    k = STYLE[s]
    keep = (x >= xlim[0]) & (x <= xlim[1])
    ax.plot(x[keep], y[keep], color=k["color"], ls=k["ls"], lw=LW, zorder=k["zorder"], marker=k["marker"], ms=MS, mew=0, mfc=k["color"],
            markevery=max(1, keep.sum() // most))


def corner(ax, text, x=0.04, y=0.95, ha="left", va="top"):
    TEXTS.append(ax.text(x, y, text, transform=ax.transAxes, va=va, ha=ha, fontsize=LABEL, bbox=dict(fc="white", ec="none", pad=1.0), zorder=6))


def key(s, label, marker=False):
    k = STYLE[s]
    return plt.Line2D([], [], color=k["color"], ls=k["ls"], lw=k["lw"], marker=k["marker"] if marker else None, ms=MS + (1.5 if s == "static" else 0), mfc=k["color"], label=label)


def rkey(label, kw=REF):
    return plt.Line2D([], [], label=label, **kw)


def loglog(ax, x, y, s, **over):
    k = STYLE[s]
    style = dict(color=k["color"], ls=k["ls"], lw=k["lw"], marker=k["marker"], ms=MS + (1.5 if s == "static" else 0), mfc=k["color"],
                 zorder=1.9 if s == "static" else k["zorder"])  # static under LUNA: its squares show round LUNA's circles
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
    xs, rs, *_ = ours("static", p, n)
    xl, rl, *_ = luna(p, n)
    w = np.load(f"{WIDE_CPU}/aro_shu-osher-wide_{2 * n}.npy")
    xw, rw = w[:, 1], w[:, 3]
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
    line(ax, xw, rw, "padded")
    ax.set_xlim(-5, 5)
    ax.set_ylim(0.4, 5.9)
    ax.set_ylabel(r"$\rho$")
    ax.legend(handles=[key("luna", "LUNA MC"), key("static", "static"), key("padded", "moving, domain [-15, 5]"),
                       rkey("LUNA, high resolution", REF), rkey("AthenaK PPMX", AK)], **LEGEND)
    z = outer[1].subgridspec(1, 4, wspace=0.3).subplots()
    runs = [(xl, rl, "luna"), (xs, rs, "static"), (xw, rw, "padded")]
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


def fig_convergence():
    rows = report()
    wide = wide_rule_l1()
    matrix = [r for r in json.load(open(MATRIX)) if r["problem"] == "shu-osher-wide" and r["config"] == "c04"]
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, FIGSIZE[1] * 1.7), sharex=True, gridspec_kw=dict(hspace=0, wspace=0.34))
    for j, (p, title) in enumerate(PROBLEMS):
        for i, (var, ylabel) in enumerate(ROWS):
            ax = axes[i, j]
            log_axes(ax)
            for s, k in (("static", "static"), ("luna", "luna"), ("moving", "moving-a04")):
                if (s, p) == ("moving", "shu-osher"):  # issue #3: on the padded domain until inflow adds cells
                    continue
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
        corner(axes[0, j], title, 0.95, 0.95, "right")
    axes[1, 0].legend(handles=[key("luna", "LUNA MC", True), key("static", "static", True), key("moving", "moving", True)], loc="lower left", fontsize=LABEL)
    axes[1, 1].legend(handles=[key("padded", "moving, LUNA rule", True), key("padded3", "moving, default rule", True)], **LEGEND)
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
                if (s, p) == ("moving", "shu-osher"):  # issue #3: on the padded domain until inflow adds cells
                    continue
                r = rows[k, p]
                loglog(ax, [x["wall_s"] for x in r], [x["L1"][var] for x in r], s)
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == 1:
                ax.set_xlabel(r"GPU time (s)")
            ax.set_xlim(4e-3, 6e2)
        corner(axes[0, j], title, 0.95, 0.95, "right")
    axes[1, 1].legend(handles=[key("luna", "LUNA MC", True), key("static", "static", True), key("moving", "moving", True)], **LEGEND)
    save(fig, "time_to_target_1d")


def fig_vortex():
    rows = vortex_rows()
    d = reference_file("luna/yee_vortex/yee_vortex_convergence.npz")
    pick = (d["dim"] == 3) & (d["limiter"] == 1) & (d["vx"] == 0) & (d["cfl"] == 0.8)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, FIGSIZE[1]), gridspec_kw=dict(wspace=0.24))
    nx = (32, 64, 128, 256)
    for i, ax, ylabel, v in ((0, axes[0], r"$L_1(\rho)$", "rho"), (1, axes[1], r"$L_1(v)$", "v")):
        log_axes(ax)
        ax.plot(nx, [rows["sta", n][1][i] for n in nx], color="k", ls=":", lw=LW, zorder=1)
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
    axes[0].legend(handles=[key("luna", "LUNA MC", True), plt.Line2D([], [], color=LUNA_C, ls="--", lw=LW, marker="o", ms=MS, label="LUNA MC, point start"),
                            key("static", "static", True), key("moving", "moving, one pass", True), key("two", "moving, two passes", True),
                            rkey(r"$O(h)$ floor", dict(color="k", ls=":", lw=LW))], **LEGEND, ncol=2)
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
    a.legend(handles=[key("luna", "LUNA MC", True), key("static", "static", True), key("moving", "moving, one pass", True), key("two", "moving, two passes", True)], **LEGEND)
    save(fig, "kh_2d")


def kh_field(kind, t, n, m):
    """Density on an m x m lattice, [x, y]: LUNA's cell values; ours, each pixel its nearest cell (periodic) plus the run's limited
    gradient from the centroid, the linear reconstruction of the scheme, as AREPO's images."""
    if kind == "luna":
        rho = np.load(f"{KH_FIELDS}/field_luna_t{t}_{n}.npz")["rho"]
        return rho if rho.shape[0] == m else rho.repeat(m // rho.shape[0], 0).repeat(m // rho.shape[0], 1)
    rows = np.fromfile(f"{KH_FIELDS}/field_{kind}_t{t}_{n}.bin").reshape(-1, 7)
    gradient = np.fromfile(f"{KH_FIELDS}/field_{kind}_t{t}_{n}.grad.bin").reshape(-1, 2)
    at = (np.arange(m) + 0.5) / m
    x, y = np.meshgrid(at, at, indexing="ij")
    pixels = np.c_[x.ravel(), y.ravel()]
    near = cKDTree(rows[:, :2] % 1.0, boxsize=1.0).query(pixels, workers=-1)[1]
    d = (pixels - rows[near, 2:4] + 0.5) % 1.0 - 0.5  # from the centroid, periodic
    return (rows[near, 5] + (gradient[near] * d).sum(1)).reshape(m, m)


def fig_kh_density():
    """Density maps by time, one colour scale; a zoom on a roll-up; the difference from LUNA at the finest N."""
    n = fine = 2048
    window = KH_WINDOW
    times = ("1.0", "1.5", "2.5", "3.0")
    last = 2  # the zoom sits directly under its t = 1.5 row
    rows = (0, 1, 3, 4)
    fig, axes = plt.subplots(5, 4, figsize=(WIDTH, WIDTH * 0.76 * 5 / 3), gridspec_kw=dict(wspace=0.05, hspace=0.07))
    names = (("luna", "LUNA MC"), ("static", "static"), ("moving", "moving"))
    norm = matplotlib.colors.SymLogNorm(linthresh=1e-3, vmin=-1, vmax=1, base=10)
    for r, t in zip(rows, times):
        ref = kh_field("luna", t, fine, fine)
        for c, (kind, name) in enumerate(names):
            ax = axes[r, c]
            rho = kh_field(kind, t, n, n)
            image = ax.imshow(rho.T, origin="lower", extent=(0, 1, 0, 1), cmap="magma", vmin=KH_VMIN, vmax=KH_VMAX, interpolation="nearest")
            corner(ax, f"{name}, $N = {n}$, $t = {t}$", y=0.97)
            if t == "1.5":
                axes[last, c].imshow(rho.T, origin="lower", extent=(0, 1, 0, 1), cmap="magma", vmin=KH_VMIN, vmax=KH_VMAX, interpolation="nearest")
                axes[last, c].set_xlim(*window[0])
                axes[last, c].set_ylim(*window[1])
                corner(axes[last, c], f"{name}, $t = 1.5$, zoom", y=0.97)
                mark(ax, window[0], window[1], "")
                to_fig = fig.transFigure.inverted()
                for x, side in ((window[0][0], 0), (window[0][1], 1)):
                    fig.add_artist(ConnectionPatch(xyA=(x, window[1][0]), coordsA=ax.transData, xyB=(side, 1), coordsB=axes[last, c].transAxes,
                                                   color=EDGE, lw=0.6, clip_on=False, zorder=10))
                    CONNECTORS.append((to_fig.transform(ax.transData.transform((x, window[1][0]))),
                                       to_fig.transform(axes[last, c].transAxes.transform((side, 1)))))
        difference = kh_field("moving", t, n, fine) - ref
        delta = axes[r, 3].imshow(difference.T, origin="lower", extent=(0, 1, 0, 1), cmap="RdBu_r", norm=norm, interpolation="nearest")
        corner(axes[r, 3], "moving $-$ LUNA, sampled", y=0.97)
    bottom = axes[last, 3]
    static = kh_field("static", "1.5", n, fine) - kh_field("luna", "1.5", fine, fine)
    bottom.imshow(static.T, origin="lower", extent=(0, 1, 0, 1), cmap="RdBu_r", norm=norm, interpolation="nearest")
    corner(bottom, "static $-$ LUNA, $t = 1.5$", y=0.97)
    for ax in axes.flat:
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.minorticks_off()
    fig.colorbar(image, ax=axes[:, :3].ravel().tolist(), location="bottom", fraction=0.03, pad=0.02, aspect=50, label=r"$\rho$")
    fig.colorbar(delta, ax=axes[:, 3].ravel().tolist(), location="bottom", fraction=0.03, pad=0.02, aspect=16, label=r"$\Delta\rho$",
                 ticks=(-1, -0.01, 0, 0.01, 1), format="%g")
    save(fig, "kh_2d_density")


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
    b.legend(handles=[key("luna", "LUNA MC", True), plt.Line2D([], [], color=LUNA_C, ls="--", lw=LW, marker="o", ms=MS, label="LUNA minmod"),
                      key("static", "static", True), key("moving", "moving", True), rkey("Sedov-Taylor", REF)], **LEGEND)
    save(fig, "sedov_3d")


KPC, MSUN, YEAR, PC, MP, KB = 3.08568e21, 1.98892e33, 3.15576e7, 3.08568e18, 1.67262e-24, 1.38065e-16
UNIT_T, UNIT_RHO, UNIT_P = KPC / 1e5, 1e10 * MSUN / KPC ** 3, 1e10 * MSUN / (KPC * (KPC / 1e5) ** 2)
BOX = 0.05  # kpc
RUNS_3D = {"moving": "moving, two passes", "static": "static", "luna": "LUNA MC"}
SIDE = 256


def sedov_files(run):
    return sorted(glob.glob(f"{SEDOV_256}/{run}_{SIDE}/snapdir_*/*.hdf5"))


def sedov_cells(path):
    """Position (kpc, from the box corner), volume, density, velocity, pressure, time of a snapshot; ours Voronoi, LUNA's grid."""
    with h5py.File(path, "r") as f:
        c = f["cells"]
        rho = np.asarray(c["density"][...], float)
        v = np.stack([np.asarray(c[f"velocity_{a}"][...], float).ravel() for a in "xyz"])
        P = np.asarray(c["pressure"][...], float).ravel()
        t = float(c.attrs["time"])
        if "x" in c:
            pos, vol = np.stack([c[a][...] for a in "xyz"]).astype(float), np.asarray(c["volume"][...], float)
        else:
            n = rho.shape[0]
            g = (np.arange(n) + 0.5) * BOX / n
            X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
            pos, vol = np.stack([X.ravel(), Y.ravel(), Z.ravel()]), np.full(n ** 3, (BOX / n) ** 3)
    return pos, vol, rho.ravel(), v, P, t


def sedov_profiles(run):
    """Volume-weighted shells about the box centre, n_x / 2 bins out to the corner (LUNA's radial_profile), per snapshot; cached."""
    cache = f"{SEDOV_256}/profiles_{run}.npz"
    if not os.path.exists(cache):
        edges = np.linspace(0, np.sqrt(3) * BOX / 2, SIDE // 2 + 1)
        rows = []
        for path in sedov_files(run):
            pos, vol, rho, v, P, t = sedov_cells(path)
            d = pos - BOX / 2
            r = np.sqrt((d ** 2).sum(0))
            weight = np.maximum(np.histogram(r, edges, weights=vol)[0], 1e-300)
            shell = lambda f: np.histogram(r, edges, weights=f * vol)[0] / weight  # noqa: E731
            rows.append((t, shell(rho), shell((v * d).sum(0) / np.maximum(r, 1e-30)), shell(P)))
        np.savez(cache, r=0.5 * (edges[1:] + edges[:-1]), t=[x[0] for x in rows], rho=[x[1] for x in rows], vr=[x[2] for x in rows], P=[x[3] for x in rows])
    return np.load(cache)


def sedov_column(run, m=2 * SIDE, samples=4 * SIDE):
    """Column density along z on m^2 rays, g cm^-2. LUNA: its cell values summed along z, as its own figure. Ours: midpoint rule with
    `samples` points along each ray, each point in its nearest cell (the Voronoi cell the ray crosses) with the run's limited
    gradient from the centroid, the linear reconstruction in cells as in AREPO's projections. Cached."""
    cache = f"{SEDOV_256}/column_{run}.npz"
    if not os.path.exists(cache):
        pos, _, rho, _, _, _ = sedov_cells(sedov_files(run)[-1])
        if run == "luna":
            n = round(rho.size ** (1 / 3))
            column = rho.reshape(n, n, n).sum(axis=2) * BOX / n * UNIT_RHO * KPC
            column = column.repeat(m // n, 0).repeat(m // n, 1)
        else:
            extra = np.fromfile(f"{SEDOV_256}/{run}_{SIDE}/gradients.bin").reshape(-1, 6)
            centroid, gradient = (pos.T + extra[:, :3]), extra[:, 3:]
            h = BOX / m
            g = (np.arange(m) + 0.5) * h
            tree = cKDTree(pos.T % BOX, boxsize=BOX)
            X, Y = np.meshgrid(g, g, indexing="ij")
            total = np.zeros((m, m))
            for z in (np.arange(samples) + 0.5) * BOX / samples:
                at = np.stack([X.ravel(), Y.ravel(), np.full(m * m, z)], axis=1)
                near = tree.query(at, workers=-1)[1]
                d = (at - centroid[near] + BOX / 2) % BOX - BOX / 2
                total += (rho[near] + (gradient[near] * d).sum(1)).reshape(m, m)
            column = total * BOX / samples * UNIT_RHO * KPC
        np.savez(cache, column=column)
    return np.load(cache)["column"]


def filled(values):
    """A shell with no generator in it has no value: leave a gap in the curve."""
    return np.where(values == 0, np.nan, values)


def fig_sedov_shock():
    prof = {run: sedov_profiles(run) for run in ("moving", "luna")}
    rho0, e0 = MP / UNIT_RHO, 1e51 / (1e10 * MSUN * 1e10)
    fig = plt.figure(figsize=(WIDTH, FIGSIZE[1] * 2.2))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.2)
    top = outer[0].subgridspec(1, 2, wspace=0.22)
    a, b = fig.add_subplot(top[0]), fig.add_subplot(top[1])
    maps = outer[1].subgridspec(1, 4, width_ratios=[1, 1, 0.42, 1], wspace=0.06)
    names = {"luna": (LUNA_C, "o"), "moving": (MOVING_C, "^")}
    for run, (color, marker) in names.items():
        p = prof[run]
        t = p["t"][1:] * UNIT_T / YEAR
        peak = p["r"][p["rho"][1:].argmax(axis=1)] * KPC / PC
        a.loglog(t, peak, color=color, marker=marker, ms=MS, lw=LW, label=f"{RUNS_3D[run]}, peak-$\\rho$ location")
    t_all = prof["luna"]["t"][1:]
    t_line = np.geomspace(500, 1.4 * t_all[-1] * UNIT_T / YEAR, 200)
    a.loglog(t_line, [exact_sedov.solution(np.array([0.0]), x * YEAR / UNIT_T, e0, rho0, 5 / 3)[3] * KPC / PC for x in t_line], color="0.5", lw=1.0,
             label=rf"$R_s(t) = \xi (E t^2/\rho_0)^{{1/5}}$, $\xi$={exact_sedov.xi0(5 / 3):.2f}")
    a.set_xlabel("t [yr]")
    a.set_ylabel(r"$R_{shock}$ [pc]")
    a.legend(loc="upper left", fontsize=LABEL)
    r_fine = np.linspace(0.02, 25, 3000) * PC / KPC
    pick = [i for i, t in enumerate(prof["luna"]["t"]) if i in (1, 2, 4, 7, 10)]
    for i in pick:
        t = prof["luna"]["t"][i]
        rho, _, _, R = exact_sedov.solution(r_fine, t, e0, rho0, 5 / 3)
        b.semilogy(r_fine * KPC / PC, np.maximum(rho, 1e-300) * UNIT_RHO / MP, **REF)
        for run, (color, _) in names.items():
            b.semilogy(prof[run]["r"] * KPC / PC, filled(prof[run]["rho"][i]) * UNIT_RHO / MP, color=color, lw=LW)
        b.text(R * KPC / PC, 4.6, f"{t * UNIT_T / YEAR / 1e3:.0f}", ha="center", fontsize=LABEL)
    b.set_xlim(0, 16)
    b.set_ylim(1e-3, 100)
    b.set_xlabel("r [pc]")
    b.set_ylabel(r"n [cm$^{-3}$]")
    b.legend(handles=[plt.Line2D([], [], color=MOVING_C, lw=LW, label=RUNS_3D["moving"]), plt.Line2D([], [], color=LUNA_C, lw=LW, label=RUNS_3D["luna"]), rkey("Sedov-Taylor", REF)],
             loc="lower right", fontsize=LABEL)
    corner(b, r"$t$ [$10^3$ yr] above each shock", 0.03, 0.97)
    ours, theirs = sedov_column("moving") / 1e-4, sedov_column("luna") / 1e-4
    lo, hi = min(ours.min(), theirs.min()), max(ours.max(), theirs.max())
    L = BOX * KPC / PC
    spread = abs(ours - theirs).max()
    panels = [(0, "moving, two passes", ours, dict(cmap="magma", vmin=lo, vmax=hi), None),
              (1, "LUNA MC", theirs, dict(cmap="magma", vmin=lo, vmax=hi), r"$\Sigma$ [$10^{-4}$ g cm$^{-2}$]"),
              (3, "moving $-$ LUNA", ours - theirs, dict(cmap="RdBu_r", vmin=-spread, vmax=spread), r"$\Delta\Sigma$ [$10^{-4}$ g cm$^{-2}$]")]
    for slot, label, field, kw, title in panels:
        ax = fig.add_subplot(maps[slot])
        im = ax.imshow(field.T, origin="lower", extent=(0, L, 0, L), aspect="equal", **kw)
        ax.grid(False)
        ax.set_xticks((0, 20, 40))
        ax.set_xlabel("x [pc]")
        if slot == 0:
            ax.set_ylabel("y [pc]")
        else:
            ax.tick_params(labelleft=False)
        corner(ax, label, 0.03, 0.97)
        if title:
            fig.colorbar(im, cax=ax.inset_axes([1.03, 0, 0.05, 1]), label=title)
    save(fig, "sedov3d_shock")


def fig_sedov_radial():
    prof = {run: sedov_profiles(run) for run in RUNS_3D}
    rho0, e0 = MP / UNIT_RHO, 1e51 / (1e10 * MSUN * 1e10)
    t = prof["luna"]["t"][-1]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, FIGSIZE[1] * 1.8), gridspec_kw=dict(hspace=0.28, wspace=0.26))
    r_fine = np.linspace(0.02, 25, 4000) * PC / KPC
    rho, u, P, R = exact_sedov.solution(r_fine, t, e0, rho0, 5 / 3)
    ex = dict(n=rho * UNIT_RHO / MP, v=u, P=P * UNIT_P)
    style = {"moving": (MOVING_C, "-"), "static": (STATIC_C, "--"), "luna": (LUNA_C, "-")}
    for run, (color, ls) in style.items():
        p = prof[run]
        r = p["r"] * KPC / PC
        n, vr, Pc = filled(p["rho"][-1]) * UNIT_RHO / MP, np.where(p["rho"][-1] == 0, np.nan, p["vr"][-1]), filled(p["P"][-1]) * UNIT_P
        kw = dict(color=color, ls=ls, lw=LW, label=RUNS_3D[run])
        axes[0, 0].plot(r, n, **kw)
        axes[0, 1].plot(r, vr, **kw)
        axes[1, 0].plot(r, Pc, **kw)
        axes[1, 1].plot(r, Pc / np.maximum(n, 1e-30) / KB, **kw)
    inside = r_fine < R
    axes[0, 0].plot(r_fine[inside] * KPC / PC, ex["n"][inside], **REF, label="Sedov-Taylor")
    axes[0, 1].plot(r_fine[inside] * KPC / PC, ex["v"][inside], **REF)
    axes[1, 0].plot(r_fine[inside] * KPC / PC, ex["P"][inside], **REF)
    meta = ((axes[0, 0], "n [cm$^{-3}$]", "log"), (axes[0, 1], r"$v_r$ [km s$^{-1}$]", "linear"), (axes[1, 0], "P [erg cm$^{-3}$]", "log"), (axes[1, 1], "T [K]", "log"))
    for ax, ylabel, scale in meta:
        ax.axvline(R * KPC / PC, color="0.5", lw=1.0, label=rf"$R_s$ analytic = {R * KPC / PC:.1f} pc" if ax is axes[0, 0] else None)
        ax.set_xlim(0, 25)
        ax.set_yscale(scale)
        ax.set_xlabel("r [pc]")
        ax.set_ylabel(ylabel)
    axes[0, 0].set_ylim(1e-4, 30)
    axes[0, 0].legend(loc="lower right", fontsize=LABEL)
    save(fig, "sedov3d_radial")


def voronoi_polygons(points, values, window, margin=0.03):
    """Exact Voronoi cells of periodic generators (unit box) touching a window, from scipy.spatial.Voronoi with periodic images: polygons and their values."""
    (x0, x1), (y0, y1) = window
    near, value = [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            q = points % 1.0 + (dx, dy)
            keep = (q[:, 0] > x0 - margin) & (q[:, 0] < x1 + margin) & (q[:, 1] > y0 - margin) & (q[:, 1] < y1 + margin)
            near.append(q[keep])
            value.append(values[keep])
    near, value = np.concatenate(near), np.concatenate(value)
    v = Voronoi(near)
    polygons, colours = [], []
    for i, region in enumerate(v.point_region):
        ring = v.regions[region]
        if not ring or -1 in ring:
            continue
        p = v.vertices[ring]
        if p[:, 0].max() > x0 and p[:, 0].min() < x1 and p[:, 1].max() > y0 and p[:, 1].min() < y1:
            polygons.append(p)
            colours.append(value[i])
    return near, polygons, np.array(colours)


def cell_collection(ax, polygons, colours, edge=None, lw=0.0, **kw):
    """Cells as polygons, coloured by density, thin edges or none."""
    cells = PolyCollection(polygons, array=colours, edgecolors=edge if edge else "face", linewidths=lw if edge else 0.15, **kw)
    ax.add_collection(cells)
    return cells


def smooth_map(rows, gradient, window, m):
    """Nearest cell plus the run's limited gradient from the centroid, periodic, on m^2 pixels of a window: the field the scheme has."""
    (x0, x1), (y0, y1) = window
    x, y = np.meshgrid(x0 + (np.arange(m) + 0.5) * (x1 - x0) / m, y0 + (np.arange(m) + 0.5) * (y1 - y0) / m, indexing="ij")
    pixels = np.c_[x.ravel(), y.ravel()]
    near = cKDTree(rows[:, :2] % 1.0, boxsize=1.0).query(pixels % 1.0, workers=-1)[1]
    d = (pixels - rows[near, 2:4] + 0.5) % 1.0 - 0.5
    return (rows[near, 5] + (gradient[near] * d).sum(1)).reshape(m, m)


def fading_edges(ax, rows, window, pixels, lw, full=8.0):
    """The Voronoi edges of a window, exact, thin and semi-transparent, fading out below `full` pixels of length."""
    (x0, x1), (y0, y1) = window
    near, _, _ = voronoi_polygons(rows[:, :2], rows[:, 5], window, margin=0.03)
    v = Voronoi(near)
    ridges = np.array([r for r in v.ridge_vertices if min(r) >= 0])
    segments = v.vertices[ridges]
    keep = (segments[:, :, 0].max(1) > x0) & (segments[:, :, 0].min(1) < x1) & (segments[:, :, 1].max(1) > y0) & (segments[:, :, 1].min(1) < y1)
    segments = segments[keep]
    length = np.hypot(*(segments[:, 1] - segments[:, 0]).T) / (x1 - x0) * pixels
    colours = np.ones((len(segments), 4))
    colours[:, 3] = 0.6 * np.clip(length / full, 0, 1)
    ax.add_collection(LineCollection(segments, colors=colours, linewidths=lw))


def fig_kh_mesh():
    """The moving mesh as AREPO's figures show it: the reconstructed field, the Voronoi edges over it, fading where cells are unresolved;
    the whole box and three nested zooms into a billow, each box marked on the panel before."""
    t, n = "4.0", 512
    rows = np.fromfile(f"{SEDOV_256}/kh_mesh512_t{t}.bin").reshape(-1, 7)
    gradient = np.fromfile(f"{SEDOV_256}/kh_mesh512_t{t}.grad.bin").reshape(-1, 2)
    windows = KH_MESH_WINDOWS
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, WIDTH * 0.235), gridspec_kw=dict(wspace=0.09, left=0.005, right=0.925, top=0.99, bottom=0.01))
    labels = (f"moving, two passes, $N = {n}$, $t = 4$", "(a)", "(b)", "(c)")
    pixels = 1000
    for ax, window, label, lw in zip(axes, windows, labels, (0.1, 0.15, 0.3, 0.5)):
        image = ax.imshow(smooth_map(rows, gradient, window, pixels).T, origin="lower", extent=(*window[0], *window[1]), cmap="magma", vmin=KH_VMIN, vmax=KH_VMAX,
                          interpolation="bilinear")
        fading_edges(ax, rows, window, pixels, lw)
        ax.set_xlim(*window[0])
        ax.set_ylim(*window[1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.minorticks_off()
        corner(ax, label, y=0.97)
    to_fig = fig.transFigure.inverted()
    for ax, target, inner, letter in zip(axes, axes[1:], windows[1:], "abc"):
        mark(ax, inner[0], inner[1], letter)
        TEXTS[-1].set_bbox(dict(fc="white", ec="none", pad=0.8))
        for y, side in ((inner[1][0], 0), (inner[1][1], 1)):
            fig.add_artist(ConnectionPatch(xyA=(inner[0][1], y), coordsA=ax.transData, xyB=(0, side), coordsB=target.transAxes, color=EDGE, lw=0.6, clip_on=False, zorder=10))
            CONNECTORS.append((to_fig.transform(ax.transData.transform((inner[0][1], y))), to_fig.transform(target.transAxes.transform((0, side)))))
    fig.colorbar(image, ax=axes.ravel().tolist(), location="right", fraction=0.012, pad=0.01, label=r"$\rho$")
    save(fig, "kh_2d_mesh", dpi=300)


def fig_kh_evolution():
    """Moving and LUNA MC at N = 2048, to t = 5: ours the reconstructed field, nearest cell plus the limited gradient; LUNA cell values."""
    n = 2048
    times = ("0", "1.0", "2.5", "4.0", "5.0")
    fig, axes = plt.subplots(2, 5, figsize=(WIDTH, WIDTH * 0.385), gridspec_kw=dict(wspace=0.04, hspace=0.04, left=0.01, right=0.92, top=0.99, bottom=0.01))
    for r, (kind, name) in enumerate((("moving", "moving"), ("luna", "LUNA MC"))):
        for c, t in enumerate(times):
            ax = axes[r, c]
            image = ax.imshow(kh_field(kind, t, n, n).T, origin="lower", extent=(0, 1, 0, 1), cmap="magma", vmin=KH_VMIN, vmax=KH_VMAX, interpolation="nearest")
            corner(ax, f"{name}, $N = {n}$, $t = {float(t):g}$", y=0.97)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.minorticks_off()
    fig.colorbar(image, ax=axes.ravel().tolist(), location="right", fraction=0.012, pad=0.01, label=r"$\rho$")
    save(fig, "kh_2d_evolution", dpi=300)


def plane_cells(run, window, pad=3.0):
    """Sections of the 3D Voronoi cells by the plane z = 25 pc, exact: scipy.spatial.Voronoi on the generators within pad pc of the window,
    each ridge cut by the plane gives one edge of both its cells. Polygons in pc, n in cm^-3, generators near the plane."""
    pos, _, rho, _, _, _ = sedov_cells(sedov_files(run)[-1])
    p = pos * KPC / PC
    (x0, x1), (y0, y1) = window
    keep = (p[0] > x0 - pad) & (p[0] < x1 + pad) & (p[1] > y0 - pad) & (p[1] < y1 + pad) & (abs(p[2] - 25) < pad)
    p, n = p[:, keep].T, rho[keep] * UNIT_RHO / MP
    v = Voronoi(p)
    edges = defaultdict(list)
    for (a, b), ridge in zip(v.ridge_points, v.ridge_vertices):
        if -1 in ridge:
            continue
        q = v.vertices[ridge]
        z = q[:, 2] - 25
        if z.min() >= 0 or z.max() <= 0:
            continue
        i, j = np.where(z > 0)[0], np.where(z <= 0)[0]
        s_ = z[i][:, None] / (z[i][:, None] - z[j][None, :])
        cut = (q[i][:, None, :2] + s_[:, :, None] * (q[j][None, :, :2] - q[i][:, None, :2])).reshape(-1, 2)
        axis = cut - cut.mean(0)
        along = axis @ np.linalg.svd(axis, full_matrices=False)[2][0]
        ends = cut[[along.argmin(), along.argmax()]]
        edges[a].append(ends)
        edges[b].append(ends)
    polygons, colours = [], []
    for cell, parts in edges.items():
        c = np.concatenate(parts)
        if c[:, 0].max() < x0 or c[:, 0].min() > x1 or c[:, 1].max() < y0 or c[:, 1].min() > y1:
            continue
        mid = c.mean(0)
        polygons.append(c[np.argsort(np.arctan2(c[:, 1] - mid[1], c[:, 0] - mid[0]))])
        colours.append(n[cell])
    return polygons, np.array(colours), p[abs(p[:, 2] - 25) < 0.5 * 50 / SIDE]


def fig_sedov_mesh():
    box, shell = ((25, 41), (17, 33)), ((34.5, 39.5), (22.5, 27.5))
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, WIDTH * 0.36), gridspec_kw=dict(wspace=0.06))
    norm = matplotlib.colors.LogNorm(1e-2, 5)
    polygons, colours, _ = plane_cells("moving", box)
    image = cell_collection(axes[0], polygons, colours, cmap="magma", norm=norm)
    axes[0].set_xlim(*box[0])
    axes[0].set_ylim(*box[1])
    axes[0].set_aspect("equal")
    corner(axes[0], "(a) moving, $z = 25$ pc", y=0.97)
    for ax, run, label in ((axes[1], "moving", "(b) moving, two passes"), (axes[2], "static", "(c) static")):
        polygons, colours, p = plane_cells(run, shell, 1.5)
        cell_collection(ax, polygons, colours, "white", 0.3, cmap="magma", norm=norm)
        ax.plot(p[:, 0], p[:, 1], ls="none", marker=".", ms=1.5, color="k", zorder=3)
        ax.set_xlim(*shell[0])
        ax.set_ylim(*shell[1])
        ax.set_aspect("equal")
        corner(ax, label, y=0.97)
    mark(axes[0], shell[0], shell[1], "")
    to_fig = fig.transFigure.inverted()
    for y, side in ((shell[1][0], 0), (shell[1][1], 1)):
        fig.add_artist(ConnectionPatch(xyA=(shell[0][1], y), coordsA=axes[0].transData, xyB=(0, side), coordsB=axes[1].transAxes, color=EDGE, lw=0.6, clip_on=False, zorder=10))
    for ax in axes:
        ax.set_xlabel("x [pc]")
    axes[0].set_ylabel("y [pc]")
    axes[1].tick_params(labelleft=False)
    axes[2].tick_params(labelleft=False)
    fig.colorbar(image, ax=axes.ravel().tolist(), location="right", fraction=0.02, pad=0.015, label=r"n [cm$^{-3}$]")
    save(fig, "sedov3d_mesh")


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
    style = dict(lw=LW, ms=MS)
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
    top.legend(loc="lower right", ncol=2, fontsize=LABEL, columnspacing=1.5)
    low.loglog(x, [ours[s, "two-pass"]["gpu_MB"] / 1e3 for s in sides], color=MOVING_C, marker="D", label="ours, two passes, GPU in use", **style)
    low.loglog(x, [ours[s, "two-pass"]["views_MB"] / 1e3 for s in sides], color=MOVING_C, ls="--", marker="D", mfc="white", label="ours, Kokkos views", **style)
    kept = [s for s in shown if memory[s] >= 10]  # LUNA prints two decimals of a GB
    low.loglog([s ** 3 for s in kept], [memory[s] / 1e3 for s in kept], color=LUNA_C, marker="o", label="LUNA, reported", **style)
    low.axhline(42.4, color="0.6", lw=0.8, ls=":")  # the A100's memory
    low.set_ylabel("peak GB")
    low.set_xlabel("cells")
    low.set_xticks([s ** 3 for s in sides], [f"${s}^3$" for s in sides])
    low.set_ylim(0.004, 40000)
    low.legend(loc="upper left", ncol=3, fontsize=LABEL, columnspacing=1.5)
    save(fig, "throughput_scaling")


def precision_log():
    """The ladder of scripts/profile_a100.sh (branch roofline): ours {(side, mesh, preset): rates and GB}, what did not fit, LUNA {(side, FP): throughput of each round}, its peak GB."""
    ours, failed, luna, memory = {}, [], defaultdict(list), {}
    for block in re.split(r"^== ", open(PRECISION).read(), flags=re.M)[1:]:
        head = block.split("\n")[0].split()
        if head[0] == "luna":
            perf = [float(x) for x in re.findall(r"perf = ([\d.]+) Mcells", block)]
            if perf:
                key = int(head[1]), "FP64" if head[2] == "x64=1" else "FP32"
                luna[key].append(statistics.median(perf) * 1e-3)
                memory[key] = float(re.findall(r"mem = ([\d.]+) GB", block)[-1])
        elif head[0] == "ours":
            side, mesh, preset = int(head[1]), head[2], head[3].split("=")[1]
            speed = re.search(r"^throughput_M_per_s: (.*)$", block, re.M)
            if not speed:
                failed.append((side, mesh, preset))
                continue
            rates = dict(re.findall(r"(\w+)=([\d.e+-]+)", speed[1]))
            mem = re.search(r"^memory: phase=steps.*?gpu_used_MB=([\d.]+)", block, re.M)
            ours[side, mesh, preset] = dict(hydro=float(rates["hydro"]) * 1e-3, step=float(rates["step"]) * 1e-3, GB=float(mem[1]) * 1e-3)
    return ours, failed, luna, memory


def fig_throughput_precision():
    ours, failed, luna, memory = precision_log()
    presets = (("default", "double", "k", "-", "o"), ("grad32", "grad32", "C1", "-", "s"), ("grad16", "grad16", "C2", "--", "^"), ("flux32", "flux32", "C3", ":", "D"), ("mixed", "mixed", "C4", "-.", "v"))
    sides = sorted({k[0] for k in ours})
    cells = [s ** 3 for s in sides]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, FIGSIZE[1] * 1.45), sharex=True, sharey="row", gridspec_kw=dict(height_ratios=(3, 1.25), hspace=0, wspace=0))
    style = dict(lw=LW, ms=MS)
    for column, (mesh, field, title) in enumerate((("static", "hydro", "hydro, static"), ("two", "step", "whole moving step, two passes"))):
        top, low = axes[0, column], axes[1, column]
        for preset, label, color, ls, marker in presets:
            have = [s for s in sides if (s, mesh, preset) in ours]
            top.loglog([s ** 3 for s in have], [ours[s, mesh, preset][field] for s in have], color=color, ls=ls, marker=marker, label=label, **style)
            low.loglog([s ** 3 for s in have], [ours[s, mesh, preset]["GB"] for s in have], color=color, ls=ls, marker=marker, **style)
        for fp, ls in (("FP64", "-"), ("FP32", "--")):
            have = [s for s in sides if (s, fp) in luna]
            top.loglog([s ** 3 for s in have], [statistics.median(luna[s, fp]) for s in have], color=LUNA_C, ls=ls, marker="x", label=f"LUNA {fp}", **style)
            kept = [s for s in have if memory[s, fp] >= 0.01]  # LUNA prints two decimals of a GB
            low.loglog([s ** 3 for s in kept], [memory[s, fp] for s in kept], color=LUNA_C, ls=ls, marker="x", **style)
        low.axhline(42.4, color="0.6", lw=0.8, ls=":")  # the A100's memory
        corner(top, title, 0.03, 0.95)
        low.set_xlabel("cells")
        low.set_xticks(cells, [f"${s}^3$" for s in sides])
        low.minorticks_off()
    for side, mesh, preset in failed:
        if mesh == "two":
            TEXTS.append(axes[1, 1].text(side ** 3, 0.02, f"{preset}: does not fit at ${side}^3$", ha="right", va="bottom", fontsize=LABEL))
    axes[0, 0].set_ylim(0.006, 3)
    axes[1, 0].set_ylim(0.005, 200)
    axes[0, 0].set_ylabel("G cell updates/s")
    axes[1, 0].set_ylabel("GPU GB")
    axes[0, 0].legend(loc="lower right", ncol=2, fontsize=LABEL, columnspacing=1.2)
    save(fig, "throughput_precision")


def dims_log(dim):
    """The sweep of scripts/dims_a100.sh in one dimension: ours {(cells, mode): dict of G cell updates/s and GB}, LUNA {cells: median of the perf lines of each round}, its GB."""
    path = sorted(glob.glob(f"{DIMS}/dims-{dim}-*.out"))[-1]
    ours, luna, memory, failed = {}, defaultdict(list), {}, []
    for block in re.split(r"^== ", open(path).read(), flags=re.M)[1:]:
        head = block.split("\n")[0].split()
        side = int(head[1])
        cells = side ** dim if dim > 1 else side
        if head[0] == "luna":
            perf = [float(x) for x in re.findall(r"perf = ([\d.]+) Mcells", block)]
            if perf:
                luna[cells].append(statistics.median(perf) * 1e-3)
                memory[cells] = float(re.findall(r"mem = ([\d.]+) GB", block)[-1])
            continue
        if head[0] != "ours":
            continue
        mode = "static" if head[2] == "static" else head[2] if dim < 3 else head[2].split("=")[-1] if "=" in head[2] else head[3].split("=")[1]
        if dim == 3:
            mode = "static" if head[2] == "static" else head[3].split("=")[1]
            speed = re.search(r"^throughput_M_per_s: (.*)$", block, re.M)
            mem = re.search(r"^memory: phase=steps.*?peak_MB=([\d.]+)", block, re.M)
            if not speed:
                failed.append((cells, mode))
                continue
            rates = dict(re.findall(r"(\w+)=([\d.e+-]+)", speed[1]))
            phase = dict(re.findall(r"(\w+)=([\d.e+-]+)", re.search(r"^phases_ms_median: (.*)$", block, re.M)[1]))
            mesh_ms = float(phase["mesh"]) + float(phase["cells"]) + float(phase["drift"])
            row = dict(hydro=float(rates["hydro"]) * 1e-3, step=float(rates["step"]) * 1e-3, mesh=cells / mesh_ms * 1e-6 if mesh_ms > 0 else 0, GB=float(mem[1]) * 1e-3)
        elif dim == 2:
            line = re.search(r"^phase: M cells/s hydro ([\d.]+) .*? mesh ([\d.]+) ", block, re.M)
            whole = re.search(r"^ours: .*?M_cell_updates_per_s=([\d.]+)", block, re.M)
            mem = re.search(r"^memory: peak ([\d.]+) B/cell, ([\d.]+) GB", block, re.M)
            if not whole or not line:
                failed.append((cells, mode))
                continue
            row = dict(hydro=float(line[1]) * 1e-3, step=float(whole[1]) * 1e-3, mesh=float(line[2]) * 1e-3, GB=float(mem[2]) if mem else 0)
        else:
            phase = dict(re.findall(r"^phase: (\w+) M_cell_updates_per_s=([\d.e+-]+)", block, re.M))
            whole = re.search(r"^ours: .*?M_cell_updates_per_s=([\d.e+-]+)", block, re.M)
            mem = re.search(r"^memory: peak_MB=([\d.]+)", block, re.M)
            if not whole:
                failed.append((cells, mode))
                continue
            row = dict(hydro=1e-3 / (1 / float(phase["time_step"]) + 1 / float(phase["hydro"])), step=float(whole[1]) * 1e-3,
                       mesh=float(phase.get("mesh", 0)) * 1e-3, GB=float(mem[1]) * 1e-3 if mem else 0)
        ours[cells, mode] = row
    clock = re.search(r"Mcells|NVIDIA[^\n]*", open(path).read())
    return ours, luna, memory, failed, clock[0]


def fig_throughput_dims():
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH * 1.75, FIGSIZE[1] * 1.45), sharex="col", gridspec_kw=dict(height_ratios=(3, 1.25), hspace=0, wspace=0.24))
    style = dict(lw=LW, ms=MS)
    curves = (("static", "hydro", STATIC_C, "-", "s", "hydro, static"), ("two-pass", "hydro", MOVING_C, ":", "^", "hydro, two passes"),
              ("one-pass", "mesh", "0.5", "-", "v", "mesh only"), ("one-pass", "step", MOVING_C, "--", "o", "whole step, one pass"),
              ("two-pass", "step", MOVING_C, "-", "D", "whole step, two passes"))
    for column, dim in enumerate((1, 2, 3)):
        ours, luna, memory, failed, _ = dims_log(dim)
        top, low = axes[0, column], axes[1, column]
        cells = sorted({k[0] for k in ours})
        for mode, field, color, ls, marker, label in curves:
            have = [c for c in cells if (c, mode) in ours and ours[c, mode][field] > 0]
            if have:
                top.loglog(have, [ours[c, mode][field] for c in have], color=color, ls=ls, marker=marker, label=label, **style)
        shown = sorted(luna)
        top.loglog(shown, [statistics.median(luna[c]) for c in shown], color=LUNA_C, marker="o", label="LUNA (MC, Pallas)", **style)
        corner(top, f"{dim}D")
        for mode, color, marker, ls in (("two-pass", MOVING_C, "D", "-"), ("one-pass", MOVING_C, "o", "--")):
            have = [c for c in cells if (c, mode) in ours and ours[c, mode]["GB"] > 0]
            if have:
                low.loglog(have, [ours[c, mode]["GB"] for c in have], color=color, ls=ls, marker=marker, **style)
                break
        kept = [c for c in shown if memory[c] >= 0.01]  # LUNA prints two decimals of a GB
        low.loglog(kept, [memory[c] for c in kept], color=LUNA_C, marker="o", **style)
        low.axhline(42.4, color="0.6", lw=0.8, ls=":")  # the A100's memory
        low.set_xlabel("cells")
        low.set_ylim(1e-4, 400)
    axes[0, 0].set_ylabel("G cell updates/s")
    axes[1, 0].set_ylabel("peak GB")
    axes[0, 0].set_ylim(top=300)
    axes[0, 1].set_ylim(0.06, 8)
    axes[0, 2].set_ylim(0.003, 3)
    axes[0, 0].legend(loc="lower right", fontsize=LABEL - 1)
    axes[0, 2].legend(loc="lower right", fontsize=LABEL - 1)
    save(fig, "throughput_dimensions")


def ladder_rows():
    """The 1D ladder (scripts/ladder_1d.py table), and the orders of its series."""
    rows = json.load(open(LADDER))
    return rows, lad.orders(rows)


LADDER_PANELS = [p for p in lad.ALL if p not in ("einfeldt-strong", "la-vacuum")]  # einfeldt-strong is Toro 2


def fig_ladder():
    """L1(rho) against N for every 1D test, ours beside LUNA's, measured order in each panel."""
    rows, res = ladder_rows()
    cols = 5
    lines = (len(LADDER_PANELS) + cols - 1) // cols
    fig, axes = plt.subplots(lines, cols, figsize=(WIDTH, FIGSIZE[1] * 0.56 * lines), sharex=True, gridspec_kw=dict(hspace=0, wspace=0.3))
    for ax in axes.ravel()[len(LADDER_PANELS):]:
        ax.set_visible(False)
    for ax, p in zip(axes.ravel(), LADDER_PANELS):
        log_axes(ax)
        labels = []
        tops = []
        for series, style, tag in (("static", "static", "S"), ("luna-mc", "luna", "L"), ("moving", "moving", "M"), ("moving-wide", "padded", "W")):
            o = res.get((p, series, 0))
            if o is None:
                continue
            loglog(ax, o["N"], o["L1"], style, ms=MS * 0.7)
            tops += o["L1"]
            if o["p_all"] is not None:
                labels.append(f"{tag} {o['p_all']:.2f}")
        ax.set_xlim(150, 3e5)
        ax.set_ylim(min(tops) / 3, max(tops) * 40)
        ax.tick_params(labelsize=LABEL)
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=4))
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        corner(ax, LADDER_NAMES.get(p, p), 0.95, 0.95, "right")
        corner(ax, "  ".join(labels), 0.04, 0.05, va="bottom")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$L_1(\\rho)$", fontsize=LABEL + 1)
    for j, ax in enumerate(axes.ravel()):
        if ax.get_visible() and j + cols >= len(LADDER_PANELS):
            ax.set_xlabel(r"$N$")
            ax.tick_params(labelbottom=True)
    axes[0, 0].legend(handles=[key("luna", "LUNA MC (L)", True), key("static", "static (S)", True), key("moving", "moving (M)", True), key("padded", "moving, wide line (W)", True)],
                      loc="center right", fontsize=LABEL - 1, bbox_to_anchor=(1.0, 0.62))
    save(fig, "ladder_1d")


def fig_hard():
    """Density of the hard cases at one N: LUNA, static, moving, exact; a zoom under each."""
    cases = (("leblanc", "LeBlanc", "moving", True, (0.72, 0.8), [0.73, 0.75, 0.77, 0.79], "contact"),
             ("einfeldt-strong", "Einfeldt strong", "moving-wide", False, (0.42, 0.58), [0.45, 0.50, 0.55], "near vacuum"),
             ("noh", "Noh", None, False, (0.0, 0.03), [0.0, 0.01, 0.02], "wall heating"),
             ("toro-3", "Toro 3", "moving", False, (0.76, 0.82), [0.77, 0.79, 0.81], "contact and shock"))
    n = 1024
    fig = plt.figure(figsize=(WIDTH, FIGSIZE[1] * 1.15))
    outer = fig.add_gridspec(2, 4, height_ratios=[2.2, 1.5], hspace=0.2, wspace=0.28)
    for j, (p, title, moving, logy, xlim, ticks, label) in enumerate(cases):
        ax, low = fig.add_subplot(outer[0, j]), fig.add_subplot(outer[1, j])
        x_l = ldr.cells(f"{LAD_RUNS}/luna-mc_{p}_{n}.npz", p)
        xl, rl = 0.5 * (x_l[0] + x_l[1]), x_l[2][0]
        x_s = ldr.cells(f"{LAD_RUNS}/static_{p}_{n}.npy", p)
        xs, rs = 0.5 * (x_s[0] + x_s[1]), x_s[2][0]
        runs = [(xl, rl, "luna"), (xs, rs, "static")]
        box = pr1d.problem(p)["box"]
        lim = (0.0, 0.4) if p == "noh" else box
        fine = np.linspace(lim[0], lim[1], 8001)
        exact = pr1d.exact(p, fine[:-1], fine[1:])[0]
        fine = 0.5 * (fine[:-1] + fine[1:])
        ax.plot(fine, exact, **REF)
        if moving:
            m = ldr.cells(f"{LAD_RUNS}/{moving}_{p}_{n}.npy", p)
            runs.append((0.5 * (m[0] + m[1]), m[2][0], "moving"))
        for x, r, s in runs:
            line(ax, x, r, s)
        ax.set_xlim(*lim)
        if logy:
            ax.set_yscale("log")
        ax.tick_params(labelsize=LABEL)
        corner(ax, f"{title}, N = {n}", 0.95, 0.95, "right")
        zoom(ax, low, xlim, ticks, runs, [(fine, exact, REF)], "abcd"[j], label)
        low.set_xlabel(r"$x$")
        if j == 0:
            ax.set_ylabel(r"$\\rho$")
            low.set_ylabel(r"$\\rho$")
    fig.axes[0].legend(handles=[key("luna", "LUNA MC"), key("static", "static"), key("moving", "moving"), rkey("exact", REF)], loc="lower left", fontsize=LABEL)
    save(fig, "hard_1d")


FIGURES = {"sod": fig_sod, "shu_osher": fig_shu_osher, "woodward_colella": fig_woodward_colella, "convergence_1d": fig_convergence,
           "time_to_target_1d": fig_time, "vortex_3d": fig_vortex, "kh_2d": fig_kh, "kh_2d_density": fig_kh_density, "sedov_3d": fig_sedov, "throughput_scaling": fig_throughput,
           "sedov3d_shock": fig_sedov_shock, "sedov3d_radial": fig_sedov_radial, "kh_2d_mesh": fig_kh_mesh, "kh_2d_evolution": fig_kh_evolution, "sedov3d_mesh": fig_sedov_mesh, "throughput_precision": fig_throughput_precision, "throughput_dimensions": fig_throughput_dims,
           "ladder_1d": fig_ladder, "hard_1d": fig_hard}


def main():
    for name in sys.argv[1:] or list(FIGURES):
        FIGURES[name]()
    if FAILED:
        raise SystemExit("overlaps: " + "; ".join(f"{name}: {what}" for name, what in FAILED))


if __name__ == "__main__":
    main()
