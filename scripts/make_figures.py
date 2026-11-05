"""The figures of docs/figures/, light and dark, from run files. Nothing is typed in: every number is read here.

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
import textwrap

import h5py
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
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

# Identity by entity, same in every figure. Hues: slots 1-3 of the palette, the only three that separate as all pairs in
# both modes; a variant of a series keeps its hue and changes line style and marker.
THEMES = {
    "": dict(surface="#fcfcfb", ink="#0b0b0b", second="#52514e", muted="#8b8a85", grid="#e4e3df",
             luna="#2a78d6", static="#eb6834", moving="#1baf7a"),
    "-dark": dict(surface="#1a1a19", ink="#f0efec", second="#c3c2b7", muted="#8b8a85", grid="#33332f",
                  luna="#3987e5", static="#d95926", moving="#199e70"),
}
T = THEMES[""]
MARK = dict(ms=4, mew=1.0)
LAYOUT = dict(left=0.075, right=0.995, top=0.94, bottom=0.1)
ROW = {**LAYOUT, "top": 0.9, "bottom": 0.16}  # one row of panels
MAX_MARKED = 110  # cells in a zoom window; beyond this, points hide each other


def series(name):
    """Line style of a series, in the current theme."""
    return {"luna": dict(color=T["luna"], marker="o", ls="-", label="LUNA MC"),
            "static": dict(color=T["static"], marker="s", ls="-", label="static"),
            "moving": dict(color=T["moving"], marker="^", ls="-", label="moving"),
            "two": dict(color=T["moving"], marker="D", ls="--", mfc="none", label="moving, two passes"),
            "padded": dict(color=T["moving"], marker="v", ls="--", mfc="none", label="moving, padded domain")}[name]


def theme(name):
    global T
    T = THEMES[name]
    plt.rcParams.update({
        "figure.facecolor": T["surface"], "savefig.facecolor": T["surface"], "axes.facecolor": T["surface"],
        "font.size": 9, "font.family": "DejaVu Sans", "text.color": T["ink"], "axes.labelcolor": T["second"],
        "axes.edgecolor": T["muted"], "axes.linewidth": 0.6, "axes.titlesize": 9.5, "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False, "xtick.color": T["second"], "ytick.color": T["second"],
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "axes.grid": True, "grid.color": T["grid"], "grid.linewidth": 0.5, "axes.axisbelow": True,
        "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 1.4, "figure.dpi": 200,
    })


def legend(fig, handles, ncol=None):
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(LAYOUT["left"], 1.01), ncol=ncol or len(handles), columnspacing=1.4,
               handlelength=2.2, borderaxespad=0, labelcolor=T["ink"])


def save(fig, name, foot):
    foot = "\n".join(textwrap.fill(p, 158) for p in foot.split("\n"))
    fig.text(0.008, 0.0, foot, fontsize=6.6, color=T["second"], va="top", linespacing=1.3)
    suffix = [k for k, v in THEMES.items() if v is T][0]
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/{name}{suffix}.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.06, dpi=200, pil_kwargs={"optimize": True})
    plt.close(fig)
    Image.open(path).convert("RGB").quantize(256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE).save(path, optimize=True)
    print(f"{path} {os.path.getsize(path) / 1024:.0f} kB")


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


def profile_panels(fig, grid, runs, ref, views, ylabel):
    """Density and two zooms: lines over the whole line, cells as points in a zoom."""
    axes = []
    for i, (title, xlim, ylim) in enumerate(views):
        ax = fig.add_subplot(grid[i])
        axes.append(ax)
        ax.set_title(title)
        zoom = i > 0
        for x, y, kw in ref:
            ax.plot(x, y, zorder=2 if zoom else 1, **kw)
        for x, y, s in runs:
            k = series(s)
            if zoom:
                keep = (x >= xlim[0]) & (x <= xlim[1])
                marked = keep.sum() <= MAX_MARKED
                ax.plot(x[keep], y[keep], color=k["color"], ls="--" if s == "static" else k["ls"], lw=0.9, marker=k["marker"] if marked else None,
                        ms=3.6, mew=0.8, mfc=T["surface"] if s == "static" else k.get("mfc", k["color"]), zorder=3 if s != "static" else 2.5)
            else:
                ax.plot(x, y, color=k["color"], ls="--" if s == "static" else k["ls"], lw=1.1, zorder=3)
        ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlabel("x")
    axes[0].set_ylabel(ylabel)
    return axes


def handles(names):
    out = []
    for s in names:
        k = series(s)
        out.append(plt.Line2D([], [], color=k["color"], ls=k["ls"], marker=k["marker"], ms=3.6, mfc=k.get("mfc", k["color"]),
                              label=LABEL.get(s, k["label"])))
    return out


LABEL = {"luna": "LUNA MC", "static": "static (same scheme, same values)", "moving": "moving, default rule, Courant 0.4"}


def ref_line(label, dashed=False):
    return plt.Line2D([], [], color=T["second"] if dashed else T["ink"], ls=(0, (3, 2)) if dashed else "-", lw=0.9, label=label)


def fig_sod():
    n = 512
    p = "sod"
    x, r, _, P = ours("moving-a04", p, n)
    xs, rs, _, Ps = ours("static", p, n)
    xl, rl, _, Pl = luna(p, n)
    ex = cmp1d.Exact()
    fine = np.linspace(0, 1, 40001)
    fig = plt.figure(figsize=(7.6, 4.4))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.5, 1, 1], hspace=0.5, wspace=0.28, **LAYOUT)
    profile_panels(fig, [grid[0, 0], grid[0, 1], grid[0, 2]], [(xs, rs, "static"), (xl, rl, "luna"), (x, r, "moving")],
                   [(fine, ex(fine)[0], dict(color=T["ink"], lw=0.8))],
                   [("Density", (0, 1), None), ("Zoom: contact", (0.67, 0.76), (0.24, 0.45)),
                    ("Zoom: shock", (0.91, 0.965), (0.11, 0.28))], "density")
    rho_d = [(xs, rs - ex(xs)[0], "static"), (xl, rl - ex(xl)[0], "luna"), (x, r - ex(x)[0], "moving")]
    P_d = [(xs, Ps - ex(xs)[2], "static"), (xl, Pl - ex(xl)[2], "luna"), (x, P - ex(x)[2], "moving")]
    a, b = profile_panels(fig, [grid[1, 0], grid[1, 1]], rho_d, [],
                          [("Density minus exact", (0, 1), None), ("Zoom: rarefaction", (0.24, 0.32), None)], "difference")
    c, = profile_panels(fig, [grid[1, 2]], P_d, [], [("Pressure minus exact", (0, 1), None)], "")
    legend(fig, handles(["luna", "static", "moving"]) + [ref_line("exact")])
    save(fig, "sod", f"Sod, t = 0.25, N = {n}. All HLLC, MUSCL-Hancock, FP64. LUNA: LIMITER 1 (MC), Pallas kernel, CFL 0.8. Static: uniform points, face limiter (MC there), "
         "LUNA's time step rule at 0.8. Moving: face limiter, default rule at 0.4, regularised. Job 31917206, voronoi-cpp 947946a. "
         "Cell means at centroids. Exact: LUNA's exact Riemann solver.")


def fig_shu_osher():
    n = 1024
    p = "shu-osher"
    x, r, *_ = ours("moving-a04", p, n)
    xs, rs, *_ = ours("static", p, n)
    xl, rl, *_ = luna(p, n)
    w = np.load(f"{WIDE_CPU}/aro_shu-osher-wide_{2 * n}.npy")
    xw, rw = w[:, 1], w[:, 3]
    lr = reference_file("luna/shu_osher/shu_osher_mc.npz")["rho"]
    ar = reference_file("athenak/shu_osher/so_ppmx_rk4_16384.npz")["ref_rho"]
    ref = [(centres(-5, 5, lr.size), lr, dict(color=T["ink"], lw=0.9)),
           (centres(-5, 5, ar.size), ar, dict(color=T["second"], lw=0.9, ls=(0, (3, 2))))]
    views = [("Density", (-5, 5), None), ("Zoom: left of the shock train", (-3.2, 0.2), (3.4, 4.3)), ("Zoom: post-shock peak", (1.74, 1.93), (4.2, 4.75))]
    fig = plt.figure(figsize=(7.6, 4.6))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.5, 1, 1], hspace=0.5, wspace=0.28, **LAYOUT)
    profile_panels(fig, [grid[0, 0], grid[0, 1], grid[0, 2]], [(xs, rs, "static"), (xl, rl, "luna"), (x, r, "moving")], ref,
                   [("Density, LUNA's domain [-5, 5]", *views[0][1:]), *views[1:]], "density")
    padded = [(xl, rl, "luna"), (xw, rw, "padded")]
    profile_panels(fig, [grid[1, 0], grid[1, 1], grid[1, 2]], padded, ref,
                   [("Density, padded domain [-15, 5]", (-5, 5), None), views[1], views[2]], "density")
    legend(fig, handles(["luna", "static", "moving", "padded"]) + [ref_line("LUNA, high resolution"), ref_line("AthenaK PPMX 16384", True)],
           ncol=3)
    save(fig, "shu_osher", f"Shu-Osher, t = 1.8, N = {n} on [-5, 5]. Top: job 31917208, 947946a, as in the Sod figure: moving FAILS here, the grown inflow cell holds disturbed gas.\n"
         "Bottom: moving on [-15, 5], default rule at 0.3 (the 0.4 default was not run padded), OpenMP run, job 31920121, branch dim-1d-diag.\n"
         "Curves from luna-reference-solutions 99a52d0: luna/shu_osher/shu_osher_mc.npz (LUNA b7b0d35, MC HLLC, 65536 on 8192 cells), athenak/shu_osher/so_ppmx_rk4_16384.npz (AthenaK c52f88f).")


def fig_woodward_colella():
    n = 2048
    p = "woodward-colella"
    x, r, _, P = ours("moving-a04", p, n)
    xs, rs, _, Ps = ours("static", p, n)
    xl, rl, _, Pl = luna(p, n)
    lr = reference_file("luna/woodward_colella/woodward_colella_mc.npz")
    ar = reference_file("athenak/woodward_colella/a9_ppmx_rk4_16384.npz")["ref_rho"]
    ref = [(centres(0, 1, lr["rho"].size), lr["rho"], dict(color=T["ink"], lw=0.9)),
           (centres(0, 1, ar.size), ar, dict(color=T["second"], lw=0.9, ls=(0, (3, 2))))]
    fig = plt.figure(figsize=(7.6, 4.4))
    grid = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.28, **LAYOUT)
    runs = [(xs, rs, "static"), (xl, rl, "luna"), (x, r, "moving")]
    profile_panels(fig, [grid[0, :]], runs, ref, [("Density", (0, 1), None)], "density")
    profile_panels(fig, [grid[1, 0], grid[1, 1], grid[1, 2]], runs, ref,
                   [("Zoom: contact", (0.588, 0.606), None), ("Zoom: contact, shocked gas", (0.755, 0.777), None), ("Zoom: shock", (0.788, 0.804), None)], "density")
    legend(fig, handles(["luna", "static", "moving"]) + [ref_line("LUNA, high resolution"), ref_line("AthenaK PPMX 16384", True)], ncol=3)
    save(fig, "woodward_colella", f"Woodward-Colella, t = 0.038, N = {n}, reflecting walls. Runs as in the Sod figure, job 31917210, 947946a. Moving passes here, ahead of LUNA at every N.\n"
         "Curves from luna-reference-solutions 99a52d0: luna/woodward_colella/woodward_colella_mc.npz (LUNA b7b0d35, MC HLLC, 65536 on 8192 cells), athenak/woodward_colella/a9_ppmx_rk4_16384.npz (AthenaK c52f88f).")


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


def log_series(ax, x, y, s, **over):
    k = series(s)
    style = dict(color=k["color"], ls=k["ls"], marker=k["marker"], mfc=k.get("mfc", k["color"]), lw=1.2, label=k["label"], **MARK)
    style.update(over)
    ax.plot(x, y, **style)


def slope_guides(ax, n, y0):
    n = np.array(n, float)
    ax.plot(n, y0 * (n / n[0]) ** -1, color=T["muted"], lw=0.7, ls=":", zorder=1)


PROBLEMS = (("sod", "Sod"), ("shu-osher", "Shu-Osher"), ("woodward-colella", "Woodward-Colella"))
VARS = ("density", "velocity", "pressure")


def fig_convergence():
    rows = report()
    wide = wide_rule_l1()
    matrix = [r for r in json.load(open(MATRIX)) if r["problem"] == "shu-osher-wide" and r["config"] == "c04"]
    fig, axes = plt.subplots(3, 3, figsize=(7.6, 6.2), sharex="col", gridspec_kw=dict(hspace=0.16, wspace=0.3, **LAYOUT))
    for j, (p, title) in enumerate(PROBLEMS):
        axes[0, j].set_title(title)
        for i, name in enumerate(VARS):
            ax = axes[i, j]
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            for s, key in (("static", "static"), ("luna", "luna"), ("moving", "moving-a04")):
                r = rows[key, p]
                log_series(ax, [x["N"] for x in r], [x["L1"][i] for x in r], s, ms=5 if s == "static" else 3.2,
                           mfc="none" if s == "static" else series(s)["color"])
            if p == "shu-osher":
                log_series(ax, [m["N"] for m in matrix], [m["L1"][i] for m in matrix], "padded", ms=3.2)
                ks = sorted(wide)
                log_series(ax, ks, [wide[k][i] for k in ks], "two", marker="x", ls=":", ms=3.6)
            n = [x["N"] for x in rows["luna", p]]
            y = rows["luna", p][0]["L1"][i]
            slope_guides(ax, n, y)
            ax.minorticks_off()
            ax.set_xticks([2 ** 8, 2 ** 12, 2 ** 16])
            if j == 0:
                ax.set_ylabel(f"field L1, {name}")
            if i == 2:
                ax.set_xlabel("N")
    h = [plt.Line2D([], [], **{k: v for k, v in series("luna").items()}, ms=3.2),
         plt.Line2D([], [], color=T["static"], marker="s", mfc="none", ms=5, label="static (on LUNA)"),
         plt.Line2D([], [], color=T["moving"], marker="^", ms=3.2, label="moving, default rule 0.4"),
         plt.Line2D([], [], color=T["moving"], marker="v", ls="--", mfc="none", ms=3.2, label="moving padded, LUNA's rule 0.4"),
         plt.Line2D([], [], color=T["moving"], marker="x", ls=":", ms=3.6, label="moving padded, default rule 0.3")]
    legend(fig, h, ncol=3)
    save(fig, "convergence_1d", "Field L1 of the reconstructed field (dotted: order 1 from LUNA's first point), integral of |f_h - f_ref| dx / L, scripts/compare_1d.py. Sod against the exact solution, Shu-Osher against LUNA MC 2^21 cells, Woodward-Colella against LUNA MC 2^20 cells.\n"
         "Jobs 31917206/8/10, 947946a. Padded Shu-Osher on [-15, 5], norm on [-5, 5], N per length 10: LUNA's rule 0.4 job 31919016 (GPU), default rule 0.3 job 31920121 (OpenMP), branch dim-1d-diag.")


def fig_time():
    rows = report()
    fig, axes = plt.subplots(3, 3, figsize=(7.6, 6.2), sharex="col", gridspec_kw=dict(hspace=0.16, wspace=0.3, **LAYOUT))
    for j, (p, title) in enumerate(PROBLEMS):
        axes[0, j].set_title(title)
        for i, (name, var) in enumerate(zip(VARS, cmp1d.VARS)):
            ax = axes[i, j]
            ax.set_xscale("log")
            ax.set_yscale("log")
            for s, key in (("static", "static"), ("luna", "luna"), ("moving", "moving-a04")):
                r = rows[key, p]
                log_series(ax, [x["wall_s"] for x in r], [x["L1"][i] for x in r], s, ms=5 if s == "static" else 3.2,
                           mfc="none" if s == "static" else series(s)["color"])
            for eps in cmp1d.TARGETS[p][var]:
                ax.axhline(eps, color=T["muted"], lw=0.7, ls=":", zorder=1)
                ax.annotate(f"{eps:g}", (1.0, eps), xycoords=("axes fraction", "data"), fontsize=6.5, color=T["second"], ha="right", va="bottom")
            ax.minorticks_off()
            if j == 0:
                ax.set_ylabel(f"field L1, {name}")
            if i == 2:
                ax.set_xlabel("GPU seconds")
    legend(fig, handles(["luna", "static", "moving"]))
    save(fig, "time_to_target_1d", "GPU seconds of the stepping loop on one A100 against field L1, one point per N. Dotted: the target errors of the notes. LUNA's compile not counted. Below ~1e5 cells a step is launch latency.\n"
         "Each panel is one job at one clock: Sod 31917206 and Shu-Osher 31917208 at 1095 MHz, Woodward-Colella 31917210 at 1305 MHz, 947946a. Moving on Shu-Osher reaches no target (grown inflow cell).")


def vortex_rows():
    out = {}
    for path, keys in ((f"{VORTEX_LOGS}/n-vortex-31920446.out", ("mov", "sta")), (f"{VORTEX_LOGS}/fin-norm-31922276.out", ("tp",))):
        for line in open(path):
            m = re.search(r"vortex (\S+) nx=(\d+) gen .*\| avg (\S+) (\S+) (\S+) \| field0 \S+ \S+ \S+ \| exact0 (\S+) (\S+) (\S+)", line)
            if m and m.group(1) in keys:
                g = m.groups()
                out[g[0], int(g[1])] = ([float(a) for a in g[2:5]], [float(a) for a in g[5:8]])
    return out


def fig_vortex():
    rows = vortex_rows()
    d = reference_file("luna/yee_vortex/yee_vortex_convergence.npz")
    pick = (d["dim"] == 3) & (d["limiter"] == 1) & (d["vx"] == 0) & (d["cfl"] == 0.8)
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.75), gridspec_kw=dict(wspace=0.3, **ROW))
    nx = (32, 64, 128, 256)
    for i, (ax, name, v) in enumerate(zip(axes, VARS, ("rho", "v", "P"))):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(name.capitalize())
        log_series(ax, nx, [rows["sta", n][1][i] for n in nx], "static", color=T["muted"], marker=None, ls=(0, (1, 2)), lw=1.0, label="floor")
        for q, over in ((8, dict()), (1, dict(ls="--", mfc="none"))):
            s = pick & (d["ic_quad"] == q)
            o = np.argsort(d["nx"][s])
            log_series(ax, d["nx"][s][o], d[f"avg_{v}_L1"][s][o], "luna", ms=3.2, **over)
        log_series(ax, nx, [rows["sta", n][0][i] for n in nx], "static", ms=3.2)
        log_series(ax, nx, [rows["mov", n][0][i] for n in nx], "moving", ms=3.2)
        log_series(ax, nx, [rows["tp", n][0][i] for n in nx], "two", ms=3.2)
        ax.set_xticks(nx)
        ax.set_xticklabels(nx)
        ax.minorticks_off()
        ax.set_xlabel("nx")
    axes[0].set_ylabel("L1 against exact cell averages")
    h = [plt.Line2D([], [], color=T["luna"], marker="o", ms=3.2, label="LUNA MC"),
         plt.Line2D([], [], color=T["luna"], marker="o", ms=3.2, ls="--", mfc="none", label="LUNA MC, point start as ours"),
         plt.Line2D([], [], color=T["static"], marker="s", ms=3.2, label="static"),
         plt.Line2D([], [], color=T["moving"], marker="^", ms=3.2, label="moving, one pass"),
         plt.Line2D([], [], color=T["moving"], marker="D", ms=3.2, ls="--", mfc="none", label="moving, two passes"),
         plt.Line2D([], [], color=T["muted"], ls=(0, (1, 2)), label="O(h) floor")]
    legend(fig, h, ncol=3)
    save(fig, "vortex_3d", "Yee vortex, 3D slab 10 x 10 x 2.5, nx x nx x nx/4 cells, t = 10. Measure: cell values against the exact average over each cell (Voronoi cell for ours, 8 x 8 Gauss-Legendre for LUNA); L1 of the vector v.\n"
         "O(h) floor: exact averages as a piecewise-constant field against the exact field; it binds a field norm, not this measure, and is drawn as the size of the first-order term.\n"
         "Ours: HLLC, face limiter, per-face, Courant 0.3, jitter 0.1, start from the point value at the centroid, jobs 31920446 (one pass, static), 31922276 (two passes, a switch on branch dim-1d-diag, not on main). "
         "LUNA: MC, HLLC, Pallas, CFL 0.8, luna-reference-solutions 99a52d0 luna/yee_vortex/yee_vortex_convergence.npz (LUNA debcdae + uncommitted, bitwise at 60cf841).")


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


def fig_kh():
    runs, block, target = kh_rows()
    eps = [float(c) for c in next(l for l in open(KH).read().splitlines() if l.startswith("| series | 0.05")).strip("| \n").split("|")[1:]]
    names = (("static", "static"), ("luna", "luna"), ("moving", "moving"), ("moving-twoflux", "two"))
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.75), gridspec_kw=dict(wspace=0.32, **ROW))
    for key, s in names:
        over = dict(ms=5, mfc="none") if s == "static" else dict(ms=3.2)
        lab = dict(label="moving, two fluxes") if s == "two" else {}
        r = np.array(runs[key])
        log_series(axes[0], r[:, 0], r[:, 2], s, **over, **lab)
        b = np.array(block[key]) if key in block else None
        if b is not None:
            log_series(axes[1], b[:, 0], b[:, 1], s, **over, **lab)
        log_series(axes[2], r[:, 1], r[:, 2], s, **over, **lab)
    for e in eps:
        axes[2].axhline(e, color=T["muted"], lw=0.7, ls=":", zorder=1)
        axes[2].annotate(f"{e:g}", (1.0, e), xycoords=("axes fraction", "data"), fontsize=6.5, color=T["second"], ha="right", va="bottom")
    for ax, t, xl in zip(axes, ("Mode amplitude, max |e|", "Field L1 rho, t = 1.5", "max |e| against GPU time"), ("N", "N", "GPU seconds")):
        ax.set_yscale("log")
        ax.set_xscale("log", base=2 if xl == "N" else 10)
        ax.set_title(t)
        ax.set_xlabel(xl)
        ax.minorticks_off()
    axes[0].set_ylabel("error")
    h = handles(["luna"]) + [plt.Line2D([], [], color=T["static"], marker="s", mfc="none", ms=5, label="static"),
                             plt.Line2D([], [], color=T["moving"], marker="^", ms=3.2, label="moving, one flux"),
                             plt.Line2D([], [], color=T["moving"], marker="D", ms=3.2, ls="--", mfc="none", label="moving, two fluxes")]
    legend(fig, h)
    save(fig, "kh_2d", "Kelvin-Helmholtz after McNally, Lyra & Passy (2012), N^2 cells, one A100 at 1305 MHz. e(t) = M / M_ref - 1 against their published reference curve, max over t <= 1.5; L1 on 128^2 block averages against AthenaK PPMX 4096^2.\n"
         "The table has no LUNA row for the L1 panel. All HLLC, FP64. LUNA MC, Pallas, CFL 0.8. Ours: face limiter, per-face, LUNA's rule 0.8. Jobs 31922404/5/6, voronoi-cpp branch dim-2d: this 2D hydro is not yet on luna-cpp main.")


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


def fig_sedov():
    KPC, MSUN = 3.08568e21, 1.98892e33
    rho0 = 1.6726219e-24 / (1e10 * MSUN / KPC ** 3)
    e0 = 1e51 / (1e10 * MSUN * 1e10)
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.75), gridspec_kw=dict(wspace=0.3, width_ratios=[1.2, 1, 1], **ROW))
    prof = {}
    for name, path in (("static", f"{SEDOV}/static_face_128"), ("luna", f"{SEDOV}/luna/luna1_128.npz"), ("moving", f"{SEDOV}/moving_face_128")):
        r, vol, rho, t = sedov_state(path)
        prof[name] = (r, vol, rho)
    R = exact_sedov.solution(np.array([0.0]), t, e0, rho0, 5 / 3)[3]
    edges = np.linspace(0, 1.15, 70) * R
    mid = 0.5 * (edges[1:] + edges[:-1]) / R
    fine = np.linspace(1e-4, 1.15, 3000) * R
    exact = exact_sedov.solution(fine, t, e0, rho0, 5 / 3)[0] / rho0
    for ax in axes[:2]:
        ax.plot(fine / R, exact, color=T["ink"], lw=0.9, zorder=1)
        for name in ("static", "luna", "moving"):
            r, vol, rho = prof[name]
            mass = np.histogram(r, edges, weights=rho * vol)[0]
            v = np.histogram(r, edges, weights=vol)[0]
            k = series(name)
            ax.plot(mid, mass / np.maximum(v, 1e-300) / rho0, color=k["color"], lw=1.1, label=k["label"])
    axes[0].set_title("Density profile, 128^3")
    axes[0].set_xlabel("r / R_s")
    axes[0].set_ylabel("rho / rho0")
    axes[1].set_title("Zoom: shell")
    axes[1].set_xlim(0.85, 1.05)
    axes[1].set_xlabel("r / R_s")
    ax = axes[2]
    lad = {**json.load(open(f"{SEDOV_LADDER}/norms_luna.json")), **json.load(open(f"{SEDOV_LADDER}/norms_vor.json"))}
    mc = json.load(open(SEDOV_L1))
    for prefix, s, over in (("sta", "static", dict()), ("luna", "luna", dict(ls="--", mfc="none")), ("mov", "moving", dict())):
        keys = [k for k in lad if re.fullmatch(prefix + r"\d+", k) or k == ("mov256cell" if prefix == "mov" else "")]
        keys.sort(key=lambda k: lad[k]["n"])
        log_series(ax, [lad[k]["n"] for k in keys], [lad[k]["field_rho"] for k in keys], s, ms=3.2, **over)
    log_series(ax, [128, 256], [mc["luna-1_128"]["rho"], mc["luna-1_256"]["rho"]], "luna", ms=3.2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([64, 128, 256])
    ax.set_xticklabels([64, 128, 256])
    ax.minorticks_off()
    ax.set_yticks([0.04, 0.06, 0.1])
    ax.set_yticklabels(["0.04", "0.06", "0.1"])
    ax.set_title("Field L1 rho ladder")
    ax.set_xlabel("n (n^3 cells)")
    h = [plt.Line2D([], [], color=T["luna"], marker="o", ms=3.2, label="LUNA MC"),
         plt.Line2D([], [], color=T["luna"], marker="o", ms=3.2, ls="--", mfc="none", label="LUNA minmod (L1 only)"),
         plt.Line2D([], [], color=T["static"], marker="s", ms=3.2, label="static"),
         plt.Line2D([], [], color=T["moving"], marker="^", ms=3.2, label="moving"),
         ref_line("Sedov-Taylor")]
    legend(fig, h)
    save(fig, "sedov_3d", "Sedov 3D, 1e51 erg, n = 1 cm^-3, 50 pc box, 1e4 yr. A shock test that cannot rank schemes: order below 1 for every code (0.66 LUNA, 0.78 moving, 128 to 256). Field L1 over r < 1.5 R_s of the piecewise-constant field, not sampled at generators.\n"
         "Ours: HLLC, jittered, Courant 0.3, face limiter. LUNA: Pallas, CFL 0.8. Profile: volume-weighted shells, branch dim-limiters, jobs 31918879 (moving), 31918880 (static), 31918732 (LUNA). Ladder: older code, jobs 31879294, 31879807 (LUNA minmod 128-448, moving 64-256, 256 with the per-cell loop, static 64-192); LUNA MC 128, 256 from dim-limiters.")


FIGURES = {"sod": fig_sod, "shu_osher": fig_shu_osher, "woodward_colella": fig_woodward_colella, "convergence_1d": fig_convergence,
           "time_to_target_1d": fig_time, "vortex_3d": fig_vortex, "kh_2d": fig_kh, "sedov_3d": fig_sedov}


def main():
    names = sys.argv[1:] or list(FIGURES)
    for suffix in THEMES:
        theme(suffix)
        for name in names:
            FIGURES[name]()


if __name__ == "__main__":
    main()
