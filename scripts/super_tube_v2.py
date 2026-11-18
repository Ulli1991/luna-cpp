"""The super-hard tube v2 (research_notes/super_tube_v2_2026-11-18.md): scores, initial cells for other codes, figures, final states.

    python super_tube_v2.py single <runs> <out>                 # the new stages alone at t = 0.1: L1, positivity, steps (single.md, .json)
    python super_tube_v2.py score <runs> <out>                  # the tube at t = 0.1 and 4, per stage (tube.md, .json)
    python super_tube_v2.py ic <N> <out stem> [m0]              # N uniform cells of the initial state: <stem>.npz and <stem>.txt
    python super_tube_v2.py figures <runs> <single runs> <out>   # one figure per stage (LUNA's figstyle)
    python super_tube_v2.py npz <runs> <out> <series> <problem> <N> <t> <meta json>   # one final state, luna-reference-solutions' keys

Run with LUNA's environment. Runs: <series>_<problem>_<N>_t<t>.npz (LUNA) or .npy (luna-cpp), series luna, static, moving; scripts/
super_tube_v2_run.sh (CPU) and super_tube_v2_gpu.sh (A100) write them. The norm is v1's (scripts/super_tube.py): over a stage's region,
sum of V |W_i - target_i|, target the exact solution's or the reference's conservative average over the same cell. At t = 0.1 the exact
stages (super_tube_layout.EXACT_V2) against the exact superposition (problems_1d.super2_exact), SO, S and H against LUNA 2^20; at t = 4
everything against LUNA 2^20. A stage's region: from the middle of the buffer on its left to the middle of the one on its right.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ladder_1d as lad  # noqa: E402
import problems_1d as pr  # noqa: E402
import super_tube_layout as lay  # noqa: E402

SERIES = ("luna", "static", "moving")
NS = (4096, 32768)
TIMES = ("0.1", "4")
REF_N, CHECK_N = 1048576, 524288
KEYS = [k for k, *_ in pr.SUPER2_STAGES]
NAMES = {k: name for k, name, *_ in lay.STAGES_V2}


def path(runs, s, p, n, t):
    return f"{runs}/{s}_{p}_{n}_t{t}" + (".npz" if s == "luna" else ".npy")


def status(f):
    """What a run reached: t, reached, steps, wall_s, smallest_h (moving), from its JSON, else (a LUNA chunk cut short) from the npz."""
    if os.path.exists(f + ".json"):
        j = json.load(open(f + ".json"))
        return dict(t=float(j["t"]), reached=bool(j.get("reached", 1)), steps=int(j["steps"]), wall_s=float(j["wall_s"]),
                    smallest_h=j.get("smallest_h"))
    if f.endswith(".npz") and os.path.exists(f):
        d = np.load(f)
        return dict(t=float(d["t"]), reached=False, steps=int(d["steps"]), wall_s=float(d["wall_s"]), smallest_h=None)
    return None


def l1(a, b, W, target, lo, hi):
    m = (0.5 * (a + b) >= lo) & (0.5 * (a + b) < hi)
    return [float(((b - a)[m] * np.abs(W[k, m] - target[k, m])).sum()) for k in range(3)]


def reference(runs, p, t, n=REF_N):
    f = path(runs, "luna", p, n, t)
    st = status(f)
    if st is None or not st["reached"]:
        return None
    d = np.load(f)
    lo, hi = pr.problem(p)["box"]
    return lad.Reference(np.linspace(lo, hi, d["rho"].size + 1), d["rho"], d["u"], d["p"])


def positivity(W):
    return dict(min_rho=float(np.nanmin(W[0])), min_P=float(np.nanmin(W[2])), finite=bool(np.isfinite(W).all()))


def stream_pressure(a, b, W, t):
    """Stage M: (max |P / 1e-6 - 1| over the carried bump, max P / 1e-6 from the bump to 0.3 past the stream's rear edge: heating
    where the stream leaves vacuum behind)."""
    x, x0 = 0.5 * (a + b), lay.BUMP_X0 - lay.V_MACH * t
    bump = (x > x0) & (x < x0 + lay.BUMP_W)
    rear = (x > x0 + lay.BUMP_W) & (x < 2.0 - lay.V_MACH * t + 0.3)
    return (float(np.abs(W[2, bump] / 1e-6 - 1.0).max()) if bump.any() else float("nan"),
            float(np.nanmax(W[2, rear]) / 1e-6) if rear.any() else float("nan"))


def single(runs, out):
    rows = []
    for k in pr.SUPER2_SINGLE:
        p = f"super-v2-{k.lower()}"
        m = pr.problem(p)["cells_4096"]
        lo, hi = pr.problem(p)["box"]
        ref = reference(runs, p, "0.1", 64 * m)
        for s in SERIES:
            for n, label in ((m, 4096), (8 * m, 32768), (64 * m, 262144)):
                f = path(runs, s, p, n, "0.1")
                st = status(f)
                if st is None or (n == 64 * m and s != "luna"):
                    continue
                a, b, W = lad.cells(f, p)
                row = dict(stage=k, series=s, N=label, cells=n, **st, **positivity(W))
                if k in lay.EXACT_V2 and st["reached"]:
                    row["L1_exact"] = l1(a, b, W, np.stack(pr.super2_exact(a, b, 0.1)), lo, hi)
                if ref is not None and n < 64 * m and st["reached"]:
                    row["L1_luna64"] = l1(a, b, W, np.stack(ref.average(a, b)), lo, hi)
                if k == "M":
                    row["stream_dP"] = stream_pressure(a, b, W, st["t"])
                rows.append(row)
    os.makedirs(out, exist_ok=True)
    json.dump(rows, open(f"{out}/single.json", "w"), indent=1)
    g = lambda r, key, i=0: f"{r[key][i]:.3e}" if key in r else "-"  # noqa: E731
    lines = ["| stage | series | N (tube) | cells | reached | t | steps | min rho | min P | finite | L1 rho exact | L1 rho LUNA 64x | L1 P exact | smallest h "
             "| M: bump dP/P; rear P/P0 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        sh = f"{r['smallest_h']:.2g}" if r.get("smallest_h") is not None else "-"
        dp = "{:.1e}; {:.1e}".format(*r["stream_dP"]) if "stream_dP" in r else "-"
        lines.append(f"| {r['stage']} | {r['series']} | {r['N']} | {r['cells']} | {int(r['reached'])} | {r['t']:.4g} | {r['steps']} | {r['min_rho']:.2e} | "
                     f"{r['min_P']:.2e} | {r['finite']} | {g(r, 'L1_exact')} | {g(r, 'L1_luna64')} | {g(r, 'L1_exact', 2)} | {sh} | {dp} |")
    open(f"{out}/single.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def score(runs, out):
    regions = {k: pr.super2_region(k) for k in KEYS}
    e = np.linspace(0.0, pr.SUPER2_LENGTH, 2 ** 16 + 1)
    rows = []
    for t in TIMES:
        ref = reference(runs, "super-tube-v2", t)
        for p in ("super-tube-v2", "super-tube-v2-m0"):
            U0 = (pr.super2_initial(e[:-1], e[1:], p.endswith("m0")) * np.diff(e)).sum(axis=1)
            for s in SERIES:
                for n in NS:
                    f = path(runs, s, p, n, t)
                    st = status(f)
                    if st is None:
                        continue
                    a, b, W = lad.cells(f, "super-tube-v2")
                    row = dict(problem=p, series=s, N=n, t_end=float(t), **st, **positivity(W), L1={})
                    U = (pr.conserved(*W, pr.G14) * (b - a)).sum(axis=1)
                    row["budget"] = dict(mass=float(U[0] - U0[0]), energy=float(U[2] - U0[2]))
                    qa = (0.5 * (a + b) > 3.0) & (0.5 * (a + b) < 4.0)
                    row["A_quiet"] = float(np.abs(W[:, qa] - np.array([[1.0], [0.0], [1.0]])).max()) if qa.any() else float("nan")
                    row["cells_in_box"] = int(a.size)
                    if st["reached"] and np.isfinite(W).all():
                        ex = np.stack(pr.super2_exact(a, b, 0.1)) if t == "0.1" else None
                        rf = np.stack(ref.average(a, b)) if ref is not None else None
                        for k, (lo, hi) in regions.items():
                            if p.endswith("m0") and k == "M":
                                continue
                            exact = t == "0.1" and k in lay.EXACT_V2 and not (p.endswith("m0") and k == "M")
                            target = ex if exact else rf
                            if target is not None:
                                row["L1"][k] = dict(against="exact" if exact else "LUNA 2^20", L1=l1(a, b, W, target, lo, hi))
                        if t == "0.1":
                            row["stream_dP"] = stream_pressure(a, b, W, 0.1) if not p.endswith("m0") else None
                    rows.append(row)
    checks = {}
    # The reference's own resolution: LUNA at 2^19 goes NaN (stage N), so luna-cpp static (LUNA's scheme, with floors) at 2^19 and
    # 2^20 against LUNA 2^20; how far stage M reaches: luna-cpp static 32768, v2 against m0.
    for t in TIMES:
        ref = reference(runs, "super-tube-v2", t)
        for s, n in (("luna", CHECK_N), ("static", CHECK_N), ("static", REF_N)):
            f = path(runs, s, "super-tube-v2", n, t)
            if ref is not None and status(f) and status(f)["reached"]:
                a, b, W = lad.cells(f, "super-tube-v2")
                rf = np.stack(ref.average(a, b))
                checks[f"L1 rho, {s} {n} vs LUNA 2^20, t = {t}"] = {k: l1(a, b, W, rf, *regions[k]) for k in KEYS}
        f0, f1 = path(runs, "static", "super-tube-v2", 32768, t), path(runs, "static", "super-tube-v2-m0", 32768, t)
        if status(f0) and status(f1) and status(f0)["reached"] and status(f1)["reached"]:
            a, b, W = lad.cells(f0, "super-tube-v2")
            _, _, W0 = lad.cells(f1, "super-tube-v2")
            checks[f"max |rho(v2) - rho(m0)|, static 32768, t = {t}"] = {
                k: float(np.abs(W[0] - W0[0])[(0.5 * (a + b) >= lo) & (0.5 * (a + b) < hi)].max()) for k, (lo, hi) in regions.items()}
    os.makedirs(out, exist_ok=True)
    json.dump(dict(rows=rows, checks=checks), open(f"{out}/tube.json", "w"), indent=1)
    cols = [(p, s, n) for p in ("super-tube-v2", "super-tube-v2-m0") for s in SERIES for n in NS]
    index = {(r["problem"], r["series"], r["N"], r["t_end"]): r for r in rows}
    cols = [c for c in cols if any((*c, float(t)) in index for t in TIMES)]
    short = lambda p: "v2" if p == "super-tube-v2" else "m0"  # noqa: E731
    lines = ["L1(rho) per stage", "", "| t | stage | against | " + " | ".join(f"{s} {n} ({short(p)})" for p, s, n in cols) + " |",
             "|---|---|---|" + "---|" * len(cols)]
    for t in TIMES:
        for k in KEYS:
            cells = [index.get((*c, float(t)), {}).get("L1", {}).get(k) for c in cols]
            if not any(cells):
                continue
            against = next(c["against"] for c in cells if c)
            lines.append(f"| {t} | {k} | {against} | " + " | ".join("-" if c is None else f"{c['L1'][0]:.3e}" for c in cells) + " |")
    lines += ["", "| problem | t_end | series | N | reached | t | steps | wall s | min rho | min P | finite | mass - initial | energy - initial | A: max abs W - ambient "
              "| M: bump dP/P; rear P/P0 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        dp = "{:.1e}; {:.1e}".format(*r["stream_dP"]) if r.get("stream_dP") is not None else "-"
        lines.append(f"| {short(r['problem'])} | {r['t_end']:g} | {r['series']} | {r['N']} | {int(r['reached'])} | {r['t']:.4g} | {r['steps']} | {r['wall_s']:.0f} | "
                     f"{r['min_rho']:.2e} | {r['min_P']:.2e} | {r['finite']} | {r['budget']['mass']:.2e} | {r['budget']['energy']:.2e} | {r['A_quiet']:.1e} | {dp} |")
    for name, per in checks.items():
        lines += ["", f"{name}:", "", "| " + " | ".join(KEYS) + " |", "|" + "---|" * len(KEYS),
                  "| " + " | ".join(f"{v[0]:.2e}" if isinstance(v, list) else f"{v:.1e}" for v in per.values()) + " |"]
    open(f"{out}/tube.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def ic(n, stem, m0=False):
    """n uniform cells on [0, L] of v2's initial state (exact cell averages): edges, centres, primitives of the averages, and the cell
    totals of mass, momentum, total energy; for codes that set up their own cells (AREPO, GIZMO), the same per cell."""
    n = int(n)
    e = np.linspace(0.0, pr.SUPER2_LENGTH, n + 1)
    a, b = e[:-1], e[1:]
    U = pr.super2_initial(a, b, m0)
    rho, u, p = pr.primitive(U, pr.G14)
    meta = dict(problem="super-tube-v2" + ("-m0" if m0 else ""), gamma=pr.G14, length=pr.SUPER2_LENGTH, t1=pr.SUPER2_T1, t_end=pr.SUPER2_T_END,
                boundary="left open (zero gradient outflow), right reflecting", note="cell averages of problems_1d.super2_initial",
                segments=json.dumps([[x, list(w)] for x, w in pr.SUPER2_SEGMENTS]),
                extras=f"B: v1's waves on [{pr.SUPER2_B}, {pr.SUPER2_B + 1}]; M: rho += {lay.BUMP_AMP} sin^2(pi (x - {lay.BUMP_X0}) / {lay.BUMP_W}) "
                       f"on [{lay.BUMP_X0}, {lay.BUMP_X0 + lay.BUMP_W}] at v = {-lay.V_MACH!r}, P = 1e-6; SO: rho += {lay.SO_AMP} sin({lay.SO_K} (x - "
                       f"{pr.SUPER2_SO[0]})) on [{pr.SUPER2_SO[0]}, {pr.SUPER2_SO[1]!r}]")
    np.savez(stem + ".npz", x_left=a, x_right=b, x=0.5 * (a + b), rho=rho, v=u, P=p, mass=U[0] * (b - a), momentum=U[1] * (b - a),
             energy=U[2] * (b - a), **meta)
    header = "\n".join(f"{k}: {v}" for k, v in meta.items()) + "\ncolumns: x_left x_right x rho v P mass momentum energy"
    np.savetxt(stem + ".txt", np.column_stack([a, b, 0.5 * (a + b), rho, u, p, U[0] * (b - a), U[1] * (b - a), U[2] * (b - a)]), fmt="%.17g",
               header=header)
    print(stem, n, "cells; mass", (U[0] * (b - a)).sum(), "energy", (U[2] * (b - a)).sum())


def figures(runs, singles, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from figstyle import FIGSIZE, apply_style, decorate
    apply_style()
    style = {"luna": dict(color="C0", ls="-", label="LUNA MC"), "static": dict(color="C1", ls="--", label="static"),
             "moving": dict(color="C2", ls="-", label="moving")}
    lw = 2.0
    os.makedirs(out, exist_ok=True)

    def run(f, p):
        st = status(f)
        if st is None:
            return None
        a, b, W = lad.cells(f, p)
        return 0.5 * (a + b), W, st

    def panel(ax, lo, hi, curves, refs, text):
        ys = []
        for x, y, kw in refs:
            m = (x >= lo) & (x <= hi)
            ax.plot(x[m], y[m], color="k", lw=lw, zorder=2.7, **kw)
            ys.append(y[m])
        for x, y, s, extra in curves:
            m = (x >= lo) & (x <= hi) & np.isfinite(y)
            ax.plot(x[m], y[m], lw=lw, zorder=3 if s == "moving" else 2, **{k: v for k, v in style[s].items() if k != "label"}, **extra)
            ys.append(y[m])
        y = np.concatenate([v for v in ys if v.size]) if ys else np.array([1.0])
        y = y[np.isfinite(y) & (y > 0)]
        if y.size and y.max() / max(y.min(), 1e-300) > 100:
            ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.text(0.03, 0.95, text, transform=ax.transAxes, ha="left", va="top", fontsize=10)
        decorate(ax)

    e = np.linspace(0.0, pr.SUPER2_LENGTH, 2 ** 17 + 1)
    xe, We = 0.5 * (e[:-1] + e[1:]), np.stack(pr.super2_exact(e[:-1], e[1:], 0.1))
    refs = {t: run(path(runs, "luna", "super-tube-v2", REF_N, t), "super-tube-v2") for t in TIMES}
    for k in KEYS:
        lo, hi = pr.super2_region(k)
        fig, axes = plt.subplots(2, 2, figsize=(FIGSIZE[0] * 2, FIGSIZE[1] * 1.6), sharex=True)
        for i, t in enumerate(TIMES):
            for j, n in enumerate(NS):
                ax = axes[i, j]
                rr = []
                if t == "0.1" and k in lay.EXACT_V2:
                    rr.append((xe, We[0], dict(ls="-", lw=1.2, label="exact")))
                if refs[t] is not None:
                    rr.append((refs[t][0], refs[t][1][0], dict(ls=":", label="LUNA $2^{20}$")))
                curves, notes = [], []
                for s in SERIES:
                    r = run(path(runs, s, "super-tube-v2", n, t), "super-tube-v2")
                    tag = ""
                    if s == "moving" and k != "M" and (r is None or not r[2]["reached"]):
                        r, tag = run(path(runs, s, "super-tube-v2-m0", n, t), "super-tube-v2"), " (m0)"
                    if r is None:
                        continue
                    if not r[2]["reached"]:
                        notes.append(f"{s}{tag} stopped at t = {r[2]['t']:.3g}")
                        continue
                    curves.append((r[0], r[1][0], s, dict(label=style[s]["label"] + tag)))
                panel(ax, lo, hi, curves, rr, f"$t = {t}$, $N = {n}$" + "".join("\n" + s for s in notes))
                if i == 1:
                    ax.set_xlabel(r"$x$")
                if j == 0:
                    ax.set_ylabel(r"$\rho$")
        handles, labels = [], []
        for ax in axes.flat:
            for h, lab in zip(*ax.get_legend_handles_labels()):
                if lab not in labels:
                    handles.append(h); labels.append(lab)
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.0))
        fig.subplots_adjust(top=0.92, hspace=0.08, wspace=0.18)
        name = f"{out}/super_tube_v2_{k}.png"
        fig.savefig(name)
        plt.close(fig)
        print(name, NAMES[k])
    for k in pr.SUPER2_SINGLE:   # the stages alone at t = 0.1
        p = f"super-v2-{k.lower()}"
        m = pr.problem(p)["cells_4096"]
        lo, hi = pr.problem(p)["box"]
        fig, axes = plt.subplots(1, 2, figsize=(FIGSIZE[0] * 2, FIGSIZE[1]), sharey=True)
        ref = run(path(singles, "luna", p, 64 * m, "0.1"), p)
        for ax, (n, label) in zip(axes, ((m, 4096), (8 * m, 32768))):
            rr = [(xe, We[0], dict(ls="-", lw=1.2, label="exact"))] if k in lay.EXACT_V2 else []
            if ref is not None:
                rr.append((ref[0], ref[1][0], dict(ls=":", label=f"LUNA {64 * m} cells")))
            curves, notes = [], []
            for s in SERIES:
                r = run(path(singles, s, p, n, "0.1"), p)
                if r is None:
                    continue
                if not r[2]["reached"]:
                    notes.append(f"{s} stopped at t = {r[2]['t']:.2g}")
                    continue
                curves.append((r[0], r[1][0], s, dict(label=style[s]["label"])))
            panel(ax, lo, hi, curves, rr, f"{k} alone, $t = 0.1$\n{n} cells (spacing of $N = {label}$)" + "".join("\n" + s for s in notes))
            ax.set_xlabel(r"$x$")
        axes[0].set_ylabel(r"$\rho$")
        axes[0].legend(frameon=False, fontsize=10, loc="lower right")
        name = f"{out}/super_tube_v2_alone_{k}.png"
        fig.savefig(name)
        plt.close(fig)
        print(name)


def npz(runs, out, s, p, n, t, meta):
    """One final state, the keys of v1's super_tube.py npz (luna-reference-solutions' luna/shu_osher format) plus provenance."""
    f = path(runs, s, p, int(n), t)
    a, b, W = lad.cells(f, p)
    info = json.load(open(meta))
    st = status(f)
    coarse = int(info.pop("coarse", 0))
    lo, hi = pr.problem(p)["box"]
    keys = dict(problem=p, lx=hi - lo, x0=lo, t=float(st["t"]), t_target=float(t), gamma=pr.G14, limiter=1, riemann_solver="hllc", precision="fp64",
                n_run=int(n), boundary="left outflow (zero gradient), right reflecting" if pr.problem(p).get("right") == pr.REFLECTIVE else "outflow both",
                ic="problems_1d.super2_initial: exact cell averages", steps=int(st["steps"]), wall_s=float(st["wall_s"]), reached=int(st["reached"]), **info)
    if coarse:
        e = np.linspace(lo, hi, coarse + 1)
        W = np.stack(lad.Reference(np.concatenate([a, b[-1:]]), *W).average(e[:-1], e[1:]))
        keys["n_coarse"] = coarse
    keys.update(rho=W[0], u=W[1], p=W[2])
    if s == "moving":
        keys.update(x_left=a, x_right=b)
    os.makedirs(out, exist_ok=True)
    name = f"{out}/{p.replace('-', '_')}_{s}_{n}_t{t}.npz"
    np.savez(name, **keys)
    print(name, {k: v for k, v in keys.items() if np.ndim(v) == 0})


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "single":
        single(*sys.argv[2:4])
    elif cmd == "score":
        score(*sys.argv[2:4])
    elif cmd == "ic":
        ic(sys.argv[2], sys.argv[3], len(sys.argv) > 4 and sys.argv[4] == "m0")
    elif cmd == "figures":
        figures(*sys.argv[2:5])
    else:
        npz(*sys.argv[2:9])
