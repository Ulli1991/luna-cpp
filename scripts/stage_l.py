"""Stage L alone (research_notes/stage_l_2026-11-19.md): a single Riemann problem to t = 4, exact at every time; luna-cpp #30.

    python stage_l.py check                          # exact waves of the three ratios; no wave reaches a wall by t = 4
    python stage_l.py score <runs> <out>             # L1 against the exact solution, orders, the pile-up at the contact (stage_l.md, .json)
    python stage_l.py figures <runs> <out dir>       # convergence, pile-up over time, density at t = 4 (LUNA's figstyle)
    python stage_l.py npz <runs> <out root> <meta dir>   # final states into luna-reference-solutions (luna/stage_l, luna-cpp/stage_l)

Run with LUNA's environment. Runs (scripts/stage_l_run.sh): luna_stage-l-<k>_<N>_t<t>.npz (LUNA, uniform cells); for static and moving
<series>_stage-l-<k>_<N>_t<t>.npy, rows (generator x, mass, momentum, energy): the cells are the 1D Voronoi cells of the sorted
generators between the walls. L1 is LUNA's norm: sum over cells of V |W - exact|, exact the conservative average over the cell
(problems_1d.averages, 256 points a cell, the bisection solver). The pile-up: the smallest inner cell and the number of generators
within one initial cell width h0 = L / N of the exact contact x_c = u* t.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import problems_1d as pr  # noqa: E402
import super_tube_layout as lay  # noqa: E402

KS = ("3", "6", "9")
NS = (1024, 2048, 4096, 8192, 16384, 32768)
SERIES = ("luna", "static", "moving")
TIMES = ("0.1", "0.5", "0.81", "2", "4")
SNAPS = ("0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.81", "0.9", "1", "1.25", "1.5", "1.75", "2", "2.5", "3", "3.5", "4")
LO, HI = -2.0, 9.0


def star(k):
    pb = pr.problem(f"stage-l-{k}")
    return lay.speeds(pb["WL"], pb["WR"], 1e-30)


def check():
    for k in KS:
        pb = pr.problem(f"stage-l-{k}")
        s = star(k)
        left, right = min(s["left"][1:]), max(s["right"][1:])
        print(f"ratio 1e{k}: p* = {s['p']:.4g}, u* = {s['u']:.4f}, rho*L = {s['rl']:.4g}, rho*R = {s['rr']:.4g}; left wave {s['left'][0]} "
              f"{s['left'][1]:.4f}..{s['left'][2]:.4f}, right {s['right'][0]} {s['right'][1]:.4f}; at t = 4 the waves span "
              f"[{4 * left:.3f}, {4 * right:.3f}] in [{LO}, {HI}]: {'inside' if LO < 4 * left and 4 * right < HI else 'REACHES A WALL'}")


def cells(f, p):
    """a, b, (rho, u, p) of a run's cells at a snapshot."""
    if f.endswith(".npz"):
        d = np.load(f)
        e = np.linspace(LO, HI, d["rho"].size + 1)
        return e[:-1], e[1:], np.stack([d["rho"], d["u"], d["p"]])
    d = np.load(f)
    d = d[np.argsort(d[:, 0])]
    x = d[:, 0]
    e = np.concatenate([[LO], 0.5 * (x[1:] + x[:-1]), [HI]])
    a, b = e[:-1], e[1:]
    U = d[:, 1:4].T / (b - a)
    return a, b, np.stack(pr.primitive(U, pr.G14)), x


def path(runs, s, k, n, t):
    return f"{runs}/{s}_stage-l-{k}_{n}_t{t}" + (".npz" if s == "luna" else ".npy")


def exact(k, a, b, t):
    pb = pr.problem(f"stage-l-{k}")
    return np.stack(pr.averages(dict(pb, t=float(t)), a, b))


def reached(runs, s, k, n, t):
    """Whether the run got to t: the snapshot exists (static, moving) or the JSON says so (LUNA)."""
    f = path(runs, s, k, n, t)
    if s == "luna":
        return os.path.exists(f + ".json") and json.load(open(f + ".json")).get("reached") == 1
    return os.path.exists(f)


def stalled_at(runs, s, k, n):
    """For static, moving: the last snapshot time, steps, wall seconds (from <stem>_times.txt), and the final JSON."""
    stem = f"{runs}/{s}_stage-l-{k}_{n}"
    rows = np.loadtxt(stem + "_times.txt", ndmin=2) if os.path.exists(stem + "_times.txt") else np.zeros((0, 3))
    meta = json.load(open(stem + ".npy.json")) if os.path.exists(stem + ".npy.json") else {}
    return rows, meta


def score(runs, out):
    rows, pile = [], []
    for k in KS:
        u_star = star(k)["u"]
        for s in SERIES:
            for n in NS:
                for t in TIMES:
                    if not reached(runs, s, k, n, t):
                        continue
                    c = cells(path(runs, s, k, n, t), f"stage-l-{k}")
                    a, b, W = c[:3]
                    ex = exact(k, a, b, t)
                    rows.append(dict(k=k, series=s, N=n, t=float(t), L1=[float(((b - a) * np.abs(W[v] - ex[v])).sum()) for v in range(3)],
                                     min_rho=float(W[0].min()), min_P=float(W[2].min()), finite=bool(np.isfinite(W).all())))
                if s == "moving":
                    h0 = (HI - LO) / n
                    for t in SNAPS:
                        f = path(runs, s, k, n, t)
                        if not os.path.exists(f):
                            continue
                        a, b, W, x = cells(f, f"stage-l-{k}")
                        V = (b - a)[1:-1]
                        i = np.argmin(V)
                        near = int((np.abs(x - u_star * float(t)) < h0).sum())
                        pile.append(dict(k=k, N=n, t=float(t), smallest_h0=float(V[i] / h0), at=float(x[1:-1][i]), contact=float(u_star * float(t)),
                                         near_contact=near, finite=bool(np.isfinite(W).all())))
    os.makedirs(out, exist_ok=True)
    stops = {}
    for k in KS:
        for s in ("static", "moving"):
            for n in NS:
                snaps, meta = stalled_at(runs, s, k, n)
                if meta:
                    stops[f"{s} 1e{k} {n}"] = dict(t=meta.get("t"), reached=meta.get("reached"), steps=meta.get("steps"), wall_s=meta.get("wall_s"))
    json.dump(dict(L1=rows, pileup=pile, runs=stops), open(f"{out}/stage_l.json", "w"), indent=1)
    index = {(r["k"], r["series"], r["N"], r["t"]): r for r in rows}
    lines = []
    for k in KS:
        lines += [f"## Density ratio 1e{k}: L1(rho) against the exact solution (order between N and 2N in brackets)", "",
                  "| t | series | " + " | ".join(str(n) for n in NS) + " |", "|---|---|" + "---|" * len(NS)]
        for t in TIMES:
            for s in SERIES:
                vals = [index.get((k, s, n, float(t))) for n in NS]
                if not any(vals):
                    continue
                cellsx = []
                for i, v in enumerate(vals):
                    if v is None:
                        cellsx.append("-")
                        continue
                    o = ""
                    if i and vals[i - 1] is not None and v["L1"][0] > 0:
                        o = f" ({np.log2(vals[i - 1]['L1'][0] / v['L1'][0]):.2f})"
                    cellsx.append(f"{v['L1'][0]:.3e}{o}")
                lines.append(f"| {t} | {s} | " + " | ".join(cellsx) + " |")
        lines += ["", f"Moving, ratio 1e{k}: smallest inner cell / h0 and generators within h0 of the contact u* t", "",
                  "| t | " + " | ".join(str(n) for n in NS) + " |", "|---|" + "---|" * len(NS)]
        pidx = {(p["k"], p["N"], p["t"]): p for p in pile}
        for t in SNAPS:
            vals = [pidx.get((k, n, float(t))) for n in NS]
            if any(vals):
                lines.append(f"| {t} | " + " | ".join("-" if v is None else f"{v['smallest_h0']:.1e}, {v['near_contact']}" for v in vals) + " |")
        lines.append("")
    lines += ["## Runs that stopped", "", "| run | t | reached | steps | wall s |", "|---|---|---|---|---|"]
    for name, m in stops.items():
        lines.append(f"| {name} | {m['t']} | {m['reached']} | {m['steps']} | {m['wall_s']} |")
    open(f"{out}/stage_l.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def figures(runs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from figstyle import FIGSIZE, apply_style, decorate
    apply_style()
    os.makedirs(out, exist_ok=True)
    d = json.load(open(f"{out}/../report/stage_l.json")) if os.path.exists(f"{out}/../report/stage_l.json") else None
    if d is None:
        raise SystemExit("run score first (report/stage_l.json)")
    style = {"luna": dict(color="C0", ls="-", marker="o", label="LUNA MC (floors on)"), "static": dict(color="C1", ls="--", marker="s", label="static"),
             "moving": dict(color="C2", ls="-", marker="^", label="moving")}
    # 1. convergence at t = 0.1 and 4, one panel per ratio
    fig, axes = plt.subplots(2, 3, figsize=(FIGSIZE[0] * 2, FIGSIZE[1] * 1.5), sharex=True)
    for j, k in enumerate(KS):
        for i, t in enumerate(("0.1", "4")):
            ax = axes[i, j]
            for s in SERIES:
                pts = sorted((r["N"], r["L1"][0]) for r in d["L1"] if r["k"] == k and r["series"] == s and r["t"] == float(t))
                if pts:
                    ax.loglog(*zip(*pts), lw=2, ms=5, **style[s])
            ax.text(0.04, 0.06, f"density ratio $10^{{{k}}}$, $t = {t}$", transform=ax.transAxes, fontsize=10)
            if i == 1:
                ax.set_xlabel("$N$")
            if j == 0:
                ax.set_ylabel(r"L1($\rho$)")
            decorate(ax)
    axes[0, 2].legend(frameon=False, fontsize=9, loc="upper right")
    fig.savefig(f"{out}/stage_l_convergence.png")
    plt.close(fig)
    # 2. pile-up over time (moving)
    fig, axes = plt.subplots(2, 3, figsize=(FIGSIZE[0] * 2, FIGSIZE[1] * 1.5), sharex=True)
    for j, k in enumerate(KS):
        for n, c in zip(NS, plt.cm.viridis(np.linspace(0, 0.9, len(NS)))):
            pts = sorted((p["t"], p["smallest_h0"], p["near_contact"]) for p in d["pileup"] if p["k"] == k and p["N"] == n)
            if pts:
                t, sm, near = zip(*pts)
                axes[0, j].semilogy(t, sm, color=c, lw=2, marker="^", ms=4, label=f"$N = {n}$")
                axes[1, j].plot(t, near, color=c, lw=2, marker="^", ms=4)
        axes[0, j].set_ylim(0.03, 0.4)
        axes[0, j].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axes[0, j].text(0.04, 0.94, f"density ratio $10^{{{k}}}$", transform=axes[0, j].transAxes, va="top", fontsize=10)
        axes[1, j].set_xlabel("$t$")
        for ax in axes[:, j]:
            decorate(ax)
    axes[0, 0].set_ylabel("smallest cell / $h_0$")
    axes[1, 0].set_ylabel("cells within $h_0$ of the contact")
    for ax in axes[1]:
        ax.set_ylim(-1, 20)
    axes[1, 1].legend(*axes[0, 0].get_legend_handles_labels(), frameon=False, fontsize=9, loc="upper right", ncol=2)
    fig.savefig(f"{out}/stage_l_pileup.png")
    plt.close(fig)
    # 3. density at t = 4, N = 1024, near the waves
    fig, axes = plt.subplots(1, 3, figsize=(FIGSIZE[0] * 2, FIGSIZE[1]), sharey=True)
    for ax, k in zip(axes, KS):
        e = np.linspace(LO, HI, 2 ** 14 + 1)
        ax.semilogy(0.5 * (e[:-1] + e[1:]), exact(k, e[:-1], e[1:], 4)[0], color="k", lw=1.2, label="exact")
        notes = [f"density ratio $10^{{{k}}}$", "$t = 4$, $N = 1024$"]
        for s in SERIES:
            f = path(runs, s, k, 1024, "4")
            if os.path.exists(f):
                c = cells(f, f"stage-l-{k}")
                ax.semilogy(0.5 * (c[0] + c[1]), c[2][0], lw=2, **{kk: v for kk, v in style[s].items() if kk != "marker"})
            elif s == "moving":
                _, meta = stalled_at(runs, s, k, 1024)
                notes.append(f"moving: stopped at $t = {meta.get('t', 0):.3g}$")
        ax.text(0.97, 0.96, "\n".join(notes), transform=ax.transAxes, va="top", ha="right", fontsize=10)
        ax.set_xlim(LO, HI)
        ax.set_xlabel("$x$")
        decorate(ax)
    axes[0].set_ylabel(r"$\rho$")
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.savefig(f"{out}/stage_l_profiles.png")
    plt.close(fig)
    print("figures in", out)


def npz(runs, out, meta_dir):
    """The last state of every run (t = 4, or where it stopped) as npz: rho, u, p, cell edges, provenance."""
    for k in KS:
        for s in SERIES:
            for n in NS:
                last = next((t for t in reversed(SNAPS if s != "luna" else TIMES) if os.path.exists(path(runs, s, k, n, t))), None)
                if last is None:
                    continue
                c = cells(path(runs, s, k, n, last), f"stage-l-{k}")
                a, b, W = c[:3]
                info = json.load(open(f"{meta_dir}/{'luna' if s == 'luna' else 'lcpp'}.json"))
                code = "luna" if s == "luna" else "luna-cpp"
                d = f"{out}/{code}/stage_l"
                os.makedirs(d, exist_ok=True)
                meta = {}
                if s != "luna":
                    _, meta = stalled_at(runs, s, k, n)
                keys = dict(problem=f"stage-l-{k}", series=s, density_ratio=float(f"1e{k}"), t=float(last), t_end=4.0, reached=int(last == "4"),
                            n_run=n, x_left=a, x_right=b, rho=W[0], u=W[1], p=W[2], gamma=pr.G14, limiter=1, riemann_solver="hllc",
                            boundary="outflow both", steps=int(meta.get("steps", -1)), **info)
                np.savez(f"{d}/stage_l_{k}_{s}_{n}_t{last}.npz", **keys)
    print("npz written under", out)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "check":
        check()
    elif cmd == "score":
        score(*sys.argv[2:4])
    elif cmd == "figures":
        figures(*sys.argv[2:4])
    else:
        npz(*sys.argv[2:5])
