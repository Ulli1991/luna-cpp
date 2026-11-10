"""The super-hard 1D tube (research_notes/super_tube_2026-11-10.md): scores, positivity, budgets, and the final states as npz.

    python super_tube.py check                              # no two stages of A to G meet by t = 0.1
    python super_tube.py score <runs dir> <out dir>         # L1 per stage and region, positivity, budgets: super_tube.json, .md
    python super_tube.py npz <runs dir> <out dir> <series> <N> <t> <meta json>    # one final state in luna-reference-solutions' format

Run with LUNA's environment. Runs: <series>_super-tube_<N>_t<t>.npz (LUNA) or .npy (ours), series luna, static, moving; t 0.1 and 2;
the references luna_super-tube_<2^k>_t<t>.npz with the largest N. The norm is ladder_1d.py's: sum over cells of V |W_i - exact_i|,
exact_i the reference's conservative average over the same cell. At t = 0.1, A to G against the exact superposition of their
Riemann solutions (problems_1d.super_exact), S and H against the LUNA reference; at t_end every region against the LUNA reference.
A stage's region: from the middle of the buffer on its left to the middle of the one on its right.
"""
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ladder_1d as lad  # noqa: E402
import problems_1d as pr  # noqa: E402

P = "super-tube"
SERIES = ("luna", "static", "moving")
NS = (4096, 32768)
TIMES = ("0.1", "2")
EXACT = ("A", "B", "C", "C2", "D", "E", "F", "G")


def regions():
    """{stage: (lo, hi)}: the stage's line between the middles of its buffers."""
    st = pr.SUPER_STAGES
    cuts = [0.0] + [0.5 * (a[2] + b[1]) for a, b in zip(st, st[1:])] + [pr.SUPER_LENGTH]
    return {s[0]: (lo, hi) for s, lo, hi in zip(st, cuts, cuts[1:])}


def run_file(runs, s, n, t):
    return f"{runs}/{s}_{P}_{n}_t{t}" + (".npz" if s == "luna" else ".npy")


def reference(runs, t):
    """The finest LUNA run at t as a ladder_1d.Reference, and its N."""
    files = [f for f in glob.glob(f"{runs}/luna_{P}_*_t{t}.npz") if int(re.search(r"_(\d+)_t", f).group(1)) > max(NS)]
    if not files:
        return None, 0
    f = max(files, key=lambda f: int(re.search(r"_(\d+)_t", f).group(1)))
    d = np.load(f)
    n = d["rho"].size
    return lad.Reference(np.linspace(0.0, pr.SUPER_LENGTH, n + 1), d["rho"], d["u"], d["p"]), n


def totals(a, b, W):
    return (pr.conserved(*W, pr.G14) * (b - a)).sum(axis=1)


def score(runs, out):
    regs = regions()
    e = np.linspace(0.0, pr.SUPER_LENGTH, 2 ** 16 + 1)
    U0 = (pr.super_initial(e[:-1], e[1:]) * np.diff(e)).sum(axis=1)  # exact: the averages are exact integrals
    rows = []
    for t in TIMES:
        ref, n_ref = reference(runs, t)
        for s in SERIES:
            for n in (*NS, n_ref):
                f = run_file(runs, s, n, t)
                if (s != "luna" and n == n_ref) or not os.path.exists(f):
                    continue
                a, b, W = lad.cells(f, P)
                x = 0.5 * (a + b)
                ex = np.stack(pr.super_exact(a, b, 0.1)) if t == "0.1" else None
                rf = np.stack(ref.average(a, b)) if ref is not None and n < n_ref else None
                row = dict(series=s, N=n, t=float(t), min_rho=float(W[0].min()), min_P=float(W[2].min()), finite=bool(np.isfinite(W).all()),
                           cells=int(a.size), reference_N=n_ref, L1={})
                U = totals(a, b, W)
                row["budget"] = dict(mass=float(U[0] - U0[0]), energy=float(U[2] - U0[2]), mass0=float(U0[0]), energy0=float(U0[2]))
                quiet = x < regs["A"][1] - 1.0
                row["A_quiet"] = float(np.abs(W[:, quiet] - np.array([[1.0], [0.0], [1.0]])).max())
                for name, (lo, hi) in [*regs.items(), ("all", (0.0, pr.SUPER_LENGTH))]:
                    m = (x >= lo) & (x < hi)
                    use_exact = t == "0.1" and name in EXACT
                    target = ex if use_exact else rf
                    if target is None:
                        continue
                    row["L1"][name] = dict(against="exact" if use_exact else f"LUNA {n_ref}",
                                           L1=[float(((b - a)[m] * np.abs(W[k, m] - target[k, m])).sum()) for k in range(3)])
                    if t == "0.1" and name == "B" and rf is not None:
                        row["L1"]["B vs LUNA"] = dict(against=f"LUNA {n_ref}", L1=[float(((b - a)[m] * np.abs(W[k, m] - rf[k, m])).sum()) for k in range(3)])
                rows.append(row)
    os.makedirs(out, exist_ok=True)
    json.dump(rows, open(f"{out}/super_tube.json", "w"), indent=1)
    lines = ["| t | region | against | " + " | ".join(f"{s} {n}" for s in SERIES for n in NS) + " |", "|---|---|---|" + "---|" * (len(SERIES) * len(NS))]
    index = {(r["series"], r["N"], r["t"]): r for r in rows}
    for t in TIMES:
        for name in [*regs, "B vs LUNA", "all"]:
            cells = [index.get((s, n, float(t)), {}).get("L1", {}).get(name) for s in SERIES for n in NS]
            if not any(cells):
                continue
            against = next(c["against"] for c in cells if c)
            lines.append(f"| {t} | {name} | {against} | " + " | ".join("-" if c is None else f"{c['L1'][0]:.3e}" for c in cells) + " |")
    lines += ["", "| t | series | N | min rho | min P | finite | mass - initial | energy - initial | A quiet, max abs W - ambient |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['t']:g} | {r['series']} | {r['N']} | {r['min_rho']:.3e} | {r['min_P']:.3e} | {r['finite']} | {r['budget']['mass']:.2e} | "
                     f"{r['budget']['energy']:.2e} | {r['A_quiet']:.1e} |")
    open(f"{out}/super_tube.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def npz(runs, out, s, n, t, meta):
    """One final state, keys as luna-reference-solutions' luna/shu_osher/shu_osher_mc.npz plus the cells of a moving mesh."""
    a, b, W = lad.cells(run_file(runs, s, int(n), t), P)
    info = json.load(open(meta))
    keys = dict(rho=W[0], u=W[1], p=W[2], problem="super-tube", lx=pr.SUPER_LENGTH, t=float(t), gamma=pr.G14, limiter=1, riemann_solver="hllc",
                precision="fp64", n_run=int(n), boundary="left outflow (zero gradient), right reflecting", **info)
    if s == "moving":
        keys.update(x_left=a, x_right=b)
    os.makedirs(out, exist_ok=True)
    name = f"{out}/{info.get('code', s)}_{s}_super_tube_{n}_t{t}.npz".replace("luna-cpp_", "")
    np.savez(name, **keys)
    print(name)


if __name__ == "__main__":
    if sys.argv[1] == "check":
        for t in (0.1, 0.129, 0.13):
            print(f"t = {t}: overlapping fans (x0 of each pair) {pr.super_meets(t)}")
    elif sys.argv[1] == "score":
        score(sys.argv[2], sys.argv[3])
    else:
        npz(*sys.argv[2:8])
