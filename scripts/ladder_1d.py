"""The convergence ladder of the 1D tests: LUNA, static and moving at N = 2^b, measured order per variable.

    python ladder_1d.py plan <series,...> <problems,...|all> <b_lo> <b_hi>     # the tasks of ladder_1d_a100.sh
    python ladder_1d.py table <runs dir> <out dir> [refs dir]                   # L1 of every run, tables, ladder_1d.json

Run with LUNA's environment. The norm is LUNA's own tests': the sum over cells of V |W_i - exact_i|, W the cell's primitives and
exact_i the exact solution's conservative average over the same cell (problems_1d.exact, 256 midpoints), for rho, v, P, on any
mesh, static or moving. Not the reconstructed field of compare_1d.py. As LUNA's tests: RCVCR and the vacuum expansions count v only
where the exact gas is; Noh counts [4 cells, 0.3] only. Linear waves: the part odd in the amplitude, (e(+eps) - e(-eps)) / (2 eps)
(linear_waves.py, errors). Shu-Osher, Woodward-Colella: against a fine LUNA MC run (compare_1d.py ref); Titarev-Toro, density only,
against AthenaK PPMX 65536, as LUNA's external gate; both averaged over the cell. Finer than a reference, both go on its cells.

Orders: p_all a fit over the ladder, p_top over its last four rungs (N >= 2^14 on the full ladder), of the rungs above a floor:
a rung is at the round-off floor when it falls short of a factor 1.1 from the rung before and is below 1e-7, or has a rung at the floor
before it.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import problems_1d as pr  # noqa: E402

SERIES = ("luna-mc", "static", "moving", "luna-minmod", "static-minmod", "moving-minmod", "luna-vanleer", "static-vanleer",
          "moving-vanleer", "static-none", "moving-none")
ALL = (*[f"toro-{k}" for k in range(1, 6)], "einfeldt-strong", "einfeldt-shear", "la-peak", "la-rcvcr", "la-stationary", "la-slow",
       "la-streams", "la-near-vacuum", "la-vacuum", "leblanc", "noh", "shu-osher", "woodward-colella", "titarev-toro",
       "sound-right", "sound-left", "entropy", "sound-advected", "simple-wave", *[f"shock-{m}" for m in pr.MACHS])
SMOOTH = ("sound-right", "sound-left", "entropy", "sound-advected", "simple-wave")
NOT_LUNA = ("simple-wave",)   # LUNA runs the setup all the same: the same scheme on our problem
VARS = ("rho", "v", "P")
SKIP, X_MAX = 4, 0.3          # Noh
REFERENCES = {"shu-osher": "ref_shu-osher.npz", "woodward-colella": "ref_woodward-colella.npz"}
ATHENAK = "athenak/titarev_toro/tt_ppmx_rk4_65536.npz"


def plan(series, problems, lo, hi):
    tasks = []
    for b in range(lo, hi + 1):
        for p in problems:
            for s in series:
                for sign in (("+", "-") if p in pr.LINEAR else ("+",)):
                    tasks.append(f"{s} {p} {1 << b} {sign}")
    return tasks


def path(runs, s, p, n, sign):
    return f"{runs}/{s}_{p}_{n}{'_m' if sign == '-' else ''}" + (".npz" if s.startswith("luna") else ".npy")


def cells(file, p):
    """a, b, (rho, u, p) of a run's cells."""
    if file.endswith(".npz"):
        d = np.load(file)
        W = np.stack([d["rho"], d["u"], d["p"]])
        lo, hi = pr.problem(p)["box"]
        e = np.linspace(lo, hi, W.shape[1] + 1)
        return e[:-1], e[1:], W
    d = np.load(file)
    return d[:, 1] - 0.5 * d[:, 2], d[:, 1] + 0.5 * d[:, 2], d[:, 3:6].T


class Reference:
    """Fine cells, piecewise constant: conservative averages over any cells from the running integrals."""

    def __init__(self, edges, rho, u=None, p=None, gamma=1.4):
        self.edges, self.gamma, self.full = edges, gamma, u is not None
        U = pr.conserved(rho, u, p, gamma) if self.full else np.stack([rho])
        self.running = np.concatenate([np.zeros((U.shape[0], 1)), np.cumsum(U * np.diff(edges), axis=1)], axis=1)

    def average(self, a, b):
        I = lambda x: np.stack([np.interp(x, self.edges, row) for row in self.running])
        U = (I(b) - I(a)) / (b - a)
        if self.full:
            return pr.primitive(U, self.gamma)
        return U[0], None, None


def references(refs_dir):
    out = {}
    for p, name in REFERENCES.items():
        f = f"{refs_dir}/{name}"
        if os.path.exists(f):
            d = np.load(f)
            out[p] = Reference(d["edges"], d["rho"], d["u"], d["p"])
    root = os.environ.get("LUNA_REFERENCE_SOLUTIONS", "/raven/u/uli/luna-reference-solutions")
    if os.path.exists(f"{root}/{ATHENAK}"):
        rho = np.asarray(np.load(f"{root}/{ATHENAK}")["ref_rho"], np.float64)
        out["titarev-toro"] = Reference(np.linspace(-5.0, 5.0, rho.size + 1), rho)
    return out


def rebin(a, b, W, ref, gamma):
    """A run finer than its reference, on the reference's cells; else the run."""
    if a.size * (b[-1] - a[0]) / (ref.edges[-1] - ref.edges[0]) <= ref.edges.size - 1:
        return a, b, W
    run = Reference(np.concatenate([a, b[-1:]]), W[0], W[1], W[2], gamma)
    A, B = ref.edges[:-1], ref.edges[1:]
    return A, B, np.stack(run.average(A, B))


def errors(p, a, b, W, refs, eps=pr.EPS):
    """Per variable e_i = W_i - exact_i, and the cells' mask per variable."""
    pb = pr.problem(p)
    if pb["kind"] == "reference":
        if p not in refs:
            return None
        a, b, W = rebin(a, b, W, refs[p], pb["gamma"])
        ex = refs[p].average(a, b)
    else:
        ex = pr.exact(p, a, b, eps)
    e = [None if x is None else w - x for w, x in zip(W, ex)]
    V = b - a
    masks = [np.ones_like(a, bool) for _ in range(3)]
    if p == "la-rcvcr":
        masks[1] = ex[0] > 0
    if p in ("la-near-vacuum", "la-vacuum"):
        masks[1] = ex[0] > 1e-3
    if p == "noh":
        keep = 0.5 * (a + b) < X_MAX
        keep[:SKIP] = False
        masks = [keep] * 3
    return e, V, masks


def measure(runs, s, p, n, refs):
    """A row: L1 of rho, v, P, steps, the time reached; None if the run is not there."""
    base = path(runs, s, p, n, "+")
    meta = base + ".json"
    if not os.path.exists(meta):
        return None
    row = json.load(open(meta))
    row.update(series=s, problem=p, N=n)
    if not row.get("reached", 1):
        row["L1"] = [None] * 3
        return row
    a, b, W = cells(base, p)
    if not np.isfinite(W).all():
        row.update(L1=[None] * 3, reached=0, nonfinite=1)
        return row
    if p in pr.LINEAR:
        minus = path(runs, s, p, n, "-")
        if not os.path.exists(minus + ".json"):
            return None
        am, bm, Wm = cells(minus, p)
        ep, V, masks = errors(p, a, b, W, refs, pr.EPS)
        em = errors(p, am, bm, Wm, refs, -pr.EPS)[0]
        e = [(x - y) / (2 * pr.EPS) for x, y in zip(ep, em)]
    else:
        out = errors(p, a, b, W, refs)
        if out is None:
            return None
        e, V, masks = out
    row["L1"] = [None if x is None else float((V[m] * np.abs(x[m])).sum()) for x, m in zip(e, masks)]
    row["min_rho"], row["min_P"] = float(W[0].min()), float(W[2].min())
    return row


def fit(ns, es):
    return float(-np.polyfit(np.log(ns), np.log(es), 1)[0]) if len(ns) >= 2 else None


def above_floor(ns, es):
    """The rungs before a round-off floor (see the head of this file)."""
    keep = len(ns)
    for i in range(1, len(ns)):
        if es[i - 1] < 1.1 * es[i] and es[i] < 1e-7:
            keep = i
            break
    return ns[:keep], es[:keep]


def orders(rows):
    """{(problem, series, variable index): dict(p_all, p_top, N, L1, floor)}."""
    out = {}
    for key in sorted({(r["problem"], r["series"]) for r in rows}):
        sel = sorted((r for r in rows if (r["problem"], r["series"]) == key), key=lambda r: r["N"])
        for k in range(3):
            ns = [r["N"] for r in sel if r["L1"][k] is not None]
            es = [r["L1"][k] for r in sel if r["L1"][k] is not None]
            if not ns or any(e == 0 for e in es):
                continue
            n2, e2 = above_floor(ns, es)
            out[(*key, k)] = dict(p_all=fit(n2, e2), p_top=fit(n2[-4:], e2[-4:]), N=ns, L1=es, floor=len(n2) < len(ns), n_fit=len(n2))
    return out


def table(runs, reports, refs_dir):
    refs = references(refs_dir)
    rows = []
    for p in ALL:
        for s in SERIES:
            for b in range(4, 18):
                r = measure(runs, s, p, 1 << b, refs)
                if r:
                    rows.append(r)
    os.makedirs(reports, exist_ok=True)
    with open(f"{reports}/ladder_1d.json", "w") as f:
        json.dump(rows, f, indent=0)
    res = orders(rows)
    print("| test | variable | series | N | L1 first | L1 last | p all | p top | rungs |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (p, s, k), o in sorted(res.items(), key=lambda kv: (ALL.index(kv[0][0]), kv[0][2], SERIES.index(kv[0][1]))):
        print(f"| {p} | {VARS[k]} | {s} | {o['N'][0]}-{o['N'][-1]} | {o['L1'][0]:.3e} | {o['L1'][-1]:.3e} | {o['p_all']:.2f} | "
              f"{o['p_top']:.2f} | {o['n_fit']}{' floor' if o['floor'] else ''} |")
    print("\n## Runs that did not reach t_end\n")
    for r in rows:
        if not r.get("reached", 1):
            print(f"{r['series']} {r['problem']} N={r['N']}: stopped at t={r.get('t')} after {r.get('steps')} steps")
    print("\n## Ratio to LUNA at the same N, same limiter: largest |ratio - 1| over N, per variable\n")
    index = {(r["series"], r["problem"], r["N"]): r for r in rows}
    for r_series, l_series in (("static", "luna-mc"), ("static-minmod", "luna-minmod"), ("static-vanleer", "luna-vanleer")):
        for p in ALL:
            worst = [0.0, 0.0, 0.0]
            count = 0
            for n in sorted({r["N"] for r in rows if r["problem"] == p}):
                a, b = index.get((r_series, p, n)), index.get((l_series, p, n))
                if a and b and a["L1"][0] is not None and b["L1"][0] is not None:
                    count += 1
                    for k in range(3):
                        if a["L1"][k] is not None and b["L1"][k]:
                            worst[k] = max(worst[k], abs(a["L1"][k] / b["L1"][k] - 1))
            if count:
                print(f"| {p} | {r_series} | {count} rungs | " + " | ".join(f"{w:.2e}" for w in worst) + " |")


if __name__ == "__main__":
    if sys.argv[1] == "plan":
        probs = ALL if sys.argv[3] == "all" else sys.argv[3].split(",")
        print("\n".join(plan(sys.argv[2].split(","), probs, int(sys.argv[4]), int(sys.argv[5]))))
    else:
        table(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else sys.argv[2])
