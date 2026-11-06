"""The convergence ladder of the 1D tests: LUNA, static and moving at N = 2^b, measured order per variable.

    python ladder_1d.py plan <series,...> <problems,...|all> <b_lo> <b_hi>     # the tasks of ladder_1d_a100.sh
    python ladder_1d.py table <runs dir> <out dir> [refs dir]                   # L1 of every run, tables, ladder_1d.json
    python ladder_1d.py gates <runs dir> <out dir> [refs dir] > src/test/luna_1d_gates.hpp    # the gates of hydro1d_cases_test, from ladder_1d.json

Run with LUNA's environment. The norm is LUNA's own tests': the sum over cells of V |W_i - exact_i|, W the cell's primitives and
exact_i the exact solution's conservative average over the same cell (problems_1d.exact, 256 midpoints), for rho, v, P, on any
mesh, static or moving. Not the reconstructed field of compare_1d.py. As LUNA's tests: RCVCR and the vacuum expansions count v only
where the exact gas is; Noh counts [4 cells, 0.3] only. Linear waves: the part odd in the amplitude, (e(+eps) - e(-eps)) / (2 eps)
(linear_waves.py, errors). Shu-Osher, Woodward-Colella: against a fine LUNA MC run (compare_1d.py ref); Titarev-Toro, density only,
against AthenaK PPMX 65536, as LUNA's external gate; both averaged over the cell. Finer than a reference, both go on its cells.

Orders: p_all a fit over the ladder, p_top over its last four rungs (N >= 2^14 on the full ladder). On the linear waves, whose errors
at amplitude 1e-6 meet rounding, only the rungs before the first that falls short of a factor 1.1 from the one before.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import problems_1d as pr  # noqa: E402

SERIES = ("luna-mc", "static", "moving", "moving-wide", "moving-fixed", "luna-minmod", "static-minmod", "moving-minmod", "luna-vanleer", "static-vanleer",
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


def pad(p):
    """The length of padding that keeps the seam's waves out of the box to t_end: twice the fastest speed of the two states."""
    pb = pr.problem(p)
    fastest = max(abs(W[1]) + np.sqrt(pb["gamma"] * W[2] / W[0]) for W in (pb["WL"], pb["WR"]) if W[0] > 0)
    return f"{2 * fastest * pb['t']:.6g}"


def plan(series, problems, lo, hi):
    tasks = []
    for b in range(lo, hi + 1):
        for p in problems:
            for s in series:
                for sign in (("+", "-") if p in pr.LINEAR else ("+",)):
                    tasks.append(f"{s} {p} {1 << b} {sign} {pad(p) if s == 'moving-wide' else 0}")
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
    lo, hi = pr.problem(p)["box"]
    if d.shape[0] > 0 and (d[0, 1] < lo or d[-1, 1] > hi) and pr.problem(p)["boundary"] != pr.PERIODIC:  # moving-wide: LUNA's box
        d = d[(d[:, 1] >= lo) & (d[:, 1] <= hi)]
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
    if row.get("steps", 16) < 16:          # Noh, moving: a step of the whole run (issue 15) is no run
        row["reached"] = 0
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


def above_floor(p, ns, es):
    """The rungs before a round-off floor (see the head of this file): linear waves at amplitude 1e-6 only."""
    keep = len(ns)
    for i in range(1, len(ns)):
        if p in pr.LINEAR and es[i - 1] < 1.1 * es[i]:
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
            n2, e2 = above_floor(key[0], ns, es)
            out[(*key, k)] = dict(p_all=fit(n2, e2), p_top=fit(n2[-4:], e2[-4:]), N=ns, L1=es, floor=len(n2) < len(ns), n_fit=len(n2))
    return out


def num(x):
    return "-" if x is None else f"{x:.2f}"


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
        print(f"| {p} | {VARS[k]} | {s} | {o['N'][0]}-{o['N'][-1]} | {o['L1'][0]:.3e} | {o['L1'][-1]:.3e} | {num(o['p_all'])} | "
              f"{num(o['p_top'])} | {o['n_fit']}{' floor' if o['floor'] else ''} |")
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


# Moving runs that cannot run yet: the issue of each (a failure must stay measured), and those waiting for the inflow fix.
KNOWN = {"toro-2": 14, "einfeldt-strong": 14, "einfeldt-shear": 14, "la-peak": 14, "la-rcvcr": 14, "la-stationary": 14, "la-slow": 14, "noh": 15, "la-streams": 16,
         "la-vacuum": 17}
WAITS = ("shu-osher", "titarev-toro", "toro-5")
GATE_N = {"titarev-toro": (1024, 2048, 4096)}
BINS = {"shu-osher": 256, "woodward-colella": 256, "titarev-toro": 1024}
FLOOR = 1e-7
LEFT = ("luna-mc", "luna-minmod", "luna-vanleer", "moving")


def binned(file, p, bins):
    """rho of a run on `bins` equal cells of its box."""
    a, b, W = cells(file, p)
    lo, hi = pr.problem(p)["box"]
    h = (hi - lo) / bins
    k = np.minimum(bins - 1, ((0.5 * (a + b) - lo) / h).astype(int))
    return np.bincount(k, (b - a) * W[0], bins) / h


def cxx(values):
    return "{" + ", ".join("nan" if v is None or not np.isfinite(v) else f"{v:.17g}" for v in values) + "}"


def gates(runs, reports, refs_dir):
    rows = json.load(open(f"{reports}/ladder_1d.json"))
    index = {(r["series"], r["problem"], r["N"]): r for r in rows}
    refs = references(refs_dir)
    out, stored, ladders = [], [], {}
    for p in ALL:
        ns = GATE_N.get(p, (256, 512, 1024))
        ref = None
        if p in BINS:
            ref = refs[p].average(*(lambda e: (e[:-1], e[1:]))(np.linspace(*pr.problem(p)["box"], BINS[p] + 1)))[0]
            ref = np.array([float(f"{v:.10g}") for v in ref])
            stored.append(f"    {{\"{p}\", {BINS[p]}, {p.replace('-', '_')}_rho}},")
            print(f"inline constexpr double {p.replace('-', '_')}_rho[] = {{" + ", ".join(f"{v:.10g}" for v in ref) + "};")
        for n in ns:
            got = {}
            for s in LEFT:
                r = index.get((s, p, n))
                if r is None or not r.get("reached", 1):
                    got[s] = [None] * 3
                elif ref is not None:
                    got[s] = [float((np.abs(binned(path(runs, s, p, n, "+"), p, BINS[p]) - ref)).sum() * (pr.problem(p)["box"][1] - pr.problem(p)["box"][0]) / BINS[p]), None, None]
                else:
                    got[s] = r["L1"]
                ladders.setdefault((p, s), {})[n] = got[s]
            issue = "waits" if p in WAITS else KNOWN.get(p, 0)
            # CI's 2 cores and sanitizers: moving at the two smallest N, a known failure at the smallest, the other limiters at the middle one.
            if n != ns[1]:
                got["luna-minmod"], got["luna-vanleer"] = [None] * 3, [None] * 3
            if n == ns[2] or (n == ns[1] and p in KNOWN):
                issue = "waits"
            tol = 1e-2 if p in pr.LINEAR else 1e-6
            if p in pr.LINEAR:   # rounding, not the scheme, below 1e-6 at amplitude 1e-6 on a moving line: that is the bound
                got["moving"] = [None if v is None else max(v, 1e-6) for v in got["moving"]]
            out.append(f"    {{\"{p}\", {n}, {str(got['luna-mc'][0] is not None).lower()}, {FLOOR}, {tol}, {cxx(got['luna-mc'])}, {cxx(got['luna-minmod'])}, "
                       f"{cxx(got['luna-vanleer'])}, {cxx(got['moving'])}, {issue}}},")
    bands = []
    tag = {"luna-mc": "", "luna-minmod": "minmod", "luna-vanleer": "vanleer", "moving": "moving"}
    for (p, s), by_n in ladders.items():
        ns = sorted(by_n)
        if s in ("luna-minmod", "luna-vanleer"):
            continue
        if s == "moving":
            ns = ns[:2]
        lo, hi = [None] * 3, [None] * 3
        for k in range(3):
            es = [by_n[n][k] for n in ns]
            if any(e is None or e < FLOOR for e in es):
                continue
            o = fit(ns, es)
            lo[k], hi[k] = o - 0.2, o + 0.2
        if any(v is not None for v in lo) and not (s == "moving" and (p in WAITS or p in KNOWN)):
            bands.append(f"    {{\"{p}\", \"{tag[s]}\", {cxx(lo)}, {cxx(hi)}}},")
    print("\ninline constexpr Reference references[] = {\n" + "\n".join(stored) + "\n};")
    print("\ninline constexpr Row rows[] = {\n" + "\n".join(out) + "\n};")
    print("\ninline constexpr Band bands[] = {\n" + "\n".join(bands) + "\n};")
    print("\n}  // namespace luna_gates")


HEADER = """#pragma once

// Generated by scripts/ladder_1d.py gates: L1 of rho, v, P in the norm of LUNA's tests; luna, minmod, vanleer: LUNA's runs; moving: ours.
// References: rho of the ladder's fine run on a few bins. nan: not measured.
// issue: a known failure on a moving mesh, its luna-cpp issue; waits: no moving run at this N.

#include <limits>

namespace luna_gates {

inline constexpr double nan = std::numeric_limits<double>::quiet_NaN();
inline constexpr int waits = -1;

struct Reference {
    const char* problem;
    int bins;
    const double* rho;
};
struct Row {
    const char* problem;
    int n;
    bool run_luna;
    double floor, tolerance;
    double luna[3], minmod[3], vanleer[3], moving[3];
    int issue;
};
struct Band {
    const char* problem;
    const char* series;
    double lo[3], hi[3];
};
"""


if __name__ == "__main__":
    if sys.argv[1] == "plan":
        probs = ALL if sys.argv[3] == "all" else sys.argv[3].split(",")
        print("\n".join(plan(sys.argv[2].split(","), probs, int(sys.argv[4]), int(sys.argv[5]))))
    elif sys.argv[1] == "gates":
        print(HEADER)
        gates(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else sys.argv[2])
    else:
        table(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else sys.argv[2])
