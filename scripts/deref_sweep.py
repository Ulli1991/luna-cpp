"""Derefinement (luna-cpp #30) on the 1D ladder: does it trigger, and what it costs where it does.

    python deref_sweep.py run <bin> <out dir> <fractions,...> <N,...> [jobs]   # every ladder problem moving (and moving-wide), each f
    python deref_sweep.py table <out dir> <refs dir>                            # removed cells, same bits as f = 0, L1 rho change

Run with LUNA's environment. Each run: hydro_1d <p> <N> --moving [--pad=...] --steps=200 N --derefine=f --out=<out>/<series>_<p>_<N>_f<f>.npy;
the log line gives the hash and the cells removed. L1 rho as the ladder's (ladder_1d.errors; linear waves: plain L1, amplitude 1e-6).
"""
import concurrent.futures
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ladder_1d as lad  # noqa: E402

SERIES = ("moving", "moving-wide")


def one(binary, out, s, p, n, f):
    name = f"{out}/{s}_{p}_{n}_f{f}.npy"
    args = [binary, p, str(n), "--moving", f"--steps={200 * n}", f"--derefine={f}", "--repeats=1", f"--out={name}"]
    if s == "moving-wide":
        args.append(f"--pad={lad.pad(p)}")
    if p in lad.pr.LINEAR:
        args.append(f"--amp={lad.pr.EPS}")  # as the ladder's "+" run; L1 here is the plain error at that amplitude
    r = subprocess.run(args, capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="1"))
    line = next((x for x in r.stdout.splitlines() if x.startswith("ours")), "")
    get = lambda k: (re.search(rf" {k}=([^ ]+)", line) or [None, None])[1]  # noqa: E731
    row = dict(series=s, problem=p, N=n, f=f, hash=get("hash"), removed=int(get("removed") or 0), reached=get("reached") == "1",
               t=float(get("t") or "nan"), steps=int(get("steps") or 0), file=name)
    json.dump(row, open(name + ".row.json", "w"))
    return row


def run(binary, out, fractions, ns, jobs=8):
    os.makedirs(out, exist_ok=True)
    riemann = lambda p: "WL" in lad.pr.problem(p)  # noqa: E731  moving-wide: two states to pad with
    tasks = [(s, p, int(n), f) for n in ns.split(",") for p in lad.ALL for s in SERIES for f in ["0", *fractions.split(",")]
             if s == "moving" or riemann(p)]
    with concurrent.futures.ThreadPoolExecutor(int(jobs)) as pool:
        for row in pool.map(lambda t: one(binary, out, *t), tasks):
            print(json.dumps(row), flush=True)


def l1_rho(row, refs):
    if not row["reached"] or not os.path.exists(row["file"]):
        return None
    a, b, W = lad.cells(row["file"], row["problem"])
    got = lad.errors(row["problem"], a, b, W, refs)
    if got is None:
        return None
    e, V, masks = got
    return float((V * np.abs(e[0]))[masks[0]].sum())


def table(out, refs_dir):
    refs = lad.references(refs_dir)
    rows = [json.load(open(f)) for f in glob.glob(f"{out}/*.row.json")]
    base = {(r["series"], r["problem"], r["N"]): r for r in rows if r["f"] == "0"}
    lines = ["| series | problem | N | f | removed | same bits as f = 0 | reached | L1 rho f = 0 | L1 rho | ratio |", "|---|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for r in sorted(rows, key=lambda r: (r["f"], r["series"], r["problem"], r["N"])):
        if r["f"] == "0":
            continue
        b = base.get((r["series"], r["problem"], r["N"]))
        same = b is not None and b["hash"] == r["hash"]
        s = summary.setdefault(r["f"], dict(runs=0, triggered=0, same=0))
        s["runs"] += 1
        s["same"] += same
        s["triggered"] += r["removed"] > 0
        if r["removed"] == 0 and same:
            continue
        l0, l1 = l1_rho(b, refs) if b else None, l1_rho(r, refs)
        ratio = f"{l1 / l0:.3f}" if l0 and l1 else "-"
        lines.append(f"| {r['series']} | {r['problem']} | {r['N']} | {r['f']} | {r['removed']} | {same} | {r['reached']} | "
                     f"{'-' if l0 is None else f'{l0:.3e}'} | {'-' if l1 is None else f'{l1:.3e}'} | {ratio} |")
    lines += ["", "| f | runs | runs with a cell removed | same bits as f = 0 |", "|---|---|---|---|"]
    for f, s in sorted(summary.items()):
        lines.append(f"| {f} | {s['runs']} | {s['triggered']} | {s['same']} |")
    open(f"{out}/deref_sweep.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(*sys.argv[2:7])
    else:
        table(*sys.argv[2:4])
