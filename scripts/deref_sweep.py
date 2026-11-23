"""Derefinement and refinement (luna-cpp #30) on the 1D ladder: do they trigger, and what they change where they do.

    python deref_sweep.py run <bin> <out dir> <configs,...> <N,...> [jobs]   # every ladder problem moving (and moving-wide), each config
    python deref_sweep.py table <out dir> <refs dir>                          # cells removed and added, same bits as off, L1 rho

A config is `off` (--amr=off) or parts joined by `+`: mass (--amr=mass, the default), off (--amr=off), t<x> (--steer=x), s<x> (--split-min=x), m<x> (--merge-max=x), d<f> (--derefine=f), r<F> (--refine=F, split
--split=constant), r<F>lin (--refine=F --split=linear); lengths without mass mean --amr=length. E.g. mass, mass+d3e-2, d3e-2,
r2, r2lin, r2+d3e-2. `off` is always run. Run with LUNA's environment. Each run: hydro_1d <p> <N> --moving [--pad=...] --steps=200 N
<config flags> --out=<out>/<series>_<p>_<N>_<config>.npy; the log line gives the hash and the cells removed and added. L1 rho as the
ladder's (ladder_1d.errors; linear waves: plain L1 at amplitude 1e-6).
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


def flags(config):
    """hydro_1d flags of a config (see the head of this file)."""
    if config == "off":
        return ["--amr=off"]
    parts = config.split("+")
    out = ["--amr=mass"] if "mass" in parts else ["--amr=off"] if "off" in parts else []
    for part in parts:
        if part in ("mass", "off"):
            continue
        if part.startswith("t"):    # steering: away from a neighbour cell narrower than x times one's own
            out.append(f"--steer={part[1:]}")
        elif part.startswith("s"):  # mass: split only cells wider than x h
            out.append(f"--split-min={part[1:]}")
        elif part.startswith("m"):  # mass: remove by mass only cells narrower than x h
            out.append(f"--merge-max={part[1:]}")
        elif part.startswith("d"):
            out.append(f"--derefine={part[1:]}")
        elif part.startswith("r"):
            linear = part.endswith("lin")
            out += [f"--refine={part[1:-3] if linear else part[1:]}", f"--split={'linear' if linear else 'constant'}"]
        else:
            raise ValueError(f"config part {part}")
    return out


def one(binary, out, s, p, n, config):
    name = f"{out}/{s}_{p}_{n}_{config}.npy"
    args = [binary, p, str(n), "--moving", f"--steps={200 * n}", "--repeats=1", f"--out={name}", *flags(config)]
    if s == "moving-wide":
        args.append(f"--pad={lad.pad(p)}")
    if p in lad.pr.LINEAR:
        args.append(f"--amp={lad.pr.EPS}")  # as the ladder's "+" run; L1 here is the plain error at that amplitude
    r = subprocess.run(args, capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="1"))
    line = next((x for x in r.stdout.splitlines() if x.startswith("ours")), "")
    get = lambda k: (re.search(rf" {k}=([^ ]+)", line) or [None, None])[1]  # noqa: E731
    row = dict(series=s, problem=p, N=n, config=config, hash=get("hash"), removed=int(get("removed") or 0), refined=int(get("refined") or 0),
               reached=get("reached") == "1", t=float(get("t") or "nan"), steps=int(get("steps") or 0), cells=int(get("final_cells") or 0),
               wall_s=float(get("wall_s") or "nan"), file=name)
    json.dump(row, open(name + ".row.json", "w"))
    return row


def run(binary, out, configs, ns, jobs=8):
    os.makedirs(out, exist_ok=True)
    riemann = lambda p: "WL" in lad.pr.problem(p)  # noqa: E731  moving-wide: two states to pad with
    tasks = [(s, p, int(n), c) for n in ns.split(",") for p in lad.ALL for s in SERIES for c in ["off", *configs.split(",")]
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
    base = {(r["series"], r["problem"], r["N"]): r for r in rows if r["config"] == "off"}
    lines = ["| series | problem | N | config | removed | added | same bits as off | reached | steps off | steps | L1 rho off | L1 rho | ratio |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    summary, every = {}, []
    for r in sorted(rows, key=lambda r: (r["config"], r["series"], r["problem"], r["N"])):
        b = base.get((r["series"], r["problem"], r["N"]))
        l0, l1 = (l1_rho(b, refs) if b else None), l1_rho(r, refs)
        every.append(dict(r, L1=l1, L1_off=l0))
        if r["config"] == "off":
            continue
        same = b is not None and b["hash"] == r["hash"]
        s = summary.setdefault(r["config"], dict(runs=0, triggered=0, same=0, reached=0, reached_off=0))
        s["runs"] += 1
        s["same"] += same
        s["triggered"] += r["removed"] + r["refined"] > 0
        s["reached"] += r["reached"]
        s["reached_off"] += bool(b and b["reached"])
        if r["removed"] + r["refined"] == 0 and same:
            continue
        ratio = f"{l1 / l0:.3f}" if l0 and l1 else "-"
        lines.append(f"| {r['series']} | {r['problem']} | {r['N']} | {r['config']} | {r['removed']} | {r['refined']} | {same} | {r['reached']} | "
                     f"{b['steps'] if b else '-'} | {r['steps']} | {'-' if l0 is None else f'{l0:.3e}'} | {'-' if l1 is None else f'{l1:.3e}'} | {ratio} |")
    lines += ["", "| config | runs | runs with a cell removed or added | same bits as off | reached t_end | off reached t_end |", "|---|---|---|---|---|---|"]
    for c, s in sorted(summary.items()):
        lines.append(f"| {c} | {s['runs']} | {s['triggered']} | {s['same']} | {s['reached']} | {s['reached_off']} |")
    open(f"{out}/deref_sweep.md", "w").write("\n".join(lines) + "\n")
    json.dump(every, open(f"{out}/deref_sweep.json", "w"), indent=0)
    print("\n".join(lines))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(*sys.argv[2:7])
    else:
        table(*sys.argv[2:4])
