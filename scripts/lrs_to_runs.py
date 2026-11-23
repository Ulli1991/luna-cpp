"""luna-reference-solutions final states of the tube v2 as a runs directory that super_tube_v2.py score/figures and make_figures.py read.

    python lrs_to_runs.py <dir of lrs npz> <runs dir>

LUNA (luna/super_tube_v2/super_tube_v2_luna_<N>_t<t>.npz) -> luna_super-tube-v2_<N>_t<t>.npz (rho, u, p on uniform cells) and its .json;
luna-cpp static (luna-cpp/super_tube_v2/super_tube_v2[_m0]_static_<N>_t<t>.npz) -> static_super-tube-v2[-m0]_<N>_t<t>.npy, rows
(x, centroid, volume, rho, u, p) as hydro_1d --out writes them, and its .json. The json: t, reached, steps, wall_s from the npz keys.
"""
import glob
import json
import os
import re
import sys

import numpy as np


def main(src, out):
    os.makedirs(out, exist_ok=True)
    for f in sorted(glob.glob(f"{src}/*.npz")):
        m = re.match(r"super_tube_v2(_m0)?_(luna|static)_(\d+)_t([0-9.]+)\.npz", os.path.basename(f))
        if not m:
            continue
        m0, series, n, t = m.groups()
        d = np.load(f)
        meta = dict(series=series, N=int(n), t=float(d["t"]), reached=int(d["reached"]), steps=int(d["steps"]), wall_s=float(d["wall_s"]))
        problem = "super-tube-v2" + ("-m0" if m0 else "")
        if series == "luna":
            name = f"{out}/luna_{problem}_{n}_t{t}.npz"
            np.savez(name, rho=d["rho"], u=d["u"], p=d["p"], t=d["t"], steps=d["steps"], wall_s=d["wall_s"])
        else:
            e = np.linspace(float(d["x0"]), float(d["x0"]) + float(d["lx"]), d["rho"].size + 1)
            c, V = 0.5 * (e[:-1] + e[1:]), np.diff(e)
            name = f"{out}/static_{problem}_{n}_t{t}.npy"
            np.save(name, np.stack([c, c, V, d["rho"], d["u"], d["p"]], axis=1))
        json.dump(meta, open(name + ".json", "w"))
        print(name, meta)


if __name__ == "__main__":
    main(*sys.argv[1:3])
