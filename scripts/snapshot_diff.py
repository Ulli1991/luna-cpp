"""How two hydro_sedov runs differ when their bits do: the last snapshot of each run dir, cell by cell.

    python3 scripts/snapshot_diff.py <run dir A> <run dir B> [<label>]

Prints, per conserved variable (rho, rho v_x, rho v_y, rho v_z, E per volume): the largest |a - b| / max(|a|, |b|) over
cells, and the volume-weighted L1 of the density difference relative to the mean density. Same cell order in both runs
(same mesh), as for a change that keeps the mesh and moves the hydro's last bits. Needs numpy and h5py.
"""
import glob
import sys

import h5py
import numpy as np


def last(run):
    path = sorted(glob.glob(f"{run}/snapdir_*/*.hdf5"))[-1]
    with h5py.File(path, "r") as f:
        return np.array(f["/cells/U"]), np.array(f["/cells/volume"]), np.array(f["/cells/x"])


def main(argv):
    a, b = argv[1], argv[2]
    label = argv[3] if len(argv) > 3 else f"{a} vs {b}"
    ua, va, xa = last(a)
    ub, vb, xb = last(b)
    if ua.shape != ub.shape or not np.array_equal(xa, xb):
        print(f"diff: {label} meshes differ (positions or shape); no cell-by-cell comparison")
        return 1
    if ua.shape[0] != 5:
        ua, ub = ua.T, ub.T
    names = ["rho", "mx", "my", "mz", "E"]
    parts = []
    for k, name in enumerate(names):
        scale = np.maximum(np.abs(ua[k]), np.abs(ub[k]))
        rel = np.where(scale > 0, np.abs(ua[k] - ub[k]) / np.where(scale > 0, scale, 1), 0)
        parts.append(f"{name}={rel.max():.3e}")
    l1 = np.sum(va * np.abs(ua[0] - ub[0])) / np.sum(va * ua[0])
    same = np.array_equal(ua, ub)
    print(f"diff: {label} max_relative " + " ".join(parts) + f" l1_rho_relative={l1:.3e} cells_differing={int(np.sum(np.any(ua != ub, axis=0)))}"
          + f" of {ua.shape[1]} identical={same}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
