"""Two states (float64 .npy of the same shape) value by value: how many differ, and the largest difference over the largest magnitude.

    python grid_compare.py <a.npy> <b.npy> [label]
"""
import sys

import numpy as np

a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
label = sys.argv[3] if len(sys.argv) > 3 else ""
same = (a == b) | (np.isnan(a) & np.isnan(b))
scale = np.nanmax(np.abs(b))
print(f"compare {label}: {int((~same).sum())} of {a.size} values differ, largest {float(np.nanmax(np.abs(a - b)) / scale):.3e} of max|U|; "
      f"per variable {[float(np.nanmax(np.abs(a[v] - b[v])) / np.nanmax(np.abs(b[v]))) for v in range(a.shape[0])]}")
