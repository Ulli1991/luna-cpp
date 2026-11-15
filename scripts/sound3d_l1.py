"""3D sound wave after one period: L1 of rho - 1 - amp sin(2 pi x) at the centroids, over amp (hydro_wave3d sound --out rows).

    python sound3d_l1.py <file.npy> <amp>
"""
import sys

import numpy as np

d = np.load(sys.argv[1])
amp = float(sys.argv[2])
x = (d[:, 3] % 1.0)
V = d[:, 6]
rho = d[:, 7]
exact = 1 + amp * np.sin(2 * np.pi * x)
print(f"{sys.argv[1]} sound L1(rho)/amp {np.sum(np.abs(rho - exact) * V) / np.sum(V) / amp:.4e}")
