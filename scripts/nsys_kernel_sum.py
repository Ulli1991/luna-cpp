"""Kernel time per step from `nsys stats -r cuda_gpu_kern_sum -f csv`: the sum over all kernels, divided by the steps run.

    python nsys_kernel_sum.py <kern_sum.csv> <steps> <cells> [label]
"""
import csv
import sys

path, steps, cells = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
label = sys.argv[4] if len(sys.argv) > 4 else ""
rows = [r for r in csv.DictReader(line for line in open(path) if line.strip() and not line.startswith("NOTICE"))]
total_ns = sum(float(r["Total Time (ns)"]) for r in rows)
ms = total_ns / steps / 1e6
top = sorted(rows, key=lambda r: -float(r["Total Time (ns)"]))[:6]
print(f"kernels {label}: {ms:.4f} ms/step = {ms * 1e6 / cells:.4f} ns/cell = {cells / (ms * 1e-3) / 1e9:.4f} G (kernel time only, {steps} steps); "
      + "; ".join(f"{r['Name'][:40]} {float(r['Total Time (ns)']) / steps / 1e6:.3f} ms x{float(r['Instances']) / steps:.1f}" for r in top))
