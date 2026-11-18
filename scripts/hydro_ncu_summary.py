"""Per hydro kernel: time, registers, warps active, local-memory and DRAM bytes per cell, from ncu --csv files.

    python3 scripts/hydro_ncu_summary.py <cells> <file.csv> ...   (the csv files of scripts/hydro_ncu_local_a100.sh)
"""
import csv
import re
import sys
from collections import defaultdict


def short(name):
    for key, label in (("MovingFlux<1>", "pass 2"), ("MovingFlux<0>", "pass 1"), ("MovingFlux<true>", "pass 2"),
                       ("MovingFlux<false>", "pass 1"), ("FaceFlux", "static"), ("Sweep", "Sweep"), ("TimeStep", "TimeStep"),
                       ("FaceStep", "FaceStep"), ("CellStep", "CellStep")):
        if key in name:
            kind = label
            break
    else:
        kind = name[:40]
    if kind in ("pass 1", "pass 2", "static"):
        kind += " fluxes" if "long long" in name else " update"
    return kind


def main(argv):
    cells = float(argv[1])
    for path in argv[2:]:
        rows = defaultdict(lambda: defaultdict(list))
        with open(path) as f:
            lines = [l for l in f if l.startswith('"')]
        for r in csv.DictReader(lines):
            key = (short(r["Kernel Name"]), r["ID"])
            rows[key][r["Metric Name"]].append(r["Metric Value"].replace(",", ""))
        agg = defaultdict(lambda: defaultdict(float))
        count = defaultdict(int)
        for (kind, _), m in rows.items():
            count[kind] += 1
            for k, v in m.items():
                try:
                    agg[kind][k] += float(v[0])
                except ValueError:
                    pass
        print(f"## {path}")
        print("kernel | launches | ms each | regs | warps % | local ld+st B/cell | DRAM B/cell")
        for kind in sorted(agg):
            a, c = agg[kind], count[kind]
            t = a["gpu__time_duration.sum"] / c
            t_ms = t / 1e6  # ns (--print-units base)
            local = (a["l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum"] + a["l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum"]) / c / cells
            dram = (a["dram__bytes_read.sum"] + a["dram__bytes_write.sum"]) / c / cells
            print(f"{kind} | {c} | {t_ms:.3f} | {a['launch__registers_per_thread'] / c:.0f} | "
                  f"{a['sm__warps_active.avg.pct_of_peak_sustained_active'] / c:.1f} | {local:.0f} | {dram:.0f}")


if __name__ == "__main__":
    main(sys.argv)
