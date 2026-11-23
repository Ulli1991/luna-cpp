"""A per-kernel table from ncu --csv --page raw output: the median over a kernel's launches of each metric, and per cell.

    python ncu_table.py <cells> <file.csv> [...]
"""
import csv
import statistics
import sys
from collections import defaultdict

cells = float(sys.argv[1])
SHOW = [
    ("ms", "gpu__time_duration.sum", 1e-6),
    ("regs", "launch__registers_per_thread", 1),
    ("block", "launch__block_size", 1),
    ("grid", "launch__grid_size", 1),
    ("warps_active%", "sm__warps_active.avg.pct_of_peak_sustained_active", 1),
    ("theor_warps%", "sm__maximum_warps_per_active_cycle_pct", 1),
    ("dram%", "dram__throughput.avg.pct_of_peak_sustained_elapsed", 1),
    ("fp64_pipe%", "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active", 1),
    ("stall_long_sb%", "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct", 1),
    ("stall_math%", "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct", 1),
    ("stall_wait%", "smsp__warp_issue_stalled_wait_per_warp_active.pct", 1),
    ("stall_barrier%", "smsp__warp_issue_stalled_barrier_per_warp_active.pct", 1),
    ("stall_branch%", "smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct", 1),
    ("issue%", "smsp__issue_active.avg.pct_of_peak_sustained_active", 1),
    ("clock_MHz", "sm__cycles_elapsed.avg.per_second", 1e-6),
]
PER_CELL = [
    ("ns/cell", "gpu__time_duration.sum", 1.0),
    ("DRAM_B/cell", ("dram__bytes_read.sum", "dram__bytes_write.sum"), 1.0),
    ("L2_B/cell", "lts__t_bytes.sum", 1.0),
    ("local_B/cell", ("l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum", "l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum"), 1.0),
    ("fp64_inst/cell", ("smsp__sass_thread_inst_executed_op_dfma_pred_on.sum", "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum",
                        "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum"), 1.0),
    ("warp_inst/cell", "smsp__inst_executed.sum", 32.0),
    ("branch_warp_inst/cell", "smsp__sass_inst_executed_op_branch.sum", 1.0),
    ("divergent_targets/cell", "smsp__sass_branch_targets_threads_divergent.sum", 1.0),
]


def number(text):
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return float("nan")


for path in sys.argv[2:]:
    rows = list(csv.reader(line for line in open(path) if line.startswith("\"")))
    head = rows[0]
    units = rows[1]
    data = defaultdict(lambda: defaultdict(list))
    for r in rows[2:]:
        if len(r) != len(head):
            continue
        name = r[head.index("Kernel Name")][:70]
        for i, h in enumerate(head):
            v = number(r[i])
            if units[i] in ("msecond",) and h == "gpu__time_duration.sum":
                v *= 1e6
            if units[i] in ("usecond",) and h == "gpu__time_duration.sum":
                v *= 1e3
            if units[i] == "Kbyte":
                v *= 1e3
            if units[i] == "Mbyte":
                v *= 1e6
            if units[i] == "Gbyte":
                v *= 1e9
            if units[i] == "Ghz" or units[i] == "cycle/nsecond":
                v *= 1e9
            if units[i] == "Mhz" or units[i] == "cycle/usecond":
                v *= 1e6
            data[name][h].append(v)
    print(f"== {path}")
    for name, m in data.items():
        med = {h: statistics.median(v) for h, v in m.items() if v}
        parts = [f"{label}={med.get(key, float('nan')) * s:.4g}" for label, key, s in SHOW]
        for label, keys, s in PER_CELL:
            keys = keys if isinstance(keys, tuple) else (keys,)
            total = sum(med.get(k, float("nan")) for k in keys)
            parts.append(f"{label}={total * s / cells:.4g}")
        print(f"{name} launches={len(next(iter(m.values())))}\n   " + " ".join(parts))
