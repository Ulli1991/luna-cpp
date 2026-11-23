"""Per-kernel speed-of-light table of the moving step's hydro kernels from an Nsight Compute raw CSV (--page raw --csv).

    python scripts/moving_sol_table.py <raw.csv> <side>

Per kernel: ns/cell, registers, block size, theoretical and achieved occupancy, eligible warps per scheduler, issue slots
busy, the top three stall reasons (share of warp-cycles per issued instruction), branch instructions per cell and the
divergent branch targets per cell, threads per instruction, local memory (spill) sectors per cell, DRAM and L2 bytes per
cell, FP64 pipe busy. Floors: DRAM at 1.37 TB/s; FP64 as the time the FP64 pipe is busy (ns x pipe %).
Values carry units in the CSV's second row (byte, Kbyte, Mbyte, Gbyte; nsecond, usecond, msecond) and are scaled.
"""
import csv
import sys

rows = list(csv.reader(open(sys.argv[1])))
head, units, data = rows[0], rows[1], rows[2:]
side = int(sys.argv[2])
cells = side ** 3
col = {name: i for i, name in enumerate(head)}
SCALE = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "Tbyte": 1e12, "nsecond": 1e-9, "usecond": 1e-6,
         "msecond": 1e-3, "second": 1.0, "ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0, "sector": 1, "Ksector": 1e3, "Msector": 1e6, "Gsector": 1e9, "inst": 1,
         "Kinst": 1e3, "Minst": 1e6, "Ginst": 1e9}


def num(r, name, default=0.0):
    i = col.get(name)
    if i is None or r[i] in ("", "n/a"):
        return default
    return float(r[i].replace(",", "")) * SCALE.get(units[i], 1)


def kind(name):
    if "Sweep<" in name:
        return "sweep"
    if "TimeStep<" in name:
        return "time_step"
    if "step_moving" in name and "MovingFlux" not in name:
        return "keep_centroids"
    if "::move(" in name:
        return "drift"
    if "primitives" in name:
        return "primitives"
    p = "2" if "MovingFlux<1>" in name else "1" if "MovingFlux<0>" in name else "?"
    return ("fluxes" if "long long" in name else "update") + p


stalls = [n for n in head if n.startswith("smsp__average_warps_issue_stalled_") and n.endswith("_per_issue_active.ratio")]
print(f"cells {cells}")
print("| kernel | ns/cell | regs | block | occ. theor./achieved % | eligible warps/sched | issue % | top stalls (% of stall) | branch inst/cell | divergent targets/cell | threads/inst | local sectors/cell | DRAM B/cell r+w | L2 B/cell | FP64 pipe % | floor DRAM ns | floor FP64 ns |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
total = 0.0
for r in data:
    name = r[col["Kernel Name"]]
    ns_cell = num(r, "gpu__time_duration.sum") * 1e9 / cells
    total += ns_cell
    s = sorted(((num(r, n), n) for n in stalls), reverse=True)
    ssum = sum(v for v, _ in s) or 1
    top = ", ".join(f"{n.split('stalled_')[1].replace('_per_issue_active.ratio', '')} {100 * v / ssum:.0f}" for v, n in s[:3])
    dram_r, dram_w = num(r, "dram__bytes_read.sum"), num(r, "dram__bytes_write.sum")
    local = num(r, "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum") + num(r, "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum")
    fp64 = num(r, "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active")
    print(f"| {kind(name)} | {ns_cell:.3f} | {num(r, 'launch__registers_per_thread'):.0f} | {num(r, 'launch__block_size'):.0f}"
          f" | {num(r, 'sm__maximum_warps_per_active_cycle_pct'):.0f}/{num(r, 'sm__warps_active.avg.pct_of_peak_sustained_active'):.0f}"
          f" | {num(r, 'smsp__warps_eligible.avg.per_cycle_active'):.2f} | {num(r, 'smsp__issue_active.avg.pct_of_peak_sustained_active'):.0f}"
          f" | {top} | {num(r, 'smsp__inst_executed_op_branch.sum') * 32 / cells:.0f}"
          f" | {num(r, 'smsp__sass_branch_targets_threads_divergent.sum') / cells:.1f}"
          f" | {num(r, 'smsp__thread_inst_executed_per_inst_executed.ratio'):.1f} | {local / cells:.0f}"
          f" | {dram_r / cells:.0f}+{dram_w / cells:.0f} | {num(r, 'lts__t_sectors.sum') * 32 / cells:.0f} | {fp64:.0f}"
          f" | {(dram_r + dram_w) / cells / 1370:.3f} | {ns_cell * fp64 / 100:.3f} |")
print(f"sum of the profiled kernels: {total:.3f} ns/cell")
