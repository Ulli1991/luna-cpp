"""Per kernel launch of hydro_derivatives_test --profile: time, registers, warps active, local (spill) and DRAM bytes per cell,
FP64 operations per cell, in launch order, policy by policy (the second round of each policy is the measured one).

    python3 scripts/derivatives_ncu_summary.py <cells> <kernels.csv>   (from scripts/derivatives_ncu_a100.sh)
"""
import csv
import sys
from collections import OrderedDict

KINDS = ("TangentSweepHook", "TangentPrimitives", "TangentSweep", "TangentFluxes", "TangentUpdate", "AdjointFaces", "AdjointCell", "AdjointGather",
         "Sweep", "TimeStep")


def kind(name):
    for k in KINDS:
        if k in name:
            return k
    if "apply" in name:
        return "fluxes" if "long long" in name else "update"
    if "primitives" in name:
        return "primitives"
    return name[:30]


def main(argv):
    cells = float(argv[1])
    launches = OrderedDict()
    with open(argv[2]) as f:
        lines = [l for l in f if l.startswith('"')]
    for r in csv.DictReader(lines):
        key = r["ID"]
        if key not in launches:
            policy = "grad16" if "Half" in r["Kernel Name"] else "double"
            launches[key] = {"kind": kind(r["Kernel Name"]), "policy": policy}
        try:
            launches[key][r["Metric Name"]] = float(r["Metric Value"].replace(",", ""))
        except ValueError:
            pass
    print(f"{'policy':7} {'kernel':18} {'ms':>8} {'regs':>5} {'warps%':>7} {'local B/c':>10} {'DRAM B/c':>9} {'GB/s':>7} {'FP64/c':>8}")
    for m in launches.values():
        t = m.get("gpu__time_duration.sum", 0) * 1e-9
        local = m.get("l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum", 0) + m.get("l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum", 0)
        dram = m.get("dram__bytes_read.sum", 0) + m.get("dram__bytes_write.sum", 0)
        fp = 2 * m.get("sm__sass_thread_inst_executed_op_dfma_pred_on.sum", 0) + m.get("sm__sass_thread_inst_executed_op_dadd_pred_on.sum", 0) \
            + m.get("sm__sass_thread_inst_executed_op_dmul_pred_on.sum", 0)
        print(f"{m['policy']:7} {m['kind']:18} {t * 1e3:8.3f} {m.get('launch__registers_per_thread', 0):5.0f} "
              f"{m.get('sm__warps_active.avg.pct_of_peak_sustained_active', 0):7.1f} {local / cells:10.0f} {dram / cells:9.0f} "
              f"{dram / t / 1e9 if t else 0:7.0f} {fp / cells:8.0f}")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
