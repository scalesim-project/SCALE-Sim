"""
Unified TIME_REPORT writer.

Consolidates the compute (GEMM/conv) per-layer times and the non-compute op
latencies into ONE report inside the run's inner directory:

    <run_dir>/TIME_REPORT.csv

columns: OpID, time_us, time_with_dispatch_us, layer, stablehlo
  OpID                 : global program-order index of the op
  time_us              : predicted device (compute-busy) time; blank if no model
  time_with_dispatch_us: time_us + per-op dispatch cost c  (eager wall-clock model)
  layer                : COMPUTE_REPORT LayerID for compute ops; 'N/A' otherwise
  stablehlo            : short 'op in->out dtype' signature (shapes only)
the final row is a TOTAL (sums of both time columns).

The compute per-op times come from the simulator/bypass (written first to
TIME_REPORT.csv as 'LayerID, Time (us)'); the non-compute times + program order
come from <stem>_op_table.json (written by the MLIR converter). This finalizer
joins them, rewrites TIME_REPORT.csv in the unified format, and removes the
now-redundant intermediate files.
"""
import csv
import glob
import json
import os

# Per-op dispatch cost for the eager-execution wall-clock model (us/op).
# Calibrated end-to-end on TPU v4 (gpt2/qwen/smollm2); see
# SCALE-Sim_TPU/reports/C_integration.md. A fused/compiled deployment -> ~0.
DEFAULT_DISPATCH_US_PER_OP = 19.75


def _read_compute_times(run_dir):
    """{layer_id: time_us} from the simulator/bypass TIME_REPORT.csv."""
    path = os.path.join(run_dir, "TIME_REPORT.csv")
    times = {}
    if not os.path.exists(path):
        return times
    for row in list(csv.reader(open(path)))[1:]:
        if len(row) >= 2 and row[0].strip().isdigit() and row[1].strip() not in ("", "N/A"):
            times[int(row[0])] = float(row[1])
    return times


def write_total_time_report(logpath, run_dir, dispatch_us_per_op=DEFAULT_DISPATCH_US_PER_OP):
    """Join op table + compute times into the single unified TIME_REPORT.csv.

    No-op (returns False) when no op table is present, e.g. a plain CSV-topology
    run, leaving the standard per-layer TIME_REPORT.csv untouched.
    """
    tables = glob.glob(os.path.join(logpath, "*_op_table.json"))
    if not tables:
        return False
    op_table = json.load(open(tables[0]))
    compute_times = _read_compute_times(run_dir)

    out = os.path.join(run_dir, "TIME_REPORT.csv")
    sum_t = sum_td = 0.0
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["OpID", "time_us", "time_with_dispatch_us", "layer", "stablehlo"])
        for r in sorted(op_table, key=lambda x: x["op_id"]):
            if r["kind"] == "compute":
                t = compute_times.get(r["layer"])
                layer = r["layer"]
            else:
                t = r["time_us"]
                layer = "N/A"
            # every op pays one dispatch, even unmodeled ones (counted in N_ops)
            td = (t if t is not None else 0.0) + dispatch_us_per_op
            if t is not None:
                sum_t += t
            sum_td += td
            w.writerow([r["op_id"], "" if t is None else f"{t:.6f}",
                        f"{td:.6f}", layer, r["stablehlo"]])
        w.writerow(["TOTAL", f"{sum_t:.6f}", f"{sum_td:.6f}", "", ""])

    # remove redundant intermediates now folded into the unified report
    for p in tables:
        os.remove(p)
    legacy = os.path.join(logpath, "NON_COMPUTE_TIME_REPORT.csv")
    if os.path.exists(legacy):
        os.remove(legacy)
    return True
