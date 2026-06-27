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

from scalesim.latency_model.compensation import (  # the whole-model latency model
    compensation_for_generation, dispatch_for_generation)


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


def write_total_time_report(logpath, run_dir, dispatch_us_per_op=None, generation=None):
    """Join op table + compute times into the single unified TIME_REPORT.csv.

    Columns:
      OpID         : program-order index
      single_op_us : the op's STANDALONE predicted latency (compute via the GEMM
                     linear model; non-compute via the per-op models) -- what the
                     op costs in isolation.
      tuned_us     : the op's compensated contribution to fused whole-model device
                     time, a_class*max(0, single_op - c_class) (MXU coeffs for
                     compute, VPU for non-compute; see COMPENSATION_BY_GEN). The
                     forward-level host constant C_forward is added ONCE, in the
                     TOTAL row only (it is not attributable to a single op).
      layer        : COMPUTE_REPORT LayerID for compute ops, else 'N/A'.
      stablehlo    : short op signature.

    If `generation` has no calibrated compensation, tuned_us == single_op_us
    (identity) and no C_forward is added, so the column is still well-defined.
    NOTE: the compensation is calibrated at batch-1; for batch>1 scale the TOTAL
    by the occupancy factor (see SCALE-Sim_TPU/e2e_work/fit_occupancy_model.py).

    No-op (returns False) when no op table is present, e.g. a plain CSV-topology
    run, leaving the standard per-layer TIME_REPORT.csv untouched.
    """
    comp = compensation_for_generation(generation)
    tables = glob.glob(os.path.join(logpath, "*_op_table.json"))
    if not tables:
        return False
    op_table = json.load(open(tables[0]))
    compute_times = _read_compute_times(run_dir)

    def tune(single, kind):
        """Per-op compensated contribution (excludes the once-per-forward C)."""
        if single is None:
            return 0.0
        if comp is None:
            return single
        a, c = ((comp["a0_mxu"], comp["c_c"]) if kind == "compute"
                else (comp["a1_vpu"], comp["c_n"]))
        return a * max(0.0, single - c)

    out = os.path.join(run_dir, "TIME_REPORT.csv")
    sum_single = sum_tuned = 0.0
    n_gemm = 0
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["OpID", "single_op_us", "tuned_us", "layer", "stablehlo"])
        for r in sorted(op_table, key=lambda x: x["op_id"]):
            if r["kind"] == "compute":
                t = compute_times.get(r["layer"])
                layer = r["layer"]
                n_gemm += 1
            else:
                t = r["time_us"]
                layer = "N/A"
            tuned = tune(t, r["kind"])
            if t is not None:
                sum_single += t
            sum_tuned += tuned
            w.writerow([r["op_id"], "" if t is None else f"{t:.6f}",
                        f"{tuned:.6f}", layer, r["stablehlo"]])
        # once-per-forward overhead, SIZE-DEPENDENT: C0 + C1*(#GEMM kernels).
        # (A fixed C over-predicts tiny models; this scales the floor with the
        #  number of device kernel launches. Falls back to a fixed C_forward if a
        #  generation still carries the old constant form.)
        if comp and "C0_forward" in comp:
            c_forward = comp["C0_forward"] + comp.get("C1_per_gemm", 0.0) * n_gemm
        else:
            c_forward = comp["C_forward"] if comp else 0.0
        sum_tuned += c_forward
        w.writerow(["TOTAL", f"{sum_single:.6f}", f"{sum_tuned:.6f}",
                    f"C_forward={c_forward:.3f}(n_gemm={n_gemm})", ""])

    # remove redundant intermediates now folded into the unified report
    for p in tables:
        os.remove(p)
    legacy = os.path.join(logpath, "NON_COMPUTE_TIME_REPORT.csv")
    if os.path.exists(legacy):
        os.remove(legacy)
    return True
