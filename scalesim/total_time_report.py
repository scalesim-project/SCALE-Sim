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

# Per-op dispatch cost for the eager-execution wall-clock model (us/op),
# calibrated end-to-end (measured whole-model wall-clock vs SCALE-Sim device
# total, fit as measured = device + dispatch * n_ops). A fused/compiled
# deployment -> ~0.
#   TPUv4 : 19.75  (gpt2/qwen/smollm2; see SCALE-Sim_TPU/reports/C_integration.md)
#   TPUv6e:  5.712 (gpt2/qwen/smollm2 measured on this VM 2026-06-22; end-to-end
#                   MAPE 3.4% with this constant vs 176% reusing the v4 value;
#                   ground truth in topologies/stablehlo/llm/measured_tpu_v6e.json)
DISPATCH_US_PER_OP_BY_GEN = {
    "TPUv4": 19.75,
    "TPUv6e": 5.712,
}
DEFAULT_DISPATCH_US_PER_OP = 19.75


def dispatch_for_generation(generation):
    """Per-op dispatch cost (us/op) for a TPU generation; DEFAULT if unknown."""
    if not generation:
        return DEFAULT_DISPATCH_US_PER_OP
    return DISPATCH_US_PER_OP_BY_GEN.get(generation, DEFAULT_DISPATCH_US_PER_OP)


# Whole-model compensation for the naive sum-of-standalone-op latencies.
# Summing each op's STANDALONE latency over-counts whole-model device time ~8x:
# the standalone latencies carry per-op host cost the fused graph doesn't re-pay,
# and matmul (MXU) vs elementwise (VPU) work survive fusion to different degrees.
# Each op's contribution to real device time is therefore:
#     tuned_i = a_class * max(0, single_op_i - c_class)
# (a0/c_c for compute=MXU, a1/c_n for non-compute=VPU), and the whole-forward
# total adds one fixed host overhead C_forward (NOT attributable to any single op,
# so it appears only in the TOTAL row). Calibrated batch-1 against torch.compile
# device-busy ground truth (3 LLMs x 4 seq); see SCALE-Sim_TPU/e2e_work/
# compensation/ (coeffs.json, fit_compensation.py, measure_c.py): ~10.9% MAPE,
# leave-one-seq-out 17.3%, per-model own-fit ~1%.
#   C_forward is MEASURED, not fit: the per-execute device-busy floor of a single
#   compiled matmul is ~65 us, flat from 1x1x1 up to 768^3 and independent of
#   kernel count (a 128-matmul chain costs the same). The earlier free-fit value
#   (185 us) was inflated -- it absorbed model-specific error (qwen/smollm own-fit
#   C ran to 319/470 us, while gpt2's 81 us matched the measured floor). Pinning C
#   to the measured 65 us and refitting a0/a1 keeps accuracy and exposes a real
#   residual (qwen/smollm under-predicted at low seq -> a missing vocab/embedding
#   term, not a bigger constant). Host dispatch (~194 us/execute) is separate and
#   correctly excluded from device_ms.
#   c_c/c_n (per-op host cost) stay 0: not identifiable from whole-model totals;
#   measure as (python_timer - xprof_kernel) per op-type to pin them.
#   SCOPE: batch-1 composition. batch>1 parallelizes across the chip -- multiply
#   the TOTAL by the occupancy factor (fit_occupancy_model.py), not modeled here.
COMPENSATION_BY_GEN = {
    # Batch-1 whole-model model: T = a0*Sc + a1*Sn + C_forward.  a0 PINNED to 1.0
    # (GEMM passes through: single_op_us is the right magnitude, validated Sum(GEMM)<
    # truth).  a1 = non-compute fusion-survival factor.  C_forward = C0 (fixed; C1_per_
    # gemm = 0 -- a size-dependent C1*n_gemm term is collinear with Sn and hurts cross-
    # model generalization, so it is dropped).  C0=83.8us = the per-execute device floor
    # (~the loop fit's 81us, physical).
    #   PURE per-op models (scalesim/model/tpuv4 = xprof single-op spans, fixed inner-
    # span trace rule, no ~10us launch floor; 60/40-sampler n=1000 re-collection,
    # trained on the LLM-bucket subset).  RESHAPE is a constant = the median standalone
    # reshape latency (~7us): reshape latency is bimodal (metadata ~0 vs relayout
    # ~1000s us) and NOT predictable from shape, and the HGBR over-predicted it ~37x at
    # vocab sizes, dominating Sn (63%) and wrecking the fit -- the median constant fixes
    # that.  Calibrated on f32 StableHLO graphs, 3 LLMs x seq{128,256,512,1024} + a
    # tiny_transformer anchor (pins C0 down) vs v4 device-busy truth via calibration_
    # pure/fit_compensation_pure.py --gen tpuv4: in-sample 10.3% MAPE, tiny +5%.
    # Batch>1 occupancy NOT modelled.
    "TPUv4": {"a0_mxu": 1.0, "a1_vpu": 0.0491,
              "c_c": 0.0, "c_n": 0.0, "C0_forward": 83.8, "C1_per_gemm": 0.0},
    # v6e: same pure-model pipeline as v4 (PURE per-op models in model/tpuv6e,
    # re-collected with the fixed inner-span collector + n=1000 sampler; f32 calib_mlir
    # graphs; device-busy truth). ONE structural difference from v4: a0 is FREE, not
    # pinned to 1. v6e is fast enough that Sum(GEMM) (the standalone fusion floors of
    # GEMM-heavy models -- smollm2 has 272 GEMMs) EXCEEDS the fused device-busy truth
    # for 4/12 points, so a0=1 forces a1 negative; freeing a0 gives a0=0.73 (GEMM
    # contributes ~73% of its standalone-fusion sum once fused) with positive a1.
    # Calibrated batch-1 on 3 LLMs x seq{128,256,512,1024} + tiny_transformer via
    # fit_compensation_pure.py --gen tpuv6e --free-a0 --mape-min -> calib_tpuv6e_pure
    # .csv, coeffs_tpuv6e_pure.json. The fit MINIMIZES MAPE directly under physical
    # bounds (0<a0<1, a1>0, C0>=0) rather than WLS -- gives a physical fit AND better
    # accuracy: in-sample 12.1% MAPE, leave-one-model-out 17.7% (vs WLS 13.5%/22.6%).
    # a0=0.73 (GEMM fusion-floor overlap on fast v6e), a1=0.0098, C0=156us. (C1=0,
    # dropped as in v4.)  NOTE: v4 reshape=median is NOT yet applied to v6e's models;
    # re-run set_reshape_median.py on model/tpuv6e if porting. Batch>1 NOT modelled.
    "TPUv6e": {"a0_mxu": 0.7263, "a1_vpu": 0.0098,
               "c_c": 0.0, "c_n": 0.0, "C0_forward": 156.2, "C1_per_gemm": 0.0},
}


def compensation_for_generation(generation):
    """Two-factor (MXU/VPU) compensation coeffs for a TPU generation, or None if
    that generation is not yet calibrated (then tuned_us == single_op_us)."""
    return COMPENSATION_BY_GEN.get(generation)


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
