#!/usr/bin/env python3
"""
End-to-end baseline: NAIVE sum-of-ops vs ground-truth device time.

For each (model, seq) at batch 1, sum every op's predicted STANDALONE latency
(compute GEMMs + non-compute ops, both kernel/fusion-level) and compare to the
torch.compile device-busy ground truth. This is the upper-bound baseline the
fusion-survival compensation (fit_compensation.py) then corrects.

Inputs (both persisted):
  calib.csv            : per (model,seq,batch) -> Sc_us (sum of GEMM fusion-time
                         predictions), Sn_us (sum of non-compute per-op predictions),
                         Nc/Nn (op counts), truth_us (device-busy ground truth).
  e2e_device_truth.csv : raw ground truth (model,seq,batch,...,device_ms).

Note on units: Sc is the GEMM linear model's FUSION/kernel time (tpu.py floor
~1.5us, NOT the ~14us whole-program span); Sn is the per-op models' kernel time
(fori_loop-subtracted). So the sum is kernel-level throughout -> the ~8.7x
over-count is genuine fusion ELIMINATION, not a program-vs-fusion units mismatch.
"""
import csv, os, statistics

HERE = os.path.dirname(__file__)
ORDER = {"gpt2": 0, "qwen2.5-0.5b": 1, "smollm2-135m": 2}


def main(batch=1):
    rows = [r for r in csv.DictReader(open(os.path.join(HERE, "calib.csv")))
            if int(r["batch"]) == batch]
    rows.sort(key=lambda r: (ORDER.get(r["model"], 9), int(r["seq"])))
    print(f"NAIVE sum-of-ops vs ground-truth device time (batch {batch})")
    print(f"{'model':14}{'seq':>5}{'compute':>9}{'noncomp':>9}{'SUM_us':>9}"
          f"{'truth_us':>9}{'ratio':>7}{'err%':>7}")
    errs, ratios, tc, tn = [], [], 0.0, 0.0
    for r in rows:
        Sc, Sn, tr = float(r["Sc_us"]), float(r["Sn_us"]), float(r["truth_us"])
        tot = Sc + Sn
        errs.append(abs(tot - tr) / tr * 100); ratios.append(tot / tr)
        tc += Sc; tn += Sn
        print(f"{r['model']:14}{int(r['seq']):>5}{Sc:>9.0f}{Sn:>9.0f}{tot:>9.0f}"
              f"{tr:>9.0f}{tot/tr:>6.1f}x{errs[-1]:>6.0f}%")
    print(f"\n  mean over-count = {statistics.mean(ratios):.1f}x   "
          f"mean abs err = {statistics.mean(errs):.0f}%")
    print(f"  summed total is compute {tc/(tc+tn)*100:.0f}% / non-compute "
          f"{tn/(tc+tn)*100:.0f}%  (non-compute dominates -> fused away in reality)")


if __name__ == "__main__":
    main()
