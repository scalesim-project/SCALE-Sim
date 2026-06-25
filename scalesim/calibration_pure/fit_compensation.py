#!/usr/bin/env python3
"""
Two-factor (MXU/VPU) whole-model compensation for the naive sum-of-ops.

Problem: summing each op's STANDALONE latency over-counts whole-model device time
by ~8x. Two physical causes (not just fusion):
  1. each standalone op latency carries a per-op host/CPU cost (python-timer, not
     xprof kernel time) that the fused graph does NOT re-pay per op  -> remove it;
  2. MXU (matmul/compute) and VPU (elementwise/non-compute) survive fusion to
     different degrees -> two factors, not one global R.

Model (per-op terms + one forward-level host constant):

    T = a0*(Sc - c_c*Nc) + a1*(Sn - c_n*Nn) + C

  Sc, Sn = sum of standalone compute / non-compute op latencies (us)
  Nc, Nn = compute / non-compute op counts
  c_c, c_n = per-op host cost baked into the standalone measurement (us/op, >=0)
             ideally MEASURED as python_timer - xprof_kernel; here fit with c>=0.
  a0 (MXU), a1 (VPU) = fusion-survival factors (different units -> different a)
  C = once-per-forward host overhead (us)

Per-op decomposition for the report: tuned_i = a_class*(x_i - c_class); C lives
only in the TOTAL row (not attributable to a single op).

Scope: batch-1 (single-array composition regime). batch>1 parallelism across the
chip is a separate total-level occupancy factor (see fit_occupancy_model.py).

Usage: python3 fit_compensation.py   (writes coeffs.json)
"""
import csv, json, os
import numpy as np

HERE = os.path.dirname(__file__)


def load(batch=1):
    rows = [r for r in csv.DictReader(open(os.path.join(HERE, "calib.csv")))
            if int(r["batch"]) == batch]
    Sc = np.array([float(r["Sc_us"]) for r in rows])
    Sn = np.array([float(r["Sn_us"]) for r in rows])
    Nc = np.array([float(r["Nc"]) for r in rows])
    Nn = np.array([float(r["Nn"]) for r in rows])
    Y = np.array([float(r["truth_us"]) for r in rows])
    lab = [f"{r['model']}/{r['seq']}" for r in rows]
    mdl = [r["model"] for r in rows]
    return Sc, Sn, Nc, Nn, Y, lab, mdl


def wls(X, y, w):
    WX = X * w[:, None]
    return np.linalg.solve(X.T @ WX, X.T @ (w * y))


def mape(y, p):
    return float(np.mean(np.abs((y - p) / y)) * 100)


def feats(Sc, Sn, Nc, Nn, cc, cn):
    """Feature matrix for given per-op host costs (cc, cn): [Sc-cc*Nc, Sn-cn*Nn, 1]."""
    return np.stack([Sc - cc * Nc, Sn - cn * Nn, np.ones(len(Sc))], 1)


def fit_full(Sc, Sn, Nc, Nn, Y, w):
    """Grid c_c, c_n >= 0; fit (a0,a1,C) linearly inside; pick by in-sample MAPE."""
    best = None
    for cc in np.linspace(0, 3.0, 31):
        for cn in np.linspace(0, 1.5, 31):
            X = feats(Sc, Sn, Nc, Nn, cc, cn)
            try:
                a0, a1, C = wls(X, Y, w)
            except np.linalg.LinAlgError:
                continue
            if a0 <= 0 or a1 <= 0:          # survival factors must be positive
                continue
            m = mape(Y, X @ [a0, a1, C])
            if best is None or m < best[0]:
                best = (m, cc, cn, a0, a1, C)
    return best


def lomo_cv(Sc, Sn, Nc, Nn, Y, mdl, cc, cn):
    """Leave-one-model-out CV at fixed (cc,cn): honest generalization to an unseen model."""
    w = 1.0 / Y ** 2
    preds = np.zeros(len(Y))
    for mo in sorted(set(mdl)):
        tr = np.array([i for i, m in enumerate(mdl) if m != mo])
        ts = np.array([i for i, m in enumerate(mdl) if m == mo])
        X = feats(Sc, Sn, Nc, Nn, cc, cn)
        a0, a1, C = wls(X[tr], Y[tr], w[tr])
        preds[ts] = X[ts] @ [a0, a1, C]
    return mape(Y, preds), preds


def main():
    Sc, Sn, Nc, Nn, Y, lab, mdl = load(batch=1)
    w = 1.0 / Y ** 2
    print(f"batch-1 calibration: {len(Y)} points ({len(set(mdl))} models x 4 seqs)\n")

    # baselines
    Xr = np.stack([Sc + Sn, np.ones(len(Sc))], 1)
    r, Cr = wls(Xr, Y, w)
    print(f"baseline single-R  : R={r:.4f} C={Cr:.0f}us   in-sample MAPE={mape(Y, Xr@[r,Cr]):.1f}%")

    m, cc, cn, a0, a1, C = fit_full(Sc, Sn, Nc, Nn, Y, w)
    print(f"two-factor (c>=0)  : a0(MXU)={a0:.3f} a1(VPU)={a1:.4f} "
          f"c_c={cc:.3f} c_n={cn:.3f} us/op  C={C:.0f}us")
    print(f"                     in-sample MAPE={m:.1f}%")
    cv, _ = lomo_cv(Sc, Sn, Nc, Nn, Y, mdl, cc, cn)
    print(f"                     leave-one-MODEL-out CV MAPE={cv:.1f}%")
    # leave-one-seq-out (interpolation across seq, models seen)
    seqs = [l.split("/")[1] for l in lab]
    Xc = feats(Sc, Sn, Nc, Nn, cc, cn); pso = np.zeros(len(Y))
    for so in sorted(set(seqs)):
        tr = np.array([i for i, s in enumerate(seqs) if s != so])
        ts = np.array([i for i, s in enumerate(seqs) if s == so])
        pso[ts] = Xc[ts] @ wls(Xc[tr], Y[tr], w[tr])
    cv_seq = mape(Y, pso)
    print(f"                     leave-one-SEQ-out   CV MAPE={cv_seq:.1f}%\n")

    print(f"{'point':16s}{'pred':>8}{'truth':>8}{'err%':>7}")
    X = feats(Sc, Sn, Nc, Nn, cc, cn)
    p = X @ [a0, a1, C]
    for l, pp, yy in zip(lab, p, Y):
        print(f"{l:16s}{pp:>8.0f}{yy:>8.0f}{abs(pp-yy)/yy*100:>6.0f}%")

    coeffs = {
        "model": "T = a0*(Sc - c_c*Nc) + a1*(Sn - c_n*Nn) + C   [batch-1 composition]",
        "a0_mxu": a0, "a1_vpu": a1, "c_c_us_per_op": cc, "c_n_us_per_op": cn,
        "C_forward_us": C, "in_sample_mape": m, "lomo_cv_mape": cv,
        "loso_cv_mape": cv_seq, "n_points": len(Y),
        "scope": "batch-1 composition; batch>1 needs the occupancy total-factor.",
        "note": "c_c/c_n are fit (c>=0) and come out 0 -> per-op host cost is NOT "
        "identifiable from whole-model totals; MEASURE it as python_timer-minus-"
        "xprof_kernel per op-type, then it pulls op-count host cost out of C and "
        "should improve cross-model generalization (per-model own-fit is ~1%).",
    }
    out = os.path.join(HERE, "coeffs.json")
    json.dump(coeffs, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
