#!/usr/bin/env python3
"""
v6e whole-model compensation fit (final v4 architecture: a0=1, SIZE-DEPENDENT C).

    T_device ~= Sum(GEMM_fusion) + a1*Sum(non-compute) + C0 + C1*(#GEMM kernels)

a0 PINNED to 1.0 (GEMM passthrough; Sum(GEMM) is the right magnitude, validated by
Sum(GEMM)<truth). We fit a1 (non-compute fusion survival), C0 (per-execute device
floor) and C1 (per-GEMM-kernel launch/drain) by weighted least squares (w=1/truth^2),
batch-1. The size-dependent C replaces a fixed C_forward, which over-predicts tiny
models (their floor scales with kernel count, not a constant).

Input : calib_tpuv6e.csv  (model,seq,batch,Sc_us,Sn_us,Nc,Nn,truth_us)
Output: coeffs_tpuv6e.json + the COMPENSATION_BY_GEN["TPUv6e"] line to paste.
"""
import csv, json, os, sys
import numpy as np

HERE = os.path.dirname(__file__)
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "calib_tpuv6e.csv")


def load(batch=1):
    rows = [r for r in csv.DictReader(open(DATA)) if int(r["batch"]) == batch]
    Sc = np.array([float(r["Sc_us"]) for r in rows])
    Sn = np.array([float(r["Sn_us"]) for r in rows])
    Nc = np.array([float(r["Nc"]) for r in rows])
    Y = np.array([float(r["truth_us"]) for r in rows])
    lab = [f"{r['model']}/{r['seq']}" for r in rows]
    mdl = [r["model"] for r in rows]
    return Sc, Sn, Nc, Y, lab, mdl


def mape(y, p):
    return float(np.mean(np.abs((y - p) / y)) * 100)


# a1 (non-compute fusion survival) is PINNED. Sn (non-compute) and Nc (#GEMM) both
# scale with model size, so a1 and C1 are confounded: the free 3-param solve is
# degenerate (drives a1 ~2x high and C1 negative). a1 is a generation-stable
# physical constant (~97% of non-compute fused away: v4=0.036; the robust v6e
# LLM-only fixed-C fit gave 0.0295), so we pin it and fit only the floor (C0, C1).
A1_PIN = 0.0295


def fit(Sc, Sn, Nc, Y, w):
    """a0=1 pinned. Free WLS for (a1, C0, C1) on r = Y - Sc ~= a1*Sn + C0 + C1*Nc.
    Physical constraints: a1 > 0, C1 >= 0. The free solve is used when it satisfies
    them (it does on the pure-device basis); if it degenerates (a1<=0 or C1<0, as on
    the loop-method basis where Sn and Nc are confounded), pin a1 = A1_PIN and refit
    (C0, C1>=0)."""
    r = Y - Sc
    X3 = np.stack([Sn, np.ones(len(Sn)), Nc], 1)
    a1, C0, C1 = np.linalg.solve(X3.T @ (X3 * w[:, None]), X3.T @ (w * r))
    if a1 > 0 and C1 >= 0:
        return a1, C0, C1
    a1 = A1_PIN                                       # degenerate -> pin a1, refit C0,C1>=0
    rp = r - a1 * Sn
    X2 = np.stack([np.ones(len(rp)), Nc], 1)
    C0, C1 = np.linalg.solve(X2.T @ (X2 * w[:, None]), X2.T @ (w * rp))
    if C1 < 0:
        C1 = 0.0; C0 = float(np.sum(w * rp) / np.sum(w))
    return a1, C0, C1


def predict(Sc, Sn, Nc, a1, C0, C1):
    return Sc + a1 * Sn + C0 + C1 * Nc


def main():
    Sc, Sn, Nc, Y, lab, mdl = load(batch=1)
    w = 1.0 / Y ** 2
    n = len(Y)
    print(f"v6e batch-1 calibration: {n} points ({len(set(mdl))} models)\n")

    viol = [lab[i] for i in range(n) if Sc[i] >= Y[i]]
    if viol:
        print(f"WARNING: Sum(GEMM) >= truth at: {viol}\n")

    a1, C0, C1 = fit(Sc, Sn, Nc, Y, w)
    pred = predict(Sc, Sn, Nc, a1, C0, C1)
    print(f"a0=1 (pinned)  a1={a1:.4f}  C0_forward={C0:.1f}us  C1_per_gemm={C1:.4f}us"
          f"   in-sample MAPE={mape(Y, pred):.1f}%")

    # leave-one-model-out CV
    if len(set(mdl)) > 1:
        pcv = np.zeros(n)
        for mo in sorted(set(mdl)):
            tr = np.array([i for i, m in enumerate(mdl) if m != mo])
            ts = np.array([i for i, m in enumerate(mdl) if m == mo])
            a, c0, c1 = fit(Sc[tr], Sn[tr], Nc[tr], Y[tr], w[tr])
            pcv[ts] = predict(Sc[ts], Sn[ts], Nc[ts], a, c0, c1)
        print(f"               leave-one-MODEL-out CV MAPE={mape(Y, pcv):.1f}%")

    print(f"\n{'point':18s}{'Nc':>4}{'Sc':>8}{'Sn':>9}{'pred':>8}{'truth':>8}{'err%':>7}")
    for i, l in enumerate(lab):
        print(f"{l:18s}{int(Nc[i]):>4}{Sc[i]:>8.0f}{Sn[i]:>9.0f}{pred[i]:>8.0f}"
              f"{Y[i]:>8.0f}{(pred[i]-Y[i])/Y[i]*100:>+6.0f}%")

    coeffs = {"model": "T = Sum(GEMM) + a1*Sum(noncompute) + C0 + C1*n_gemm  [batch-1]",
              "a0_mxu": 1.0, "a1_vpu": float(a1), "c_c": 0.0, "c_n": 0.0,
              "C0_forward": float(C0), "C1_per_gemm": float(C1),
              "in_sample_mape": mape(Y, pred), "n_points": n}
    json.dump(coeffs, open(os.path.join(HERE, "coeffs_tpuv6e.json"), "w"), indent=2)
    print(f'\nCOMPENSATION_BY_GEN["TPUv6e"] = {{"a0_mxu": 1.0, "a1_vpu": {a1:.4f}, '
          f'"c_c": 0.0, "c_n": 0.0, "C0_forward": {C0:.1f}, "C1_per_gemm": {C1:.4f}}}')


if __name__ == "__main__":
    main()
