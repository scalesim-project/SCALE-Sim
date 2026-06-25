#!/usr/bin/env python3
"""
v6e whole-model compensation fit (final v4 architecture: pin a0=1, fit a1, C).

    T_device ~= a0*Sum(GEMM_fusion) + a1*Sum(non-compute) + C_forward,  a0 = 1.0

a0 is PINNED to 1.0 (GEMM passthrough): Sum(GEMM) is already the right magnitude
(incompressibility bound -> Sum(GEMM) < truth), and freeing a0 was unstable on v4
(low LOMO). We fit a1 (non-compute fusion-survival) and C_forward (once-per-forward
host/launch overhead) by weighted least squares (w = 1/truth^2), batch-1.

Input : calib_tpuv6e.csv  (model,seq,batch,Sc_us,Sn_us,Nc,Nn,truth_us)
Output: coeffs_tpuv6e.json + printed table; reports in-sample + leave-one-model-out.
"""
import csv, json, os, sys
import numpy as np

HERE = os.path.dirname(__file__)
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "calib_tpuv6e.csv")


def load(batch=1):
    rows = [r for r in csv.DictReader(open(DATA)) if int(r["batch"]) == batch]
    Sc = np.array([float(r["Sc_us"]) for r in rows])
    Sn = np.array([float(r["Sn_us"]) for r in rows])
    Y = np.array([float(r["truth_us"]) for r in rows])
    lab = [f"{r['model']}/{r['seq']}" for r in rows]
    mdl = [r["model"] for r in rows]
    return Sc, Sn, Y, lab, mdl


def mape(y, p):
    return float(np.mean(np.abs((y - p) / y)) * 100)


def wls(X, y, w):
    WX = X * w[:, None]
    return np.linalg.solve(X.T @ WX, X.T @ (w * y))


def main():
    Sc, Sn, Y, lab, mdl = load(batch=1)
    w = 1.0 / Y ** 2
    n = len(Y)
    print(f"v6e batch-1 calibration: {n} points ({len(set(mdl))} models)\n")

    # incompressibility guardrail: Sum(GEMM) must be < truth everywhere
    viol = [(l, Sc[i], Y[i]) for i, l in enumerate(lab) if Sc[i] >= Y[i]]
    if viol:
        print("WARNING: Sum(GEMM) >= truth (compute incompressibility violated):")
        for l, sc, y in viol:
            print(f"    {l}: Sc={sc:.0f} >= truth={y:.0f}")
        print()

    # pin a0=1, fit a1 + C
    r = Y - Sc
    a1, C = wls(np.stack([Sn, np.ones(n)], 1), r, w)
    pred = Sc + a1 * Sn + C
    print(f"pinned a0=1 : a1(VPU)={a1:.4f}  C_forward={C:.0f}us   in-sample MAPE={mape(Y, pred):.1f}%")

    # leave-one-model-out CV
    if len(set(mdl)) > 1:
        pcv = np.zeros(n)
        for mo in sorted(set(mdl)):
            tr = np.array([i for i, m in enumerate(mdl) if m != mo])
            ts = np.array([i for i, m in enumerate(mdl) if m == mo])
            a1t, Ct = wls(np.stack([Sn[tr], np.ones(len(tr))], 1), (Y - Sc)[tr], w[tr])
            pcv[ts] = Sc[ts] + a1t * Sn[ts] + Ct
        print(f"            leave-one-MODEL-out CV MAPE={mape(Y, pcv):.1f}%")

    print(f"\n{'point':16s}{'Sc':>8}{'Sn':>9}{'pred':>8}{'truth':>8}{'err%':>7}")
    for i, l in enumerate(lab):
        print(f"{l:16s}{Sc[i]:>8.0f}{Sn[i]:>9.0f}{pred[i]:>8.0f}{Y[i]:>8.0f}{abs(pred[i]-Y[i])/Y[i]*100:>6.0f}%")

    coeffs = {"model": "T = 1.0*Sum(GEMM) + a1*Sum(noncompute) + C_forward  [batch-1]",
              "a0_mxu": 1.0, "a1_vpu": float(a1), "c_c": 0.0, "c_n": 0.0,
              "C_forward": float(C), "in_sample_mape": mape(Y, pred), "n_points": n}
    out = os.path.join(HERE, "coeffs_tpuv6e.json")
    json.dump(coeffs, open(out, "w"), indent=2)
    print(f"\nwrote {out}")
    print(f'COMPENSATION_BY_GEN["TPUv6e"] = {{"a0_mxu": 1.0, "a1_vpu": {a1:.4f}, '
          f'"c_c": 0.0, "c_n": 0.0, "C_forward": {C:.1f}}}')


if __name__ == "__main__":
    main()
