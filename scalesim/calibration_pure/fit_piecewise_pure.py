#!/usr/bin/env python3
"""
Fit the GEMM linear model on GOLDEN pure-device latency (xprof kernel time from
gemm_pure_master.csv, column latency_us_device). Compares:
  - single global G_roof              A*max(cyc, bytes/BW)+B
  - piecewise G_roof, 8 regions       region = (foldK>1)*4+(foldN>=16)*2+(foldM>=16)
on an 80/20 held-out split (relative-error weighted, robust-trimmed BW grid).
Reports held-out MAPE = accuracy of predicting the golden pure device latency,
and dumps the 8-region table to paste into tpu.py.
"""
import csv, math, os, sys
import numpy as np

HERE = os.path.dirname(__file__)
S = 128
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "gemm_pure_master.csv")


def load():
    M, N, K, cyc, lat = [], [], [], [], []
    for r in csv.DictReader(open(DATA)):
        if r["status"] != "ok" or not r["latency_us_device"]:
            continue
        m, n, k = int(r["M"]), int(r["N"]), int(r["K"])
        M.append(m); N.append(n); K.append(k)
        cyc.append(float(r["cycles_compute"])); lat.append(float(r["latency_us_device"]))
    M, N, K = np.array(M), np.array(N), np.array(K)
    cyc, lat = np.array(cyc, float), np.array(lat, float)
    bytes_ = 2.0 * (M * K + K * N + M * N)
    return M, N, K, cyc, bytes_, lat


def region8(M, N, K):
    fM = np.ceil(M / S); fN = np.ceil(N / S); fK = np.ceil(K / S)
    return ((fK > 1) * 4 + (fN >= 16) * 2 + (fM >= 16)).astype(int)


def fit_groof(cyc, bytes_, y, trim=0.05, iters=3):
    """Relative-weighted (w=1/y^2) fit of y ~= A*max(cyc, bytes/BW)+B over a BW grid,
    with robust trimming of the worst `trim` residuals each iteration."""
    keep = np.ones(len(y), bool)
    best = None
    for _ in range(iters):
        for BW in np.logspace(np.log10(2), np.log10(600), 60):
            feat = np.maximum(cyc, bytes_ / BW)
            X = np.stack([feat, np.ones(len(y))], 1)
            w = 1.0 / y ** 2
            idx = keep
            WX = (X * w[:, None])[idx]
            try:
                A, B = np.linalg.solve(X[idx].T @ WX, WX.T @ y[idx])
            except np.linalg.LinAlgError:
                continue
            p = X @ [A, B]
            mape = np.mean(np.abs((y[idx] - p[idx]) / y[idx]))
            if best is None or mape < best[0]:
                best = (mape, A, B, BW)
        _, A, B, BW = best
        feat = np.maximum(cyc, bytes_ / BW)
        res = np.abs((y - (A * feat + B)) / y)
        thr = np.quantile(res[keep], 1 - trim)
        keep = res <= thr
    return best[1], best[2], best[3]   # A, B, BW


def mape(y, p):
    return float(np.mean(np.abs((y - p) / y)) * 100)


def predict(cyc, bytes_, A, B, BW):
    return A * np.maximum(cyc, bytes_ / BW) + B


def main():
    M, N, K, cyc, bytes_, lat = load()
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(lat))
    ntr = int(0.8 * len(idx)); tr, te = idx[:ntr], idx[ntr:]
    print(f"golden pure-device GEMMs: {len(lat)}  (train {len(tr)} / test {len(te)})\n")

    # --- single global G_roof ---
    A, B, BW = fit_groof(cyc[tr], bytes_[tr], lat[tr])
    p = predict(cyc, bytes_, A, B, BW)
    print(f"single G_roof : A={A:.4e} B={B:.3f}us BW={BW:.1f}  "
          f"held-out MAPE={mape(lat[te], p[te]):.1f}%")

    # --- piecewise 8-region G_roof ---
    rid = region8(M, N, K)
    table = {}
    pe = np.zeros(len(lat))
    for r in range(8):
        m_all = rid == r
        m_tr = m_all & np.isin(np.arange(len(lat)), tr)
        if m_tr.sum() < 8:                       # too few -> fall back to global
            table[r] = (A, B, BW)
        else:
            table[r] = fit_groof(cyc[m_tr], bytes_[m_tr], lat[m_tr])
        Ar, Br, BWr = table[r]
        pe[m_all] = predict(cyc[m_all], bytes_[m_all], Ar, Br, BWr)
    print(f"piecewise(8)  : held-out MAPE={mape(lat[te], pe[te]):.1f}%\n")

    print("TPUV4_REGION_TABLE (PURE device latency) = {")
    for r in range(8):
        Ar, Br, BWr = table[r]
        print(f"    {r}: ({Ar:.6e}, {Br:.4f}, {BWr:.3f}),   n={int((rid==r).sum())}")
    print("}")


if __name__ == "__main__":
    main()
