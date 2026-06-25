#!/usr/bin/env python3
"""
Fit the v6e GEMM fusion-time model with the SAME 12-region scheme tpu.py uses
(`_tpuv4_region`: foldK>1, foldN>1, foldM>1, plus a 2nd M bar at foldM>=6), on the
golden xprof fusion labels (gemm_pure_master_tpuv6e.csv, latency_us_device).

Reuses fit_piecewise_pure.fit_groof (relative-weighted, trimmed BW grid) and the
production region selector from scalesim.linear_model.tpu so the fitted table drops
straight into TPUV6E_REGION_TABLE. Prints held-out MAPE (single vs piecewise-12)
and the region table.
"""
import csv, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from scalesim.linear_model.tpu import _tpuv4_region
from fit_piecewise_pure import fit_groof, predict, mape

HERE = os.path.dirname(__file__)
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "gemm_pure_master_tpuv6e.csv")
REGIONS = [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15]  # ids producible by _tpuv4_region


def load():
    M, N, K, cyc, lat = [], [], [], [], []
    for r in csv.DictReader(open(DATA)):
        if r["status"] != "ok" or not r["latency_us_device"]:
            continue
        M.append(int(r["M"])); N.append(int(r["N"])); K.append(int(r["K"]))
        cyc.append(float(r["cycles_compute"])); lat.append(float(r["latency_us_device"]))
    M, N, K = np.array(M), np.array(N), np.array(K)
    cyc, lat = np.array(cyc, float), np.array(lat, float)
    bytes_ = 2.0 * (M * K + K * N + M * N)
    rid = np.array([_tpuv4_region(int(m), int(n), int(k)) for m, n, k in zip(M, N, K)])
    return cyc, bytes_, lat, rid


def main():
    cyc, bytes_, lat, rid = load()
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(lat)); ntr = int(0.8 * len(idx))
    tr, te = idx[:ntr], idx[ntr:]
    trmask = np.zeros(len(lat), bool); trmask[tr] = True
    print(f"v6e golden fusion GEMMs: {len(lat)}  (train {len(tr)} / test {len(te)})\n")

    # single global G_roof baseline
    A, B, BW = fit_groof(cyc[tr], bytes_[tr], lat[tr])
    p = predict(cyc, bytes_, A, B, BW)
    print(f"single G_roof  : A={A:.4e} B={B:.3f} BW={BW:.1f}  held-out MAPE={mape(lat[te], p[te]):.1f}%")

    # piecewise-12
    table, pe = {}, np.zeros(len(lat))
    for r in REGIONS:
        m_all = rid == r
        m_tr = m_all & trmask
        if m_tr.sum() < 8:
            table[r] = (max(A, 0.0), B, BW)              # too few -> global
        else:
            Ar, Br, BWr = fit_groof(cyc[m_tr], bytes_[m_tr], lat[m_tr])
            table[r] = (max(Ar, 0.0), Br, BWr)           # clamp negative slope (floor-only)
        Ar, Br, BWr = table[r]
        if m_all.any():
            pe[m_all] = predict(cyc[m_all], bytes_[m_all], Ar, Br, BWr)
    print(f"piecewise(12)  : held-out MAPE={mape(lat[te], pe[te]):.1f}%\n")

    print("TPUV6E_REGION_TABLE = {")
    for r in REGIONS:
        Ar, Br, BWr = table[r]
        print(f"    {r}: ({Ar:.6e}, {Br:.4f}, {BWr:.3f}),   # n={int((rid==r).sum())}")
    print("}")


if __name__ == "__main__":
    main()
