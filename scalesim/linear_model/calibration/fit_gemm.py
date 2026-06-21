#!/usr/bin/env python3
"""
Workstream B: fit time = a*cycles + b (a=effective clock, b=overhead) from the
collected TPU v4 GEMM data, compare candidate strategies, run the placebo test,
and emit the chosen coefficients + a report.

Candidates (all keep linear pieces => a/b stay physical):
  G0      : single global a*cyc + b
  G_roof  : a*max(cyc, bytes/bpc) + b        (roofline; mem term absorbs regime)
  G_seg   : piecewise a*cyc+b, segments by arithmetic intensity (data-chosen)
Placebo : refit G0 with cycles replaced by const / M*N*K  -> must get worse.
"""
import argparse, json, math, os
import numpy as np, pandas as pd

S = 128


def load(path):
    d = pd.read_csv(path)
    d = d[d.get("status", "ok") == "ok"].copy()
    d = d[(d.latency_us_device > 0) & (d.cycles_compute > 0)]
    d["intensity"] = (d.M * d.N * d.K) / (d.M * d.K + d.K * d.N + d.M * d.N)
    d["bytes"] = 2.0 * (d.M * d.K + d.K * d.N + d.M * d.N)  # bf16
    return d.reset_index(drop=True)


def mape(y, p):
    return float(np.mean(np.abs((y - p) / np.clip(y, 1e-9, None))))


def fit_linear(cyc, y, relative=True):
    """Fit y = a*cyc + b. With relative=True, minimize Σ((y-pred)/y)^2 (weighted
    least squares, w=1/y^2) so the objective matches the MAPE we report — small
    GEMMs (the bulk of LLM layers) stop being drowned out by huge ones under OLS.
    The model form stays linear, so a (clock) and b (overhead) keep their meaning."""
    cyc = np.asarray(cyc, float); y = np.asarray(y, float)
    if relative:
        w = 1.0 / np.clip(y, 1e-9, None) ** 2
    else:
        w = np.ones_like(y)
    # weighted normal equations for [a, b]
    X = np.stack([cyc, np.ones_like(cyc)], axis=1)
    WX = X * w[:, None]
    a, b = np.linalg.solve(X.T @ WX, WX.T @ y)
    return float(a), float(b)


def fit_roof(d, y):
    """Grid-search bytes_per_cycle; for each, linear-fit a,b on max(cyc,cyc_mem)."""
    best = None
    for bpc in np.geomspace(1.0, 4096.0, 60):
        cyc_mem = d["bytes"].values / bpc
        cm = np.maximum(d.cycles_compute.values, cyc_mem)
        a, b = fit_linear(cm, y)
        p = a * cm + b
        e = mape(y, p)
        if best is None or e < best[0]:
            best = (e, a, b, bpc)
    return best  # (mape, a, b, bpc)


def fit_seg(d, y, n_seg=3):
    q = np.quantile(d.intensity.values, np.linspace(0, 1, n_seg + 1))
    q[0], q[-1] = -np.inf, np.inf
    segs = []
    pred = np.zeros_like(y)
    for i in range(n_seg):
        m = (d.intensity.values > q[i]) & (d.intensity.values <= q[i + 1])
        if m.sum() < 5:
            segs.append((q[i], q[i + 1], 0.0, float(np.median(y[m]) if m.any() else 0)))
            if m.any(): pred[m] = np.median(y[m])
            continue
        a, b = fit_linear(d.cycles_compute.values[m], y[m])
        segs.append((float(q[i]), float(q[i + 1]), float(a), float(b)))
        pred[m] = a * d.cycles_compute.values[m] + b
    return segs, mape(y, pred)


def split(d, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(d))
    cut = int(0.8 * len(d))
    return d.iloc[idx[:cut]].reset_index(drop=True), d.iloc[idx[cut:]].reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "gemm_master.csv"))
    p.add_argument("--signal", default="latency_us_device")
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "gemm_linear_tpuv4.json"))
    args = p.parse_args()

    d = load(args.data)
    d["y"] = d[args.signal]
    tr, va = split(d)
    ytr, yva = tr.y.values, va.y.values
    R = {"n_total": len(d), "n_train": len(tr), "n_val": len(va), "signal": args.signal}

    # G0
    a0, b0 = fit_linear(tr.cycles_compute.values, ytr)
    R["G0"] = {"a": float(a0), "b": float(b0),
               "val_mape": mape(yva, a0 * va.cycles_compute.values + b0)}

    # G_roof
    e, a, b, bpc = fit_roof(tr, ytr)
    cm_va = np.maximum(va.cycles_compute.values, va["bytes"].values / bpc)
    R["G_roof"] = {"a": float(a), "b": float(b), "bytes_per_cycle": float(bpc),
                   "val_mape": mape(yva, a * cm_va + b)}

    # G_seg (3 intensity segments)
    segs, _ = fit_seg(tr, ytr, 3)
    def seg_pred(df):
        out = np.zeros(len(df))
        for (lo, hi, aa, bb) in segs:
            m = (df.intensity.values > lo) & (df.intensity.values <= hi)
            out[m] = aa * df.cycles_compute.values[m] + bb
        return out
    R["G_seg"] = {"segments_by_intensity": segs, "val_mape": mape(yva, seg_pred(va))}

    # Placebo: cycles -> constant, and -> M*N*K
    R["placebo_const"] = {"val_mape": mape(yva, np.full(len(va), ytr.mean()))}
    amn, bmn = fit_linear((tr.M * tr.N * tr.K).values.astype(float), ytr)
    R["placebo_MNK"] = {"val_mape": mape(yva, amn * (va.M * va.N * va.K).values + bmn)}

    # LLM held-out shapes
    llm = d[d.shape_class == "llm"]
    if len(llm):
        cm = np.maximum(llm.cycles_compute.values, llm["bytes"].values / bpc)
        R["G_roof_on_llm_mape"] = mape(llm.y.values, a * cm + b)

    # pick best of real candidates
    cands = {k: R[k]["val_mape"] for k in ["G0", "G_roof", "G_seg"]}
    R["chosen"] = min(cands, key=cands.get)
    R["effective_clock_ns_G_roof"] = float(a * 1000.0)  # us/cyc -> ns/cyc

    json.dump(R, open(args.out, "w"), indent=2)
    print(json.dumps({k: (v if not isinstance(v, dict) else
                          {kk: round(vv, 5) if isinstance(vv, float) else vv
                           for kk, vv in v.items() if kk != "segments_by_intensity"})
                      for k, v in R.items()}, indent=2))
    print(f"\nchosen={R['chosen']}  -> {args.out}")


if __name__ == "__main__":
    main()
