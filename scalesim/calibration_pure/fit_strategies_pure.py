#!/usr/bin/env python3
"""
Try MANY piecewise / feature strategies to predict golden pure device latency,
and measure the achievable ceiling with gradient boosting. If even GBR can't beat
the piecewise G_roof much, the residual is irreducible launch jitter (not shape).
All evaluated on the same 80/20 held-out split, relative MAPE.
"""
import csv, math, os
import numpy as np
from fit_piecewise_pure import load, fit_groof, predict, mape, S

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAVE_SK = True
except Exception:
    HAVE_SK = False


def folds(M, N, K):
    return np.ceil(M / S), np.ceil(N / S), np.ceil(K / S)


def eval_regions(rid, cyc, bytes_, y, tr, te):
    """Fit a G_roof per region id on train, eval held-out MAPE."""
    p = np.zeros(len(y))
    gA, gB, gBW = fit_groof(cyc[tr], bytes_[tr], y[tr])
    for r in np.unique(rid):
        m = rid == r
        mtr = m & np.isin(np.arange(len(y)), tr)
        A, B, BW = fit_groof(cyc[mtr], bytes_[mtr], y[mtr]) if mtr.sum() >= 8 else (gA, gB, gBW)
        p[m] = predict(cyc[m], bytes_[m], A, B, BW)
    return mape(y[te], p[te]), len(np.unique(rid))


def fit_linear(X, y, tr):
    w = 1.0 / y ** 2
    WX = (X * w[:, None])[tr]
    coef = np.linalg.solve(X[tr].T @ WX, WX.T @ (w[tr] * y[tr]) if False else (X[tr] * w[tr, None]).T @ y[tr])
    return coef


def main():
    M, N, K, cyc, bytes_, y = load()
    fM, fN, fK = folds(M, N, K)
    inten = cyc / (bytes_ / 2)            # arithmetic intensity proxy
    ntiles = fM * fN * fK                 # # of array-tile launches
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y)); ntr = int(0.8 * len(idx))
    tr, te = idx[:ntr], idx[ntr:]
    cb = cyc[te] > 1e6                    # compute-bound subset of the test set

    def report(name, p_or_mape, nreg=""):
        if isinstance(p_or_mape, np.ndarray):
            m, mcb = mape(y[te], p_or_mape[te]), mape(y[te][cb], p_or_mape[te][cb])
        else:
            m, mcb = p_or_mape, float("nan")
        print(f"  {name:34s} {str(nreg):>4}  MAPE={m:5.1f}%   compute-bound={mcb:4.1f}%")

    print(f"golden pure-device GEMMs={len(y)}  (test {len(te)}, compute-bound {cb.sum()})\n")
    print("--- G_roof region schemes (selector from M,N,K) ---")
    schemes = {
        "single G_roof": np.zeros(len(y), int),
        "K-band (foldK 1/2-?/big)": np.clip(np.digitize(fK, [2, 8]), 0, 2),
        "KxNxM (8) [current]": ((fK > 1) * 4 + (fN >= 16) * 2 + (fM >= 16)).astype(int),
        "cycle-magnitude bins (6)": np.digitize(np.log10(np.maximum(cyc, 1)), [2.7, 3.3, 4, 4.7, 5.5]),
        "intensity bins (4)": np.digitize(inten, [4, 16, 64]),
        "ntiles bins (5)": np.digitize(np.log2(np.maximum(ntiles, 1)), [1, 3, 6, 9]),
        "KxNxM x cyc-small (16)": (((fK > 1) * 4 + (fN >= 16) * 2 + (fM >= 16)) * 2
                                   + (cyc > 5e3)).astype(int),
    }
    for nm, rid in schemes.items():
        m, nreg = eval_regions(rid.astype(int), cyc, bytes_, y, tr, te)
        report(nm, m, nreg)

    print("\n--- feature-form changes (global) ---")
    # G_roof + tile-count floor term: lat = A*max(cyc,bytes/BW) + C*ntiles + B
    bestm = None
    for BW in np.logspace(np.log10(2), np.log10(600), 40):
        feat = np.maximum(cyc, bytes_ / BW)
        Xl = np.stack([feat, ntiles, np.ones(len(y))], 1)
        coef = fit_linear(Xl, y, tr); p = Xl @ coef
        mm = mape(y[tr], p[tr])
        if bestm is None or mm < bestm[0]:
            bestm = (mm, p, BW)
    report("G_roof + C*ntiles floor", bestm[1])

    if HAVE_SK:
        print("\n--- ML ceiling (HistGBR on log features; non-physical) ---")
        F = np.stack([np.log1p(M), np.log1p(N), np.log1p(K), np.log1p(cyc),
                      np.log1p(bytes_), np.log1p(inten), fM, fN, fK, np.log1p(ntiles)], 1)
        gb = HistGradientBoostingRegressor(loss="absolute_error", max_iter=400,
                                           learning_rate=0.06)
        gb.fit(F[tr], y[tr]); p = gb.predict(F)
        report("HistGBR (10 features)", p)
        # GBR on log-target (helps relative error on the floor-dominated small ones)
        gb2 = HistGradientBoostingRegressor(loss="absolute_error", max_iter=400,
                                            learning_rate=0.06)
        gb2.fit(F[tr], np.log(y[tr])); p2 = np.exp(gb2.predict(F))
        report("HistGBR (log-target)", p2)


if __name__ == "__main__":
    main()
