#!/usr/bin/env python3
"""
Train per-op latency models from <op>_dataset.csv, matching the shipped
scalesim/model/tpuv4/*.pkl recipe:
  HistGradientBoostingRegressor(loss='absolute_error', learning_rate=0.06,
                                early_stopping='auto'), 80/20 split.
Package as {'model','op_name','metadata'} pickle named <stablehlo_op>.pkl.
"""
import argparse, glob, math, os, pickle
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

FEATURES = ["d0", "d1", "d2", "size", "log2_size"]

# dataset filename stem -> StableHLO op last-token (predictor auto-matches this)
NAME = {"broadcast": "broadcast_in_dim"}


def train_one(csv_path, seed=42):
    df = pd.read_csv(csv_path).dropna()
    df = df[df.latency_us > 0]
    X, y = df[FEATURES], df["latency_us"].values
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=seed)
    m = HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.06,
                                      early_stopping="auto", random_state=seed)
    m.fit(Xtr, ytr)
    pred = m.predict(Xva)
    mae = float(mean_absolute_error(yva, pred))
    mape = float(np.mean(np.abs((yva - pred) / np.clip(yva, 1e-9, None))))
    return m, {"train_rows": len(Xtr), "val_rows": len(Xva),
               "val_mae": mae, "val_mape": mape, "seed": seed,
               "features": FEATURES, "train_csv": os.path.basename(csv_path)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datadir", default=os.path.dirname(__file__))
    p.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "models_tpuv4"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    for csv_path in sorted(glob.glob(os.path.join(args.datadir, "*_dataset.csv"))):
        stem = os.path.basename(csv_path).replace("_dataset.csv", "")
        op_name = NAME.get(stem, stem)
        try:
            model, md = train_one(csv_path, args.seed)
        except Exception as e:
            print(f"  {stem:14s} FAILED: {e}"); continue
        md["op_name"] = op_name
        with open(os.path.join(args.outdir, f"{op_name}.pkl"), "wb") as f:
            pickle.dump({"model": model, "op_name": op_name, "metadata": md}, f)
        rows.append((op_name, md["train_rows"] + md["val_rows"],
                     md["val_mae"], md["val_mape"]))
        print(f"  {op_name:18s} rows={md['train_rows']+md['val_rows']:4d} "
              f"val_mae={md['val_mae']:.4f}us  val_mape={md['val_mape']*100:.2f}%")

    if rows:
        rows.sort(key=lambda r: r[3])
        print("\nMAPE summary (best->worst):")
        for op, n, mae, mape in rows:
            print(f"  {op:18s} {mape*100:6.2f}%  (mae {mae:.3f}us, n={n})")


if __name__ == "__main__":
    main()
