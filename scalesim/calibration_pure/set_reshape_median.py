#!/usr/bin/env python3
"""Replace the trained reshape model with a CONSTANT = median standalone reshape
latency. reshape latency is bimodal (metadata ~0us vs physical relayout ~1000s us)
and not predictable from shape; a fitted regressor over-predicts it ~37x at vocab
sizes and then dominates the whole-model Sn (~63%), wrecking the compensation fit.
The median constant (~7us on v4) restores a clean fit and a meaningful a1.

Run AFTER train_ops.py, BEFORE fit_compensation_pure.py.
Usage:
  python3 set_reshape_median.py --model-dir ../model/tpuv4 \
      --dataset <datasets_dir>/reshape_dataset.csv
"""
import argparse, csv, pickle, os
import numpy as np
from sklearn.dummy import DummyRegressor

FEATURES = ["d0", "d1", "d2", "size", "log2_size"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, help="scalesim/model/<gen>")
    p.add_argument("--dataset", required=True, help="<gen> reshape_dataset.csv")
    args = p.parse_args()

    rows = list(csv.DictReader(open(args.dataset)))
    lat = np.array([float(r["latency_us"]) for r in rows])
    X = np.array([[float(r["d0"]), float(r["d1"]), float(r["d2"]),
                   float(r["size"]), np.log2(float(r["size"]))] for r in rows])
    dm = DummyRegressor(strategy="median").fit(X, lat)
    med = float(np.median(lat))
    out = os.path.join(args.model_dir, "reshape.pkl")
    pickle.dump({"model": dm, "op_name": "reshape",
                 "metadata": {"strategy": "constant median", "median_us": med,
                              "features": FEATURES, "train_rows": len(rows)}},
                open(out, "wb"))
    print(f"reshape -> constant median {med:.2f}us  (n={len(rows)})  wrote {out}")


if __name__ == "__main__":
    main()
