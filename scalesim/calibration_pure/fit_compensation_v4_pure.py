#!/usr/bin/env python3
"""
Re-tune the TPU v4 whole-model compensation for the PURE per-op models.

The non-compute predictor now loads scalesim/model/tpuv4 = the PURE (xprof
single-op span) models, so the summed non-compute latency Sn changes vs the old
loop-method fit -> a1/C0/C1 must be refit.

Pipeline (fully regenerated with the CURRENT converter + GEMM model, so Sc/Sn are
self-consistent; only the hardware truth is reused):
  for each (model, seq) at batch=1:
    run `scale.py -b -c configs/tpuv4.cfg -t <mlir> -i gemm`  (bypass)
    Sc   = sum single_op_us over COMPUTE ops      (GEMM fusion-time, tpu.py)
    Sn   = sum single_op_us over NON-COMPUTE ops  (PURE per-op models)
    Ngemm= compute op count
  truth_us = e2e_device_truth.csv device_ms*1000  (torch.compile device-busy, v4)

Fit (a0 PINNED to 1.0 -- GEMM passes through):
    T = Sc + a1*Sn + C0 + C1*Ngemm
linear in (a1, C0, C1); WLS with w = 1/truth^2 (relative error).

Usage:
  python3 fit_compensation_v4_pure.py --mlir-root /tmp/v4cal --truth <e2e_device_truth.csv>
"""
import argparse, csv, glob, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CFG = os.path.join(ROOT, "configs", "tpuv4.cfg")
SEQS = [128, 256, 512, 1024]
MODELS = ["gpt2", "qwen2.5-0.5b", "smollm2-135m"]


def run_bypass_sums(mlir, outdir):
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH=ROOT)
    subprocess.run([sys.executable, os.path.join(ROOT, "scalesim", "scale.py"),
                    "-b", "-c", CFG, "-t", mlir, "-p", outdir, "-i", "gemm"],
                   env=env, cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rep = glob.glob(os.path.join(outdir, "*", "TIME_REPORT.csv"))
    if not rep:
        raise RuntimeError(f"no TIME_REPORT for {mlir}")
    Sc = Sn = 0.0; Nc = Nn = 0
    for r in csv.DictReader(open(rep[0])):
        if r["OpID"].strip() == "TOTAL":
            continue
        s = r.get("single_op_us", "").strip()
        if not s:
            continue
        v = float(s); layer = r.get("layer", "").strip()
        if layer and layer != "N/A":
            Sc += v; Nc += 1
        else:
            Sn += v; Nn += 1
    return Sc, Sn, Nc, Nn


def load_truth(path):
    t = {}
    for r in csv.DictReader(open(path)):
        if r.get("status") == "ok" and int(r["batch"]) == 1 and r["device_ms"].strip():
            t[(r["model"], int(r["seq"]))] = float(r["device_ms"]) * 1e3
    return t


# tiny_transformer small-model anchor: (mlir, measured v4 device-busy us)
TINY_MLIR = os.path.join(ROOT, "topologies", "stablehlo", "llm",
                         "tiny_transformer_pytorch.stablehlo.mlir")
TINY_TRUTH = 138.2


def build_calib(mlir_root, truth, out_csv):
    rows = []
    for m in MODELS:
        for s in SEQS:
            mlir = os.path.join(mlir_root, f"s{s}", f"{m}.stablehlo.mlir")
            if not os.path.exists(mlir) or (m, s) not in truth:
                continue
            Sc, Sn, Nc, Nn = run_bypass_sums(mlir, f"/tmp/v4cal_run/{m}_s{s}")
            rows.append(dict(model=m, seq=s, batch=1, Sc_us=Sc, Sn_us=Sn,
                             Nc=Nc, Nn=Nn, truth_us=truth[(m, s)]))
            print(f"  {m:14}{s:>5}  Sc={Sc:>9.1f}  Sn={Sn:>9.1f}  "
                  f"Nc={Nc} Nn={Nn}  truth={truth[(m,s)]:.1f}")
    if os.path.exists(TINY_MLIR):                       # small-model anchor
        Sc, Sn, Nc, Nn = run_bypass_sums(TINY_MLIR, "/tmp/v4cal_run/tiny")
        rows.append(dict(model="tiny_transformer", seq=128, batch=1, Sc_us=Sc,
                         Sn_us=Sn, Nc=Nc, Nn=Nn, truth_us=TINY_TRUTH))
        print(f"  {'tiny_transformer':14}{128:>5}  Sc={Sc:>9.1f}  Sn={Sn:>9.1f}  "
              f"Nc={Nc} Nn={Nn}  truth={TINY_TRUTH:.1f}")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model","seq","batch","Sc_us","Sn_us","Nc","Nn","truth_us"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return rows


def wls(X, y, w):
    WX = X * w[:, None]
    return np.linalg.solve(X.T @ WX, X.T @ (w * y))


def mape(y, p):
    return float(np.mean(np.abs((y - p) / y)) * 100)


def fit(rows):
    Sc = np.array([r["Sc_us"] for r in rows])
    Sn = np.array([r["Sn_us"] for r in rows])
    Y = np.array([r["truth_us"] for r in rows])
    mdl = [r["model"] for r in rows]
    w = 1.0 / Y ** 2
    # a0 pinned to 1 (GEMM passthrough); structure T = Sc + a1*Sn + C0 (C1=0: a
    # size-dependent C1*n_gemm is collinear with Sn and hurts generalization).
    R = Y - Sc
    X = np.stack([Sn, np.ones(len(Sn))], 1)            # [a1, C0]
    a1, C0 = wls(X, R, w)
    pred = Sc + X @ [a1, C0]
    ins = mape(Y, pred)
    preds = np.zeros(len(Y))                            # leave-one-model-out CV
    for mo in sorted(set(mdl)):
        tr = np.array([i for i, mm in enumerate(mdl) if mm != mo])
        ts = np.array([i for i, mm in enumerate(mdl) if mm == mo])
        preds[ts] = Sc[ts] + X[ts] @ wls(X[tr], R[tr], w[tr])
    cv = mape(Y, preds)
    return a1, C0, 0.0, ins, cv, pred, Y, mdl, [r["seq"] for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlir-root", default="/tmp/v4cal")
    ap.add_argument("--truth", default=os.path.join(
        ROOT, "..", "SCALE-Sim_TPU", "e2e_work", "e2e_device_truth.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "calib_v4_pure.csv"))
    args = ap.parse_args()

    truth = load_truth(args.truth)
    print(f"truth points (batch=1): {len(truth)}")
    print("building calib (current pipeline, PURE models):")
    rows = build_calib(args.mlir_root, truth, args.out)

    a1, C0, C1, ins, cv, pred, Y, mdl, seqs = fit(rows)
    print(f"\nFIT (a0=1 pinned):  a1={a1:.4f}  C0={C0:.2f}  C1={C1:.4f}")
    print(f"  in-sample MAPE = {ins:.1f}%   leave-one-model-out CV = {cv:.1f}%\n")
    print(f"{'model':14}{'seq':>5}{'pred':>9}{'truth':>9}{'err%':>7}")
    for m, s, pp, yy in zip(mdl, seqs, pred, Y):
        print(f"{m:14}{s:>5}{pp:>9.0f}{yy:>9.0f}{abs(pp-yy)/yy*100:>6.0f}%")

    import json
    json.dump({"a0_mxu": 1.0, "a1_vpu": a1, "c_c": 0.0, "c_n": 0.0,
               "C0_forward": C0, "C1_per_gemm": C1,
               "in_sample_mape": ins, "lomo_cv_mape": cv,
               "basis": "PURE per-op models (scalesim/model/tpuv4); current converter+GEMM; v4 truth",
               "n_points": len(rows)},
              open(os.path.join(HERE, "coeffs_v4_pure.json"), "w"), indent=2)
    print(f"\nwrote {args.out} and coeffs_v4_pure.json")


if __name__ == "__main__":
    main()
