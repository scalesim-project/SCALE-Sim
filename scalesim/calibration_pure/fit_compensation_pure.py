#!/usr/bin/env python3
"""
Whole-model compensation fit for the PURE per-op models, parameterized by TPU
generation (--gen tpuv4 | tpuv6e). Run this AFTER re-collecting + retraining the
pure per-op models (collect_pure_device_tpu.py -> train_ops.py -> model/<gen>).

Model (a0 PINNED to 1.0 -- GEMM passes through):
    T = Sc + a1*Sn + C0
  Sc = sum of COMPUTE (GEMM) single_op_us  (tpu.py fusion time; dtype-independent)
  Sn = sum of NON-COMPUTE single_op_us     (the PURE per-op models being calibrated)
  C0 = fixed per-execute device floor      (C1*n_gemm dropped: collinear with Sn)

Pipeline (fully regenerated with the CURRENT converter so Sc/Sn are self-consistent;
only the hardware device-busy truth is reused):
  for each (model, seq) at batch=1 + a tiny_transformer anchor:
    run `scale.py -b -c configs/<gen>.cfg -t <f32 mlir> -i gemm`  (bypass)
    Sc = sum single_op_us over compute ops ; Sn = sum over non-compute ops
  truth_us = e2e device-busy (device_ms*1000) for that generation.

The calibration StableHLO graphs are committed (calib_mlir/, f32, target-independent),
so no re-export is needed -- the same graphs feed v4 and v6e. The fit itself is CPU-only.

Usage:
  python3 fit_compensation_pure.py --gen tpuv6e --tiny-truth <measured_v6e_tiny_us>
  # -> prints a1, C0; paste into COMPENSATION_BY_GEN["TPUv6e"] in total_time_report.py
"""
import argparse, csv, glob, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SEQS = [128, 256, 512, 1024]
MODELS = ["gpt2", "qwen2.5-0.5b", "smollm2-135m"]
# Shared, target-INDEPENDENT StableHLO graphs committed in the repo (f32), so no
# re-export is needed on a new TPU generation -- same MLIRs feed v4 and v6e.
CALIB_MLIR_ROOT = os.path.join(HERE, "calib_mlir")          # calib_mlir/s<seq>/<model>.mlir
TINY_MLIR = os.path.join(ROOT, "topologies", "stablehlo", "llm",
                         "tiny_transformer_pytorch.stablehlo.mlir")

# Per-generation: bypass config, in-repo device-busy truth CSV, and a default tiny-
# anchor truth (measured on that generation; override with --tiny-truth).
GEN = {
    "tpuv4": {
        "cfg": os.path.join(ROOT, "configs", "tpuv4.cfg"),
        "truth": os.path.join(HERE, "e2e_device_truth_tpuv4.csv"),
        "tiny_truth": 138.2,
    },
    "tpuv6e": {
        "cfg": os.path.join(ROOT, "configs", "tpuv6e.cfg"),
        "truth": os.path.join(HERE, "e2e_device_truth_tpuv6e.csv"),
        "tiny_truth": None,        # measure on v6e, pass --tiny-truth
    },
}


def run_bypass_sums(mlir, outdir, cfg):
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH=ROOT)
    subprocess.run([sys.executable, os.path.join(ROOT, "scalesim", "scale.py"),
                    "-b", "-c", cfg, "-t", mlir, "-p", outdir, "-i", "gemm"],
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
    """device-busy us (= device_ms*1000) for batch-1 rows. Works for both the v4
    (e2e_work) and v6e truth schemas (both carry a device_ms column)."""
    t = {}
    for r in csv.DictReader(open(path)):
        if r.get("status") == "ok" and int(r["batch"]) == 1 and r.get("device_ms", "").strip():
            t[(r["model"], int(r["seq"]))] = float(r["device_ms"]) * 1e3
    return t


def build_calib(mlir_root, truth, cfg, tiny_truth, out_csv):
    rows = []
    for m in MODELS:
        for s in SEQS:
            mlir = os.path.join(mlir_root, f"s{s}", f"{m}.stablehlo.mlir")
            if not os.path.exists(mlir) or (m, s) not in truth:
                continue
            Sc, Sn, Nc, Nn = run_bypass_sums(mlir, f"/tmp/calpure_run/{m}_s{s}", cfg)
            rows.append(dict(model=m, seq=s, batch=1, Sc_us=Sc, Sn_us=Sn,
                             Nc=Nc, Nn=Nn, truth_us=truth[(m, s)]))
            print(f"  {m:14}{s:>5}  Sc={Sc:>9.1f}  Sn={Sn:>9.1f}  "
                  f"Nc={Nc} Nn={Nn}  truth={truth[(m,s)]:.1f}")
    if tiny_truth is not None and os.path.exists(TINY_MLIR):     # small-model anchor
        Sc, Sn, Nc, Nn = run_bypass_sums(TINY_MLIR, "/tmp/calpure_run/tiny", cfg)
        rows.append(dict(model="tiny_transformer", seq=128, batch=1, Sc_us=Sc,
                         Sn_us=Sn, Nc=Nc, Nn=Nn, truth_us=tiny_truth))
        print(f"  {'tiny_transformer':14}{128:>5}  Sc={Sc:>9.1f}  Sn={Sn:>9.1f}  "
              f"Nc={Nc} Nn={Nn}  truth={tiny_truth:.1f}")
    elif tiny_truth is None:
        print("  (no --tiny-truth given: skipping the small-model anchor -- C0 may "
              "over-fit and over-predict tiny models; measure tiny on this gen and pass it)")
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
    R = Y - Sc                                         # a0 pinned to 1
    X = np.stack([Sn, np.ones(len(Sn))], 1)            # [a1, C0]
    a1, C0 = wls(X, R, w)
    pred = Sc + X @ [a1, C0]
    ins = mape(Y, pred)
    preds = np.zeros(len(Y))                           # leave-one-model-out CV
    for mo in sorted(set(mdl)):
        tr = np.array([i for i, mm in enumerate(mdl) if mm != mo])
        ts = np.array([i for i, mm in enumerate(mdl) if mm == mo])
        preds[ts] = Sc[ts] + X[ts] @ wls(X[tr], R[tr], w[tr])
    cv = mape(Y, preds)
    return a1, C0, ins, cv, pred, Y, mdl, [r["seq"] for r in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", required=True, choices=list(GEN))
    ap.add_argument("--mlir-root", default=CALIB_MLIR_ROOT,
                    help="dir with f32 graphs in s{128,256,512,1024}/<model>.stablehlo.mlir "
                         "(default: the in-repo shared calib_mlir/)")
    ap.add_argument("--truth", default=None, help="override the per-gen device-busy CSV")
    ap.add_argument("--tiny-truth", type=float, default=None,
                    help="measured tiny_transformer device-busy us on this gen (anchors C0)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    g = GEN[args.gen]
    truth_path = args.truth or g["truth"]
    tiny_truth = args.tiny_truth if args.tiny_truth is not None else g["tiny_truth"]
    out_csv = args.out or os.path.join(HERE, f"calib_{args.gen}_pure.csv")

    truth = load_truth(truth_path)
    print(f"gen={args.gen}  cfg={os.path.basename(g['cfg'])}  truth points(batch=1)={len(truth)}")
    print("building calib (current pipeline, PURE models):")
    rows = build_calib(args.mlir_root, truth, g["cfg"], tiny_truth, out_csv)

    a1, C0, ins, cv, pred, Y, mdl, seqs = fit(rows)
    print(f"\nFIT (a0=1 pinned):  a1={a1:.4f}  C0={C0:.2f}  C1=0")
    print(f"  in-sample MAPE={ins:.1f}%   leave-one-model-out CV={cv:.1f}%\n")
    print(f"{'model':16}{'seq':>5}{'pred':>9}{'truth':>9}{'err%':>7}")
    for m, s, pp, yy in zip(mdl, seqs, pred, Y):
        print(f"{m:16}{s:>5}{pp:>9.0f}{yy:>9.0f}{abs(pp-yy)/yy*100:>6.0f}%")

    import json
    coeffs = {"a0_mxu": 1.0, "a1_vpu": float(a1), "c_c": 0.0, "c_n": 0.0,
              "C0_forward": float(C0), "C1_per_gemm": 0.0,
              "in_sample_mape": ins, "lomo_cv_mape": cv, "n_points": len(rows),
              "gen": args.gen, "basis": "PURE per-op models; T=Sc+a1*Sn+C0; f32 graphs"}
    jp = os.path.join(HERE, f"coeffs_{args.gen}_pure.json")
    json.dump(coeffs, open(jp, "w"), indent=2)
    print(f"\nwrote {out_csv} and {jp}")
    print(f'paste into COMPENSATION_BY_GEN["{"TPUv4" if args.gen=="tpuv4" else "TPUv6e"}"]: '
          f'a1_vpu={a1:.4f}, C0_forward={C0:.1f}, C1_per_gemm=0.0')


if __name__ == "__main__":
    main()
