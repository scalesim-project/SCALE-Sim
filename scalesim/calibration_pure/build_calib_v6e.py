#!/usr/bin/env python3
"""
Build calib_tpuv6e.csv for the whole-model compensation fit.

For each (model, seq) StableHLO graph, run the SCALE-Sim bypass with configs/
tpuv6e.cfg and sum the per-op STANDALONE predictions from the unified TIME_REPORT:
  Sc_us = sum of single_op_us over COMPUTE ops (GEMM fusion-time, tpuv6e table)
  Sn_us = sum of single_op_us over NON-COMPUTE ops (per-op models)
  Nc/Nn = op counts
Then join the torch.compile device-busy ground truth (e2e_device_truth_tpuv6e.csv,
device_ms -> truth_us). Output schema matches fit_compensation_v6e.py.

Bypass is run with JAX_PLATFORMS=cpu so it never touches the TPU.

Usage:
  python3 build_calib_v6e.py --mlir-dir <dir of {model}_s{seq}.mlir> \
      --truth e2e_device_truth_tpuv6e.csv --out calib_tpuv6e.csv
"""
import argparse, csv, glob, os, re, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CFG = os.path.join(ROOT, "configs", "tpuv6e.cfg")


def run_bypass(mlir, outdir):
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH=ROOT)
    subprocess.run([sys.executable, os.path.join(ROOT, "scalesim", "scale.py"),
                    "-b", "-c", CFG, "-t", mlir, "-p", outdir, "-i", "gemm"],
                   env=env, cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # find the unified TIME_REPORT
    hits = glob.glob(os.path.join(outdir, "*", "TIME_REPORT.csv"))
    return hits[0] if hits else None


def sums_from_report(path):
    Sc = Sn = 0.0; Nc = Nn = 0
    for r in csv.DictReader(open(path)):
        if r["OpID"].strip() == "TOTAL":
            continue
        s = r.get("single_op_us", "").strip()
        if not s:
            continue
        v = float(s)
        layer = r.get("layer", "").strip()
        if layer and layer != "N/A":          # compute op (has a COMPUTE_REPORT layer)
            Sc += v; Nc += 1
        else:
            Sn += v; Nn += 1
    return Sc, Sn, Nc, Nn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mlir-dir", required=True)
    p.add_argument("--truth", default=os.path.join(HERE, "e2e_device_truth_tpuv6e.csv"))
    p.add_argument("--out", default=os.path.join(HERE, "calib_tpuv6e.csv"))
    args = p.parse_args()

    truth = {}
    for r in csv.DictReader(open(args.truth)):
        if r.get("status") == "ok":
            truth[(r["model"], int(r["seq"]), int(r["batch"]))] = float(r["device_ms"]) * 1e3

    rows = []
    for mlir in sorted(glob.glob(os.path.join(args.mlir_dir, "*.mlir"))):
        base = os.path.basename(mlir)
        m = re.match(r"(.+)_s(\d+)\.mlir$", base)
        if not m:
            print(f"  skip (name): {base}"); continue
        model, seq = m.group(1), int(m.group(2))
        if (model, seq, 1) not in truth:
            print(f"  skip (no truth): {model} seq{seq}"); continue
        with tempfile.TemporaryDirectory() as td:
            rep = run_bypass(mlir, td)
            if not rep:
                print(f"  skip (no report): {base}"); continue
            Sc, Sn, Nc, Nn = sums_from_report(rep)
        rows.append({"model": model, "seq": seq, "batch": 1,
                     "Sc_us": f"{Sc:.4f}", "Sn_us": f"{Sn:.4f}", "Nc": Nc, "Nn": Nn,
                     "truth_us": f"{truth[(model, seq, 1)]:.4f}"})
        print(f"  {model:14s} seq{seq:<5d} Sc={Sc:9.1f} Sn={Sn:10.1f} Nc={Nc} Nn={Nn} "
              f"truth={truth[(model, seq, 1)]:.1f}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "seq", "batch", "Sc_us", "Sn_us", "Nc", "Nn", "truth_us"])
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
