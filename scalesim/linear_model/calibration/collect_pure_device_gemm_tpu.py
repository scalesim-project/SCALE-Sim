#!/usr/bin/env python3
"""
Collect PURE device kernel latency for bf16 GEMMs via the xprof trace.

Trace-authoritative complement to `collect_gemm_tpu.py`. That script derives
`latency_us_device` indirectly, by an in-JIT fori_loop and a wall-clock
subtraction `(t_K - t_1)/(K-1)` (cancels the fixed host dispatch). This script
reads the GEMM kernel's span **straight from the TPU:0 device timeline** under
`jax.profiler.trace` -- host/PJRT dispatch and the host->device sync floor are
excluded by construction, not by subtraction.

Output schema is a DROP-IN for `fit_gemm.py` (same columns as `gemm_master.csv`):
`latency_us_device` is set to the pure xprof kernel time, `latency_us_wallclock`
to the python-timer execute+block, plus an extra `host_us = wall - kernel` column
(the per-execute non-TPU cost). So you can fit G_roof on pure-kernel labels:
    python3 fit_gemm.py --data gemm_pure_master.csv

Findings on TPU v4 (see scalesim/model/README.md f and
SCALE-Sim_TPU/e2e_work/compensation/measure_xprof.py): pure GEMM kernel time is
~`1.14e-4 us/cyc * cycles + ~12.6 us` floor; host_us is a flat ~90 us per execute,
work-independent. The wall-clock "device" signal conflates the ~12 us kernel floor
with ~52 us of device sync -- xprof separates them.

Requirements: exclusive PJRT TPU access, `pip install jax[tpu]`, bf16.
Usage:
    PJRT_DEVICE=TPU python3 collect_pure_device_gemm_tpu.py --shuffle \
        --shapes shapes.csv --out gemm_pure_master.csv --iters 30
Resumable / crash-safe (skips shapes already in --out).
"""
import os
os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("XLA_USE_SPMD", "0")

import argparse, csv, glob, gzip, json, math, shutil, time
import jax, jax.numpy as jnp

TRACE_ROOT = "/tmp/xprof_gemm"


def matmul_scale_sim_model(m, n, k, S=128):
    """SCALE-Sim closed-form compute cycles (M-streaming, weight-stationary)."""
    return (2 * S + S + m - 2) * math.ceil(n / S) * math.ceil(k / S) - 1


def device_pid_tpu0(events):
    """PID of the TPU:0 device op stream, auto-detected from process_name meta
    (NOT a hardcoded `pid == 3`; core 0 only, so no cross-core double-count)."""
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "process_name":
            if "device:TPU:0" in e.get("args", {}).get("name", ""):
                return e["pid"]
    return None


def kernel_us_from_trace(folder):
    """Per-call pure kernel time = mean dur of the dominant device span on the
    TPU:0 timeline (the executed HLO module). Taking the single largest-total-dur
    span name avoids double-counting nested fusion ops."""
    files = []
    for root, _, fs in os.walk(folder):
        files += [os.path.join(root, f) for f in fs if f.endswith(".trace.json.gz")]
    if not files:
        return None
    data = json.load(gzip.open(max(files, key=os.path.getmtime), "rt"))
    events = data.get("traceEvents", [])
    pid = device_pid_tpu0(events)
    if pid is None:
        return None
    by_name = {}
    for e in events:
        if e.get("pid") == pid and e.get("ph") == "X" and "dur" in e:
            by_name.setdefault(e["name"], []).append(e["dur"])
    if not by_name:
        return None
    durs = by_name[max(by_name, key=lambda n: sum(by_name[n]))]
    return sum(durs) / len(durs)


def measure(m, n, k, warmup, iters, reps):
    a = jnp.ones((m, k), jnp.bfloat16)
    b = jnp.ones((k, n), jnp.bfloat16)
    a.block_until_ready(); b.block_until_ready()
    compiled = jax.jit(lambda x, y: x @ y).lower(a, b).compile()
    for _ in range(warmup):
        compiled(a, b).block_until_ready()

    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); compiled(a, b).block_until_ready()
        ts.append(time.perf_counter() - t0)
    ts.sort(); wall = ts[len(ts) // 2] * 1e6

    folder = os.path.join(TRACE_ROOT, f"{m}_{n}_{k}")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)
    with jax.profiler.trace(folder):
        for _ in range(iters):
            compiled(a, b).block_until_ready()
    kernel = kernel_us_from_trace(folder)
    shutil.rmtree(folder, ignore_errors=True)
    return wall, kernel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", default=os.path.join(os.path.dirname(__file__), "shapes.csv"))
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "gemm_pure_master.csv"))
    p.add_argument("--limit", type=int, default=0, help="0 = all")
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30, help="executes traced per shape")
    p.add_argument("--reps", type=int, default=15, help="wall-clock reps (median)")
    p.add_argument("--max-bytes", type=float, default=8e9)
    args = p.parse_args()

    rows = list(csv.DictReader(open(args.shapes)))
    if args.shuffle:
        import random; random.Random(0).shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]

    print(f"device={jax.devices()[0].device_kind}  shapes={len(rows)}  out={args.out}")
    done = set()
    if os.path.exists(args.out):
        for r in csv.DictReader(open(args.out)):
            done.add((int(r["M"]), int(r["N"]), int(r["K"])))

    fields = ["M", "N", "K", "shape_class", "dtype", "latency_us_wallclock",
              "latency_us_device", "host_us", "cycles_compute", "status"]
    new = not os.path.exists(args.out)
    fh = open(args.out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=fields)
    if new:
        w.writeheader()

    t_start = time.perf_counter(); n_ok = n_skip = n_err = 0
    for i, r in enumerate(rows):
        m, n, k = int(r["M"]), int(r["N"]), int(r["K"])
        if (m, n, k) in done:
            continue
        rec = {"M": m, "N": n, "K": k, "shape_class": r.get("shape_class", ""),
               "dtype": "bf16", "latency_us_wallclock": "", "latency_us_device": "",
               "host_us": "", "cycles_compute": matmul_scale_sim_model(m, n, k),
               "status": "ok"}
        if 2 * (m * k + k * n + m * n) > args.max_bytes:
            rec["status"] = "skip_oom"; n_skip += 1
        else:
            try:
                wall, kernel = measure(m, n, k, args.warmup, args.iters, args.reps)
                if kernel is None:
                    rec["status"] = "err:no_trace"; n_err += 1
                else:
                    rec["latency_us_wallclock"] = f"{wall:.4f}"
                    rec["latency_us_device"] = f"{kernel:.4f}"     # PURE xprof kernel
                    rec["host_us"] = f"{wall - kernel:.4f}"
                    n_ok += 1
            except Exception as e:
                rec["status"] = "err:" + repr(e)[:60]; n_err += 1
        w.writerow(rec); fh.flush()
        if (i + 1) % 25 == 0:
            el = time.perf_counter() - t_start
            print(f"  [{i+1}/{len(rows)}] ok={n_ok} skip={n_skip} err={n_err} "
                  f"elapsed={el:.0f}s rate={(n_ok+n_err)/max(el,1e-9):.2f}/s")
    fh.close()
    print(f"DONE ok={n_ok} skip={n_skip} err={n_err} -> {args.out}")


if __name__ == "__main__":
    main()
