#!/usr/bin/env python3
"""
Collect PURE device kernel latency for StableHLO ops via the xprof trace.

This is the trace-authoritative complement to `collect_ops_tpu.py`. That script
isolates kernel time by an in-JIT fori_loop and a wall-clock subtraction
`(t_K - t_1)/(K-1)` (it cancels the fixed host dispatch indirectly). This script
instead reads the **device timeline directly**: it compiles the op, runs it under
`jax.profiler.trace`, and sums the `dur` of the op's span on the TPU:0 device
stream -- the true on-device kernel time, with host/PJRT dispatch and the
host->device sync floor excluded by construction (not by subtraction).

For each op x shape it records three quantities:
    kernel_us : pure device kernel time   (xprof device-span, per call)
    wall_us   : python-timer execute+block (host + device), per call
    host_us   : wall_us - kernel_us       (per-op/per-execute non-TPU cost)
`latency_us` is set == kernel_us so the CSV is a drop-in for `train_ops.py`
(same [d0,d1,d2,size,log2_size,latency_us] schema; wall/host are extra columns).

Why both methods: the fori_loop method gives the *marginal* per-op cost (floor
removed) at high throughput; this xprof method gives the *true device span* of a
single kernel invocation (floor included) and, via host_us, a direct measurement
of the per-execute host cost we otherwise have to fit. Use the fori_loop method
to TRAIN the size-driven per-op models; use this one to AUDIT them and to pin the
host / floor constants (see SCALE-Sim_TPU/e2e_work/compensation/measure_xprof.py
and the whole-model compensation in scalesim/total_time_report.py).

Requirements: exclusive PJRT TPU access, `pip install jax[tpu]`, bf16.
Usage:
    PJRT_DEVICE=TPU python3 collect_pure_device_tpu.py --ops add multiply reduce \
        --n 1000 --outdir datasets_pure_tpuv4   (defaults: n=1000 iters=10 warmup=1 reps=1)
Outputs one `<op>_pure_dataset.csv` per op.
"""
import os
os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("XLA_USE_SPMD", "0")

import argparse, csv, glob, gzip, json, math, shutil, time
import jax, jax.numpy as jnp

from collect_ops_tpu import OPS, build, sample_shapes

TRACE_ROOT = "/tmp/xprof_collect"


def device_pid_tpu0(events):
    """PID of the TPU:0 device op stream, auto-detected from process_name meta
    (replaces the fragile hardcoded `pid == 3`). Core 0 only, so a single-device
    jit is not double-counted across cores."""
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "process_name":
            if "device:TPU:0" in e.get("args", {}).get("name", ""):
                return e["pid"]
    return None


def kernel_us_from_trace(folder, iters):
    """Per-call device times from the TPU:0 timeline. Returns (kernel_us,
    program_us): kernel = sum of the INNER op spans (`fusion`, copy, ...) = the
    op's real device work; program = the whole jit-module span = kernel + the
    fixed per-LAUNCH overhead. The outer `jit_*` wrapper NESTS the inner ops, so
    taking it as the kernel time double-counts the launch overhead (the bug that
    inflated tiny ops to ~14us). program-kernel ~= per-execute launch overhead."""
    files = []
    for root, _, fs in os.walk(folder):
        files += [os.path.join(root, f) for f in fs if f.endswith(".trace.json.gz")]
    if not files:
        return None, None
    data = json.load(gzip.open(max(files, key=os.path.getmtime), "rt"))
    events = data.get("traceEvents", [])
    pid = device_pid_tpu0(events)
    if pid is None:
        return None, None
    inner = prog = 0.0
    for e in events:
        if e.get("pid") == pid and e.get("ph") == "X" and "dur" in e:
            if e["name"].startswith("jit"):
                prog += e["dur"]
            else:
                inner += e["dur"]
    if inner == 0 and prog == 0:
        return None, None
    return inner / iters, prog / iters    # (kernel_us, program_us)


def measure(kind, fn, d0, d1, d2, warmup, iters, reps):
    """Compile the op once, time wall (python timer) and pure kernel (xprof)."""
    inputs, apply = build(kind, fn, d0, d1, d2)
    compiled = jax.jit(apply).lower(*inputs).compile()
    for _ in range(warmup):
        compiled(*inputs).block_until_ready()

    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); compiled(*inputs).block_until_ready()
        ts.append(time.perf_counter() - t0)
    ts.sort(); wall = ts[len(ts) // 2] * 1e6

    folder = os.path.join(TRACE_ROOT, f"{kind}_{d0}_{d1}_{d2}")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)
    with jax.profiler.trace(folder):
        for _ in range(iters):
            compiled(*inputs).block_until_ready()
    kernel, program = kernel_us_from_trace(folder, iters)
    shutil.rmtree(folder, ignore_errors=True)        # keep disk bounded
    return kernel, program, wall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ops", nargs="+", required=True)
    p.add_argument("--n", type=int, default=1000, help="shapes/op")
    p.add_argument("--iters", type=int, default=10,
                   help="executes traced per shape; kernel_us = mean device span over these")
    p.add_argument("--warmup", type=int, default=1, help="throwaway executes before timing")
    p.add_argument("--reps", type=int, default=1,
                   help="wall-clock reps for wall_us/host_us (audit only; not the kernel label)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default=os.path.dirname(__file__))
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    shapes = sample_shapes(args.n, args.seed)
    print(f"device={jax.devices()[0].device_kind}  shapes/op={len(shapes)}  "
          f"iters={args.iters}  ops={args.ops}")
    for op in args.ops:
        if op not in OPS:
            print(f"  SKIP unknown op {op}"); continue
        kind, fn = OPS[op]
        # named <op>_dataset.csv so train_ops.py's stem->op mapping works as-is
        # (separate --outdir keeps these pure-kernel sets distinct from the loop sets)
        out = os.path.join(args.outdir, f"{op}_dataset.csv")
        if os.path.exists(out) and sum(1 for _ in open(out)) > len(shapes) // 2:
            print(f"  {op:12s} already done ({out}), skipping"); continue
        fh = open(out, "w", newline=""); w = csv.writer(fh)
        w.writerow(["d0", "d1", "d2", "size", "log2_size",
                    "latency_us", "kernel_us", "program_us", "wall_us", "host_us"])
        t0 = time.perf_counter(); ok = 0
        for (d0, d1, d2) in shapes:
            try:
                k, program, wall = measure(kind, fn, d0, d1, d2,
                                           args.warmup, args.iters, args.reps)
                if k is None:
                    continue
                size = d0 * d1 * d2
                # latency_us == kernel_us (inner fusion span) for train_ops drop-in
                w.writerow([d0, d1, d2, size, f"{math.log2(size):.6f}",
                            f"{k:.6f}", f"{k:.6f}", f"{program:.6f}",
                            f"{wall:.6f}", f"{wall - program:.6f}"])
                ok += 1
                if ok % 50 == 0:
                    fh.flush()
            except Exception:
                pass
        fh.close()
        print(f"  {op:12s} {ok}/{len(shapes)} rows  {time.perf_counter()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
