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
        --n 300 --iters 30 --outdir datasets_pure_tpuv4
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
    """Per-call pure kernel time = mean dur of the dominant device span on the
    TPU:0 timeline (the executed HLO module, once per iteration). Taking the
    single largest-total-dur span name avoids double-counting nested fusion ops."""
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
    return sum(durs) / len(durs)          # mean device span per call


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
    kernel = kernel_us_from_trace(folder, iters)
    shutil.rmtree(folder, ignore_errors=True)        # keep disk bounded
    return kernel, wall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ops", nargs="+", required=True)
    p.add_argument("--n", type=int, default=300,
                   help="shapes/op (xprof is slower than the loop method; keep modest)")
    p.add_argument("--iters", type=int, default=30, help="executes traced per shape")
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--reps", type=int, default=15, help="wall-clock reps (median)")
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
        out = os.path.join(args.outdir, f"{op}_pure_dataset.csv")
        fh = open(out, "w", newline=""); w = csv.writer(fh)
        w.writerow(["d0", "d1", "d2", "size", "log2_size",
                    "latency_us", "kernel_us", "wall_us", "host_us"])
        t0 = time.perf_counter(); ok = 0
        for (d0, d1, d2) in shapes:
            try:
                k, wall = measure(kind, fn, d0, d1, d2,
                                  args.warmup, args.iters, args.reps)
                if k is None:
                    continue
                size = d0 * d1 * d2
                w.writerow([d0, d1, d2, size, f"{math.log2(size):.6f}",
                            f"{k:.6f}", f"{k:.6f}", f"{wall:.6f}", f"{wall - k:.6f}"])
                ok += 1
                if ok % 50 == 0:
                    fh.flush()
            except Exception:
                pass
        fh.close()
        print(f"  {op:12s} {ok}/{len(shapes)} rows  {time.perf_counter()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
