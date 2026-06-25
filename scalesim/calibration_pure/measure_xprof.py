#!/usr/bin/env python3
"""
Measure PURE device kernel latency via the xprof trace (jax.profiler), per the
approach the user supplied. Compiles a named kernel (jit().lower().compile() +
jax.named_call so the kernel is identifiable in the trace), profiles it, then
reads the device-timeline trace events and sums their `dur` = true on-device
kernel time (excludes host dispatch / Python / sync floor by construction).

Fixes the supplied code's hardcoded TPU `pid == 3`: the device-stream PID is
auto-detected from the trace's process_name metadata.

Outputs, per matmul shape:
  kernel_us : pure device kernel time (xprof dur, per call)
  wall_us   : python-timer wall of compiled execute+block (host+device)
  host_us   : wall - kernel  (the per-op non-TPU cost = candidate `c`)
and compares the pure-kernel floor to the wall-clock floor (my earlier ~65us).
"""
import os, glob, gzip, json, time, math, statistics, shutil
import numpy as np
import jax, jax.numpy as jnp

S = 128
ITERS = 50
TRACE_ROOT = "/tmp/xprof_kernel"


def closed_form_cycles(M, N, K):
    return (2 * S + S + M - 2) * math.ceil(N / S) * math.ceil(K / S) - 1


def compile_matmul(M, N, K):
    a = jax.random.uniform(jax.random.key(0), (M, K), jnp.bfloat16)
    b = jax.random.uniform(jax.random.key(1), (K, N), jnp.bfloat16)
    a.block_until_ready(); b.block_until_ready()
    fn = jax.jit(jax.named_call(lambda x, y: x @ y, name="compiled_kernel_function"))
    compiled = fn.lower(a, b).compile()
    return compiled, a, b


def device_pid_tpu0(events):
    """PID of the TPU:0 device op stream (auto-detected from process_name meta).
    Uses core 0 only so a single-device jit isn't double-counted across cores."""
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "process_name":
            nm = e.get("args", {}).get("name", "")
            if "device:TPU:0" in nm:
                return e["pid"]
    return None


def kernel_us_from_trace(folder):
    """Per-call pure device kernel time = mean dur of the compiled-module span on
    the TPU:0 device timeline (the top-level executed HLO module, repeated once
    per iteration). Picking the single dominant span name avoids double-counting
    nested fusion sub-ops."""
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
    # group device-timeline X-events by name; the module span dominates total dur
    by_name = {}
    for e in events:
        if e.get("pid") == pid and e.get("ph") == "X" and "dur" in e:
            by_name.setdefault(e["name"], []).append(e["dur"])
    if not by_name:
        return None
    name = max(by_name, key=lambda n: sum(by_name[n]))   # dominant device span
    durs = by_name[name]
    return sum(durs) / len(durs)      # per-call kernel time (mean over iterations)


def wall_us(compiled, a, b):
    for _ in range(8):
        compiled(a, b).block_until_ready()
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter(); compiled(a, b).block_until_ready()
        ts.append(time.perf_counter() - t0)
    ts.sort(); return ts[len(ts) // 2] * 1e6


def measure(M, N, K):
    compiled, a, b = compile_matmul(M, N, K)
    w = wall_us(compiled, a, b)
    folder = os.path.join(TRACE_ROOT, f"{M}_{N}_{K}")
    if os.path.exists(folder): shutil.rmtree(folder)
    os.makedirs(folder)
    with jax.profiler.trace(folder):
        for _ in range(ITERS):
            compiled(a, b).block_until_ready()
    k = kernel_us_from_trace(folder)
    return k, w


SHAPES = [(1, 1, 1), (32, 32, 32), (128, 128, 128), (256, 256, 256),
          (512, 512, 512), (768, 768, 768), (1024, 1024, 1024),
          (1536, 1536, 1536), (2048, 2048, 2048)]


def main():
    print(f"device: {jax.devices()[0].device_kind}, iters={ITERS}\n")
    print(f"{'M,N,K':>16}{'cycles':>10}{'kernel_us':>11}{'wall_us':>10}{'host_us':>9}")
    cyc, ker = [], []
    for (M, N, K) in SHAPES:
        k, w = measure(M, N, K)
        if k is None:
            print(f"{f'{M}x{N}x{K}':>16}  (no trace events)"); continue
        cyc.append(closed_form_cycles(M, N, K)); ker.append(k)
        print(f"{f'{M}x{N}x{K}':>16}{closed_form_cycles(M,N,K):>10d}"
              f"{k:>11.3f}{w:>10.1f}{w-k:>9.1f}")
    cyc = np.array(cyc, float); ker = np.array(ker, float)
    A, C = np.polyfit(cyc, ker, 1)
    print(f"\npure-kernel fit: kernel_us = {A*1e3:.4f}e-3*cyc + {C:.3f}")
    print(f"  PURE-KERNEL floor (intercept)   = {C:.3f} us")
    print(f"  PURE-KERNEL 1x1x1                = {ker[0]:.3f} us")
    print(f"  (compare: wall-clock full-issue floor measured earlier = ~65 us)")


if __name__ == "__main__":
    main()
