#!/usr/bin/env python3
"""
MEASURE the fixed per-forward overhead C (and the device-busy floor) directly,
instead of reading it off a whole-model regression intercept.

Method (matches the ground-truth methodology in devicetruth_worker.py):
  device-busy time = wall(run + wait_device_ops) - wall(run, no wait)   [ms]
We sweep a single compiled matmul from tiny to large work and fit
  device_us ~= A * cycles + C
The intercept C is the work-independent floor = fixed per-forward overhead.
We also report the absolute floor: device-busy time of the smallest matmul.

All on TPU via torch.compile(backend="openxla") so we time real device-busy
execution, not torch_xla host tracing.
"""
import os, sys, time
os.environ["PJRT_DEVICE"] = "TPU"; os.environ["XLA_USE_SPMD"] = "0"
import math
import numpy as np
import torch
import torch._dynamo
torch._dynamo.config.recompile_limit = 128   # allow a distinct executable per shape
torch._dynamo.config.accumulated_recompile_limit = 1024
import torch_xla.core.xla_model as xm

S = 128
dev = xm.xla_device()


def closed_form_cycles(M, N, K):
    return (2 * S + S + M - 2) * math.ceil(N / S) * math.ceil(K / S) - 1


def make_compiled(M, N, K):
    a = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
    b = torch.randn(K, N, dtype=torch.bfloat16, device=dev)
    f = torch.compile(lambda x, y: x @ y, backend="openxla")
    return f, a, b


def med_ms(fn, K=30):
    ts = []
    for _ in range(K):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    ts.sort(); return ts[len(ts) // 2] * 1e3


def device_us(M, N, K):
    f, a, b = make_compiled(M, N, K)
    def issue():
        with torch.no_grad(): f(a, b)
        xm.mark_step()
    def full(): issue(); xm.wait_device_ops()
    for _ in range(8): full()                  # warm + compile
    tf = med_ms(full)
    ti = med_ms(issue)                          # host-dispatch only (no wait)
    return max(tf - ti, 0.0) * 1e3, tf * 1e3    # (device_us, full_wall_us)


# sweep work from a 1x1x1 matmul up to large, several points per decade
SHAPES = [(1, 1, 1), (8, 8, 8), (32, 32, 32), (64, 64, 64), (128, 128, 128),
          (256, 256, 256), (512, 512, 512), (768, 768, 768), (1024, 1024, 1024),
          (1536, 1536, 1536), (2048, 2048, 2048)]


def make_chain(n, sz=128):
    """A graph of n sequential small matmuls -> n distinct MXU kernel launches."""
    ws = [torch.randn(sz, sz, dtype=torch.bfloat16, device=dev) for _ in range(n)]
    x0 = torch.randn(sz, sz, dtype=torch.bfloat16, device=dev)
    def body(x, ws):
        for w in ws:
            x = x @ w
        return x
    f = torch.compile(body, backend="openxla")
    return f, x0, ws


def chain_device_us(n):
    f, x0, ws = make_chain(n)
    def issue():
        with torch.no_grad(): f(x0, ws)
        xm.mark_step()
    def full(): issue(); xm.wait_device_ops()
    for _ in range(8): full()
    return max(med_ms(full) - med_ms(issue), 0.0) * 1e3


def main():
    print("=== (1) single-matmul work sweep: device-busy vs compute ===")
    print(f"{'M,N,K':>16} {'cycles':>10} {'device_us':>11} {'full_wall_us':>13}")
    cyc, dev_us = [], []
    for (M, N, K) in SHAPES:
        c = closed_form_cycles(M, N, K)
        d, fw = device_us(M, N, K)
        cyc.append(c); dev_us.append(d)
        print(f"{f'{M}x{N}x{K}':>16} {c:>10d} {d:>11.2f} {fw:>13.2f}")
    cyc = np.array(cyc, float); dev_us = np.array(dev_us, float)
    A, C = np.polyfit(cyc, dev_us, 1)
    print(f"  fit device_us = {A*1e3:.4f}e-3*cyc + C  ->  per-execute floor C0 = {C:.1f} us")
    print(f"  device-busy floor (1x1x1) = {dev_us[0]:.1f} us")
    print(f"  full-wall floor incl. host dispatch (1x1x1) = {dev_us[0]:.1f}+ "
          f"-> {device_us(1,1,1)[1]:.1f} us  (host = the non-TPU cost)")

    print("\n=== (2) N-op chain: does the floor scale with #kernels? ===")
    print(f"{'n_matmuls':>10} {'device_us':>11}")
    ns = [1, 2, 4, 8, 16, 32, 64, 128]
    ds = []
    for n in ns:
        d = chain_device_us(n); ds.append(d)
        print(f"{n:>10d} {d:>11.2f}")
    ns = np.array(ns, float); ds = np.array(ds, float)
    per_kernel, base = np.polyfit(ns, ds, 1)
    print(f"  fit device_us = {per_kernel:.3f}*n_kernels + {base:.1f}")
    print(f"  -> per-kernel floor = {per_kernel:.2f} us/op,  per-execute base = {base:.1f} us")

    print(f"\n=== insight ===")
    print(f"  whole-model regression C_forward (3 LLMs)   = 184.7 us")
    print(f"  measured per-execute floor (single matmul)  = {C:.1f} us")
    print(f"  measured per-kernel floor                   = {per_kernel:.2f} us/kernel")
    print(f"  => C is NOT one constant: ~{base:.0f}us base + ~{per_kernel:.1f}us x (#fused kernels)")


if __name__ == "__main__":
    main()
