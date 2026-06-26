#!/usr/bin/env python3
"""Measure the tiny_transformer whole-model DEVICE-BUSY latency on whatever TPU this
runs on (v4, v6e, ...). It is the small-model anchor for fit_compensation_pure.py
(pins C0 down). Same device-busy method as profile_model_on_tpu.py:
    device_busy = median(full = run + wait) - median(issue = run, no wait)
Pass the printed number to `fit_compensation_pure.py --tiny-truth <us>`.

Run on the target TPU (exclusive):  PJRT_DEVICE=TPU python3 measure_tiny_truth.py
"""
import os, sys, time
os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("XLA_USE_SPMD", "0")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "topologies", "stablehlo", "llm"))
import torch
import torch_xla.core.xla_model as xm
from export_tiny_transformer_pytorch import TinyTransformer, SEQ, VOCAB


def main(warmup=8, iters=30):
    dev = xm.xla_device()
    m = TinyTransformer().eval().to(dev)
    cm = torch.compile(m, backend="openxla")
    ids = torch.arange(SEQ, dtype=torch.long).remainder(VOCAB).reshape(1, SEQ).to(dev)

    def issue():
        with torch.no_grad():
            cm(ids)
        xm.mark_step()

    def full():
        issue(); xm.wait_device_ops()

    for _ in range(warmup):
        full()

    def med(fn):
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
        ts.sort(); return ts[len(ts) // 2] * 1e6        # us

    device_busy = max(med(full) - med(issue), 0.0)
    print(f"device={xm.xla_device_hw(dev)}  tiny_transformer device-busy = "
          f"{device_busy:.1f} us")
    print(f"-> pass to: fit_compensation_pure.py --tiny-truth {device_busy:.1f}")


if __name__ == "__main__":
    main()
