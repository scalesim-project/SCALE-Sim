#!/usr/bin/env python3
"""
torch.compile device-busy ground truth for the tiny_transformer (the small-model
anchor for the size-dependent C_forward fit), measured the SAME way as
devicetruth_worker.py. Prints a calib-style line: model,seq,batch,vocab,device_ms,...
Run with venv_xla python on the TPU.
"""
import os, sys, time
os.environ["PJRT_DEVICE"] = "TPU"; os.environ["XLA_USE_SPMD"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "topologies", "stablehlo", "llm"))
import torch
import torch_xla.core.xla_model as xm
from export_tiny_transformer_pytorch import TinyTransformer, SEQ, VOCAB

m = TinyTransformer().eval()
dev = xm.xla_device(); m = m.to(dev)
cm = torch.compile(m, backend="openxla")
ids = torch.arange(SEQ, dtype=torch.long).remainder(VOCAB).reshape(1, SEQ).to(dev)


def issue():
    with torch.no_grad():
        cm(ids)
    xm.mark_step()


def full():
    issue(); xm.wait_device_ops()


for _ in range(6):
    full()


def med(fn, K=20):
    ts = []
    for _ in range(K):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    ts.sort(); return ts[len(ts) // 2] * 1e3


tf = med(full)
print(f"tiny_transformer,{SEQ},1,{VOCAB},{tf:.4f},,ok")
