#!/usr/bin/env python3
"""Measure real whole-model DEVICE-BUSY latency of the LLMs on a TPU (bf16, eager
attention), to validate SCALE-Sim's predicted TIME_REPORT. Needs the TPU free.

Uses torch.compile(backend="openxla") so the graph is traced/compiled ONCE and then
re-executes the cached executable -- otherwise plain eager torch_xla re-traces every
call and the wall-clock is ~99% host tracing (e.g. gpt2 reads ~19 ms, not its ~0.5 ms
device time). Device-busy is isolated as median(full) - median(issue):
    full()  = run + wait_device_ops()   (host dispatch + device)
    issue() = run, NO wait              (host dispatch only)
Output: measured_<gen>.json next to this file."""

import os
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["XLA_USE_SPMD"] = "0"

import argparse, json, time
from pathlib import Path
import torch, torch_xla
import torch_xla.core.xla_model as xm
from transformers import AutoModelForCausalLM

MODELS = {
    "gpt2": "gpt2",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M",
}


def median_ms(fn, iters):
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2] * 1e3


def run_one(model_id, seq_len, warmup, iters, dev):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="eager").eval().to(dev)
    ids = (torch.arange(seq_len, dtype=torch.long)
           .remainder(model.config.vocab_size).reshape(1, seq_len).to(dev))
    compiled = torch.compile(model, backend="openxla")

    def issue():
        with torch.no_grad():
            compiled(input_ids=ids, use_cache=False)
        xm.mark_step()

    def full():
        issue(); xm.wait_device_ops()

    for _ in range(warmup):
        full()
    return max(median_ms(full, iters) - median_ms(issue, iters), 0.0)   # device-busy ms


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS) + ["all"])
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--gen", default="tpu_v4")
    p.add_argument("--warmup", type=int, default=6)
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()

    chosen = list(MODELS) if "all" in args.models else args.models
    dev = torch_xla.device()
    out = {}
    for name in chosen:
        device_us = run_one(MODELS[name], args.seq_len, args.warmup, args.iters, dev) * 1e3
        out[name] = round(device_us, 1)
        print(f"{name}: {device_us:.1f} us")
    path = Path(__file__).resolve().parent / f"measured_{args.gen}.json"
    json.dump({"device_busy_us": out}, open(path, "w"), indent=2)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
