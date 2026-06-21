#!/usr/bin/env python3
"""
Measure REAL whole-model latency of the LLMs on a TPU, to validate SCALE-Sim's
predicted TIME_REPORT against hardware.

Robust steady-state: warm up (absorb XLA compile), then time the compiled forward
over many iterations, syncing with wait_device_ops() (NOT .cpu(), which would add
a large logits D2H transfer). Reports median + min ms. At whole-model scale the
~55us host-dispatch floor is <1%, so median wall-clock is a faithful device-busy
time and is the quantity SCALE-Sim's per-op device times sum toward.

Output: measured_<gen>.json next to this file (default gen = tpu_v4). Re-run on a
different TPU and pass --gen to record that generation's reference numbers.

Deps (different from the simulator!): torch, torch_xla, transformers. Needs the
TPU free (exclusive PJRT access).
"""
import os
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["XLA_USE_SPMD"] = "0"

import argparse, json, time
from pathlib import Path
import torch, torch_xla
import torch_xla.core.xla_model as xm
from transformers import AutoModelForCausalLM

MODELS = {"gpt2": "gpt2", "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
          "smollm2-135m": "HuggingFaceTB/SmolLM2-135M"}


def run_one(name, model_id, seq_len, batch, dev, warmup, iters):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="eager").eval().to(dev)
    vocab = model.config.vocab_size
    input_ids = (torch.arange(batch * seq_len, dtype=torch.long)
                 .remainder(vocab).reshape(batch, seq_len).to(dev))

    def forward():
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
        xm.mark_step()
        xm.wait_device_ops()
    for _ in range(warmup):
        forward()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); forward(); ts.append(time.perf_counter() - t0)
    ts.sort()
    med, mn = ts[len(ts) // 2] * 1e3, ts[0] * 1e3
    print(f"  {name:14s} median={med:.3f}ms  min={mn:.3f}ms  (n={iters})")
    return {"median_ms": med, "min_ms": mn, "median_us": med * 1e3}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--gen", default="tpu_v4", help="TPU generation tag for the output filename")
    args = p.parse_args()

    dev = torch_xla.device()
    print("TPU device:", dev, " gen:", args.gen)
    out = {"_meta": {"tpu_generation": args.gen, "dtype": "bfloat16",
                     "seq_len": args.seq_len, "batch": args.batch,
                     "attn": "eager", "metric": "median wall-clock of compiled forward "
                     "(device-busy; wait_device_ops, no logits D2H)",
                     "iters": args.iters, "warmup": args.warmup}}
    for name in args.models:
        out[name] = run_one(name, MODELS[name], args.seq_len, args.batch,
                            dev, args.warmup, args.iters)
    path = Path(__file__).resolve().parent / f"measured_{args.gen}.json"
    json.dump(out, open(path, "w"), indent=2)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
