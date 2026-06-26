#!/usr/bin/env python3
"""Export LLM forward graphs to StableHLO MLIR for SCALE-Sim (f32, seq fixed, eager
attention so matmuls stay explicit). StableHLO is target-independent; export runs on
CPU and is identical to what XLA-TPU consumes.

f32 (uniform dtype): the graph is one dtype throughout, so it has none of the f16/bf16
softmax/GELU/LayerNorm f16<->f32 convert islands (gpt2: 5 converts vs 137 in f16).
SCALE-Sim is shape-only for non-compute ops and uses a fixed bf16 byte width for GEMM,
so the element dtype does not change any predicted shape or cycle count."""

import argparse
import os

os.environ.setdefault("PJRT_DEVICE", "CPU")
os.environ.setdefault("XLA_USE_SPMD", "0")

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from torch_xla.stablehlo import exported_program_to_stablehlo

MODELS = {
    "gpt2": "gpt2",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M",
}


class LogitsOnly(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        return self.model(input_ids=input_ids, use_cache=False).logits


def export_one(name, model_id, seq_len, out_dir):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, attn_implementation="eager").eval()
    vocab = model.config.vocab_size
    input_ids = torch.arange(seq_len, dtype=torch.long).remainder(vocab).reshape(1, seq_len)
    with torch.no_grad():
        ep = torch.export.export(LogitsOnly(model), (input_ids,))
    text = exported_program_to_stablehlo(ep).get_stablehlo_text()
    out_path = os.path.join(out_dir, f"{name}.stablehlo.mlir")
    with open(out_path, "w") as f:
        f.write(text)
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS) + ["all"])
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = p.parse_args()

    chosen = list(MODELS) if "all" in args.models else args.models
    os.makedirs(args.out_dir, exist_ok=True)
    for name in chosen:
        print(f"{name}: {export_one(name, MODELS[name], args.seq_len, args.out_dir)}")


if __name__ == "__main__":
    main()
