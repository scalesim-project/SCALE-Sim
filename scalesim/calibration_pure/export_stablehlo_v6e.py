#!/usr/bin/env python3
"""
Export a HF causal-LM forward to StableHLO MLIR at a given (model, seq), batch=1,
bf16, eager attention -- matching how topologies/stablehlo/llm/*.mlir were made,
so the bypass can compute per-(model,seq) op sums for the whole-model fit.

Usage: python3 export_stablehlo_v6e.py <model> <seq> <out.mlir>
Run with venv_xla python (torch + torch_xla + transformers).
"""
import os, sys
os.environ.setdefault("PJRT_DEVICE", "CPU")   # export is graph capture; no TPU needed
import torch
from transformers import AutoModelForCausalLM
from torch_xla.stablehlo import exported_program_to_stablehlo

MODELS = {"gpt2": "gpt2", "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
          "smollm2-135m": "HuggingFaceTB/SmolLM2-135M"}


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, ids):
        return self.m(input_ids=ids, use_cache=False).logits


def main():
    name, seq, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    mid = MODELS[name]
    # Export in float32: torch_xla's StableHLO state_dict export can't convert bf16
    # to numpy, and the GRAPH STRUCTURE / SHAPES (all the bypass needs -- M,N,K per
    # GEMM, op shapes) are identical across dtype. The v6e timing model applies the
    # bf16 cost to these shapes regardless of the export dtype.
    m = AutoModelForCausalLM.from_pretrained(
        mid, dtype=torch.float32, attn_implementation="eager").eval()
    vocab = m.config.vocab_size
    ids = torch.arange(seq, dtype=torch.long).remainder(vocab).reshape(1, seq)
    ep = torch.export.export(Wrap(m), (ids,), strict=False)
    shlo = exported_program_to_stablehlo(ep)
    text = shlo.get_stablehlo_text()
    with open(out, "w") as f:
        f.write(text)
    print(f"{name} seq{seq}: wrote {out} ({len(text)} chars)")


if __name__ == "__main__":
    main()
