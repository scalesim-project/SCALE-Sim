#!/usr/bin/env python3
"""
Export a TINY synthetic transformer to StableHLO -- small enough to run in SCALE-Sim's
FULL cycle-accurate sim (<2 min), but with the real LLM non-compute ops so the per-op
models are exercised: LayerNorm, softmax, GELU, residual adds, embedding.
dims: hidden=512, 4 heads, ff=768, 3 layers, seq=128, vocab=2048 (f32; shapes are
dtype-independent).  Run on CPU (no TPU needed for export).
"""
import os, re
import torch, torch.nn as nn
from torch_xla.stablehlo import exported_program_to_stablehlo

H, HEADS, FF, LAYERS, SEQ, VOCAB = 512, 4, 768, 3, 128, 2048


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.hd = H // HEADS
        self.qkv = nn.Linear(H, 3 * H)
        self.o = nn.Linear(H, H)

    def forward(self, x):
        B, S, _ = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, HEADS, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                 # B, heads, S, hd
        att = (q @ k.transpose(-2, -1)) * (self.hd ** -0.5)   # batched dot_general
        att = att.softmax(-1)                            # exp / reduce / divide
        out = (att @ v).transpose(1, 2).reshape(B, S, H)
        return self.o(out)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(H); self.attn = Attention()
        self.ln2 = nn.LayerNorm(H)
        self.fc1 = nn.Linear(H, FF); self.act = nn.GELU(); self.fc2 = nn.Linear(FF, H)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))                   # residual add
        x = x + self.fc2(self.act(self.fc1(self.ln2(x))))  # LN -> GEMM -> GELU -> GEMM -> add
        return x


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, H)              # gather
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.ln_f = nn.LayerNorm(H)
        self.head = nn.Linear(H, VOCAB, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))


def main():
    m = TinyTransformer().eval()
    ids = torch.arange(SEQ, dtype=torch.long).remainder(VOCAB).reshape(1, SEQ)
    with torch.no_grad():
        ep = torch.export.export(m, (ids,))
    text = exported_program_to_stablehlo(ep).get_stablehlo_text()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "tiny_transformer.stablehlo.mlir")
    open(out, "w").write(text)
    print(f"wrote {out}")
    print(f"  dot_general={len(re.findall('stablehlo.dot_general', text))}  "
          f"layernorm/batch_norm={len(re.findall('batch_norm', text))}  "
          f"exp={len(re.findall('stablehlo.exponential', text))}  "
          f"add={len(re.findall('stablehlo.add', text))}  "
          f"tanh={len(re.findall('stablehlo.tanh', text))}  "
          f"reduce={len(re.findall('stablehlo.reduce', text))}")


if __name__ == "__main__":
    main()
