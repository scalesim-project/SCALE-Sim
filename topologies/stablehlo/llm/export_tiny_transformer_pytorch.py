#!/usr/bin/env python3
"""Export a tiny synthetic transformer (LayerNorm/softmax/GELU/residual/embedding) to
StableHLO via PyTorch -- small enough for SCALE-Sim's full cycle-accurate sim. CPU."""
import os
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
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1) * self.hd ** -0.5).softmax(-1)
        return self.o((att @ v).transpose(1, 2).reshape(B, S, H))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(H); self.attn = Attention()
        self.ln2 = nn.LayerNorm(H)
        self.fc1 = nn.Linear(H, FF); self.act = nn.GELU(); self.fc2 = nn.Linear(FF, H)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.fc2(self.act(self.fc1(self.ln2(x))))


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, H)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.ln_f = nn.LayerNorm(H)
        self.head = nn.Linear(H, VOCAB, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))


def main():
    ids = torch.arange(SEQ, dtype=torch.long).remainder(VOCAB).reshape(1, SEQ)
    with torch.no_grad():
        ep = torch.export.export(TinyTransformer().eval().half(), (ids,))   # f16
    text = exported_program_to_stablehlo(ep).get_stablehlo_text()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny_transformer_pytorch.stablehlo.mlir")
    open(out, "w").write(text)
    print(out)


if __name__ == "__main__":
    main()
