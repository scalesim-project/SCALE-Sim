#!/usr/bin/env python3
"""
JAX version of the tiny-transformer StableHLO export (mirrors the PyTorch
export_tiny_transformer.py). Same architecture and dims so the GEMM shapes +
non-compute ops match: hidden=512, 4 heads, ff=768, 3 layers, seq=128, vocab=2048.
Includes the full LLM op set: embedding (gather), LayerNorm, softmax, GELU,
residual adds, linear projections.

JAX lowers to StableHLO natively -- no torch/torch_xla needed. Runs on CPU.
Output: tiny_transformer_jax.stablehlo.mlir
"""
import os, re
os.environ.setdefault("JAX_PLATFORMS", "cpu")   # device-independent lowering
import jax, jax.numpy as jnp

H, HEADS, FF, LAYERS, SEQ, VOCAB = 512, 4, 768, 3, 128, 2048
HD = H // HEADS


def layernorm(x, g, b, eps=1e-5):
    mu = jnp.mean(x, -1, keepdims=True)
    var = jnp.mean((x - mu) ** 2, -1, keepdims=True)
    return (x - mu) * jax.lax.rsqrt(var + eps) * g + b


def linear(x, w, b=None):
    y = x @ w
    return y + b if b is not None else y


def block(x, p):
    B, S, _ = x.shape
    # --- attention ---
    h = layernorm(x, p["ln1_g"], p["ln1_b"])
    qkv = linear(h, p["qkv_w"], p["qkv_b"]).reshape(B, S, 3, HEADS, HD)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]            # B,S,heads,HD
    q = q.transpose(0, 2, 1, 3); k = k.transpose(0, 2, 1, 3); v = v.transpose(0, 2, 1, 3)
    att = (q @ k.transpose(0, 1, 3, 2)) * (HD ** -0.5)           # batched dot_general
    att = jax.nn.softmax(att, -1)                                # exp / reduce / divide
    o = (att @ v).transpose(0, 2, 1, 3).reshape(B, S, H)
    x = x + linear(o, p["o_w"], p["o_b"])                        # residual add
    # --- MLP ---
    h = layernorm(x, p["ln2_g"], p["ln2_b"])
    h = jax.nn.gelu(linear(h, p["fc1_w"], p["fc1_b"]))           # GELU
    x = x + linear(h, p["fc2_w"], p["fc2_b"])                    # residual add
    return x


def init():
    key = jax.random.key(0)
    def rn(k, *s): return jax.random.normal(k, s, jnp.float32) * 0.02
    ks = jax.random.split(key, 4 + LAYERS * 9)
    i = iter(ks)
    p = {"embed": rn(next(i), VOCAB, H),
         "lnf_g": jnp.ones(H), "lnf_b": jnp.zeros(H),
         "head_w": rn(next(i), H, VOCAB), "blocks": []}
    for _ in range(LAYERS):
        p["blocks"].append({
            "ln1_g": jnp.ones(H), "ln1_b": jnp.zeros(H),
            "qkv_w": rn(next(i), H, 3 * H), "qkv_b": jnp.zeros(3 * H),
            "o_w": rn(next(i), H, H), "o_b": jnp.zeros(H),
            "ln2_g": jnp.ones(H), "ln2_b": jnp.zeros(H),
            "fc1_w": rn(next(i), H, FF), "fc1_b": jnp.zeros(FF),
            "fc2_w": rn(next(i), FF, H), "fc2_b": jnp.zeros(H)})
    return p


def main():
    p = init()
    ids = jnp.arange(SEQ, dtype=jnp.int32).reshape(1, SEQ) % VOCAB

    # params passed as an ARGUMENT (not closed-over) so weights stay symbolic func
    # args in the MLIR instead of being inlined as multi-MB constants.
    def forward(p, ids):
        x = p["embed"][ids]                                      # gather (embedding)
        for blk in p["blocks"]:
            x = block(x, blk)
        x = layernorm(x, p["lnf_g"], p["lnf_b"])
        return x @ p["head_w"]                                   # LM head

    text = jax.jit(forward).lower(p, ids).as_text()              # -> StableHLO MLIR
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "tiny_transformer_jax.stablehlo.mlir")
    open(out, "w").write(text)
    print(f"wrote {out}")
    print(f"  dot_general={len(re.findall('stablehlo.dot_general', text))}  "
          f"exp={len(re.findall('stablehlo.exponential', text))}  "
          f"rsqrt={len(re.findall('stablehlo.rsqrt', text))}  "
          f"add={len(re.findall('stablehlo.add', text))}  "
          f"reduce={len(re.findall('stablehlo.reduce', text))}  "
          f"gather={len(re.findall('stablehlo.gather', text))}")


if __name__ == "__main__":
    main()
