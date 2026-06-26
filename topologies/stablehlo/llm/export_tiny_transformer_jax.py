#!/usr/bin/env python3
"""Export a tiny synthetic transformer to StableHLO via JAX (same arch/dims as the
PyTorch export_tiny_transformer.py; 19 GEMMs + LayerNorm/softmax/GELU/embedding).
No torch needed; lowers on CPU."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax, jax.numpy as jnp

H, HEADS, FF, LAYERS, SEQ, VOCAB = 512, 4, 768, 3, 128, 2048
HD = H // HEADS


def layernorm(x, g, b, eps=1e-5):
    mu = jnp.mean(x, -1, keepdims=True)
    var = jnp.mean((x - mu) ** 2, -1, keepdims=True)
    return (x - mu) * jax.lax.rsqrt(var + eps) * g + b


def linear(x, w, b):
    return x @ w + b


def block(x, p):
    B, S, _ = x.shape
    h = layernorm(x, p["ln1_g"], p["ln1_b"])
    qkv = linear(h, p["qkv_w"], p["qkv_b"]).reshape(B, S, 3, HEADS, HD)
    q, k, v = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3))   # B,heads,S,HD
    att = jax.nn.softmax(q @ k.transpose(0, 1, 3, 2) * HD ** -0.5, -1)
    o = (att @ v).transpose(0, 2, 1, 3).reshape(B, S, H)
    x = x + linear(o, p["o_w"], p["o_b"])
    h = jax.nn.gelu(linear(layernorm(x, p["ln2_g"], p["ln2_b"]), p["fc1_w"], p["fc1_b"]))
    return x + linear(h, p["fc2_w"], p["fc2_b"])


DT = jnp.float32


def init():
    ks = iter(jax.random.split(jax.random.key(0), 2 + LAYERS * 4))
    rn = lambda *s: jax.random.normal(next(ks), s, DT) * 0.02
    one, zero = lambda: jnp.ones(H, DT), lambda: jnp.zeros(H, DT)
    p = {"embed": rn(VOCAB, H), "lnf_g": one(), "lnf_b": zero(),
         "head_w": rn(H, VOCAB), "blocks": []}
    for _ in range(LAYERS):
        p["blocks"].append({
            "ln1_g": one(), "ln1_b": zero(),
            "qkv_w": rn(H, 3 * H), "qkv_b": jnp.zeros(3 * H, DT),
            "o_w": rn(H, H), "o_b": zero(),
            "ln2_g": one(), "ln2_b": zero(),
            "fc1_w": rn(H, FF), "fc1_b": jnp.zeros(FF, DT),
            "fc2_w": rn(FF, H), "fc2_b": zero()})
    return p


def forward(p, ids):
    x = p["embed"][ids]
    for blk in p["blocks"]:
        x = block(x, blk)
    return layernorm(x, p["lnf_g"], p["lnf_b"]) @ p["head_w"]


def main():
    ids = jnp.arange(SEQ, dtype=jnp.int32).reshape(1, SEQ) % VOCAB
    # params passed as args (not closed-over) so weights stay symbolic, not inlined as MB constants
    text = jax.jit(forward).lower(init(), ids).as_text()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny_transformer_jax.stablehlo.mlir")
    open(out, "w").write(text)
    print(out)


if __name__ == "__main__":
    main()
