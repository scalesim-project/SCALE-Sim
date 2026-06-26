#!/usr/bin/env python3
"""
Collect TPU v4 device-time latency for non-compute StableHLO ops, to train the
per-op latency models that SCALE-Sim's NonComputeLatencyPredictor uses.

Same recipe as the shipped add/sub/mul/max/min models: features
[d0,d1,d2,size,log2_size] from the cost-driving (largest) tensor, label = µs.

Measurement: in-JIT fori_loop (one dispatch, on-device) with index-perturbed
first operand so XLA can't hoist the op; device time = (t_K - t_1)/(K-1).
This isolates kernel time from Python/host dispatch (validated on GEMM: the naive
per-call method read a flat ~55us dispatch floor; this reads true ~1us).
"""
import os
os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("XLA_USE_SPMD", "0")

import argparse, csv, math, time
import jax, jax.numpy as jnp

BF16 = jnp.bfloat16

# op -> (kind, fn). kind decides how inputs/output are built from a primary
# (cost-driver) 3D shape (d0,d1,d2) = the LARGEST tensor involved.
OPS = {
    # binary elementwise (two same-shape inputs)
    "add":      ("binary", lambda x, y: x + y),
    "subtract": ("binary", lambda x, y: x - y),
    "multiply": ("binary", lambda x, y: x * y),
    "divide":   ("binary", lambda x, y: x / y),
    "maximum":  ("binary", jnp.maximum),
    "minimum":  ("binary", jnp.minimum),
    # unary elementwise (output same shape as input = driver)
    "negate":      ("unary", lambda x: -x),
    "rsqrt":       ("unary", jax.lax.rsqrt),
    "exponential": ("unary", jnp.exp),
    "logistic":    ("unary", jax.nn.sigmoid),
    "tanh":        ("unary", jnp.tanh),
    "power":       ("unary", lambda x: x * x * x),   # x**3 (GELU-style)
    "sine":        ("unary", jnp.sin),
    "cosine":      ("unary", jnp.cos),
    "convert":     ("unary", lambda x: x.astype(jnp.float32)),  # bf16 -> f32 cast
    # elementwise binary/ternary returning a value we can reduce to a scalar carry
    "compare":     ("binary", lambda x, y: (x > y).astype(BF16)),
    "and":         ("binary", lambda x, y: jnp.logical_and(x > 0, y > 0).astype(BF16)),
    "select":      ("ternary", None),             # where(cond, x, y); built below
    # composite normalization (StableHLO batch_norm_training == LLM LayerNorm)
    "batch_norm_training": ("batchnorm", None),   # built below
    # shape / reduction (driver = larger of in/out, set below)
    "reduce":      ("reduce", None),       # sum over last axis: (d0,d1,d2)->(d0,d1)
    "slice":       ("slice", None),        # (d0,d1,d2)->(d0,d1,d2//2)
    "transpose":   ("transpose", None),    # (d0,d1,d2)->(d0,d2,d1)
    "reshape":     ("reshape", None),      # (d0,d1,d2)->(d0,d1*d2)
    "broadcast":   ("broadcast", None),    # (d0,1,d2)->(d0,d1,d2)  (driver=output)
    "concatenate": ("concat", None),       # 2x(d0,d1,d2//2)->(d0,d1,d2) (driver=out)
}


def build(kind, fn, d0, d1, d2):
    """Return (inputs_tuple, apply_fn). Primary shape (d0,d1,d2) = cost driver."""
    ones = lambda *s: jnp.ones(s, BF16)
    if kind == "binary":
        return (ones(d0, d1, d2), ones(d0, d1, d2)), (lambda x, y: fn(x, y))
    if kind == "unary":
        return (ones(d0, d1, d2),), (lambda x: fn(x))
    if kind == "reduce":
        return (ones(d0, d1, d2),), (lambda x: jnp.sum(x, axis=-1))
    if kind == "slice":
        h = max(1, d2 // 2)
        return (ones(d0, d1, d2),), (lambda x: x[:, :, :h])
    if kind == "transpose":
        return (ones(d0, d1, d2),), (lambda x: jnp.transpose(x, (0, 2, 1)))
    if kind == "reshape":
        return (ones(d0, d1, d2),), (lambda x: jnp.reshape(x, (d0, d1 * d2)))
    if kind == "broadcast":
        return (ones(d0, 1, d2),), (lambda x: jnp.broadcast_to(x, (d0, d1, d2)))
    if kind == "concat":
        h = max(1, d2 // 2)
        return (ones(d0, d1, h), ones(d0, d1, d2 - h)), \
               (lambda x, y: jnp.concatenate([x, y], axis=-1))
    if kind == "ternary":   # select(cond, x, y); first input (cond) drives shape
        return (jnp.ones((d0, d1, d2), bool), ones(d0, d1, d2), ones(d0, d1, d2)), \
               (lambda c, x, y: jnp.where(c, x, y))
    if kind == "batchnorm":  # LayerNorm over the last axis (== stablehlo.batch_norm_training)
        return (ones(d0, d1, d2),), \
               (lambda x: (x - jnp.mean(x, -1, keepdims=True))
                          * jax.lax.rsqrt(jnp.var(x, -1, keepdims=True) + 1e-5))
    raise ValueError(kind)


# Pure carry-chain bodies: ONE op per iteration, carry-dependent (no hoist),
# no extra add/perturb. y starts at 1.001 (avoids identity/fixed-point folding).
CHAIN = {
    "add":      lambda y, c: y + c,
    "subtract": lambda y, c: y - c,
    "multiply": lambda y, c: y * c,
    "divide":   lambda y, c: y / c,
    "maximum":  lambda y, c: jnp.maximum(y, c),
    "minimum":  lambda y, c: jnp.minimum(y, c),
    "negate":      lambda y, c: -y,
    "rsqrt":       lambda y, c: jax.lax.rsqrt(y),
    "exponential": lambda y, c: jnp.exp(y),
    "logistic":    lambda y, c: jax.nn.sigmoid(y),
    "tanh":        lambda y, c: jnp.tanh(y),
    "power":       lambda y, c: y * y * y,
    # reduce over last axis + broadcast back (dominant cost = the reduction)
    "reduce":   lambda y, c: y + jnp.sum(y, axis=-1, keepdims=True),
}


def measure_chain(op, d0, d1, d2, warmup, reps):
    body_fn = CHAIN[op]
    y0 = jnp.full((d0, d1, d2), 1.001, BF16)
    c = jnp.full((d0, d1, d2), 1.0009766, BF16)  # ~1, bf16-representable

    def run_K(y, c, K):
        return jax.lax.fori_loop(0, K, lambda i, acc: body_fn(acc, c), y)
    g = jax.jit(run_K, static_argnums=2)

    numel = max(d0 * d1 * d2, 1)
    K = int(min(4000, max(200, 5e8 // numel)))
    for _ in range(warmup):
        g(y0, c, K).block_until_ready()

    def t_loop(KK):
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); g(y0, c, KK).block_until_ready()
            ts.append(time.perf_counter() - t0)
        ts.sort(); return ts[len(ts) // 2]
    tK, t1 = t_loop(K), t_loop(max(1, K // 10))
    K1 = max(1, K // 10)
    return (tK - t1) / (K - K1) * 1e6, K  # us/op (subtract cancels fixed dispatch)


def measure(kind, fn, d0, d1, d2, warmup, reps):
    """Scalar-accumulate, in-JIT loop. Each iter: do the op on an index-perturbed
    input and reduce its output to a scalar carry. These ops are memory-bound, so
    this measures bytes-moved cost at the right scale (op + output read), with no
    huge full-tensor accumulator. Device-only loop => isolates from host dispatch.
    Subtracting the K1-loop time cancels the fixed per-dispatch overhead."""
    inputs, apply = build(kind, fn, d0, d1, d2)
    in0 = inputs[0]; rest = inputs[1:]

    def run_K(args, K):
        a0 = args[0]; r = args[1:]
        def body(i, s):
            p = a0 + (i % 16).astype(BF16)
            return s + jnp.sum(apply(p, *r)).astype(jnp.float32)
        return jax.lax.fori_loop(0, K, body, jnp.float32(0))
    g = jax.jit(run_K, static_argnums=1)

    numel = max(d0 * d1 * d2, 1)
    K = int(min(20000, max(500, 5e8 // numel)))
    K1 = max(1, K // 8)
    for _ in range(warmup):
        g(inputs, K).block_until_ready()

    def t_loop(KK):
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); g(inputs, KK).block_until_ready()
            ts.append(time.perf_counter() - t0)
        return min(ts)  # min = least-interfered estimate
    return (t_loop(K) - t_loop(K1)) / (K - K1) * 1e6, K  # us/op


def sample_shapes(n, seed, max_numel=4e7, llm_frac=0.6):
    """Distinct 3-D shapes for op-latency sampling, in two buckets so the models are
    sharp for LLMs (the priority -- the only regime validated against whole-model
    truth) yet degrade gracefully off-distribution:

      llm_frac  (default 60%)  LLM-anchored: small batch/head d0, a seq-length menu
                d1, and d2 log-uniform up to 160k -- covers hidden -> FFN -> VOCAB
                (the big dim in LLMs is the vocab/logits axis, e.g. gpt2 50k,
                qwen 152k, llama 128k; cap 160k so qwen/llama are in-distribution,
                not extrapolated).
      1-llm_frac (40%)  broad/general: wide log-uniform d0/d1/d2 with varied
                aspect ratios, so CNN/ViT/diffusion/arbitrary shapes are covered
                enough to avoid wild extrapolation.

    Note: the models are shape-only (features [d0,d1,d2,size,log2_size], no axis/
    permutation feature). Elementwise ops generalize by total size, so the broad
    bucket helps them directly; layout/axis ops (transpose/reduce/broadcast/slice)
    still bake in the sampled config -- broad coverage limits, not eliminates, their
    off-LLM error (a real fix needs an axis feature)."""
    import random
    rng = random.Random(seed)
    n_llm = int(round(n * llm_frac))
    shapes = set()
    # --- LLM-anchored bucket (60%) ---
    d1_llm = [1, 8, 16, 64, 128, 256, 512, 1024]
    while len(shapes) < n_llm:
        d0 = rng.choice([1, 1, 1, 2, 4, 8, 12, 14, 16, 32])
        d1 = rng.choice(d1_llm)
        d2 = int(round(math.exp(rng.uniform(math.log(8), math.log(160000)))))  # ..vocab
        if d2 >= 2 and d0 * d1 * d2 <= max_numel:
            shapes.add((d0, d1, d2))
    # --- broad / general bucket (40%) ---
    while len(shapes) < n:
        d0 = int(round(math.exp(rng.uniform(0.0, math.log(2048)))))   # 1 .. 2048
        d1 = int(round(math.exp(rng.uniform(0.0, math.log(4096)))))   # 1 .. 4096
        d2 = int(round(math.exp(rng.uniform(0.0, math.log(50000)))))  # 1 .. 50000
        if d2 >= 2 and d0 * d1 * d2 <= max_numel:
            shapes.add((d0, d1, d2))
    return sorted(shapes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ops", nargs="+", required=True)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default=os.path.dirname(__file__))
    args = p.parse_args()

    shapes = sample_shapes(args.n, args.seed)
    print(f"device={jax.devices()[0]}  shapes/op={len(shapes)}  ops={args.ops}")
    for op in args.ops:
        if op not in OPS:
            print(f"  SKIP unknown op {op}"); continue
        kind, fn = OPS[op]
        out = os.path.join(args.outdir, f"{op}_dataset.csv")
        fh = open(out, "w", newline=""); w = csv.writer(fh)
        w.writerow(["d0", "d1", "d2", "size", "log2_size", "latency_us"])
        t0 = time.perf_counter(); ok = 0
        for (d0, d1, d2) in shapes:
            try:
                lat, _ = measure(kind, fn, d0, d1, d2, args.warmup, args.reps)
                size = d0 * d1 * d2
                w.writerow([d0, d1, d2, size, f"{math.log2(size):.6f}", f"{lat:.6f}"])
                ok += 1
            except Exception as e:
                pass
            if ok % 100 == 0 and ok:
                fh.flush()
        fh.close()
        print(f"  {op:12s} {ok}/{len(shapes)} rows  {time.perf_counter()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
