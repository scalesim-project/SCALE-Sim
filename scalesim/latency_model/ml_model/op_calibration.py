#!/usr/bin/env python3
"""
Non-compute op latency-model calibration — one script, three stages:

  collect         pure device kernel latency per (op, shape) via xprof   (TPU)
                  -> <outdir>/<op>_dataset.csv
  train           fit one HistGradientBoostingRegressor per op           (CPU)
                  -> model/<gen>/<op>.pkl
  reshape-median  override the reshape model with a constant = median    (CPU)
                  (reshape isn't shape-predictable; see note in that stage)

"Pure device" = the op's kernel span on the TPU:0 xprof timeline (sum of the inner op
spans, EXCLUDING the outer `jit_*` wrapper = the per-launch floor). bf16. This replaces
the old loop-method `(t_K-t_1)/(K-1)` collector; only the pure-device signal is kept.

Usage:
  PJRT_DEVICE=TPU python3 op_calibration.py collect --ops add multiply reduce ... --outdir datasets_<gen>
  python3 op_calibration.py train --datadir datasets_<gen> --outdir ../<gen>
  python3 op_calibration.py reshape-median --model-dir ../<gen> --dataset datasets_<gen>/reshape_dataset.csv
"""
import argparse, csv, glob, math, os, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["d0", "d1", "d2", "size", "log2_size"]
# dataset filename stem -> StableHLO op last-token (predictor auto-matches this)
NAME = {"broadcast": "broadcast_in_dim"}


def sample_shapes(n, seed, max_numel=4e7, llm_frac=0.6):
    """Distinct 3-D shapes, two buckets: 60% LLM-anchored (small batch/head d0, a
    seq-length menu d1, d2 log-uniform up to 160k = hidden->FFN->vocab) + 40% broad
    (wide log-uniform d0/d1/d2) so non-LLM shapes degrade gracefully. Shape-only
    models, so layout/axis ops still bake in the sampled config."""
    import random
    rng = random.Random(seed)
    n_llm = int(round(n * llm_frac))
    shapes = set()
    d1_llm = [1, 8, 16, 64, 128, 256, 512, 1024]
    while len(shapes) < n_llm:                                   # LLM-anchored
        d0 = rng.choice([1, 1, 1, 2, 4, 8, 12, 14, 16, 32])
        d1 = rng.choice(d1_llm)
        d2 = int(round(math.exp(rng.uniform(math.log(8), math.log(160000)))))
        if d2 >= 2 and d0 * d1 * d2 <= max_numel:
            shapes.add((d0, d1, d2))
    while len(shapes) < n:                                       # broad / general
        d0 = int(round(math.exp(rng.uniform(0.0, math.log(2048)))))
        d1 = int(round(math.exp(rng.uniform(0.0, math.log(4096)))))
        d2 = int(round(math.exp(rng.uniform(0.0, math.log(50000)))))
        if d2 >= 2 and d0 * d1 * d2 <= max_numel:
            shapes.add((d0, d1, d2))
    return sorted(shapes)


# ===================================================================== collect (TPU)
TRACE_ROOT = "/tmp/xprof_ops"


def _build_ops():
    """op -> (kind, fn). kind decides how inputs/output are built from a primary
    (cost-driver) 3D shape. Imports jax lazily so train/reshape-median need no TPU."""
    import jax, jax.numpy as jnp
    BF16 = jnp.bfloat16
    OPS = {
        "add": ("binary", lambda x, y: x + y), "subtract": ("binary", lambda x, y: x - y),
        "multiply": ("binary", lambda x, y: x * y), "divide": ("binary", lambda x, y: x / y),
        "maximum": ("binary", jnp.maximum), "minimum": ("binary", jnp.minimum),
        "negate": ("unary", lambda x: -x), "rsqrt": ("unary", jax.lax.rsqrt),
        "exponential": ("unary", jnp.exp), "logistic": ("unary", jax.nn.sigmoid),
        "tanh": ("unary", jnp.tanh), "power": ("unary", lambda x: x * x * x),
        "sine": ("unary", jnp.sin), "cosine": ("unary", jnp.cos),
        "convert": ("unary", lambda x: x.astype(jnp.float32)),
        "compare": ("binary", lambda x, y: (x > y).astype(BF16)),
        "and": ("binary", lambda x, y: jnp.logical_and(x > 0, y > 0).astype(BF16)),
        "select": ("ternary", None), "batch_norm_training": ("batchnorm", None),
        "reduce": ("reduce", None), "slice": ("slice", None),
        "transpose": ("transpose", None), "reshape": ("reshape", None),
        "broadcast": ("broadcast", None), "concatenate": ("concat", None),
    }

    def build(kind, fn, d0, d1, d2):
        ones = lambda *s: jnp.ones(s, BF16)
        if kind == "binary":  return (ones(d0, d1, d2), ones(d0, d1, d2)), (lambda x, y: fn(x, y))
        if kind == "unary":   return (ones(d0, d1, d2),), (lambda x: fn(x))
        if kind == "reduce":  return (ones(d0, d1, d2),), (lambda x: jnp.sum(x, axis=-1))
        if kind == "slice":   h = max(1, d2 // 2); return (ones(d0, d1, d2),), (lambda x: x[:, :, :h])
        if kind == "transpose": return (ones(d0, d1, d2),), (lambda x: jnp.transpose(x, (0, 2, 1)))
        if kind == "reshape": return (ones(d0, d1, d2),), (lambda x: jnp.reshape(x, (d0, d1 * d2)))
        if kind == "broadcast": return (ones(d0, 1, d2),), (lambda x: jnp.broadcast_to(x, (d0, d1, d2)))
        if kind == "concat":  h = max(1, d2 // 2); return (ones(d0, d1, h), ones(d0, d1, d2 - h)), \
            (lambda x, y: jnp.concatenate([x, y], axis=-1))
        if kind == "ternary": return (jnp.ones((d0, d1, d2), bool), ones(d0, d1, d2), ones(d0, d1, d2)), \
            (lambda c, x, y: jnp.where(c, x, y))
        if kind == "batchnorm": return (ones(d0, d1, d2),), \
            (lambda x: (x - jnp.mean(x, -1, keepdims=True)) * jax.lax.rsqrt(jnp.var(x, -1, keepdims=True) + 1e-5))
        raise ValueError(kind)
    return OPS, build


def _device_pid_tpu0(events):
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "process_name" \
                and "device:TPU:0" in e.get("args", {}).get("name", ""):
            return e["pid"]
    return None


def _kernel_us_from_trace(folder, iters):
    """(kernel_us, program_us): kernel = sum of inner op spans (real work); program =
    the outer jit-module span (kernel + per-launch overhead). Reading the wrapper as
    the kernel was the bug that floored every op at ~10us."""
    import gzip, json
    files = []
    for root, _, fs in os.walk(folder):
        files += [os.path.join(root, f) for f in fs if f.endswith(".trace.json.gz")]
    if not files:
        return None, None
    events = json.load(gzip.open(max(files, key=os.path.getmtime), "rt")).get("traceEvents", [])
    pid = _device_pid_tpu0(events)
    if pid is None:
        return None, None
    inner = prog = 0.0
    for e in events:
        if e.get("pid") == pid and e.get("ph") == "X" and "dur" in e:
            if e["name"].startswith("jit"):
                prog += e["dur"]
            else:
                inner += e["dur"]
    if inner == 0 and prog == 0:
        return None, None
    return inner / iters, prog / iters


def _measure(build, kind, fn, d0, d1, d2, warmup, iters, reps):
    import shutil, time
    import jax
    inputs, apply = build(kind, fn, d0, d1, d2)
    compiled = jax.jit(apply).lower(*inputs).compile()
    for _ in range(warmup):
        compiled(*inputs).block_until_ready()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); compiled(*inputs).block_until_ready()
        ts.append(time.perf_counter() - t0)
    ts.sort(); wall = ts[len(ts) // 2] * 1e6
    folder = os.path.join(TRACE_ROOT, f"{kind}_{d0}_{d1}_{d2}")
    shutil.rmtree(folder, ignore_errors=True); os.makedirs(folder, exist_ok=True)
    with jax.profiler.trace(folder):
        for _ in range(iters):
            compiled(*inputs).block_until_ready()
    kernel, program = _kernel_us_from_trace(folder, iters)
    shutil.rmtree(folder, ignore_errors=True)
    return kernel, program, wall


def cmd_collect(args):
    os.environ.setdefault("PJRT_DEVICE", "TPU"); os.environ.setdefault("XLA_USE_SPMD", "0")
    import time
    import jax
    OPS, build = _build_ops()
    os.makedirs(args.outdir, exist_ok=True)
    shapes = sample_shapes(args.n, args.seed)
    print(f"device={jax.devices()[0].device_kind}  shapes/op={len(shapes)}  ops={args.ops}")
    for op in args.ops:
        if op not in OPS:
            print(f"  SKIP unknown op {op}"); continue
        kind, fn = OPS[op]
        out = os.path.join(args.outdir, f"{op}_dataset.csv")
        if os.path.exists(out) and sum(1 for _ in open(out)) > len(shapes) // 2:
            print(f"  {op:14s} already done, skipping"); continue
        fh = open(out, "w", newline=""); w = csv.writer(fh)
        w.writerow(["d0", "d1", "d2", "size", "log2_size",
                    "latency_us", "kernel_us", "program_us", "wall_us", "host_us"])
        t0 = time.perf_counter(); ok = 0
        for (d0, d1, d2) in shapes:
            try:
                k, program, wall = _measure(build, kind, fn, d0, d1, d2,
                                            args.warmup, args.iters, args.reps)
                if k is None:
                    continue
                size = d0 * d1 * d2
                w.writerow([d0, d1, d2, size, f"{math.log2(size):.6f}",
                            f"{k:.6f}", f"{k:.6f}", f"{program:.6f}",
                            f"{wall:.6f}", f"{wall - program:.6f}"])
                ok += 1
                if ok % 50 == 0:
                    fh.flush()
            except Exception:
                pass
        fh.close()
        print(f"  {op:14s} {ok}/{len(shapes)} rows  {time.perf_counter()-t0:.0f}s -> {out}")


# ======================================================================== train (CPU)
def cmd_train(args):
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    os.makedirs(args.outdir, exist_ok=True)
    rows = []
    for csv_path in sorted(glob.glob(os.path.join(args.datadir, "*_dataset.csv"))):
        stem = os.path.basename(csv_path).replace("_dataset.csv", "")
        op_name = NAME.get(stem, stem)
        try:
            df = pd.read_csv(csv_path).dropna()
            df = df[df.latency_us > 0]
            X, y = df[FEATURES], df["latency_us"].values
            Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=args.seed)
            m = HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.06,
                                              early_stopping="auto", random_state=args.seed)
            m.fit(Xtr, ytr)
            pred = m.predict(Xva)
            md = {"train_rows": len(Xtr), "val_rows": len(Xva),
                  "val_mae": float(mean_absolute_error(yva, pred)),
                  "val_mape": float(np.mean(np.abs((yva - pred) / np.clip(yva, 1e-9, None)))),
                  "seed": args.seed, "features": FEATURES, "op_name": op_name,
                  "train_csv": os.path.basename(csv_path)}
        except Exception as e:
            print(f"  {stem:14s} FAILED: {e}"); continue
        with open(os.path.join(args.outdir, f"{op_name}.pkl"), "wb") as f:
            pickle.dump({"model": m, "op_name": op_name, "metadata": md}, f)
        rows.append((op_name, md["train_rows"] + md["val_rows"], md["val_mae"], md["val_mape"]))
        print(f"  {op_name:18s} rows={md['train_rows']+md['val_rows']:4d} "
              f"val_mae={md['val_mae']:.4f}us  val_mape={md['val_mape']*100:.2f}%")
    if rows:
        rows.sort(key=lambda r: r[3])
        print("\nMAPE summary (best->worst):")
        for op, n, mae, mape in rows:
            print(f"  {op:18s} {mape*100:6.2f}%  (mae {mae:.3f}us, n={n})")


# =============================================================== reshape-median (CPU)
def cmd_reshape_median(args):
    """Override reshape with a CONSTANT = median standalone reshape latency. reshape is
    bimodal (metadata ~0us vs relayout ~1000s us) and not shape-predictable; a fitted
    regressor over-predicts it ~37x at vocab sizes and then dominates the whole-model
    Sn (~63%), wrecking the compensation fit. The median constant fixes that."""
    from sklearn.dummy import DummyRegressor
    rows = list(csv.DictReader(open(args.dataset)))
    lat = np.array([float(r["latency_us"]) for r in rows])
    X = np.array([[float(r["d0"]), float(r["d1"]), float(r["d2"]),
                   float(r["size"]), np.log2(float(r["size"]))] for r in rows])
    dm = DummyRegressor(strategy="median").fit(X, lat)
    med = float(np.median(lat))
    out = os.path.join(args.model_dir, "reshape.pkl")
    pickle.dump({"model": dm, "op_name": "reshape",
                 "metadata": {"strategy": "constant median", "median_us": med,
                              "features": FEATURES, "train_rows": len(rows)}},
                open(out, "wb"))
    print(f"reshape -> constant median {med:.2f}us  (n={len(rows)})  wrote {out}")


# ============================================================================== cli
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)

    c = sub.add_parser("collect", help="measure pure device op latency (needs a TPU)")
    c.add_argument("--ops", nargs="+", required=True)
    c.add_argument("--n", type=int, default=1000, help="shapes/op")
    c.add_argument("--iters", type=int, default=10, help="executes traced per shape")
    c.add_argument("--warmup", type=int, default=1)
    c.add_argument("--reps", type=int, default=1, help="wall-clock reps (audit cols only)")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--outdir", default=HERE)
    c.set_defaults(func=cmd_collect)

    t = sub.add_parser("train", help="train one HGBR per op")
    t.add_argument("--datadir", default=HERE)
    t.add_argument("--outdir", default=os.path.join(HERE, "models"))
    t.add_argument("--seed", type=int, default=42)
    t.set_defaults(func=cmd_train)

    r = sub.add_parser("reshape-median", help="override reshape with a constant median")
    r.add_argument("--model-dir", required=True)
    r.add_argument("--dataset", required=True)
    r.set_defaults(func=cmd_reshape_median)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
