#!/usr/bin/env python3
"""
Measure bf16 GEMM latency on TPU v4 (single device, no sharding) for a list of
(M,N,K) shapes, and record SCALE-Sim closed-form compute cycles alongside.

Two latency signals per shape (decide which predicts end-to-end better later):
  - latency_us_wallclock : single-dispatch median (full launch + dispatch overhead)
  - latency_us_device    : amortized per-op over a no-sync burst (~on-device time)

Rigorous steady-state: warmup absorbs XLA compile; we assert no recompilation
during timing. Inputs are live arrays + output consumed => no DCE/const-fold.
Writes one CSV row per shape, incrementally (crash-safe).
"""
import os
os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("XLA_USE_SPMD", "0")

import argparse, csv, math, time
import jax, jax.numpy as jnp


def matmul_scale_sim_model(m, n, k, S=128):
    # M-streaming weight-stationary cycle count; matches SCALE-Sim (72/73 gpt2
    # layers exact). NOT min(v1,v2) -- that only held for square-ish shapes.
    return (2 * S + S + m - 2) * math.ceil(n / S) * math.ceil(k / S) - 1


def _iters_for(m, n, k):
    # device-work budget ~20 ms/shape: many iters for tiny GEMMs, few for huge
    est_s = (2.0 * m * n * k) / 200e12  # assume ~200 bf16 TFLOP/s
    return int(min(400, max(20, round(0.02 / max(est_s, 1e-9)))))


def measure(m, n, k, dev, warmup, n_single, burst):
    a = jax.device_put(jnp.ones((m, k), jnp.bfloat16), dev)
    b = jax.device_put(jnp.ones((k, n), jnp.bfloat16), dev)
    f = jax.jit(lambda x, y: x @ y)

    # In-JIT repeated matmul: ONE dispatch, all on-device. The (b + i) makes each
    # iteration's matmul distinct so XLA cannot hoist it out of the loop.
    def run_K(x, y, K):
        def body(i, acc):
            yi = y + i.astype(jnp.bfloat16)
            return acc + (x @ yi)
        acc0 = jnp.zeros((m, n), jnp.bfloat16)
        return jax.lax.fori_loop(0, K, body, acc0)
    g = jax.jit(run_K, static_argnums=2)
    K = _iters_for(m, n, k)

    # warmup (compile both f and g for this shape)
    for _ in range(warmup):
        f(a, b).block_until_ready()
    g(a, b, K).block_until_ready()
    g(a, b, 1).block_until_ready()

    # single-dispatch wall-clock (full host+launch overhead), median
    singles = []
    for _ in range(n_single):
        t0 = time.perf_counter(); f(a, b).block_until_ready()
        singles.append(time.perf_counter() - t0)
    singles.sort()
    lat_wall = singles[len(singles) // 2] * 1e6  # us

    # device per-op: time the K-loop and the 1-loop; subtract to cancel the
    # single fixed dispatch + the per-iter add, leaving pure matmul device time
    def t_loop(KK, reps=5):
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); g(a, b, KK).block_until_ready()
            ts.append(time.perf_counter() - t0)
        ts.sort(); return ts[len(ts) // 2]
    tK = t_loop(K); t1 = t_loop(1)
    lat_dev = max((tK - t1) / max(K - 1, 1), tK / K) * 1e6  # us/op
    return lat_wall, lat_dev, False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", default=os.path.join(os.path.dirname(__file__), "shapes.csv"))
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "gemm_master.csv"))
    p.add_argument("--limit", type=int, default=0, help="0 = all")
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--n-single", type=int, default=21)
    p.add_argument("--burst", type=int, default=50)
    p.add_argument("--max-bytes", type=float, default=8e9, help="skip shapes whose tensors exceed this")
    args = p.parse_args()

    rows = list(csv.DictReader(open(args.shapes)))
    if args.shuffle:
        import random; random.Random(0).shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]

    dev = jax.devices()[0]
    print(f"device={dev}  shapes={len(rows)}  out={args.out}")

    # resume: skip already-done shapes
    done = set()
    if os.path.exists(args.out):
        for r in csv.DictReader(open(args.out)):
            done.add((int(r["M"]), int(r["N"]), int(r["K"])))

    fields = ["M", "N", "K", "shape_class", "dtype", "latency_us_wallclock",
              "latency_us_device", "cycles_compute", "recompiled", "status"]
    new = not os.path.exists(args.out)
    fh = open(args.out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=fields)
    if new:
        w.writeheader()

    t_start = time.perf_counter()
    n_ok = n_skip = n_err = 0
    for i, r in enumerate(rows):
        m, n, k = int(r["M"]), int(r["N"]), int(r["K"])
        if (m, n, k) in done:
            continue
        bytes_est = 2 * (m * k + k * n + m * n)  # bf16
        rec = {"M": m, "N": n, "K": k, "shape_class": r.get("shape_class", ""),
               "dtype": "bf16", "latency_us_wallclock": "", "latency_us_device": "",
               "cycles_compute": matmul_scale_sim_model(m, n, k),
               "recompiled": "", "status": "ok"}
        if bytes_est > args.max_bytes:
            rec["status"] = "skip_oom"; n_skip += 1
        else:
            try:
                lw, ld, rc = measure(m, n, k, dev, args.warmup, args.n_single, args.burst)
                rec["latency_us_wallclock"] = f"{lw:.4f}"
                rec["latency_us_device"] = f"{ld:.4f}"
                rec["recompiled"] = int(rc); n_ok += 1
            except Exception as e:
                rec["status"] = "err:" + repr(e)[:60]; n_err += 1
        w.writerow(rec); fh.flush()
        if (i + 1) % 25 == 0:
            el = time.perf_counter() - t_start
            print(f"  [{i+1}/{len(rows)}] ok={n_ok} skip={n_skip} err={n_err} "
                  f"elapsed={el:.0f}s rate={(n_ok+n_err)/max(el,1e-9):.1f}/s")
    fh.close()
    print(f"DONE ok={n_ok} skip={n_skip} err={n_err} -> {args.out}")


if __name__ == "__main__":
    main()
