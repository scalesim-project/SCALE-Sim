#!/usr/bin/env python3
"""
GEMM time-model calibration — one script, four stages (run in order; only `collect`
and `batch` touch the TPU):

  sample   region-stratified (M,N,K) shapes               -> shapes.csv          (CPU)
  collect  pure device kernel latency per shape (xprof)    -> gemm_master.csv     (TPU)
  fit      piecewise region-table G_roof fit (tpu.py's     -> prints the table    (CPU)
           12-region scheme); same fit for any generation
  batch    level-2 batch-reduction sweep                   -> batch_reduction.csv (TPU)

"Pure device" = the matmul `fusion` span on the TPU:0 xprof timeline (sum of the inner
op spans, EXCLUDING the outer `jit_*` wrapper = the per-launch floor). bf16 throughout.

Usage:
  python3 gemm_calibration.py sample  --out shapes.csv
  PJRT_DEVICE=TPU python3 gemm_calibration.py collect --shapes shapes.csv --out gemm_master.csv
  python3 gemm_calibration.py fit     --data gemm_master.csv        # paste table into tpu.py
  PJRT_DEVICE=TPU python3 gemm_calibration.py batch --out batch_reduction.csv
"""
import argparse, csv, math, os
import numpy as np

S = 128  # systolic tile / array dim
HERE = os.path.dirname(os.path.abspath(__file__))


def cycles_scale_sim(m, n, k, s=S):
    """SCALE-Sim closed-form compute cycles (M-streaming, weight-stationary)."""
    return (2 * s + s + m - 2) * math.ceil(n / s) * math.ceil(k / s) - 1


# ============================================================ sample (stratified)
# Random sampling starves the rare regions, so we fill EACH region to a quota: every
# region's 3-param G_roof gets enough points, log-uniform over its sub-ranges with size
# spread so the floor->compute slope is identifiable. Stratify by the coarse 8-region
# scheme (foldK>1, foldN>=16, foldM>=16); the fit re-bins into tpu.py's finer 12 regions.
_QUOTA = {0: 150, 1: 150, 2: 150, 3: 180, 4: 150, 5: 150, 6: 150, 7: 180}
_MAX_BYTES = 8e9
_M_SMALL, _M_LARGE = (1, 1920), (1921, 6144)
_N_SMALL, _N_LARGE = (1, 1920), (1921, 6144)
_K_SHALLOW, _K_DEEP = (1, 128), (129, 6144)


def _region8(M, N, K):
    return ((math.ceil(K / S) > 1) * 4 + (math.ceil(N / S) >= 16) * 2
            + (math.ceil(M / S) >= 16))


def _logu(rng, lo, hi):
    return int(round(math.exp(rng.uniform(math.log(lo), math.log(max(lo + 1, hi))))))


def cmd_sample(args):
    import random
    rng = random.Random(args.seed)
    rows, seen = [], set()
    for r in range(8):
        mr = _M_LARGE if r & 1 else _M_SMALL
        nr = _N_LARGE if r & 2 else _N_SMALL
        kr = _K_DEEP if r & 4 else _K_SHALLOW
        q = _QUOTA[r]; got = tries = 0
        while got < q and tries < q * 200:
            tries += 1
            M, N, K = _logu(rng, *mr), _logu(rng, *nr), _logu(rng, *kr)
            if _region8(M, N, K) != r or (M, N, K) in seen:
                continue
            if 2 * (M * K + K * N + M * N) > _MAX_BYTES:
                continue
            seen.add((M, N, K)); rows.append((M, N, K, f"r{r}")); got += 1
        print(f"  region {r}: {got}/{q}")
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["M", "N", "K", "shape_class"]); w.writerows(rows)
    print(f"wrote {len(rows)} stratified shapes -> {args.out}")


# ===================================================================== collect (TPU)
TRACE_ROOT = "/tmp/xprof_gemm"


def _device_pid_tpu0(events):
    """PID of the TPU:0 device op stream (auto-detected from process_name meta)."""
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "process_name" \
                and "device:TPU:0" in e.get("args", {}).get("name", ""):
            return e["pid"]
    return None


def _trace_kernel_us(folder, iters):
    """(kernel_us, program_us) per call from the TPU:0 timeline. kernel = inner op
    spans (real matmul work); program = the outer jit-module span (kernel + per-launch
    overhead). Reading the wrapper as the kernel was the bug that inflated the floor."""
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


def _traced(fn, inputs, warmup, iters, tag):
    """Compile fn, warm up, trace `iters` calls, return (kernel_us, program_us)."""
    import shutil
    import jax
    compiled = jax.jit(fn).lower(*inputs).compile()
    for _ in range(warmup):
        compiled(*inputs).block_until_ready()
    folder = os.path.join(TRACE_ROOT, tag)
    shutil.rmtree(folder, ignore_errors=True); os.makedirs(folder, exist_ok=True)
    with jax.profiler.trace(folder):
        for _ in range(iters):
            compiled(*inputs).block_until_ready()
    k, p = _trace_kernel_us(folder, iters)
    shutil.rmtree(folder, ignore_errors=True)
    return k, p


def cmd_collect(args):
    os.environ.setdefault("PJRT_DEVICE", "TPU"); os.environ.setdefault("XLA_USE_SPMD", "0")
    import time
    import jax, jax.numpy as jnp
    rows = list(csv.DictReader(open(args.shapes)))
    if args.shuffle:
        import random; random.Random(0).shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]
    print(f"device={jax.devices()[0].device_kind}  shapes={len(rows)}  out={args.out}")
    done = set()
    if os.path.exists(args.out):                                  # resumable / crash-safe
        for r in csv.DictReader(open(args.out)):
            done.add((int(r["M"]), int(r["N"]), int(r["K"])))
    fields = ["M", "N", "K", "shape_class", "dtype", "latency_us_wallclock",
              "latency_us_device", "program_us", "host_us", "cycles_compute", "status"]
    new = not os.path.exists(args.out)
    fh = open(args.out, "a", newline=""); w = csv.DictWriter(fh, fieldnames=fields)
    if new:
        w.writeheader()
    t0 = time.perf_counter(); n_ok = n_skip = n_err = 0
    for i, r in enumerate(rows):
        m, n, k = int(r["M"]), int(r["N"]), int(r["K"])
        if (m, n, k) in done:
            continue
        rec = {"M": m, "N": n, "K": k, "shape_class": r.get("shape_class", ""),
               "dtype": "bf16", "latency_us_wallclock": "", "latency_us_device": "",
               "program_us": "", "host_us": "",
               "cycles_compute": cycles_scale_sim(m, n, k), "status": "ok"}
        if 2 * (m * k + k * n + m * n) > args.max_bytes:
            rec["status"] = "skip_oom"; n_skip += 1
        else:
            try:
                a = jnp.ones((m, k), jnp.bfloat16); b = jnp.ones((k, n), jnp.bfloat16)
                a.block_until_ready(); b.block_until_ready()
                import time as _t
                ts = []
                cfn = jax.jit(lambda x, y: x @ y).lower(a, b).compile()
                for _ in range(args.warmup):
                    cfn(a, b).block_until_ready()
                for _ in range(args.reps):
                    t = _t.perf_counter(); cfn(a, b).block_until_ready(); ts.append(_t.perf_counter() - t)
                ts.sort(); wall = ts[len(ts) // 2] * 1e6
                kernel, program = _traced(lambda x, y: x @ y, (a, b),
                                          0, args.iters, f"{m}_{n}_{k}")
                if kernel is None:
                    rec["status"] = "err:no_trace"; n_err += 1
                else:
                    rec["latency_us_wallclock"] = f"{wall:.4f}"
                    rec["latency_us_device"] = f"{kernel:.4f}"     # matmul `fusion` span
                    rec["program_us"] = f"{program:.4f}"
                    rec["host_us"] = f"{wall - program:.4f}"
                    n_ok += 1
            except Exception as e:
                rec["status"] = "err:" + repr(e)[:60]; n_err += 1
        w.writerow(rec); fh.flush()
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(rows)}] ok={n_ok} skip={n_skip} err={n_err} "
                  f"elapsed={time.perf_counter()-t0:.0f}s")
    fh.close()
    print(f"DONE ok={n_ok} skip={n_skip} err={n_err} -> {args.out}")


# ========================================================================= fit (CPU)
# Piecewise region-table G_roof on the golden pure-device latency, using tpu.py's
# production 12-region selector (foldK>1, foldN>1, foldM>1, + 2nd M bar at foldM>=6).
# Same fit for any generation -- run on v4 data -> TPUV4_REGION_TABLE, on v6e -> v6e.
_REGIONS = [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15]   # ids _tpuv4_region can produce


def _fit_groof(cyc, bytes_, y, trim=0.05, iters=3):
    """Relative-weighted (w=1/y^2) fit y ~= A*max(cyc, bytes/BW)+B over a BW grid,
    robustly trimming the worst `trim` residuals each pass."""
    keep = np.ones(len(y), bool); best = None
    for _ in range(iters):
        for BW in np.logspace(np.log10(2), np.log10(600), 60):
            feat = np.maximum(cyc, bytes_ / BW)
            X = np.stack([feat, np.ones(len(y))], 1); w = 1.0 / y ** 2
            WX = (X * w[:, None])[keep]
            try:
                A, B = np.linalg.solve(X[keep].T @ WX, WX.T @ y[keep])
            except np.linalg.LinAlgError:
                continue
            p = X @ [A, B]
            m = np.mean(np.abs((y[keep] - p[keep]) / y[keep]))
            if best is None or m < best[0]:
                best = (m, A, B, BW)
        _, A, B, BW = best
        res = np.abs((y - (A * np.maximum(cyc, bytes_ / BW) + B)) / y)
        keep = res <= np.quantile(res[keep], 1 - trim)
    return best[1], best[2], best[3]


def _mape(y, p):
    return float(np.mean(np.abs((y - p) / y)) * 100)


def cmd_fit(args):
    import sys
    sys.path.insert(0, HERE)
    from tpu import _tpuv4_region                                 # the shipped 12-region scheme
    M, N, K, cyc, lat = [], [], [], [], []
    for r in csv.DictReader(open(args.data)):
        if r.get("status") != "ok" or not r.get(args.signal):
            continue
        M.append(int(r["M"])); N.append(int(r["N"])); K.append(int(r["K"]))
        cyc.append(float(r["cycles_compute"])); lat.append(float(r[args.signal]))
    M, N, K = np.array(M), np.array(N), np.array(K)
    cyc, lat = np.array(cyc, float), np.array(lat, float)
    bytes_ = 2.0 * (M * K + K * N + M * N)
    rid = np.array([_tpuv4_region(int(m), int(n), int(k)) for m, n, k in zip(M, N, K)])

    rng = np.random.default_rng(0); idx = rng.permutation(len(lat)); ntr = int(0.8 * len(idx))
    tr, te = idx[:ntr], idx[ntr:]; trmask = np.zeros(len(lat), bool); trmask[tr] = True
    print(f"golden GEMMs: {len(lat)}  (train {len(tr)} / test {len(te)})\n")

    A, B, BW = _fit_groof(cyc[tr], bytes_[tr], lat[tr])           # single global baseline
    p = A * np.maximum(cyc, bytes_ / BW) + B
    print(f"single G_roof : A={A:.4e} B={B:.3f} BW={BW:.1f}  held-out MAPE={_mape(lat[te], p[te]):.1f}%")

    table, pe = {}, np.zeros(len(lat))                            # piecewise 12-region
    for r in _REGIONS:
        m_all = rid == r; m_tr = m_all & trmask
        if m_tr.sum() < 8:
            table[r] = (max(A, 0.0), B, BW)
        else:
            Ar, Br, BWr = _fit_groof(cyc[m_tr], bytes_[m_tr], lat[m_tr])
            table[r] = (max(Ar, 0.0), Br, BWr)
        Ar, Br, BWr = table[r]
        if m_all.any():
            pe[m_all] = Ar * np.maximum(cyc[m_all], bytes_[m_all] / BWr) + Br
    print(f"piecewise(12) : held-out MAPE={_mape(lat[te], pe[te]):.1f}%\n")

    print("REGION_TABLE = {   # paste into tpu.py (TPUV4_REGION_TABLE / TPUV6E_REGION_TABLE)")
    for r in _REGIONS:
        Ar, Br, BWr = table[r]
        print(f"    {r}: ({Ar:.6e}, {Br:.4f}, {BWr:.3f}),   # n={int((rid==r).sum())}")
    print("}")


# ======================================================================= batch (TPU)
def cmd_batch(args):
    os.environ.setdefault("PJRT_DEVICE", "TPU"); os.environ.setdefault("XLA_USE_SPMD", "0")
    import jax.numpy as jnp
    BATCHES = [1, 2, 4, 8, 16, 32, 64]
    SHAPES = [(128, 128, 64), (256, 256, 64), (512, 512, 64), (1024, 1024, 64),
              (128, 128, 128), (256, 256, 128), (512, 512, 128),
              (256, 128, 64), (512, 128, 128), (128, 512, 64)]
    fh = open(args.out, "w", newline=""); w = csv.writer(fh)
    w.writerow(["b", "M", "N", "K", "einsum_us", "single_us", "b_single_us", "R"])
    single = {}
    for (M, N, K) in SHAPES:                                      # single-matmul time (b-independent)
        a = jnp.ones((M, K), jnp.bfloat16); b_ = jnp.ones((K, N), jnp.bfloat16)
        a.block_until_ready(); b_.block_until_ready()
        single[(M, N, K)] = _traced(lambda x, y: x @ y, (a, b_), 6, args.iters, f"s_{M}_{N}_{K}")[0]
    n = 0
    for (M, N, K) in SHAPES:
        s = single[(M, N, K)]
        for b in BATCHES:
            A = jnp.ones((b, M, K), jnp.bfloat16); B = jnp.ones((b, K, N), jnp.bfloat16)
            A.block_until_ready(); B.block_until_ready()
            e = _traced(lambda x, y: jnp.einsum('bmk,bkn->bmn', x, y), (A, B),
                        6, args.iters, f"b{b}_{M}_{N}_{K}")[0]
            bs = s * b
            w.writerow([b, M, N, K, f"{e:.4f}", f"{s:.4f}", f"{bs:.4f}", f"{e/bs:.5f}"]); n += 1
        fh.flush()
    fh.close()
    print(f"wrote {args.out}: {n} rows ({len(SHAPES)} shapes x {len(BATCHES)} batches). "
          f"Fit R = u + (1-u)/b, u = nt^p/(nt^p+c) on this; see calibration/README.md / tpu.py.")


# ============================================================================== cli
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)

    s = sub.add_parser("sample", help="region-stratified (M,N,K) shapes")
    s.add_argument("--out", default=os.path.join(HERE, "shapes_stratified.csv"))
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_sample)

    c = sub.add_parser("collect", help="measure pure device latency (needs a TPU)")
    c.add_argument("--shapes", default=os.path.join(HERE, "shapes_stratified.csv"))
    c.add_argument("--out", default=os.path.join(HERE, "gemm_master.csv"))
    c.add_argument("--limit", type=int, default=0, help="0 = all")
    c.add_argument("--shuffle", action="store_true")
    c.add_argument("--warmup", type=int, default=10)
    c.add_argument("--iters", type=int, default=30, help="executes traced per shape")
    c.add_argument("--reps", type=int, default=15, help="wall-clock reps (median)")
    c.add_argument("--max-bytes", type=float, default=8e9)
    c.set_defaults(func=cmd_collect)

    f = sub.add_parser("fit", help="fit the piecewise region table (any generation)")
    f.add_argument("--data", default=os.path.join(HERE, "gemm_master.csv"))
    f.add_argument("--signal", default="latency_us_device")
    f.set_defaults(func=cmd_fit)

    b = sub.add_parser("batch", help="level-2 batch-reduction sweep (needs a TPU)")
    b.add_argument("--out", default=os.path.join(HERE, "batch_reduction.csv"))
    b.add_argument("--iters", type=int, default=15)
    b.set_defaults(func=cmd_batch)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
