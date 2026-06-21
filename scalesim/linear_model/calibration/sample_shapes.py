#!/usr/bin/env python3
"""
Generate a broad set of GEMM shapes (M,N,K) for TPU v4 latency collection.

Coverage built for generality (Phase 0a of GEMM_LINEAR_PLAN.md):
  - log-spaced random over [1, MAX] for each dim
  - tile-alignment: multiples of 128 AND off-tile 128q+r
  - aspect-ratio classes: square / tall / fat / deep-K / GEMV / wide-output / thin-K
  - real LLM GEMM shapes from ../LLM/*.topology.csv (held-out anchor)
Outputs a CSV: M,N,K,shape_class  (deduplicated).
"""
import argparse, csv, glob, math, os, random

S = 128  # systolic tile


def logspace_ints(lo, hi, rng):
    return int(round(math.exp(rng.uniform(math.log(lo), math.log(hi)))))


def snap_variants(v):
    """A value plus tile-alignment variants around it."""
    out = {max(1, v)}
    q = max(1, round(v / S))
    out.add(q * S)                       # aligned
    for r in (1, 8, 33, 64, 96, 127):    # off-tile residuals
        out.add(q * S + r)
    return out


def gen(max_dim, n_random, seed):
    rng = random.Random(seed)
    shapes = {}  # (M,N,K) -> class

    def add(m, n, k, cls):
        m, n, k = int(m), int(n), int(k)
        if m >= 1 and n >= 1 and k >= 1:
            shapes.setdefault((m, n, k), cls)

    # 1) log-uniform random, all aspect ratios
    for _ in range(n_random):
        m = logspace_ints(1, max_dim, rng)
        n = logspace_ints(1, max_dim, rng)
        k = logspace_ints(1, max_dim, rng)
        add(m, n, k, "random")

    # 2) tile-boundary grid in the small/medium zone (most nonlinear region)
    base = [1, 2, 4, 8, 16, 32, 64, 96, 127, 128, 129, 192, 256, 257, 384,
            512, 768, 1024, 1536, 2048, 4096]
    base = [b for b in base if b <= max_dim]
    # sparse cross-product (cap to avoid explosion): random subset of the grid
    grid = [(m, n, k) for m in base for n in base for k in base]
    rng.shuffle(grid)
    for (m, n, k) in grid[: 3 * n_random]:
        add(m, n, k, "grid")

    # 3) explicit aspect-ratio classes with alignment variants
    bigs = [b for b in (256, 512, 1024, 2048, 4096, 8192, 16384) if b <= max_dim]
    smalls = [1, 8, 16, 64, 128, 256]
    for big in bigs:
        for sm in smalls:
            for M in snap_variants(big):
                add(M, sm, sm, "tall")        # tall-skinny
                add(sm, M, sm, "fat")         # fat (wide N)
                add(sm, sm, M, "deepK")       # deep reduction
                add(1, M, sm, "gemv_wideN")   # GEMV-ish
                add(sm, M, 64, "wide_output") # lm_head-like
                add(M, big, 64, "thinK")      # memory-bound-ish

    # 4) real LLM GEMM shapes (anchor / held-out)
    for f in glob.glob(os.path.join(os.path.dirname(__file__), "..", "LLM", "*.topology.csv")):
        try:
            with open(f) as fh:
                for row in csv.DictReader(fh):
                    keys = {k.strip().lower(): v for k, v in row.items() if k}
                    def gv(*names):
                        for nm in names:
                            if nm in keys and str(keys[nm]).strip().isdigit():
                                return int(keys[nm])
                        return None
                    M, N, K = gv("m"), gv("n"), gv("k")
                    if M and N and K:
                        add(M, N, K, "llm")
        except Exception:
            pass

    return shapes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-dim", type=int, default=8192)
    p.add_argument("--n-random", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "shapes.csv"))
    args = p.parse_args()

    shapes = gen(args.max_dim, args.n_random, args.seed)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["M", "N", "K", "shape_class"])
        for (m, n, k), cls in sorted(shapes.items()):
            w.writerow([m, n, k, cls])
    # class histogram
    from collections import Counter
    hist = Counter(shapes.values())
    print(f"wrote {len(shapes)} unique shapes -> {args.out}")
    for cls, c in sorted(hist.items(), key=lambda x: -x[1]):
        print(f"  {cls:14s} {c}")


if __name__ == "__main__":
    main()
