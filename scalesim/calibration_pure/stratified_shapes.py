#!/usr/bin/env python3
"""
Stratified GEMM shape sampler for the 8-region piecewise model. Random sampling
starves the rare regions (3,7 got only ~40-67 of 7097); this fills EACH region to
a quota so every region's 3-param G_roof has enough points. ~1260 shapes total,
each region log-uniform over its allowed sub-ranges, with good size spread so the
floor->compute transition (the slope) is identifiable within the region.

region = (foldK>1)*4 + (foldN>=16)*2 + (foldM>=16),  foldX = ceil(X/128)
  bit0 M>=16 tiles -> M>=1921 ;  bit1 N>=16 tiles -> N>=1921 ;  bit2 K>1 tile -> K>=129
"""
import csv, math, os, random

S = 128
QUOTA = {0: 150, 1: 150, 2: 150, 3: 180, 4: 150, 5: 150, 6: 150, 7: 180}
MAX_BYTES = 8e9
SEED = 0
OUT = os.path.join(os.path.dirname(__file__), "shapes_stratified.csv")

# per-region (M,N,K) ranges as (lo,hi) for log-uniform draws; large dims capped
# at 6144 to keep per-shape compile/run time reasonable while staying compute-bound.
M_SMALL, M_LARGE = (1, 1920), (1921, 6144)
N_SMALL, N_LARGE = (1, 1920), (1921, 6144)
K_SHALLOW, K_DEEP = (1, 128), (129, 6144)


def region(M, N, K):
    return ((math.ceil(K / S) > 1) * 4 + (math.ceil(N / S) >= 16) * 2
            + (math.ceil(M / S) >= 16))


def logu(rng, lo, hi):
    return int(round(math.exp(rng.uniform(math.log(lo), math.log(max(lo + 1, hi))))))


def ranges_for(r):
    return ((M_LARGE if r & 1 else M_SMALL),
            (N_LARGE if r & 2 else N_SMALL),
            (K_DEEP if r & 4 else K_SHALLOW))


def main():
    rng = random.Random(SEED)
    rows = []
    seen = set()
    for r in range(8):
        (mr, nr, kr) = ranges_for(r)
        q = QUOTA[r]
        tries = 0
        got = 0
        while got < q and tries < q * 200:
            tries += 1
            M, N, K = logu(rng, *mr), logu(rng, *nr), logu(rng, *kr)
            if region(M, N, K) != r:           # boundary rounding -> re-draw
                continue
            if 2 * (M * K + K * N + M * N) > MAX_BYTES:
                continue
            if (M, N, K) in seen:
                continue
            seen.add((M, N, K))
            rows.append((M, N, K, f"r{r}"))
            got += 1
        print(f"  region {r}: {got}/{q}")
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["M", "N", "K", "shape_class"])
        w.writerows(rows)
    print(f"wrote {len(rows)} stratified shapes -> {OUT}")


if __name__ == "__main__":
    main()
