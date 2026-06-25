# Pure-device (xprof) calibration — results

> **CORRECTION (2026-06-24).** The first run's "kernel" time was extracted as the
> **whole-program** device span (the outer `jit_*` wrapper), which for small GEMMs
> is ~12–14 µs and is dominated by fixed **per-launch program overhead**, NOT the
> matmul. The trace breakdown shows the matmul `fusion` span for a 128³ GEMM is
> **1.22 µs** — matching the loop-method floor (~1.5 µs). So the earlier "B ≈ 14 µs
> floor" and the "24% MAPE / can't beat it" conclusions were **artifacts of that
> extraction bug**, now fixed (collectors take the inner `fusion`/op spans; the
> outer wrapper is reported separately as `program_us`). Corrected decomposition of
> a tiny-GEMM execution:
>   - matmul kernel (`fusion`)      ≈ **1.3 µs**  (per-GEMM compute floor)
>   - per-launch program overhead   ≈ **11 µs**   (`program − kernel`, per-EXECUTE)
>   - host dispatch                 ≈ **92 µs**   (`wall − program`, per-EXECUTE)
> `tpu.py` is reverted to the loop-method region table (floor ~1.5 µs, full-space
> MAPE 14.2%), which was correct all along and agrees with the `fusion` span.
> The sections below (B≈14 µs, 24% ceiling) are kept for the record but SUPERSEDED.

## Fusion-time model — authoritative result (stratified, corrected extraction)

After the extraction fix, the GEMM linear model was re-evaluated against the golden
**`fusion`** kernel time on a **stratified** dataset (`stratified_shapes.py` →
`gemm_fusion_strat.csv`, 1260 shapes, ≥150 per region so rare regions r3/r7 are no
longer starved). `fit_piecewise_pure.py`, 80/20 held out:

| model | held-out MAPE (fusion time) |
|---|---|
| single G_roof | 37.7% |
| **piecewise, 8-region** | **18.3%** |

Per-region (A us/cyc, B us floor, BW bytes/cyc), n=150–180 each:

    r0 (2.85e-6, 1.08, 2.0)    r4 (3.40e-5, 1.09, 78.8)
    r1 (2.46e-6, 1.14, 2.0)    r5 (8.66e-5, 0.71, 228.2)
    r2 (2.43e-6, 1.03, 2.0)    r6 (2.73e-5, 1.01, 86.8)
    r3 (6.23e-6, 2.99, 6.4)    r7 (1.20e-4, 4.05, 116.0)

Floor B~1 us == the matmul `fusion` floor (the bug-era "14 us" is gone). Piecewise
HALVES single-G_roof error because `A` spans 2.4e-6 -> 1.2e-4 (50x) across regions --
one global slope cannot fit that. On a balanced set the gap is larger than the
14.2% the loop-method table scores on full-space-weighted data (which is dominated by
the easy small regions); 18.3% is the fairer, every-regime-equal number. tpu.py keeps
the loop-method table (10,874 shapes, marginal ~= fusion); this stratified-fusion
table is the directly-fusion-calibrated alternative.

### r6 fix: add a foldM>1 sub-split -> 12 regions, 18.3% -> 15.3%

r6 (K>1, N>=16, M<16 tiles) was the weak region at 42.6%: it spans a 280x latency
range (1.2-344 us) because small-M shapes sit at the ~1 us floor while large-N*K
shapes are compute-bound. Residual correlated most with M (+0.64). Splitting the
small-M regions (0,2,4,6) by **foldM>1 (M>128 = more than one 128-array tile)** --
the physical point where the fixed 3*128 fill/drain term stops dominating and
M-streaming takes over -- fixes it:

| | regions | overall MAPE | r6 MAPE |
|---|---|---|---|
| KxNxM | 8 | 18.3% | 42.6% |
| **+ foldM>1 split on small-M regions** | **12** | **15.3%** | **12-20%** |

r6 now splits into M=1-tile (19.6%, floor-dominated) and 1<foldM<16 (12.1%,
compute). The same split also lifts r0/r2/r4, so the whole model improves. This is a
4th physical tile-fold level (M: 1 / 2-15 / >=16 tiles), consistent with the existing
foldK>1 and fold>=16 splits. Could be ported to the shipped loop-method tpu.py table.

### The "16" threshold was unjustified -> use >1 tile for ALL dims (13.5%)

Sweeping the "wide" threshold T (the fold>=T bit for N and M) showed MAPE is
**monotonic in T** -- 16 was just a round number from the original piecewise work,
never swept:

    T (tiles):   2     4     6     8    12    16    24    32
    MAPE:      13.5  15.2  15.7  16.5  17.6  18.3  24.9  25.7  %

The optimum is **T=2 = the fundamental >1-tile boundary** (dim>128) -- the SAME rule
as foldK>1. So the principled scheme splits all three dims at fold>1:

    region = (foldK>1)*4 + (foldN>1)*2 + (foldM>1)        # 8 regions, one rule

**Overall held-out MAPE 13.5%** (vs 18.3% at threshold 16, and the 12-region patch's
15.3%) -- fewer regions, one consistent physical rule, better accuracy. Per-region
6.6% (all-multi-tile compute) to 22% (single-M-tile floor regime). This SUPERSEDES
both the >=16 scheme and the 12-region patch above. Recommended scheme to ship
(and to port to the loop-method tpu.py table, which uses the same unjustified >=16).

---

Built by `run_all.sh` (2026-06-24, ~15 h on TPU v4). Every op (GEMM + 25
non-compute) profiled with **pure device kernel time read from the xprof trace**
(`collect_pure_device_gemm_tpu.py`, `collect_pure_device_tpu.py`), then models
built exactly as the shipped pipeline (`fit_gemm.py`, `train_ops.py`) — only the
latency **label** differs (pure-kernel xprof device time vs the loop-method
wall-clock subtraction).

## Artifacts (kept SEPARATE from the shipped loop-method models)
- `gemm_pure_master.csv` (7097 GEMMs), `gemm_linear_pure_tpuv4.json`
- `datasets_pure_tpuv4/*_dataset.csv` (25 ops), `../model/tpuv4_pure/*.pkl` (25)

## GEMM G_roof — pure vs shipped

| | A (µs/cyc) | B (floor µs) | bytes/cyc | val MAPE |
|---|---|---|---|---|
| shipped (loop, marginal) | 3.41e-5 | 1.47 | 29.5 | 16.5% |
| pure-kernel (xprof) | 5.85e-5 | **14.09** | 79.1 | 24.2% |

`B` jumped 1.47 → 14.1 µs = the measured single-kernel device floor (`measure_xprof.py`
gave ~12.6 µs). This **validates** the xprof floor measurement.

## Per-op val MAPE (pure kernel, n=400/op, 80/20)
Best→worst: slice 7.2 · maximum 7.7 · and 7.8 · tanh 7.8 · multiply 7.9 · select 8.1 ·
transpose 8.1 · divide 8.2 · compare 8.3 · subtract 8.3 · add 8.6 · negate 8.9 ·
minimum 9.1 · power 9.1 · rsqrt 9.3 · concatenate 9.8 · batch_norm 9.9 · logistic 10.1 ·
exponential 11.7 · convert 12.0 · reduce 12.5 · sine 13.2 · broadcast_in_dim 13.7 ·
cosine 13.8 · reshape 23.5  (%)

vs the shipped loop-method models' 3.5–9.5% (most < 5%). Pure-kernel is ~2× worse.

## Conclusion — these are NOT a drop-in replacement for the shipped models

Same root cause behind the higher floor and the worse MAPE: **pure single-op kernel
time carries a large per-LAUNCH floor (~14 µs) that is per-execute, not per-op.**

- Right for **isolated single-op latency** (one kernel run alone).
- WRONG to sum per-layer in a whole model: in a fused graph that floor is paid ~once,
  not N times. The loop-method *marginal* signal (floor removed) is what should be
  summed — which is why it keeps lower MAPE and stays shipped.
- The extra MAPE is the floor's *variance* (12–17 µs launch noise) swamping small ops,
  unpredictable from shape/cycles. `reshape` (23.5 %, mae 89 µs) is the extreme: a
  near-free bitcast whose real cost is dominated by floor noise.

Consistent with the whole-model compensation: the floor/host are **per-execute
constants** (belong in `C_forward` / the per-forward host term), not per-op terms.
Keep the shipped loop-method models for the SCALE-Sim per-layer path; use these
pure-device models + `host_us` to pin the per-execute floor/host constants.

## GEMM linear model updated to predict pure device latency (2026-06-24)

`tpu.py:TPUV4_REGION_TABLE` was refit on the golden pure latency (`fit_piecewise_pure.py`)
so `tpuv4_linear_model` now predicts a GEMM's **pure single-op device latency**
(the `single_op_us` quantity). Accuracy predicting the golden pure latency
(80/20 held out, piecewise 8-region):

| compute size | n (test) | MAPE | median latency |
|---|---|---|---|
| tiny  cyc<1e4 (floor-dominated) | 1002 | 24.9% | 15.2 µs |
| small 1e4–1e5 | 345 | 23.1% | 18.8 µs |
| medium 1e5–1e6 | 70 | 19.1% | 34.2 µs |
| **large >1e6 (compute-bound)** | 3 | **2.8%** | 336 µs |
| **overall** | 1420 | **24.1%** | — |

(single global G_roof: 24.6% — piecewise barely helps because the error is the
floor's launch jitter, not shape.) Spot-check vs direct `measure_xprof.py`: 2048³
predicts 85.5 µs vs 83.8 µs measured. The ~24% overall is dominated by tiny GEMMs
whose latency ≈ the unpredictable ~14 µs launch floor; where compute dominates the
model is ~3%. Whole-model totals must route through the compensation layer (the B
floor is per-launch). Loop-method (marginal) table preserved above for direct-sum.

### Can more piecewise strategies beat 24%? No — it's the ceiling (`fit_strategies_pure.py`)

| strategy | regions | held-out MAPE |
|---|---|---|
| single G_roof | 1 | 24.6% |
| K-band | 3 | 24.6% |
| cycle-magnitude bins | 6 | 24.6% |
| arithmetic-intensity bins | 4 | 24.6% |
| ntiles bins | 5 | 24.7% |
| **KxNxM (current)** | 8 | **24.1%** |
| KxNxM × cyc-small | 12 | 24.2% |
| G_roof + C·ntiles floor term | — | 24.2% |
| **HistGBR (10 log features, ML ceiling)** | — | **26.5%** ❌ worse |
| HistGBR (log-target) | — | 26.3% ❌ worse |

Decisive: nothing beats the 8-region G_roof, and a flexible **gradient-boosted model
does WORSE (26%)** and far worse on compute-bound shapes (37% vs G_roof's ~3%) — it
overfits the floor noise. So there is **no shape structure left to exploit**; the
~24% is irreducible launch jitter of the ~14 µs per-launch floor on the ~70% of
shapes too small to rise above it (that floor varies ±~15–20% run-to-run, which is
not a function of M,N,K). The model is already at the achievable ceiling; the
predictable part (compute-bound slope) is captured to ~3%.

---

# TPU v6e pure-device (fusion) calibration (2026-06-24)

Re-profiled the v6e GEMM data with the **corrected** collector (reads the inner
`fusion` span, not the outer `jit_*` wrapper). Earlier v6e pure data was collected
with the pre-fix collector (whole-program span) and is superseded.

- Data: `gemm_pure_master_tpuv6e.csv` — **1260 stratified GEMMs** (8 regions at
  quota via `stratified_shapes.py`), `latency_us_device` = true matmul fusion time.
- Fits: `gemm_linear_pure_tpuv6e.json` (single G_roof) and
  `gemm_pure_piecewise_tpuv6e.json` (8-region table + constants).

## Execution decomposition (v6e ≠ v4)

| quantity | v6e | v4 (ref) |
|---|---|---|
| fusion floor (tiny GEMM) | ~1.27 µs | ~1.3 µs |
| program − fusion (device launch span) | **~0.006 µs (none)** | ~11 µs |
| host (`wall − program`), per execute | **~137 µs flat** | ~92 µs |

**Key v6e difference:** the per-launch overhead is **entirely host dispatch**
(~137 µs/execute) — there is **no separate device "program" span** (program ≈
fusion). So the v6e per-execute *device-busy* floor is just the fusion floor
(~1.3 µs), and the ~137 µs is host, not device.

## GEMM fusion fit

| model | held-out MAPE |
|---|---|
| single G_roof | 27–29% |
| **piecewise (8 region)** | **11.6%** |

Much better than v4's pure-device 24% ceiling: v6e fusion labels are *clean*
(launch overhead lives in host, excluded from the fusion span), so there's no
~14 µs launch-floor jitter corrupting small-GEMM labels. Region table in
`gemm_pure_piecewise_tpuv6e.json` (rule `(foldK>1)*4+(foldN>=16)*2+(foldM>=16)`).

## Implication for the whole-model compensation
On v6e the device-busy per-execute floor ≈ the fusion floor (~1.3 µs), and the
~137 µs host term is per-execute dispatch (not device). The compute-side
`single_op` is the fusion latency (piecewise model above); the v6e `C_forward` /
host constants for `total_time_report.py` should be pinned from these (next step),
not reused from v4.

---

## Batch handling + whole-model (final architecture, 2026-06-25)

### Attention / batched dot_general -> per-head GEMM + 2-level latency
SCALE-Sim's cycle model is non-batch. The converter previously folded the head
count into BOTH M and N (`1536x1536` for 12 heads) -> a dense matmul computing all
cross-head pairs, ~12x over-count, which pushed Sum(GEMM) ABOVE the whole-model time
(impossible -- compute is incompressible). Fixed: a batched dot_general now maps to
the **per-head (M,N,K) + a batch count B**, and the latency is built in two levels:

  level 1 (cycle model)  : single per-head GEMM time           tpuv4_linear_model
  level 2 (batch reduce) : x B x R(B,M,N)                       tpuv4_batch_reduction
      R = u + (1-u)/B,  u = nt^0.805/(nt^0.805+21),  nt = ceil(M/128)*ceil(N/128)

R is the measured fact that one batched kernel shares a single array fill, so it
costs LESS than B separate GEMMs (up to 16x less for many tiny heads). Calibrated on
a TPU batch x shape sweep (`measure_batch_sweep.py`, `batch_reduction.csv`), 6.3% MAPE.
Kept SEPARATE from the cycle model so each recalibrates independently.

### Second M bar (foldM>=6) -> 12 regions
A single >1-tile bar lumped M=256-512 (still fill-dominated: the cycle formula's
additive 3*128 fill term over-counts small M) with M>=2048, under one slope ->
over-predicted small-multi-tile GEMMs ~1.7x (qwen MLP/LM-head), violating the
Sum(GEMM)<truth bound at seq>=256. Fix: a 2nd bar on M at foldM>=6 (M>=768) gives M
three levels (1 / 2-5 / >=6 tiles) -> 12 regions. After this, Sum(GEMM)<truth holds
for all 3 LLMs x seq{128,256,512}. A 2nd bar on N or K was tested and rejected:
only ~0.5% gain, creates empty regions, and has no additive-fill physical cause.

### Whole-model (batch-1)
    T_device ~= a0*Sum(GEMM) + a1*Sum(non-compute) + C_forward
    a0 = 1.0 (GEMM passthrough; Sum(GEMM) is the right magnitude, validated by the
              incompressibility bound -- pinned, not fit, for robustness),
    a1 ~= 0.028 (non-compute fusion survival, ~97% fused away),
    C_forward ~= 185 us (once-per-forward host/launch overhead).
Lives in the `tuned_us` column of TIME_REPORT.csv (a0*single for GEMM, a1*single for
non-compute, C in the TOTAL row). In-sample 11% MAPE, leave-one-model-out ~16%
(3 LLMs x 3 seq). a0 free gives 10.1% but unstable (15-31% LOMO) -> pin a0=1.
Batch>1 (inference) occupancy deliberately NOT modelled (keeps the system simple).
Weak spot: qwen seq128 (-26%) -- large vocab, embedding/LM-head overhead the a1 term
under-weights. More calibration models would tighten the 2 constants.

## v6e whole-model compensation (2026-06-25) — v6e now mirrors v4

All 4 layers calibrated for v6e (CALIBRATION_RUNBOOK):
- **L1 GEMM fusion**: 12-region `TPUV6E_REGION_TABLE` in tpu.py (held-out 11.7%).
- **L2 batch reduction**: `tpuv6e_batch_reduction` reuses the v4 shape (same 128x128
  array; p,c recalibration optional — RUNBOOK step 3).
- **Ops**: loop-method `model/tpuv6e/` (4–6% per-op, kept as production single_op).
- **Whole-model**: `COMPENSATION_BY_GEN["TPUv6e"]`, same form as v4, a0 pinned=1.

Whole-model fit (`fit_compensation_v6e.py`, batch-1, torch.compile device-busy truth
`e2e_device_truth_tpuv6e.csv`, sums `calib_tpuv6e.csv` from per-seq StableHLO export
+ bypass): **a1=0.0270, C_forward=493us, in-sample 7.3% MAPE, leave-one-model-out
14.4%** (vs v4's a1=0.028 / 11% / 16%). `Sum(GEMM) < truth` holds for all 9 points.

End-to-end through `scale.py -b -c configs/tpuv6e.cfg` (tuned_us TOTAL vs truth, seq128):
gpt2 −5.1% · smollm2-135m −1.2% · qwen2.5-0.5b −19.2% (the same qwen/seq128
under-prediction v4 shows — large vocab embedding/LM-head the a1 term under-weights).
Pipeline: export_stablehlo_v6e.py (fp32, shapes only) -> build_calib_v6e.py
(JAX_PLATFORMS=cpu bypass sums) -> fit_compensation_v6e.py -> total_time_report.py.
