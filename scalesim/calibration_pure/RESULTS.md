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
