# Non-compute op latency models

Per-op latency regressors for the StableHLO ops SCALE-Sim does **not** simulate
cycle-accurately (everything except `dot_general`/`convolution`). The
`NonComputeLatencyPredictor` (in `stablehlo_converter.py`) loads these and predicts
each op's latency from its tensor shape.

- `tpuv4/*.pkl` — one model per op, for TPU v4. Auto-loaded by filename: a file
  `reshape.pkl` is matched to `stablehlo.reshape` (last name-token), so new ops are
  drop-in with no converter changes.
- `calibration/` — scripts to measure and train the models on a TPU (below).

Each `.pkl` is `{"model": HistGradientBoostingRegressor, "op_name": str,
"metadata": {val_mae, val_mape, train_rows, ...}}`.

---

## a. Methodology

Same recipe for every op (matches the originally-shipped 5 models):

- **Features (5):** `d0, d1, d2, size = d0·d1·d2, log2_size`, taken from the op's
  **first input** shape padded/flattened to 3-D (`_opinfo_to_shape`).
- **Label:** measured **device time** in µs (kernel-only; host/PJRT dispatch removed).
- **Model:** `HistGradientBoostingRegressor(loss="absolute_error",
  learning_rate=0.06, early_stopping="auto")`, 80/20 split.
- **Measurement:** an in-JIT `fori_loop` runs the op many times on-device in a single
  dispatch; device time = `(t_K − t_1)/(K−1)` cancels the fixed ~55 µs host dispatch,
  isolating kernel time. bf16, single device, no SPMD. (A trace-authoritative
  cross-check, `collect_pure_device_tpu.py`, reads kernel time straight from the
  xprof device timeline instead of subtracting — see §f.)
- **Sampling:** ~1500 shapes/op, log-uniform over an activation-like 3-D space.

**Shape-only by design:** the models take no op *attributes* (transpose permutation,
reduce axis, …). For size-driven ops this is exact; for a few ops it is an
approximation that bakes in the sampled configuration (see limitations).

---

## b. Accuracy (TPU v4, bf16; held-out val MAPE)

25 ops. Target was the existing 5 models' ~5–7%; most ops meet it.

| op | MAPE | | op | MAPE |
|----|-----:|-|----|-----:|
| concatenate | 3.54% | | reshape | 4.95% |
| slice | 3.62% | | maximum | 4.97% |
| negate | 3.77% | | multiply | 5.16% |
| batch_norm_training | 4.13% | | rsqrt | 5.26% |
| exponential | 4.19% | | divide | 5.29% |
| power | 4.20% | | select | 5.37% |
| tanh | 4.22% | | logistic | 5.49% |
| compare | 4.34% | | and | 5.56% |
| transpose | 4.43% | | cosine | 8.48% |
| reduce | 4.50% | | sine | 8.49% |
| convert | 4.92% | | broadcast_in_dim | 9.54% |
| add 4.06 · subtract 4.12 · minimum 4.25 | | | | |

- **21 / 25 within 5–7%.** Outliers: `sine`/`cosine` (~8.5%, transcendental, high
  variance) and `broadcast_in_dim` (9.5%).
- **Methodology check (M0):** the re-collected `add` reaches 4.06% vs the
  originally-shipped 6.6% — the pipeline reproduces and slightly beats the reference.

**Limitations (read before trusting a per-op number):**
- **Train/serve shape skew** for `broadcast_in_dim` (and would-be `gather`): trained
  on the output/driver shape, but the converter feeds the *first-input* shape. The
  fix is a converter feature change (pass output shape), not more data.
- **Not yet modeled (predict 0):** `gather`, `reduce_window` (need the output-shape /
  window feature change); `constant`, `func.return` are correctly free.
- **Whole-model context:** for an *eager* run, summed device times are a minority of
  the measured wall-clock — per-op kernel **dispatch (~20 µs/op)** dominates. These
  models are the device-busy/fused-regime component. See
  `SCALE-Sim_TPU/reports/C_integration.md`.

---

## c. How to measure & train on another TPU

To build the models for a different TPU, re-run `calibration/` **on that TPU** and
write the `.pkl`s into a new `model/<generation>/` directory.

**Requirements:** exclusive PJRT access (no other process on `/dev/accel*`),
`pip install jax[tpu] scikit-learn pandas`, bf16. ~25 ops × 1500 shapes ≈ ~1 hr.

```bash
cd scalesim/model/calibration

# 1. measure device-time latency per op on the target TPU (resumable; one CSV per op)
PJRT_DEVICE=TPU python3 collect_ops_tpu.py \
    --ops add subtract multiply maximum minimum divide negate rsqrt exponential \
          logistic tanh power reduce slice transpose reshape broadcast concatenate \
          sine cosine convert compare and select batch_norm_training \
    --n 1500                                  # -> <op>_dataset.csv

# 2. train one HistGBR per dataset -> .pkl with {model, op_name, metadata}
python3 train_ops.py --outdir ../<generation>     # e.g. ../tpuv5e
```

**Install:** the trainer names each file by the op's StableHLO last-token
(`reshape.pkl`, `broadcast_in_dim.pkl`, …) so `NonComputeLatencyPredictor` matches
them automatically — just point it at the new `model/<generation>/` dir (or make it
the default). To add a brand-new op: add it to the `OPS`/`build()` tables in
`collect_ops_tpu.py`, then collect + train.

> Fuller writeup: `SCALE-Sim_TPU/reports/A_noncompute_ops.md`.

---

## d. TPU v6e calibration progress (in progress)

> Live log for building the **TPU v6e** per-op `.pkl` models on this VM. Updated as
> each step lands so a VM crash leaves a recoverable breadcrumb. Started 2026-06-21.

**Target device:** `TPU v6 lite` (v6e), 8 devices, jax 0.6.2, bf16, single-device.
**Env:** system `python3` (sees the TPU); `scikit-learn`+`pandas` installed.

| # | step | command | output | status |
|---|------|---------|--------|--------|
| 0 | env + log | `pip install scikit-learn pandas`; verify TPU | — | ✅ done |
| 3 | collect 25 ops | `PJRT_DEVICE=TPU python3 collect_ops_tpu.py --ops … --n 1000 --outdir datasets_tpuv6e` (run in 5 op-batches for crash-safety) | `calibration/datasets_tpuv6e/<op>_dataset.csv` × 25 | ✅ done |
| 4 | train | `python3 train_ops.py --datadir datasets_tpuv6e --outdir ../tpuv6e` | `model/tpuv6e/*.pkl` × 25 | ✅ done |
| 4 | integrate | `NonComputeLatencyPredictor` is now generation-aware: `convert_mlir_if_needed(config_file=…)` reads `TimeLinearModel` and selects `model/tpuv6e/` (falls back to `tpuv4` if absent) | `stablehlo_converter.py`, `scale.py` | ✅ done |

**Per-op val MAPE (TPU v6e, n=1000/op, 80/20 split), best→worst:**
`concatenate 3.39 · convert 4.54 · rsqrt 4.60 · negate 4.61 · slice 4.61 ·
exponential 4.70 · transpose 4.86 · tanh 4.96 · reshape 4.99 · power 5.01 ·
reduce 5.34 · logistic 5.34 · maximum 5.47 · and 5.60 · compare 5.61 ·
subtract 5.74 · multiply 5.78 · minimum 5.79 · divide 5.81 · add 6.05 ·
batch_norm_training 7.23 · cosine 8.96 · sine 9.30 · select 9.32 ·
broadcast_in_dim 21.76`  (%)

Most ops land in the 4–6% band (matching the v4 reference). `sine`/`cosine` (~9%,
transcendental variance) and `broadcast_in_dim` (21.8%) are the known outliers —
the latter is the documented train/serve shape-skew (trained on the driver/output
shape, but the converter feeds the first-input shape; see Limitations above).

**End-to-end validated** (2026-06-21): `python3 scalesim/scale.py -b -c
configs/tpuv6e.cfg -t topologies/stablehlo/llm/smollm2-135m.stablehlo.mlir` modeled
2781/2800 ops; the predictor resolved `TimeLinearModel: TPUv6e` → `model/tpuv6e/`
and the reported per-op latencies match the tpuv6e `.pkl`s (not the tpuv4
fallback). Data: `calibration/datasets_tpuv6e/*.csv`.

### e. End-to-end accuracy vs real v6e silicon (2026-06-22)

Measured whole-model wall-clock on this v6e (`topologies/stablehlo/llm/run_groundtruth.py
--gen tpu_v6e`, torch_xla, seq=128, batch=1, 50 iters; ground truth in
`measured_tpu_v6e.json`) vs the SCALE-Sim bypass TOTAL (`-b -c configs/tpuv6e.cfg`):

| model | measured | device-only | +v4 dispatch (19.75µs) | +v6e dispatch (5.712µs) |
|-------|---------:|------------:|-----------------------:|------------------------:|
| gpt2 | 7,049 µs | −69.8% | +172.3% | **+0.2%** |
| qwen2.5-0.5b | 18,818 µs | −66.9% | +184.7% | **+5.9%** |
| smollm2-135m | 22,376 µs | −75.7% | +171.4% | **−4.2%** |
| **MAPE** | | 70.8% | 176.1% | **3.4%** |

Whole-model wall-clock = `device_compute + dispatch · n_ops`. Pure device-compute
is only ~30% of measured time; per-op overhead dominates. The inherited TPU v4
dispatch (19.75 µs/op) is ~3.5× too high for v6e — fitting `measured = device +
d·n_ops` gives **d = 5.712 µs/op** (consistent across all 3 models), for **3.4%
end-to-end MAPE**. This constant is wired generation-aware in
`scalesim/total_time_report.py` (`DISPATCH_US_PER_OP_BY_GEN`).

---

## f. Pure-device (xprof) measurement — trace-authoritative kernel time

The per-op labels in §a are isolated by an **indirect** wall-clock subtraction
(`(t_K − t_1)/(K−1)` over an in-JIT loop). `collect_pure_device_tpu.py` is the
**direct** alternative: it compiles the op (`jit().lower().compile()`), runs it
under `jax.profiler.trace`, and reads the op's span on the **TPU:0 device
timeline** straight from the trace — host/PJRT dispatch and the host→device sync
floor are excluded *by construction*, not by subtraction. It records three columns
per shape:

| column | meaning |
|--------|---------|
| `kernel_us` (=`latency_us`) | pure device kernel time (xprof device-span, per call) |
| `wall_us` | python-timer execute+block (host + device), per call |
| `host_us` | `wall − kernel` = the per-execute non-TPU cost |

`latency_us` is set equal to `kernel_us`, so the CSV is a drop-in for
`train_ops.py` (same `[d0,d1,d2,size,log2_size,latency_us]` schema; `wall/host`
are extra columns).

**What it found (TPU v4, bf16):**
- **`host_us` is flat at ~90–94 µs** across every op type and 4 orders of magnitude
  of tensor size. The host cost is **per-execute, not per-op** — a single op pays
  it once, and so does a whole fused model. (This is *why* a fitted per-op host
  constant `c` collapses to 0 in the whole-model compensation; the cost is a single
  per-forward term, not a per-op one.)
- **Pure kernel time has a small per-kernel device floor** (~10–13 µs) plus a
  size-proportional term; for matmul the slope is `1.14e-4 µs/cyc`, matching the
  GEMM linear model's effective clock. The wall-clock "floor" (~65 µs) measured by
  subtraction was **kernel floor (~12 µs) + device sync (~52 µs)** conflated; xprof
  separates them.

**Method notes / gotchas (baked into the script):**
- The TPU device-stream PID is **auto-detected** from the trace `process_name`
  metadata (`/device:TPU:0`). Do **not** hardcode `pid == 3` — it varies across
  libtpu/multi-device.
- `jax.named_call(name=…)` does **not** propagate to the device op; the device span
  is named after the jit function. The script takes the **dominant** device span
  (max total `dur`) on TPU:0, which also avoids double-counting nested fusion ops.
- Summing per-event `dur` overstates a *multi-op* graph's span (MXU/VPU overlap);
  fine for a single-kernel op, but for a whole graph use `max(end)−min(start)`.
- xprof tracing is slower than the loop method (one trace/parse per shape), so keep
  `--n` modest (default 300) — this is an **audit / constant-pinning** tool, while
  the §a loop method remains the one used to *train* the size-driven models.

**Usage:**
```bash
cd scalesim/model/calibration
PJRT_DEVICE=TPU python3 collect_pure_device_tpu.py \
    --ops add multiply reduce transpose reshape broadcast \
    --n 300 --iters 30 --outdir datasets_pure_tpuv4     # -> <op>_pure_dataset.csv
```

> Whole-model use of these constants (the two-factor MXU/VPU + per-forward-host
> compensation) is in `scalesim/total_time_report.py`; the matmul floor/host
> measurement that seeded it is `SCALE-Sim_TPU/e2e_work/compensation/measure_xprof.py`.

---

## g. TPU v6e pure-device single-op set (`model/tpuv6e_pure/`) (2026-06-24)

Rebuilt all 25 v6e per-op models on **trace-authoritative pure-kernel labels**
(`collect_pure_device_tpu.py`, xprof device-span, n=300/op, iters=12; data in
`calibration/datasets_pure_tpuv6e/*_pure_dataset.csv`), trained with the same
`train_ops.py` recipe and written to **`model/tpuv6e_pure/`**.

**These are an AUDIT / constant-pinning set, not the production models.** As §f
predicts, pure-kernel labels are noisier per shape than the loop-averaged marginal
labels, so the fitted MAPEs are higher than the loop-method `model/tpuv6e/` set:

| set | label source | n/op | typical val MAPE |
|-----|--------------|-----:|-----------------:|
| `model/tpuv6e/` (production) | loop method (marginal) | 1000 | **4–6%** |
| `model/tpuv6e_pure/` (audit) | xprof pure kernel | 300 | **12–29%** |

Pure-set per-op val MAPE, best→worst: `power 12.4 · tanh 12.4 · exponential 12.6
· rsqrt 12.7 · transpose 12.7 · negate 12.9 · logistic 13.5 · divide 13.6 · and
13.8 · multiply 13.9 · slice 14.0 · subtract 14.1 · compare 14.2 · select 14.2 ·
convert 14.3 · minimum 14.3 · add 14.6 · maximum 14.6 · concatenate 16.0 · cosine
17.9 · sine 18.6 · batch_norm_training 20.0 · broadcast_in_dim 23.2 · reduce 25.4
· reshape 28.8` (%).

The elevated MAPE is the *measurement* (single-kernel xprof span, n=300) being
noisier than the loop method — not a worse op. The pure set gives the true
standalone device kernel time (floor included), which is the right quantity for
auditing the loop models and for pinning the whole-model compensation constants
(C_forward, MXU/VPU factors in `total_time_report.py`); the loop-method
`model/tpuv6e/` remains the production single-op predictor.
