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
  isolating kernel time. bf16, single device, no SPMD.
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
