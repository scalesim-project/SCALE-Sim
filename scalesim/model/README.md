# Non-compute op latency models

Per-op latency regressors for the StableHLO ops SCALE-Sim does **not** simulate
cycle-accurately (everything except `dot_general`/`convolution`). The
`NonComputeLatencyPredictor` (in `stablehlo_converter.py`) loads these and predicts
each op's latency from its tensor shape.

- `tpuv4/*.pkl` — one model per op, for TPU v4. These are the **pure** (xprof
  single-op device-span) models: the standalone per-op kernel time read straight from
  the device timeline (§f), which is what the whole-model compensation now sums.
  Auto-loaded by filename: a file `reshape.pkl` is matched to `stablehlo.reshape`
  (last name-token), so new ops are drop-in with no converter changes.
- `calibration/` — scripts to measure and train the models on a TPU (below).

Each `.pkl` is `{"model": HistGradientBoostingRegressor, "op_name": str,
"metadata": {val_mae, val_mape, train_rows, ...}}`.

---

## a. Methodology

Same recipe for every op (matches the originally-shipped 5 models):

- **Features (5):** `d0, d1, d2, size = d0·d1·d2, log2_size`, taken from the op's
  **first input** shape padded/flattened to 3-D (`_opinfo_to_shape`).
- **Label:** the op's **pure device-span** in µs — its standalone kernel time read
  directly from the xprof device timeline (NOT host/PJRT dispatch).
- **Model:** `HistGradientBoostingRegressor(loss="absolute_error",
  learning_rate=0.06, early_stopping="auto")`, 80/20 split.
- **Measurement (§f):** `collect_pure_device_tpu.py` reads the single-op kernel span
  straight from the xprof device timeline — the lean, standalone kernel time. This is
  consistently *smaller* than the loop-method `(t_K − t_1)/(K−1)` marginal (≈0.7–0.95×
  for elementwise, ~0.3× for `reduce`): the loop marginal carries per-iteration
  loop-body/control overhead that a single clean kernel does not. The pure span is the
  honest standalone device cost, so the whole-model fit sums it. bf16, single device,
  no SPMD.
- **Sampling (`sample_shapes`):** distinct 3-D shapes in two buckets — **60%
  LLM-anchored** (small batch/head `d0`, seq-menu `d1`, `d2` log-uniform up to 160k =
  hidden → FFN → vocab, covering qwen/llama vocab too) + **40% broad/general** (wide
  `d0/d1/d2` so non-LLM shapes degrade gracefully). The shipped pure models use
  **1000 shapes/op** (the loop-method sets used ~1500).

**Shape-only by design:** the models take no op *attributes* (transpose permutation,
reduce axis, …). For size-driven ops this is exact; for a few ops it is an
approximation that bakes in the sampled configuration (see limitations).

---

## b. Accuracy (TPU v4, bf16; held-out val MAPE)

25 ops, **deflated pure device-span** label (fixed inner-span trace rule, LLM-bucket
n~600/op): **mean 7.0%, median 5.7%** (was ~10% with the pre-fix floored labels).

| op | MAPE | | op | MAPE |
|----|-----:|-|----|-----:|
| power | 4.0% | | divide | 6.1% |
| tanh | 4.5% | | rsqrt | 6.1% |
| negate | 4.5% | | select | 7.0% |
| exponential | 5.1% | | slice | 7.4% |
| multiply | 5.4% | | convert | 8.5% |
| add | 5.5% | | broadcast_in_dim | 11.0% |
| logistic | 5.5% | | cosine | 12.0% |
| transpose | 5.5% | | sine | 12.1% |
| maximum | 5.6% | | concatenate | 15.5% |
| batch_norm_training | 5.7% | | reduce | 15.7% |
| and 5.7 · subtract 5.7 · compare 5.9 · minimum 6.0 | | **reshape** | **const median ~7µs** |

- **22 elementwise/compute ops are 4–8%.** The floor fix (inner-span trace rule)
  brought these down from ~8–12%.
- **`reshape` is a constant = the median (~7µs)**, NOT a shape model. reshape latency
  is bimodal (metadata ~0 vs relayout ~1000s µs; dataset mean 276µs) and unpredictable
  from shape — the regressor over-predicted it ~37× at vocab sizes and dominated the
  whole-model `Sn` (63%). The median constant fixes the whole-model (10.3% vs 16% with
  the HGBR reshape). See `total_time_report.py` and RESULTS.md.
- **Layout/transcendental ops** (`reduce`, `concatenate`, `sine`, `cosine`,
  `broadcast_in_dim`) are 11–16% — the shape-only model has no axis feature.

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
`pip install jax[tpu] scikit-learn pandas`, bf16. ~25 ops × 1000 shapes.

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

Measured whole-model wall-clock on this v6e (`topologies/stablehlo/llm/profile_model_on_tpu.py
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

The collector records three nested levels per op (cols `latency_us`=`kernel_us` /
`program_us` / `wall_us`), exactly like the GEMM case (see `linear_model/README.md`
§e for the table): the op's **`fusion` kernel** span, the **whole-program** (`jit_*`
wrapper) span, and the python-timer wall.

**What it found (TPU v4, bf16):**
- **`host_us` is flat at ~90–94 µs** (`wall − program`) across every op type and 4
  orders of magnitude of tensor size. Per-**execute**, not per-op — a single op pays
  it once, and so does a whole fused model. (This is *why* a fitted per-op host
  constant `c` collapses to 0 in the whole-model compensation; it's one per-forward
  term, not a per-op one.) The per-launch program overhead (`program − kernel`) is
  similarly a flat ~11 µs, also per-execute.
- **The matmul/op `fusion` kernel floor is ~1.2–1.5 µs** (not the ~12 µs an earlier
  version reported). See the correction note below.

> **Extraction bug fixed (2026-06-24).** The first version took the *dominant*
> device span = the outer `jit_*` **whole-program** wrapper, so it reported a ~12 µs
> "kernel floor" that was really 1.2 µs of `fusion` + ~11 µs per-launch overhead.
> The collector now sums the **inner op spans** (`fusion`, copy, …) for `kernel_us`
> and reports the wrapper separately as `program_us`. This also fixes ops like
> `reshape` — a memory copy whose real kernel is ~1.3 µs (128×768) to ~26 µs
> (128×50257), but was reported at ~13–460 µs when the ~10 µs wrapper floor was
> read as the kernel. (The first re-collection's datasets still had this floor; the
> *datasets* were re-collected 2026-06-26 with the rule below.)

**Per-op inclusion rule (audited across all 24 ops, 2026-06-26).**
`kernel_us` = **sum of the `dur` of every device-stream event nested inside the
`jit_*` wrapper, excluding the wrapper itself**. Verified per op:
- **Simple ops** (add/mul/…/slice/convert/select, and `transpose`→`copy`,
  `reshape`→`copy`, `concatenate`→`pad_*_fusion`): exactly **one** real kernel.
- **Composite ops** emit several genuine sub-kernels that all belong to the op and
  are all summed: `batch_norm_training` (LayerNorm) = 5 fusions (`convert_reduce` +
  `fusion.1` + `subtract_multiply` + `fusion.3` + `fusion.4`); `reduce` = `reduce`+
  `reshape`; `broadcast` = `broadcast`+`reduce`.
- **Scheduling markers** (`dependency-wait`, `copy-start`, `copy-done`) are ~0 µs and
  harmless to include. No op needs special-casing — the uniform sum is correct.

**Method notes / gotchas (baked into the script):**
- The TPU device-stream PID is **auto-detected** from the trace `process_name`
  metadata (`/device:TPU:0`). Do **not** hardcode `pid == 3` — it varies across
  libtpu/multi-device.
- `jax.named_call(name=…)` does **not** propagate to the device op; the device span
  is named after the jit function. Take the **inner** (non-`jit_`) op spans as the
  kernel, NOT the dominant span (the wrapper nests them — that was the bug above).
- Summing per-event `dur` overstates a *multi-op* graph's span (MXU/VPU overlap);
  fine for a single-kernel op, but for a whole graph use `max(end)−min(start)`.
- xprof tracing is slower than the loop method (one trace/parse per shape), so `--n`
  is kept modest (400 here vs the loop method's ~1500). This is now the method used to
  **train** the shipped `tpuv4` models (the pure device-span is the honest standalone
  cost the whole-model sum needs); the §a loop method is the floor-removed alternate.

**Usage:**
```bash
cd scalesim/model/calibration
PJRT_DEVICE=TPU python3 collect_pure_device_tpu.py \
    --ops add multiply reduce transpose reshape broadcast \
    --outdir datasets_pure_tpuv4     # defaults: n=1000, iters=10, warmup=1, reps=1
```

> Whole-model use of these constants (the two-factor MXU/VPU + per-forward-host
> compensation) is in `scalesim/total_time_report.py`; the matmul floor/host
> measurement that seeded it is `SCALE-Sim_TPU/e2e_work/compensation/measure_xprof.py`.

---

## g. TPU v6e pure-device single-op set (`model/tpuv6e_pure/`) (2026-06-24)

> **SUPERSEDED (2026-06-26).** The framing below ("pure = audit, loop = production")
> is outdated: pure is now the **production** method (§f), and the v4 pure models were
> re-collected with the **fixed inner-span trace rule** + the 60/40 `n=1000` sampler.
> The numbers in this section are the **pre-fix** v6e audit collection (n=300, and the
> old ~10µs-floor labels), kept for history; re-collect v6e per the CALIBRATION_RUNBOOK
> to refresh them.

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
