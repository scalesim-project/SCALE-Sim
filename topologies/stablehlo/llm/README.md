# LLM StableHLO graphs + real TPU reference performance

Exported LLM forward graphs (StableHLO MLIR) to validate SCALE-Sim's TPU time
prediction end-to-end, plus their **measured device-busy latency** on real hardware.

| file | model | layers | dot_general | non-compute ops |
|------|-------|-------:|------------:|----------------:|
| `gpt2.stablehlo.mlir`             | GPT-2 (124M)        | 12 | 73  | 791  |
| `qwen2.5-0.5b.stablehlo.mlir`     | Qwen2.5-0.5B (GQA)  | 24 | 218 | 2179 |
| `smollm2-135m.stablehlo.mlir`     | SmolLM2-135M (GQA)  | 30 | 272 | 2527 |
| `tiny_transformer_pytorch.stablehlo.mlir` | synthetic tiny      | 3  | 19  | 195  |

## Scripts
- **`export_LLM.py`** — export the three LLMs to StableHLO MLIR (**f32**, seq fixed,
  eager attention so attention matmuls stay explicit `dot_general`). Runs on CPU.
  ```bash
  python3 export_LLM.py --models all --seq-len 128
  ```
  f32 (uniform dtype): the whole graph is one dtype, so it has none of the f16/bf16
  softmax/GELU/LayerNorm `f16↔f32` convert islands (gpt2: 5 converts vs 137 in f16 —
  those upcasts are inherent to the model's numerics and become explicit converts in a
  16-bit export). SCALE-Sim is shape-only for non-compute ops and uses a fixed bf16
  byte width for GEMM, so the element dtype changes no predicted shape or cycle count.
- **`profile_model_on_tpu.py`** — measure real **device-busy** latency on a TPU (bf16,
  eager). Needs the TPU free. Output: `measured_<gen>.json` (key `device_busy_us`).
  ```bash
  PJRT_DEVICE=TPU python3 profile_model_on_tpu.py --models all --gen tpu_v4
  ```

### tiny_transformer (synthetic, runs in the FULL cycle-accurate sim)
A 3-layer transformer (hidden=512, 4 heads, ff=768, seq=128, vocab=2048) with the full
LLM op set (LayerNorm, softmax, GELU, residual adds, embedding), small enough to run in
SCALE-Sim's **full cycle-accurate sim** (no `--bypass`) in **~98 s** — the three real
LLMs need bypass. Two equivalent exporters:
- `export_tiny_transformer_pytorch.py` → `tiny_transformer_pytorch.stablehlo.mlir` (PyTorch)
- `export_tiny_transformer_jax.py` → `tiny_transformer_jax.stablehlo.mlir` (JAX, CPU-only)

Same 19 GEMM `dot_general` layers; JAX decomposes LayerNorm/softmax/GELU into more
primitives (377 vs 195 non-compute), so only the non-compute count differs.

## Predict with SCALE-Sim (no TPU needed)
Run from the repo root (so the `scalesim` package is importable — use `-m` or `PYTHONPATH=.`):
```bash
python3 -m scalesim.scale -b -c configs/tpuv4.cfg \
  -t topologies/stablehlo/llm/gpt2.stablehlo.mlir -p results/gpt2_bypass
# -> results/gpt2_bypass/<run>/TIME_REPORT.csv
#    columns: single_op_us (standalone per-op) | tuned_us (whole-model contribution)
#    TOTAL row's tuned_us = the device-time prediction (C_forward added once there)
```
(`-c configs/tpuv6e.cfg` for TPU v6e.)

## Reference: measured device-busy latency (TPU v4, bf16, seq=128, batch=1)

Predictions use the **pure** per-op models (`scalesim/model/tpuv4`, xprof single-op
spans, deflated; `reshape` = constant median ~7µs); the whole-model compensation is
`T = Sc + a1·Sn + C0` (a0=1, a1=0.0491, C0=83.8 µs), refit on the **f32** graphs of
these 3 LLMs × seq{128,256,512,1024} + the tiny anchor.

| model | measured device-busy | SCALE-Sim `tuned_us` TOTAL (seq 128) |
|-------|---------------------:|---------------------------:|
| gpt2          | ~495 µs  | ~413 µs |
| qwen2.5-0.5b  | ~1462 µs | (see bypass) |
| smollm2-135m  | ~950 µs  | (see bypass) |
| tiny_transformer | ~138 µs | ~145 µs |

(seq=128 is the worst case — short-seq embedding/vocab overhead the flat `a1`
under-weights; e.g. gpt2 lands within ~1–2% at seq≥256. In-sample MAPE ~10.3%.)

**Device-busy, not wall-clock.** `profile_model_on_tpu.py` uses
`torch.compile(backend="openxla")` (compile once, re-execute) and isolates device time
as `median(run+wait) − median(run, no-wait)`. Plain eager torch_xla re-traces every call,
so its wall-clock is ~99% host tracing (gpt2 reads ~19 ms, not its ~0.5 ms device time)
— that is **not** the quantity to compare against SCALE-Sim. Compare the measured
`device_busy_us` against the `tuned_us` TOTAL (whole-model batch-1 prediction, ~11.7% MAPE).

## Re-calibrate / re-measure on another TPU generation
Latency is per-generation; tag with `--gen` (writes `measured_<gen>.json`). Full
recalibration of the prediction model is in `scalesim/calibration_pure/CALIBRATION_RUNBOOK.md`.
