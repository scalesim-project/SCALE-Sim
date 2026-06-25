# LLM StableHLO graphs + real TPU reference performance

Exported LLM forward graphs (StableHLO MLIR) used to validate SCALE-Sim's TPU time
prediction end-to-end, plus their **measured** latency on real hardware.

| file | model | layers | dot_general | non-compute ops |
|------|-------|-------:|------------:|----------------:|
| `gpt2.stablehlo.mlir`         | GPT-2 (124M)        | 12 | 73  | 791  |
| `qwen2.5-0.5b.stablehlo.mlir` | Qwen2.5-0.5B (GQA)  | 24 | 218 | 2179 |
| `smollm2-135m.stablehlo.mlir` | SmolLM2-135M (GQA)  | 30 | 272 | 2527 |
| `tiny_transformer.stablehlo.mlir` | synthetic tiny  | 3  | 19  | 195  |

Export config for the three LLMs: **batch=1, seq_len=128, eager attention** (bf16).

`tiny_transformer.stablehlo.mlir` is a synthetic 3-layer transformer (hidden=512,
4 heads, ff=768, seq=128, vocab=2048) with the full LLM op set (LayerNorm, softmax,
GELU, residual adds, embedding). It is small enough to run in SCALE-Sim's **FULL
cycle-accurate sim** (no `--bypass`) in **~98 s** -- the three real LLMs need bypass.
Measured device-busy on TPU v4 ~133 us; the size-dependent whole-model predictor
lands it at +4% (a fixed C_forward gave +75% -- this tiny model motivated making C
scale with GEMM-kernel count).

Two equivalent exports of the same tiny transformer are provided:
- `export_tiny_transformer.py`     -> `tiny_transformer.stablehlo.mlir`     (PyTorch / torch.export + torch_xla)
- `export_tiny_transformer_jax.py` -> `tiny_transformer_jax.stablehlo.mlir` (JAX / `jax.jit(...).lower().as_text()`)

Both have the **same 19 dot_general (GEMM) layers**; the JAX lowering decomposes
LayerNorm/softmax/GELU into more primitive ops (377 vs 195 non-compute), so the
GEMM/compute path is identical and only the non-compute count differs. The JAX
exporter needs no torch (CPU-only: `JAX_PLATFORMS=cpu python3 export_tiny_transformer_jax.py`),
and passes weights as function args (keep the MLIR small, ~38 KB, not multi-MB).

## Predict with SCALE-Sim (no TPU needed)
```bash
JAX_PLATFORMS=cpu python3 -m scalesim.scale \
  -t topologies/stablehlo/llm/gpt2.stablehlo.mlir \
  -c configs/tpuv4.cfg -p ./results/llm_gpt2 -i gemm --bypass
# -> results/llm_gpt2/<run_name>/TIME_REPORT.csv  (TOTAL row: device_us, with-dispatch_us)
```

## Reference: real TPU performance
`measured_<gen>.json` holds the measured whole-model latency (with the measurement
conditions in its `_meta`). Shipped reference — **TPU v4**, bf16, seq=128, batch=1,
eager, median of 50 compiled forwards:

| model | measured median |
|-------|----------------:|
| gpt2 | ~19.0 ms |
| qwen2.5-0.5b | ~57 ms |
| smollm2-135m | ~66.8 ms |

These are **eager wall-clock** and hardware-specific. Compare against the
`time_with_dispatch_us` TOTAL from SCALE-Sim (the `device_us` TOTAL is the
compute/fused-regime quantity). See `SCALE-Sim_TPU/reports/C_integration.md`.

## Re-measure on another TPU VM
Needs `torch`, `torch_xla`, `transformers` and an idle TPU (exclusive PJRT access):
```bash
python3 topologies/stablehlo/llm/run_groundtruth.py --gen tpu_v5e   # -> measured_tpu_v5e.json
```
(Latency is per-TPU-generation, so tag the output with `--gen`.)
