# LLM StableHLO graphs + real TPU reference performance

Three exported LLM forward graphs (StableHLO MLIR) used to validate SCALE-Sim's
TPU time prediction end-to-end, plus their **measured** latency on real hardware.

| file | model | layers | dot_general | non-compute ops |
|------|-------|-------:|------------:|----------------:|
| `gpt2.stablehlo.mlir`         | GPT-2 (124M)        | 12 | 73  | 791  |
| `qwen2.5-0.5b.stablehlo.mlir` | Qwen2.5-0.5B (GQA)  | 24 | 218 | 2179 |
| `smollm2-135m.stablehlo.mlir` | SmolLM2-135M (GQA)  | 30 | 272 | 2527 |

Export config for all three: **batch=1, seq_len=128, eager attention** (run in bf16).

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
