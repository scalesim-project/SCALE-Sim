# LLM StableHLO graphs

Exported LLM forward graphs (StableHLO MLIR) used as SCALE-Sim inputs, plus the
scripts that produce them and a script to measure real TPU latency for comparison.

## Exported graphs (SCALE-Sim inputs)

These `.mlir` files are what you feed to SCALE-Sim. They are **f32**, single forward
pass, sequence length fixed (default 128), eager attention (so attention matmuls stay
explicit `dot_general`). StableHLO is target-independent — the same file is used for
any TPU generation.

| file | what it is |
|------|------------|
| `gpt2.stablehlo.mlir`           | GPT-2 (124M) forward graph |
| `qwen2.5-0.5b.stablehlo.mlir`   | Qwen2.5-0.5B (GQA) forward graph |
| `smollm2-135m.stablehlo.mlir`   | SmolLM2-135M (GQA) forward graph |
| `tiny_transformer_pytorch.stablehlo.mlir` | small synthetic transformer (PyTorch export) |
| `tiny_transformer_jax.stablehlo.mlir`     | the same tiny transformer (JAX export) |

The **tiny transformer** (3 layers, hidden 512, 4 heads, ff 768, seq 128, vocab 2048)
is small enough to finish the full cycle-accurate sim quickly; the three real LLMs are
large, so their full sim is lengthy (many large GEMMs). The PyTorch and JAX exporters
produce the same arithmetic (same `dot_general` layers); only the surrounding non-compute op
decomposition differs slightly between the two frontends.

## Scripts

| script | what it does |
|--------|--------------|
| `export_LLM.py` | Export the three real LLMs to StableHLO MLIR (f32, CPU). `--models {gpt2,qwen2.5-0.5b,smollm2-135m,all}` `--seq-len N` `--out-dir DIR`. |
| `export_tiny_transformer_pytorch.py` | Export the synthetic tiny transformer via PyTorch → `tiny_transformer_pytorch.stablehlo.mlir`. |
| `export_tiny_transformer_jax.py` | Export the same tiny transformer via JAX (CPU-only) → `tiny_transformer_jax.stablehlo.mlir`. |
| `profile_model_on_tpu.py` | Measure the **real device-busy latency** of the three LLMs by running them on **actual TPU hardware** (bf16, `torch.compile`), to compare against SCALE-Sim's prediction. **Must run on a TPU VM** (with the TPU free). This is the ground-truth measurement — it runs the real model, it does **not** use SCALE-Sim. Writes `measured_<gen>.json`. |

`measured_<gen>.json` (e.g. `measured_tpu_v4.json`) — output of `profile_model_on_tpu.py`:
per-model measured device-busy latency for the tagged TPU generation. Not present until
the script is run.

## Usage

Export the graphs (CPU, no TPU needed):
```bash
python3 export_LLM.py --models all --seq-len 128
python3 export_tiny_transformer_pytorch.py
```

Predict with SCALE-Sim (from the repo root):
```bash
python3 -m scalesim.scale -c configs/tpuv4.cfg \
  -t topologies/stablehlo/llm/gpt2.stablehlo.mlir -p results/gpt2
python3 -m scalesim.scale -c configs/tpuv4.cfg \
  -t topologies/stablehlo/llm/tiny_transformer_pytorch.stablehlo.mlir -p results/tiny
```
> **Note:** this runs the full cycle-accurate systolic-array simulation. The
> `tiny_transformer` finishes in a couple of minutes; the **real LLMs take a long
> time** (their many large GEMMs are simulated cell-by-cell) — expect a lengthy run.

The whole-model prediction is the `tuned_us` TOTAL in `results/<run>/TIME_REPORT.csv`.
(`-c configs/tpuv6e.cfg` for TPU v6e.)

Measure real TPU latency — **run this on a TPU VM** (it executes the real model on
hardware; no SCALE-Sim):
```bash
PJRT_DEVICE=TPU python3 profile_model_on_tpu.py --models all --gen tpu_v4
```

## Notes
- **f32, not f16/bf16:** a single dtype keeps the graph clean (no `f16↔f32` convert
  islands around softmax/GELU/LayerNorm). The dtype does not affect SCALE-Sim's shapes
  or GEMM cycle counts, which assume the bf16 MXU regardless.
- These graphs are *forward-only, single sequence length*. Re-run the exporter with a
  different `--seq-len` to get other lengths.
