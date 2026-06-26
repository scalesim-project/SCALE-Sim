# Shared calibration StableHLO graphs (f32)

The 3 reference LLMs × seq {128, 256, 512, 1024}, exported as **f32** StableHLO
(`export_LLM.py`). Layout: `s<seq>/<model>.stablehlo.mlir`.

These are **target-independent** — the same graphs feed the whole-model compensation
fit for every TPU generation (v4, v6e, …), so no re-export is needed per generation.
Used by `fit_compensation_pure.py` (default `--mlir-root`).

Regenerate (only if the models/exporter change):
```bash
for s in 128 256 512 1024; do
  python3 ../../../topologies/stablehlo/llm/export_LLM.py --models all --seq-len $s --out-dir s$s
done
```
