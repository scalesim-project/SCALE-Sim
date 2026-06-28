# TPU GEMM/Conv linear time model

Converts SCALE-Sim **compute cycles → wall-clock microseconds** for a TPU. This is a
time-prediction layer only — it never changes how cycles are computed. It is selected
by a config's `TimeLinearModel:` key (`TPUv4`, `TPUv6e`, or `None` to disable).

## Files

| file | what it is |
|------|------------|
| `tpu.py` | The model. `tpuv4_linear_model(...)` / `tpuv6e_linear_model(...)` map cycles → µs; `tpuv4_batch_reduction(...)` / `tpuv6e_batch_reduction(...)` handle batched (multi-head) matmuls. The per-generation coefficient tables are inlined here. |
| `gemm_calibration.py` | Calibration script — `sample` (stratified shapes) / `collect` (pure device latency) / `fit` (piecewise region table) / `batch` (batch-reduction sweep) — to refit the model for a new generation. |
| `__init__.py` | Package marker. |

The fit-output / provenance JSONs (`tpu*_coeffs.json`, `gemm_pure_piecewise_tpuv6e.json`,
`gemm_linear_pure_*.json`) — the record of where `tpu.py`'s constants came from — are
archived in `SCALE-Sim_TPU/calibration_data_backup/fit_outputs/` (none are loaded at
runtime).

## How the model works

The per-piece form is **linear in cycles**, so the coefficients stay physical —
`A` is an effective clock (µs/cycle) and `B` a fixed overhead (µs):

```
time_us = A · cycles + B
```

The shipped model is **`G_roof`** — a roofline that adds one memory term so a single
form covers both compute-bound and bandwidth-bound GEMMs:

```
cyc_mem  = bytes_moved / bytes_per_cycle          # bytes_moved = 2·(M·K + K·N + M·N), bf16
time_us  = A · max(cycles, cyc_mem) + B           # max() routes memory-bound GEMMs through the BW term
```

`(A, B, bytes_per_cycle)` are **region-selected**: the `(M,N,K)` space is split into
regions by a tile-fold rule — does each dim span more than one 128×128 array tile? —
and each region has its own coefficients (`TPUV4_REGION_TABLE` / `TPUV6E_REGION_TABLE`,
selected by `_tpuv4_region(M,N,K)`). Different shapes (thin-K, wide-vocab, GEMV…) sit in
different regions so one global slope doesn't have to fit them all.

When `M,N,K` aren't available (conv layers / old call sites), it degrades to the single
global fit `A0·cycles + B0`.

### Worked example

```python
from scalesim.latency_model.linear_model.tpu import tpuv4_linear_model

# A GEMM layer with M=128, N=2304, K=768. SCALE-Sim gives its compute cycles
# (closed form: (2·128 + 128 + M − 2)·⌈N/128⌉·⌈K/128⌉ − 1 = 55079):
cycles = 55079
us = tpuv4_linear_model(cycles, M=128, N=2304, K=768)
```

Internally that call:
1. `region = _tpuv4_region(128, 2304, 768)` — picks the region from the tile folds
   (`⌈K/128⌉, ⌈N/128⌉, ⌈M/128⌉`).
2. `A, B, bytes_per_cycle = TPUV4_REGION_TABLE[region]`.
3. `cyc_mem = 2·(128·768 + 768·2304 + 128·2304) / bytes_per_cycle`.
4. `time_us = A · max(cycles, cyc_mem) + B`.

### Batched matmuls (level 2)

SCALE-Sim's cycle model is for a *single* GEMM. A batched matmul (e.g. multi-head
attention = `batch` independent per-head GEMMs) is cheaper than `batch ×` one GEMM
because the array fill/drain is amortized over the batch. So the whole-op latency is:

```
batched_us = batch · single_GEMM_us · R(batch, M, N, K)
```

where `single_GEMM_us` is the level-1 prediction for one per-head `(M,N,K)` and `R ≤ 1`
is `tpuv4_batch_reduction(batch, M, N, K)` (`R = 1` at `batch ≤ 1`). This level is kept
separate from the cycle model so the two recalibrate independently.

## How to use

**From the simulator** — set the config key; the model is applied automatically and
its prediction appears in `TIME_REPORT.csv`:
```ini
# configs/tpuv4.cfg  (or tpuv6e.cfg)
[run_presets]
TimeLinearModel: TPUv4
```
```bash
python3 -m scalesim.scale -b -c configs/tpuv4.cfg \
  -t topologies/stablehlo/llm/gpt2.stablehlo.mlir -p results/gpt2
```

**From code** — call directly with a layer's cycles and dims (see the worked example
above).

**Recalibrate for a new TPU** — run the `gemm_calibration.py` stages on that TPU, then
paste the printed **region table** into the matching `*_REGION_TABLE` in `tpu.py` (fill
the stub + add a `TimeLinearModel:` option for a new generation):
```bash
python3 gemm_calibration.py sample  --out shapes.csv                       # CPU (stratified)
PJRT_DEVICE=TPU python3 gemm_calibration.py collect --shapes shapes.csv --out gemm_master.csv
python3 gemm_calibration.py fit     --data gemm_master.csv                  # CPU; prints region table
PJRT_DEVICE=TPU python3 gemm_calibration.py batch                           # batch-reduction sweep
```
`fit` uses `tpu.py`'s 12-region scheme, so the same command produces the v4 *or* v6e
table depending on the data. Full multi-layer recipe:
`scalesim/latency_model/calibration/README.md`.
