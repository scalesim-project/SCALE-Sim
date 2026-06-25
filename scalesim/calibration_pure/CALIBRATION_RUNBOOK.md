# Calibrating SCALE-Sim's TPU time model on a new generation (e.g. TPU v6e)

End-to-end recipe to reproduce the whole layered latency model (see
`SCALE-Sim_TPU/E2E_ACCURACY_REPORT.md` "Methodology") on a fresh TPU VM. Every
script referenced here lives **inside the repo**, so a VM that clones the repo has
the full pipeline. Run it **on the target TPU** (the constants are hardware-specific;
the *structure* transfers, the *numbers* are recalibrated).

The model has 4 layers, calibrated in order. `<GEN>` = `TPUv6e` (or `TPUv5e`, ...).

```
[exact]   SCALE-Sim closed-form cycles            (no calibration; identical all gens)
[level 1] cycles -> per-op GEMM fusion latency    -> <GEN>_REGION_TABLE in tpu.py
[level 2] batched-op reduction R(B,M,N)           -> <GEN> batch-reduction coeffs in tpu.py
[whole]   a0*GEMM + a1*noncompute + C             -> COMPENSATION_BY_GEN[<GEN>] in total_time_report.py
[ops]     non-compute per-op models               -> scalesim/model/<gen>/*.pkl
```

## 0. Prereqs (on the target TPU VM)
- Exclusive PJRT access (`PJRT_DEVICE=TPU`, no other process on `/dev/accel*`).
- `pip install "jax[tpu]" scikit-learn pandas` ; `pip install torch torch_xla transformers` (for whole-model ground truth).
- Clone this repo; `cd scalesim`. bf16, single device throughout.
- xprof note: the collectors read the device-timeline `fusion` span from the trace
  (PID auto-detected from `/device:TPU:0` process_name). Verify a smoke run prints
  non-empty kernel times before launching the full sweeps.

## 1. Level-1 GEMM fusion-time model  (-> `tpu.py` region table)
```bash
cd scalesim/calibration_pure
python3 stratified_shapes.py                 # -> shapes_stratified.csv (~1260, region-balanced)
cd ../linear_model/calibration
PJRT_DEVICE=TPU python3 collect_pure_device_gemm_tpu.py --shuffle \
    --shapes ../../calibration_pure/shapes_stratified.csv \
    --out   ../../calibration_pure/gemm_fusion_strat_<gen>.csv --iters 12   # golden `fusion` time
cd ../../calibration_pure
python3 fit_piecewise_pure.py gemm_fusion_strat_<gen>.csv     # prints 8-region table; see note
```
- The fit script's region scheme = `(foldK>1)*4+(foldN>1)*2+(foldM>1)`. To get the
  **12-region** scheme (with the 2nd M bar at `foldM>=6`) used for v4, apply the same
  `+ (foldM>=6)*8` split when fitting (see how v4's table was produced in RESULTS.md
  "Second M bar"). Clamp any negative `A` to 0 (floor-only thin regions).
- **Wire in:** paste the 12-region table into a new `TPUV6E_REGION_TABLE` and point
  `tpuv6e_linear_model` at it via `_tpuv4_region` (the tile-fold selector is the SAME
  -- 128x128 array -- so reuse it; only the coefficients change).
- **Validate:** `Sum(GEMM) < whole-model device time` for every model/seq (step 4). If
  violated, the small-M slope is too high -> check the 2nd M bar.

## 2. Non-compute per-op models  (-> `model/<gen>/*.pkl`)
```bash
cd scalesim/model/calibration
PJRT_DEVICE=TPU python3 collect_ops_tpu.py --ops add subtract multiply maximum minimum \
    divide negate rsqrt exponential logistic tanh power sine cosine convert compare and \
    select batch_norm_training reduce slice transpose reshape broadcast concatenate \
    --n 1000 --outdir datasets_<gen>          # one CSV per op (run in batches of 5 for crash-safety)
python3 train_ops.py --datadir datasets_<gen> --outdir ../<gen>     # -> model/<gen>/*.pkl
```
`NonComputeLatencyPredictor` auto-selects `model/<gen>/` from the config's
`TimeLinearModel:` key (falls back to tpuv4 if absent).

## 3. Level-2 batch reduction  (-> `tpu.py` batch-reduction coeffs)
```bash
cd scalesim/calibration_pure
PJRT_DEVICE=TPU python3 measure_batch_sweep.py        # -> batch_reduction.csv (einsum vs batch*single)
```
Fit `R = u + (1-u)/B`, `u = nt^p/(nt^p + c)`, `nt = ceil(M/128)*ceil(N/128)` (the fit
loop is in the RESULTS.md "batch" section / fit_piecewise helpers). **Wire in:** a
`tpuv6e_batch_reduction(B,M,N,K)` with the fitted `p,c` (mirror `tpuv4_batch_reduction`),
and point `bypass_compute._batch_reduction` at it for `<GEN>`.

## 4. Whole-model compensation  (-> `COMPENSATION_BY_GEN[<GEN>]`)
```bash
cd scalesim/calibration_pure
# 4a. ground truth: torch.compile device-busy per (model,seq), batch=1
PJRT_DEVICE=TPU python3 devicetruth_worker.py <model> <seq> 1   # repeat over models x seqs -> e2e_device_truth_<gen>.csv
# 4b. predicted sums: run the bypass on each model's StableHLO at each seq
PYTHONPATH=.. python3 ../scale.py -b -c ../../configs/<gen>.cfg -t <model>.stablehlo.mlir -p out/   # -> COMPUTE/TIME report
#     collect Sum(GEMM) (bypass "total predicted compute time") and Sum(non-compute) per run
# 4c. fit  T ~= a0*Sum(GEMM) + a1*Sum(noncompute) + C  (pin a0=1; fit a1, C)
python3 fit_compensation.py        # adapt its calib.csv inputs to the <gen> sums + truth
```
**Wire in:** `COMPENSATION_BY_GEN["<GEN>"] = {a0_mxu:1.0, a1_vpu:<fit>, c_c:0, c_n:0,
C_forward:<fit>}` in `total_time_report.py`. Batch-1 only (inference batch>1 occupancy
is deliberately out of scope).

## 5. Validate
- Per-GEMM: `gemm_model_vs_golden` style check (pred vs measured `fusion`).
- Bound: `Sum(GEMM) < whole-model truth` for all (model,seq) -- the incompressibility
  guardrail that catches batch-mapping / small-M-slope errors.
- Whole-model: `tuned_us` TOTAL in `TIME_REPORT.csv` vs `e2e_device_truth_<gen>.csv`
  (target ~10-15% MAPE; leave-one-model-out to check generalization).

## What's hardware-INDEPENDENT (do NOT recalibrate)
- The closed-form cycle count, the tile-fold region *structure* (`foldX>1`, 2nd M bar
  at `foldM>=6`), the batch-reduction *form* (`u+(1-u)/B`), the whole-model *form*
  (`a0*G+a1*N+C`, a0=1), and the `Sum(GEMM)<truth` guardrail. Only the **constants**
  (region table A/B/BW, batch `p,c`, `a1,C`, per-op `.pkl`s) are per-generation.
