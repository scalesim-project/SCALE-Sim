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
- xprof note: the collectors read the op's real kernel time as the **sum of the inner
  device spans nested in the `jit_*` wrapper, EXCLUDING the wrapper** (PID auto-detected
  from `/device:TPU:0`). Reading the wrapper instead = a ~10 us per-launch floor on
  every op (the old bug). Verify a smoke run prints non-empty, sub-10us kernel times.
- The whole-model calibration graphs are **committed** (`calib_mlir/s{128,256,512,1024}/
  <model>.stablehlo.mlir`, f32) and are target-independent, so step 4 needs **no
  re-export** -- the same graphs calibrate every generation.

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
PURE device-span method (`collect_pure_device_tpu.py`): reads each op's real kernel
time off the xprof device timeline = **sum of the inner spans nested in the `jit_*`
wrapper, EXCLUDING the wrapper** (the wrapper is the ~10 us per-launch floor; reading
it as the kernel was the old bug that inflated every op ~10 us). Audited across all 24
ops -- uniform rule, no per-op special-casing (composite ops like `batch_norm_training`
just sum their several real fusions). See `model/README.md` §f.
```bash
cd scalesim/model/calibration
OPS="add subtract multiply divide maximum minimum negate rsqrt exponential logistic \
     tanh power sine cosine convert compare and select batch_norm_training reduce \
     slice transpose reshape broadcast concatenate"
PJRT_DEVICE=TPU python3 collect_pure_device_tpu.py --ops $OPS \
    --outdir ../../calibration_pure/datasets_pure_<gen>_fixed
    # defaults: n=1000 (60% LLM-anchored incl. vocab to 160k + 40% broad/general),
    #           iters=10 warmup=1 reps=1. Verify it prints "device=TPU <gen>".
python3 train_ops.py \
    --datadir ../../calibration_pure/datasets_pure_<gen>_fixed --outdir ../<gen>   # -> model/<gen>/*.pkl
```
`NonComputeLatencyPredictor` auto-selects `model/<gen>/` from the config's
`TimeLinearModel:` key (falls back to tpuv4 if absent), so the new models go live once
written. (The loop-method `collect_ops_tpu.py` is the floor-removed alternate; the
shipped models use the pure span -- the honest standalone cost the whole-model sum needs.)

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
Model: `T = Sc + a1*Sn + C0` (a0 pinned to 1; C1*n_gemm dropped -- collinear with Sn).
The calibration StableHLO graphs are committed (`calib_mlir/s{128,256,512,1024}/`, f32,
target-independent) and the fit is one **CPU-only** command parameterized by `--gen`,
so the only TPU work here is measuring truth.
```bash
cd scalesim/calibration_pure
# 4a. whole-model truth (torch.compile device-busy), batch=1, per (model,seq):
PJRT_DEVICE=TPU python3 devicetruth_worker.py <model> <seq> 1   # -> rows of e2e_device_truth_<gen>.csv
#     (reuse the committed e2e_device_truth_<gen>.csv if this VM is the same hardware)
# 4b. tiny small-model anchor truth on this gen:
PJRT_DEVICE=TPU python3 measure_tiny_truth.py                   # prints tiny device-busy us
# 4c. fit (CPU; uses the in-repo calib_mlir + model/<gen> from step 2):
python3 fit_compensation_pure.py --gen <gen> --tiny-truth <us_from_4b>
```
It prints `a1`, `C0` and writes `coeffs_<gen>_pure.json` + `calib_<gen>_pure.csv`.
**Wire in:** paste into `COMPENSATION_BY_GEN["<GEN>"] = {a0_mxu:1.0, a1_vpu:<a1>, c_c:0,
c_n:0, C0_forward:<C0>, C1_per_gemm:0.0}` in `total_time_report.py`. Batch-1 only
(inference batch>1 occupancy is deliberately out of scope).

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
