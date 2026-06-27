#!/bin/bash
# One-shot PURE-device (xprof) calibration of the GEMM + non-compute per-op models for
# one TPU generation. Mirrors calibration/README.md steps 1-2 (the data-collection +
# fit phases that need the TPU). Whole-model compensation (step 4) is a separate CPU
# fit (fit_compensation_pure.py) run afterwards. Phases run SEQUENTIALLY (each needs
# exclusive TPU). Resumable: collectors skip already-done work.
#
# Usage:  GEN=tpuv4 bash run_all.sh      (GEN in {tpuv4,tpuv6e})
set -u
ROOT="/home/Owner/work/SCALE-Sim/SCALE-Sim"
GEN="${GEN:-tpuv4}"
LM="$ROOT/scalesim/latency_model/linear_model"            # GEMM calibration (sample/collect/fit)
OM="$ROOT/scalesim/latency_model/ml_model"       # non-compute op calibration
OPS_OUT="$OM/datasets_pure_${GEN}_fixed"
export PJRT_DEVICE=TPU XLA_USE_SPMD=0
ALL_OPS="add subtract multiply divide maximum minimum negate rsqrt exponential \
logistic tanh power sine cosine convert compare and select batch_norm_training \
reduce slice transpose reshape broadcast concatenate"
say(){ echo "[$(date +%H:%M:%S)] $*"; }
say "=== PURE-DEVICE CALIBRATION ($GEN) START ==="

# ---- Phase 1: GEMM fusion-time (sample -> collect -> piecewise region fit) ----
say "Phase 1: GEMM xprof collection + region fit"
cd "$LM" || exit 1
python3 gemm_calibration.py sample --out shapes_stratified.csv
PJRT_DEVICE=TPU python3 gemm_calibration.py collect \
    --shapes shapes_stratified.csv --out "gemm_fusion_strat_${GEN}.csv" --iters 12
python3 gemm_calibration.py fit --data "gemm_fusion_strat_${GEN}.csv"   # prints the region table
say "Phase 1 done (paste the region table into tpu.py)"

# ---- Phase 2: non-compute ops (collect -> train -> reshape=median) ----
say "Phase 2: non-compute xprof collection -> $OPS_OUT"
cd "$OM" || exit 1
PJRT_DEVICE=TPU python3 op_calibration.py collect --ops $ALL_OPS --outdir "$OPS_OUT"
python3 op_calibration.py train --datadir "$OPS_OUT" --outdir "$ROOT/scalesim/latency_model/ml_model/${GEN}"
python3 op_calibration.py reshape-median --model-dir "$ROOT/scalesim/latency_model/ml_model/${GEN}" \
    --dataset "$OPS_OUT/reshape_dataset.csv"
say "Phase 2 done ($(ls "$ROOT/scalesim/latency_model/ml_model/${GEN}"/*.pkl 2>/dev/null | wc -l) op models)"

say "=== DONE. Next: measure truth + fit_compensation_pure.py --gen $GEN (RUNBOOK step 4) ==="
