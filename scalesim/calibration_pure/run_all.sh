#!/bin/bash
# Full pure-device (xprof) calibration: profile every op (GEMM + non-compute) and
# build a per-op model for each, exactly as the loop-method pipeline does -- only
# the latency LABEL is pure-kernel xprof device time instead of the wall-clock
# subtraction. Runs the 4 phases SEQUENTIALLY (each needs exclusive TPU access).
# Resumable: every collector skips already-done work, so re-running continues.
set -u
ROOT="/home/Owner/work/SCALE-Sim/SCALE-Sim"
LM="$ROOT/scalesim/linear_model/calibration"
OM="$ROOT/scalesim/model/calibration"
PURE="$ROOT/scalesim/calibration_pure"
OPS_OUT="$PURE/datasets_pure_tpuv4"
LOG="$PURE/run.log"
mkdir -p "$OPS_OUT"
export PJRT_DEVICE=TPU XLA_USE_SPMD=0
ITERS=12; REPS=8; WARM=6; NOPS=400
ALL_OPS="add subtract multiply divide maximum minimum negate rsqrt exponential \
logistic tanh power sine cosine convert compare and select batch_norm_training \
reduce slice transpose reshape broadcast concatenate"

say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "=== PURE-DEVICE CALIBRATION START ==="

# ---- Phase 1: GEMM pure-device collection (full shapes.csv) ----
say "Phase 1: GEMM xprof collection -> gemm_pure_master.csv"
cd "$LM" && python3 collect_pure_device_gemm_tpu.py --shuffle \
    --shapes "$LM/shapes.csv" --out "$PURE/gemm_pure_master.csv" \
    --iters $ITERS --reps $REPS --warmup $WARM >>"$LOG" 2>&1
say "Phase 1 done ($(wc -l < "$PURE/gemm_pure_master.csv") rows)"

# ---- Phase 2: non-compute ops pure-device collection (op batches of 5) ----
say "Phase 2: non-compute xprof collection -> datasets_pure_tpuv4/"
set -- $ALL_OPS
while [ $# -gt 0 ]; do
  BATCH="$1 ${2:-} ${3:-} ${4:-} ${5:-}"; shift $(( $#<5 ? $# : 5 ))
  say "  ops batch: $BATCH"
  cd "$OM" && python3 collect_pure_device_tpu.py --ops $BATCH \
      --n $NOPS --iters $ITERS --reps $REPS --warmup $WARM \
      --outdir "$OPS_OUT" >>"$LOG" 2>&1
done
say "Phase 2 done ($(ls "$OPS_OUT"/*_dataset.csv 2>/dev/null | wc -l) op datasets)"

# ---- Phase 3: fit GEMM G_roof on pure-kernel labels ----
say "Phase 3: fit_gemm on pure-kernel labels -> gemm_linear_pure_tpuv4.json"
cd "$LM" && python3 fit_gemm.py --data "$PURE/gemm_pure_master.csv" \
    --signal latency_us_device --out "$PURE/gemm_linear_pure_tpuv4.json" >>"$LOG" 2>&1
say "Phase 3 done"

# ---- Phase 4: train per-op models on pure-kernel labels ----
say "Phase 4: train_ops on pure datasets -> model/tpuv4_pure/"
cd "$OM" && python3 train_ops.py --datadir "$OPS_OUT" \
    --outdir "$ROOT/scalesim/model/tpuv4_pure" >>"$LOG" 2>&1
say "Phase 4 done ($(ls "$ROOT/scalesim/model/tpuv4_pure"/*.pkl 2>/dev/null | wc -l) pkl models)"
say "=== PURE-DEVICE CALIBRATION COMPLETE ==="
