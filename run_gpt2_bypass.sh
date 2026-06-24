#!/bin/bash
# Run the GPT-2 StableHLO graph through SCALE-Sim using the analytical bypass
# (closed-form cycles instead of the cycle-accurate systolic-array sim -> seconds,
# not minutes). Produces ONE unified report:
#   results/llm_gpt2_bypass/<run_name>/TIME_REPORT.csv
#     columns: OpID, time_us, time_with_dispatch_us, layer, stablehlo  (+ TOTAL row)
#
# Run from the SCALE-Sim repo root:  ./run_gpt2_bypass.sh
set -e
cd "$(dirname "$0")"

MLIR=topologies/stablehlo/llm/gpt2.stablehlo.mlir
CONFIG=configs/tpuv4.cfg
OUT=./results/llm_gpt2

# JAX_PLATFORMS=cpu: parsing the MLIR uses JAX, but the bypass needs no TPU, so we
# keep JAX off the accelerator. -i gemm: treat dot_general layers as GEMM.
JAX_PLATFORMS=cpu python3 -m scalesim.scale -t "$MLIR" -c "$CONFIG" -p "$OUT" -i gemm --bypass

echo
echo "Unified report: $OUT/$(ls "$OUT" | grep -v '\.csv$' | head -1)/TIME_REPORT.csv"
