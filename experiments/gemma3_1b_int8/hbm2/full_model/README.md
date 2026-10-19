# Real-weight Gemma3-1B numerical execution and HBM2 timing

This path executes the **actual locally available Google Gemma 3 1B IT weights**,
including all transformer operators and token selection. It then runs a matching
operator graph against native Ramulator2 HBM2 at the requested simulated clock.
The numerical and timing runs are separate: numerical outputs establish the
workload, while dependent memory callbacks determine its simulated schedule.
This is not a full-transformer RTL implementation or a hardware benchmark.

## Configurations and measurements

The completed default run reports **41.6419 decode tokens/s**, or
**24.0142 ms/token**, over the same three cached decode steps at contexts
513–515 after a 512-token prompt. This is **4.6338x** the original 8.9865 TPS.
Individual steps reached 41.6336, 41.6359 and 41.6563 TPS. The result is saved
in `build/fast_decode/result.json`, with a retained copy in
[`fast_decode_result.json`](fast_decode_result.json).

| Measurement | Original 32x32 | Default fast decode |
| --- | ---: | ---: |
| Decode tokens/s | 8.9865 | 41.6419 |
| Total ms/token | 111.2776 | 24.0142 |
| Compute ms/token | 94.7512 | 13.5693 |
| Memory wait ms/token | 16.5264 | 10.4450 |
| Assumed core clock | 1 GHz | 1 GHz |

The native run completed all 3,462 operators, 482,253 commands and
104,717,591 memory callbacks over 64,844,950 DRAM ticks. The dependent
schedule took 72,042,740 core cycles. All 17 configured timing checks and
three configuration regression checks passed. The original operator sequence,
705 GEMM shapes and logical GEMM read/write byte counts are preserved; the
numerical trace and HBM2 configuration hashes match the baseline. Regenerating
the baseline configuration produces its original 808,197 commands byte for byte.

The default timing configuration is `configs/fast_decode.json`. It uses a
256x256 WS array, separate 256 KiB A/B buffers, 128 FP32 vector lanes and a
128 KiB vector buffer. The assumed core clock remains 1 GHz, with the same
four SFU lanes, HBM2 profile, memory admission limits and serial dependency
policy. This is a larger simulated accelerator: 65,536 MAC units versus the
baseline's 1,024, and 640 KiB of operand/vector staging versus 96 KiB.
These resource counts do not establish FPGA fit or timing closure.

`run.py` now accepts an architecture JSON via `--config`. Its resolved values
are saved in `configuration.json`, `manifest.json` and `result.json`. Array
dimensions control the native WS schedules, packed weight layout, tied embedding
gather and SRAM tile sizes. K chunks end on array-row boundaries so splitting
a matrix does not silently remove or add partial compute folds. Every
transformer operator and the original real-weight token trace remain included.

The original configuration is retained as `configs/baseline_32x32.json`.

The original saved run in `build/full_1ghz/result.json` reports **8.9865 decode tokens/s**
or **111.2776 ms/token**, averaging three cached decode steps at contexts 513–515.
It includes the non-GEMM operations under the explicit architecture assumptions
below. Native Ramulator completed 104,719,475 memory callbacks over 300,479,515
DRAM ticks; the dependent schedule took 333,832,742 core cycles in total.

The separate real-weight numerical execution generated `The sky looks blue`.
The FP32 implementation matched Hugging Face exactly for the saved 512-token
prefill and one cached decode step. INT8 quantization changes logits and can
change greedy tokens; this is not an accuracy evaluation. All eight numerical
checks and thirteen original timing checks passed. Simulated prefill/TTFT and
full-model RTL cycle equivalence remain outside these results.

## Reproduce

From the SCALE-Sim repository root:

```sh
env -u PYTHONPATH .venv/bin/python -m pip install -r experiments/gemma3_1b_int8/hbm2/full_model/requirements.txt
# The existing pinned native HBM2 setup is required:
bash experiments/gemma3_1b_int8/hbm2/setup_ramulator2.sh

env -u PYTHONPATH .venv/bin/python experiments/gemma3_1b_int8/hbm2/full_model/test_numerical.py
env -u PYTHONPATH .venv/bin/python experiments/gemma3_1b_int8/hbm2/full_model/run_numerical.py --prompt-tokens 512 --decode-steps 3
env -u PYTHONPATH .venv/bin/python experiments/gemma3_1b_int8/hbm2/full_model/test_configuration.py
# Uses configs/fast_decode.json and saves results under build/fast_decode:
env -u PYTHONPATH .venv/bin/python experiments/gemma3_1b_int8/hbm2/full_model/run.py
env -u PYTHONPATH .venv/bin/python experiments/gemma3_1b_int8/hbm2/full_model/test_timing.py
env -u PYTHONPATH .venv/bin/python experiments/gemma3_1b_int8/hbm2/full_model/validate_checkpoint.py

# Reproduce the baseline separately, preserving its original saved result:
env -u PYTHONPATH .venv/bin/python experiments/gemma3_1b_int8/hbm2/full_model/run.py --config experiments/gemma3_1b_int8/hbm2/full_model/configs/baseline_32x32.json
```

`run_numerical.py --checkpoint PATH` selects a **BF16/FP32 Hugging Face checkpoint**
to quantize. The default is the complete local snapshot at revision
`dcc83ea841ab6100d6b47a070329e1ba4cf78752`. The loader uses local files only. It
never replaces missing trained weights with random values. Externally quantized
checkpoint formats need their own format-specific loader; this is our explicit
post-training quantization scheme, not RedHatAI's calibrated checkpoint.

`run.py --numerical PATH/result.json --out PATH` selects a numerical run and
output directory. `--prepare-only` writes and validates the complete command
stream without running memory simulation. `--summarize-only` verifies and
summarizes an already-completed native run, using its recorded configuration.
It rejects a different clock/configuration, changed input artifacts or changed
timing output. `--core-hz` overrides the assumed clock for a new simulation.
`test_timing.py --run-dir PATH` validates every distinct tile shape actually
used by that run against SCALE-Sim's native demand generator.

## Numerical execution

`numerical.py` implements all 26 decoder layers, tied embedding/head weights,
Q/K normalization, local/global RoPE, five-local/one-global attention, KV writes
and reads, stable masked softmax, gated tanh-GELU MLPs, all four block RMSNorms,
residual additions, final normalization and greedy token selection.

- Weights: symmetric INT8, scale per output channel.
- Linear activations: dynamic symmetric INT8, scale per input row.
- GEMMs: **exact INT8 inputs / INT32 accumulation** through `torch._int_mm`.
- KV cache: INT8 with a separate scale for each token/head.
- QK: query and key scales applied to INT32 products.
- AV: cached V is dequantized and requantized per output feature before its
  INT8 product. Per-token V scales cannot simply be applied after the dot product,
  because they vary along the reduction dimension.
- Norms, RoPE, masking, softmax, GELU, residuals and scale arithmetic: FP32.
- Generation: greedy argmax. No stochastic top-k/top-p sampler is enabled.

There is no learned quantization calibration. V feature scales depend on the
available prompt block, so prefill chunking can affect quantization. The checks
report differences from FP32, not bitwise equality to the original model or
model-wide accuracy guarantees.

The real run numerically prefills 512 tokens, then executes three cached decode
steps at contexts 513–515. The first generated token comes from prefill; the
three decode steps produce the next three. The prompt, actual token IDs, logits,
operator trace, checkpoint hash and FP32 comparison are saved. CPU wall-clock
measurements are explicitly separate from simulated accelerator time.

## Timing and memory contract

SCALE-Sim's bundled GPT/Llama topologies describe GEMMs. Its native Ramulator
example uses trace generation, memory replay and a second simulator pass.
`code-examples/systolic-array-rtl` supplies sample arithmetic RTL without control
logic. Those examples do not supply a complete non-GEMM decoder executor.

This extension uses their WS scheduling convention, with native-demand validation
for the tile shapes, and adds an explicit operator/traffic scheduler:

| Component | Modeled implementation |
|---|---|
| Array | Configured WS array; baseline 32x32, default 256x256; assumed 1 GHz |
| Vector unit | Baseline 32 / default 128 FP32 lanes; assumed one elementwise instruction/lane/clock |
| Special functions | Four pipelined lanes, issue interval 1; exp/tanh latency 12, sin/cos and rsqrt 16, reciprocal 8 clocks |
| Dispatch | Four core clocks per operator |
| Operand staging | Separate A/B buffers, baseline 32 KiB / default 256 KiB each; A retained across output tiles |
| Vector staging | Baseline 32 KiB / default 128 KiB across live input/output chunks; reductions use scalar state |
| Partial sums | Retained locally across K chunks for one output tile |
| Memory | Real pinned Ramulator2 HBM2, F2-derived 16 GiB / 16 physical channels |
| Frontend | Aggregate 16 admissions/DRAM tick, finite 512 in-flight requests |
| Scheduling | Serial dependent operations; serial tile load, compute and store |
| Intermediates | HBM-resident FP32 tensors, INT8 tensors, scales, KV and layout conversions |

Weights are prepacked into array-column-width tiles. The embedding gathers actual
token rows from the same tied storage. Attention uses strided reads/transposes,
not a minimum weight/KV byte count substituted for real requests. Identical input
quantizations for Q/K/V and gate/up are shared. Four attention heads are fused
as M=4 GEMMs, preserving their numerical products. Every matrix shape is checked
against the executed numerical operator trace.

Every native DRAM clock is stepped, including refresh during compute. There is
no idle-time fast-forward or fixed-bandwidth replacement. A dependent phase
starts only after its actual callbacks complete; writes also wait
`nCWL+nBL+nWR`. The program has no source timestamps fixed before memory delays.
Byte-enabled partial writes are assumed, without ECC read-modify-write traffic.

The vector/SFU costs and 1 GHz frequency are **architecture assumptions**. The
array durations come from native-validated WS schedules with staged operands;
this is not the existing RTL's clocked-SRAM/byte-unpacking policy. The aggregate
frontend is also different from that RTL's single AXI master. Neither full RTL
cycle equivalence nor AMD F2 controller/PHY equivalence is claimed.

The measurement covers device-side cached decode. Tokenization and host I/O are
outside the accelerator timing model. Numerical prefill is executed, but its
accelerator timing is not simulated here, so this run does not establish TTFT.
Softcapping is disabled in the actual 1B config, inference dropout is disabled,
training loss and vision are inapplicable, and contiguous views/GQA sharing need
no physical copies. These cases are listed in the manifest rather than silently
charged as GEMMs or omitted.

## Artifacts and checks

- `build/numerical_512/result.json`: checkpoint identity, generated text/token
  IDs, numerical scheme, host timing, FP32 comparisons and source hashes.
- `build/numerical_512/operator_trace.json`: executed numerical operations.
- `build/full_1ghz/result.json`: simulated decode throughput and category totals.
- `build/fast_decode/result.json`: throughput and category totals for the default
  larger array/vector configuration.
- `fast_decode_result.json`: retained copy of that completed result outside the
  ignored build directory.
- `build/fast_decode/configuration.json`: resolved architecture inputs.
- `build/fast_decode/execution.json`: completion status and hashes of run inputs
  and the native timing output.
- `build/fast_decode/timing_tests.json`: native tile, operator coverage and HBM
  timing validation for the configured run.
- `build/full_1ghz/manifest.json`: every timed operator, tensor address and cost.
- `build/full_1ghz/commands.txt`: dependency-ordered compute and memory commands.
- `build/full_1ghz/timing.csv`: native per-operator start/end clocks, transactions
  and admission retries; `operations.json` adds names and memory wait cycles.
- `build/numerical_tests.json`: exact integer checks and independent Hugging Face
  checks of prefill/cached decoding, local masks and global layers.
- `build/checkpoint_validation.json`: full real-weight FP32 comparison with
  Hugging Face on the 512-token prompt and a cached decode step.
- `build/timing_tests.json`: native WS tile schedules, executed matrix shapes,
  read/write completion checks against the existing live bridge, and clock phase.

The memory run persists HBM state across all layers and all three decode steps.
This small sample is a measured simulator run for one prompt and one specified
architecture, not an average over 127 decode steps or a general quality benchmark.
