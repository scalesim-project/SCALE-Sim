# TPU GEMM/Conv linear time model

Converts SCALE-Sim **compute cycles → wall-clock microseconds** for a TPU. This is
the time-prediction layer only; it never changes how cycles are computed.

- `tpu.py` — `tpuv4_linear_model(cycles, s_row, s_col, t_time, M, N, K)` (and
  `tpuv5e_/tpuv6e_` stubs). Selected by a config's `TimeLinearModel:` key.
- `tpuv4_coeffs.json` — provenance/metadata for the shipped TPU v4 coefficients
  (not loaded at runtime; the constants are inlined in `tpu.py`).
- `calibration/` — the scripts to measure and refit the model on a TPU (below).

---

## a. Methodology

**The one hard constraint:** the per-piece form is **linear in cycles**

```
time_us = A · cycles + B
```

so `A` reads as an *effective clock* (µs/cycle) and `B` as a *fixed overhead* (µs).
Keeping it linear is what makes the coefficients physically interpretable.

**Why linear works at all** — raw `(M,N,K) → latency` is highly nonlinear (tiling
cliffs from the `⌈N/128⌉⌈K/128⌉` folding), but SCALE-Sim's cycle count already
absorbs that nonlinearity, so within a regime `latency ≈ A·cycles + B` is genuinely
linear (corr ≈ 0.996–0.9997 in-regime).

**The shipped model — `G_roof`** adds one physical term for the memory-bound regime:

```
cycles_mem = bytes_moved / bytes_per_cycle          # bytes_moved = 2·(MK+KN+MN) for bf16
time_us    = A · max(cycles, cycles_mem) + B
```

The `max(·)` routes bandwidth-limited GEMMs (large-N, low arithmetic intensity)
through a *physical* memory term (`bytes_per_cycle` = effective HBM bandwidth) so
`A` stays the clock and `B` the overhead **everywhere** — no per-regime slopes.
Three numbers (`A, B, bytes_per_cycle`) cover both compute- and memory-bound GEMMs.
When `M,N,K` are unavailable (conv layers / old call sites) it degrades to the
single global fit `A0·cycles + B0`.

**Cycle source** (used by the calibration + the bypass; equals SCALE-Sim exactly):

```
cycles = (2·S + S + M − 2) · ⌈N/S⌉ · ⌈K/S⌉ − 1          # S = array dim (128)
```

M is the streamed (ofmap-pixel) dim; the filter (K×N) is weight-stationary.
Verified == SCALE-Sim `Total Cycles` on 72/73 gpt2 layers and 4096/4096
`validation/data_matmul` rows. (It is **not** `min(v1,v2)` over orientations — that
only holds for square-ish shapes.)

**Model selection** — chosen by held-out CV, favoring the fewest interpretable
pieces; `G_roof` beat both a plain global fit and a 3-segment piecewise fit.

---

## b. Accuracy (TPU v4, bf16, `configs/tpuv4.cfg`)

Coefficients: `A = 3.408e-5 µs/cyc`, `B = 1.471 µs`, `bytes_per_cycle = 29.47`.
Trained on 5694 GEMMs, validated on 1424 held-out (relative-error weighted fit).

| candidate | #coeffs | val MAPE |
|-----------|--------:|---------:|
| global `A·cyc+B` (no roofline) | 2 | 19.5% |
| **G_roof `A·max(cyc, bytes/BW)+B`** | 3 | **16.5%** |
| 3-segment piecewise | 6 | 20.4% |

**Placebo / ablation** (refit the same form with cycles replaced) — confirms the
cycles are load-bearing, not the coefficients:

| regressor | val MAPE |
|-----------|---------:|
| SCALE-Sim cycles (G_roof) | **16.5%** |
| constant (mean) | 187.9% |
| raw M·N·K | 25.7% |

- Held-out **real LLM GEMM shapes**: 16.5% MAPE.
- Per shape-class: compute-bound bulk ~15%; residual concentrates in skewed
  memory-bound shapes (gemv ~40%, thin-K/deep-K ~27%) where a single global
  bandwidth is imperfect.
- **Physical sanity:** fitted `A ≈ 0.034 ns/cyc`. This is *not* the ~0.95 ns TPU v4
  period — it means the closed-form *sim cycle* ≠ *hardware cycle* (the fill/drain
  terms over-count for the small tiles that dominate LLM GEMMs). `A` correctly maps
  sim-cycles → time; we document this rather than force 0.95 ns.

**Scope caveat:** this models the **compute/device-busy** time. For a full *eager*
model run at small batch/seq, the measured wall-clock is dominated by per-op kernel
dispatch (~20 µs/op), not compute — see `SCALE-Sim_TPU/reports/C_integration.md`.
The linear model is the load-bearing term in the **compute-bound / fused** regime.

---

## c. How to measure & refit on another TPU

The model is calibrated to **one TPU + one config** (128×128 weight-stationary,
compute-bound; memory effects handled by the roofline, not SCALE-Sim stall cycles).
To produce coefficients for a different TPU (v5e/v6e, other array size), re-run the
`calibration/` pipeline **on that TPU**:

**Requirements:** a TPU with exclusive PJRT access (no other process on `/dev/accel*`),
`pip install jax[tpu]`, bf16. ~7k shapes ≈ tens of minutes.

```bash
cd scalesim/linear_model/calibration

# 1. generate broad shape coverage (Sobol+grid+skew classes + real LLM shapes)
python3 sample_shapes.py            # -> shapes.csv

# 2. measure latency on the target TPU (records device + wall-clock per shape;
#    closed-form cycles alongside). Resumable / crash-safe.
PJRT_DEVICE=TPU python3 collect_gemm_tpu.py --shuffle    # -> gemm_master.csv

# 3. fit candidates (G0/G_roof/G_seg) + placebo + LLM held-out, pick by CV
python3 fit_gemm.py                 # -> gemm_linear_tpuv4.json  (A, B, bytes_per_cycle)
```

**Install the result:** copy `A`, `B`, `bytes_per_cycle` from the JSON into the
corresponding `*_linear_model()` in `tpu.py` (for a new generation, fill the
`tpuv5e_`/`tpuv6e_` stub and add a `TimeLinearModel:` config option). Update
`tpuv4_coeffs.json` (provenance) alongside.

Notes: train on **device time** (additive, composable with the compute path). The
`bytes_per_cycle` grid in `fit_gemm.py` is generation-agnostic. Keep the array
dims / dataflow in the config matching what you calibrated, or refit.

> Fuller writeup: `SCALE-Sim_TPU/reports/B_gemm_linear.md`.

---

## d. TPU v6e calibration progress (in progress)

> Live log for building the **TPU v6e** `G_roof` coefficients on this VM. Updated
> as each step lands so a VM crash leaves a recoverable breadcrumb. Started 2026-06-21.

**Target device:** `TPU v6 lite` (v6e), 8 devices, jax 0.6.2, bf16, single-device.
**Env:** system `python3` (sees the TPU); `scikit-learn`+`pandas` installed for the fit.

| # | step | command | output | status |
|---|------|---------|--------|--------|
| 0 | env + log | `pip install scikit-learn pandas`; verify TPU | — | ✅ done |
| 1 | sample shapes | `python3 sample_shapes.py` | `shapes.csv` (7097 shapes) | ✅ done |
| 1 | collect on v6e | `PJRT_DEVICE=TPU python3 collect_gemm_tpu.py --shuffle --out gemm_master_tpuv6e.csv` | `gemm_master_tpuv6e.csv` (7097 ok) | ✅ done |
| 2 | fit G_roof | `python3 fit_gemm.py --data gemm_master_tpuv6e.csv --out gemm_linear_tpuv6e.json` | `gemm_linear_tpuv6e.json` | ✅ done |
| 2 | integrate | rewrite `tpuv6e_linear_model` → G_roof; add M,N,K to sig; fix call sites; write `tpuv6e_coeffs.json` | `tpu.py`, `simulator.py`, `bypass_compute.py` | ✅ done |

**Resulting coefficients:** `A = 3.048e-5 µs/cyc   B = 0.879 µs   bytes_per_cycle = 39.07`
(G_roof val MAPE **25.0%** on 1420 held-out GEMMs; vs G0 27.9%, placebo M·N·K 35.5%.
A 3-segment fit scored 22.5% but its breakpoints are unmotivated — G_roof shipped for
interpretability, matching v4. Fallback G0 `A0=7.489e-5, B0=0.855`.)

**Config + end-to-end validated** (2026-06-21): added `configs/tpuv6e.cfg`
(`TimeLinearModel: TPUv6e`, 128×128 WS, compute-bound — matching the calibration).
Verified the model drives `TIME_REPORT.csv` for both GEMM topologies
(`topologies/GEMM_mnk/NCF.csv -i gemm`) and a full StableHLO model
(`topologies/stablehlo/llm/smollm2-135m.stablehlo.mlir -b`).

---

## e. Pure-device (xprof) measurement — trace-authoritative kernel time

`collect_gemm_tpu.py` derives its `latency_us_device` **indirectly** — an in-JIT
`fori_loop` and a wall-clock subtraction `(t_K − t_1)/(K−1)` that cancels the fixed
host dispatch. `collect_pure_device_gemm_tpu.py` is the **direct** alternative: it
compiles the GEMM (`jit().lower().compile()`), runs it under `jax.profiler.trace`,
and reads the kernel's span on the **TPU:0 device timeline** straight from the trace
— host/PJRT dispatch and the host→device sync floor excluded *by construction*.

Its CSV is a **drop-in for `fit_gemm.py`** (same `gemm_master.csv` columns):
`latency_us_device` = pure xprof kernel time, `latency_us_wallclock` = python-timer
execute+block, plus an extra `host_us = wall − kernel`. So you can refit G_roof on
pure-kernel labels with `python3 fit_gemm.py --data gemm_pure_master.csv`.

**What the trace shows (TPU v4, bf16):**
- **Pure kernel time `≈ 1.14e-4 µs/cyc · cycles + 12.6 µs` floor** — and the
  `1.14e-4` slope matches the shipped `A` once the sim-cycle↔hardware-cycle factor
  is accounted for, an independent confirmation of the cycle model.
- **`host_us` is flat ~90 µs per execute**, work-independent — the per-execute (not
  per-op) launch cost. It is correctly *outside* the device label.
- The wall-clock "device" signal (`collect_gemm_tpu.py`) conflates the **~12 µs
  kernel floor** with **~52 µs device sync**; xprof separates them. So for the
  *compute-bound slope* the two methods agree, but they differ on the floor `B`:
  the trace gives the true kernel floor, the subtraction gives kernel+sync.

**Method notes (baked into the script):** the TPU device PID is auto-detected from
the trace `process_name` (`/device:TPU:0`) — never hardcode `pid == 3`;
`jax.named_call` does not propagate to the device op, so it takes the dominant
device span (which also avoids double-counting nested fusions); summing per-event
`dur` would overstate a multi-op graph (use `max(end)−min(start)` there), but a
single GEMM is one kernel so the mean span is exact. xprof tracing is slower than
the loop method (one trace/parse per shape) — use it to **audit** the G_roof fit
and **pin the floor/host constants**, while `collect_gemm_tpu.py` remains the
fast bulk collector for refitting.

**Usage:**
```bash
cd scalesim/linear_model/calibration
PJRT_DEVICE=TPU python3 collect_pure_device_gemm_tpu.py --shuffle \
    --shapes shapes.csv --out gemm_pure_master.csv --iters 30   # -> drop-in for fit_gemm.py
```

> Matmul floor/host/slope measurement that seeded this:
> `SCALE-Sim_TPU/e2e_work/compensation/measure_xprof.py`. The non-compute-op
> equivalent + the kernel/host findings are in `scalesim/model/README.md` §f.
