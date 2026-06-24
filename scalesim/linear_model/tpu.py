import math

# --- TPU v4 piecewise G_roof region table -----------------------------------
# A single global G_roof has a systematic, shape-dependent bias: the effective
# slope varies ~20x across the (M,N,K) space, so one (A,B,BW) cannot serve every
# shape. We partition the WHOLE space by three physical tile-fold splits and fit
# an independent G_roof per region. Each split is a hardware quantity -- whether
# a dimension spans more than one 128x128 array tile (foldK>1) or is "wide" (>=16
# tiles, the regime where output/weight streaming dominates):
#
#     region = (ceil(K/128) > 1) * 4         # deep-K  (weights exceed one tile row)
#            + (ceil(N/128) >= 16) * 2        # wide-N  (many output-column folds)
#            + (ceil(M/128) >= 16)            # tall-M  (long ifmap stream)
#
# Calibrated on 10,874 bf16 GEMMs measured on a real TPU v4 spanning the full
# space (log-uniform M,N,K in [1,16384] + per-dim sweeps + tile-boundary + extreme
# corners; SCALE-Sim_TPU/e2e_work/gemm_pw/). Selected over single/K-band/KxN/
# intensity/aspect schemes by held-out MAPE (CV+BIC): full-space 18.8% -> 14.2%,
# OOD-large 29.3% -> 20.4%, and LLM anchors 19.1% -> 13.3% (never a fitting target).
# Per region: (A = eff. clock us/cyc, B = fixed overhead us, BW = bytes/cycle).
TPUV4_REGION_TABLE = {
    0: (5.652130e-06, 1.4190,   5.261),   # K=1, N<16, M<16  (small / thin)        n=3593
    1: (1.623043e-05, 1.5792,  20.377),   # K=1, N<16, M>=16 (tall, thin-K)        n=658
    2: (7.744915e-06, 1.5994,   9.399),   # K=1, N>=16, M<16 (wide-out, thin-K)    n=859
    3: (1.653324e-04, 1.4678, 305.666),   # K=1, N>=16, M>=16 (large, thin-K)      n=168
    4: (3.533055e-05, 1.4699,  24.725),   # K>1, N<16, M<16  (deep-K, compact)     n=3822
    5: (1.269842e-04, 1.6457, 116.200),   # K>1, N<16, M>=16 (deep-K, tall)        n=739
    6: (8.638114e-05, 1.4473,  78.920),   # K>1, N>=16, M<16 (deep-K, wide-out)    n=835
    7: (1.240715e-04, 4.2773, 140.998),   # K>1, N>=16, M>=16 (large compute)      n=200
}


def _tpuv4_region(M, N, K):
    """Map a GEMM shape to its region id via three physical tile-fold splits."""
    return ((math.ceil(K / 128) > 1) * 4
            + (math.ceil(N / 128) >= 16) * 2
            + (math.ceil(M / 128) >= 16))


def tpuv4_linear_model(cycles, s_row=1, s_col=1, t_time=1, M=None, N=None, K=None):
    """
    TPUv4 linear model: convert SCALE-Sim compute cycles to time (microseconds).

    Model = piecewise (region-selected) G_roof. The (M,N,K) space is partitioned
    into 8 regions by three physical tile-fold splits (deep-K / wide-N / tall-M;
    see TPUV4_REGION_TABLE) and each region carries its own calibrated G_roof:

        cyc_mem = bytes_moved / bytes_per_cycle[r]         # memory roofline term
        time_us = A[r] * max(cycles, cyc_mem) + B[r]       # A=eff. clock, B=overhead

    where bytes_moved = 2*(M*K + K*N + M*N) for bf16 and r = region(M,N,K). A
    single global G_roof had a ~20x slope spread across shapes (systematic bias on
    thin-K / large-vocab / GEMV corners); the per-region fit removes it while
    keeping every coefficient physical and the selector a few integer comparisons.
    Calibrated on 10,874 full-space bf16 GEMMs (SCALE-Sim_TPU/e2e_work/gemm_pw/);
    held-out full-space MAPE 14.2% (vs 18.8% single), OOD-large 20.4% (vs 29.3%).

    M,N,K are optional and passed by simulator.py for GEMM layers. If absent
    (e.g. conv layers or older call sites), it degrades gracefully to the
    single global fit A0*cycles + B0 (G0). Fit metadata: see tpuv4_coeffs.json.

    Args:
        cycles: SCALE-Sim total compute cycles for the layer
        s_row, s_col, t_time: spatiotemporal dims (kept for signature compat)
        M, N, K: GEMM dims (enable the roofline term when provided)

    Returns:
        Time in microseconds
    """
    A0, B0 = 0.00010941119711042656, 1.4237668165026633  # fallback (no M,N,K)

    if M is not None and N is not None and K is not None:
        A, B, BYTES_PER_CYCLE = TPUV4_REGION_TABLE[_tpuv4_region(M, N, K)]
        bytes_moved = 2.0 * (M * K + K * N + M * N)       # bf16
        cyc_mem = bytes_moved / BYTES_PER_CYCLE
        return A * max(cycles, cyc_mem) + B
    return A0 * cycles + B0
def tpuv5e_linear_model(cycles, s_row=1, s_col=1, t_time=1):
    """
    TPUv5e linear model for converting cycles to time in microseconds.
    
    Args:
        cycles: Total compute cycles
        s_row: Spatial dimension rows
        s_col: Spatial dimension columns
        t_time: Temporal dimension
    
    Returns:
        Time in microseconds
    """
    # TODO: Modify for V5
    if s_row <=128 and s_col <=128 and t_time <=128:
        return  0.002133 * cycles - 0.168796
    elif s_row <=1024 and s_col <=1024 and t_time <=1024:
        return 0.000167 * cycles + 1.158923
    else:
        return 0.000159 * cycles -0.380696

def tpuv6e_linear_model(cycles, s_row=1, s_col=1, t_time=1, M=None, N=None, K=None):
    """
    TPUv6e linear model: convert SCALE-Sim compute cycles to time (microseconds).

    Model = G_roof, calibrated on 7097 bf16 GEMMs measured on a real TPU v6e
    ("TPU v6 lite") single device (see scalesim/linear_model/calibration/, data
    in gemm_master_tpuv6e.csv, coeffs in gemm_linear_tpuv6e.json /
    tpuv6e_coeffs.json). Same single-linear-piece form as the TPU v4 model so the
    coefficients stay physical:

        cyc_mem = bytes_moved / bytes_per_cycle           # memory roofline term
        time_us = A * max(cycles, cyc_mem) + B            # A=eff. clock, B=overhead

    where bytes_moved = 2*(M*K + K*N + M*N) for bf16. The memory term absorbs the
    memory-bound regime so A stays the effective clock and B the fixed overhead
    EVERYWHERE (no arbitrary thresholds). This replaces the previous hand-stubbed
    3-segment model keyed on s_row/s_col/t_time, whose breakpoints/coefficients
    were placeholders, not calibrated.

    Validation (held-out 1420 of 7097 GEMMs, relative-error weighted fit):
        G_roof val MAPE 25.0%  vs  G0 (no roofline) 27.9%  vs  placebo M*N*K 35.5%.
    A data-chosen 3-segment fit scored marginally lower (22.5%) but its intensity
    breakpoints are unmotivated; we keep G_roof for the same interpretability
    reason as TPU v4.

    M,N,K are optional and passed by simulator.py / bypass_compute.py for GEMM
    layers. If absent (e.g. conv layers or older call sites) it degrades to the
    single global fit A0*cycles + B0 (G0).

    Args:
        cycles: SCALE-Sim total compute cycles for the layer
        s_row, s_col, t_time: spatiotemporal dims (kept for signature compat)
        M, N, K: GEMM dims (enable the roofline term when provided)

    Returns:
        Time in microseconds
    """
    A, B, BYTES_PER_CYCLE = 3.048108888396471e-05, 0.8789530113345402, 39.07396064038605
    A0, B0 = 7.488700448879183e-05, 0.8550086502536669  # fallback (no M,N,K)

    if M is not None and N is not None and K is not None:
        bytes_moved = 2.0 * (M * K + K * N + M * N)       # bf16
        cyc_mem = bytes_moved / BYTES_PER_CYCLE
        return A * max(cycles, cyc_mem) + B
    return A0 * cycles + B0
