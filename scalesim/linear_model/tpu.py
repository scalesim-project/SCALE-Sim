import math

# --- TPU v4 piecewise G_roof region table -----------------------------------
# Predicts a GEMM's device COMPUTE-kernel time (us) -- the matmul `fusion` span on
# the device, NOT including the fixed per-program launch wrapper (that ~12us
# per-execute overhead is a single C_forward term, handled by the compensation
# layer, never per layer). The WHOLE (M,N,K) space is partitioned by ONE physical
# rule applied to each dimension -- does it span more than one 128x128 array tile
# (fold = ceil(dim/128) > 1)? -- giving 8 regions:
#
#     region = (ceil(K/128) > 1) * 4         # deep-K  (weights exceed one tile row)
#            + (ceil(N/128) > 1) * 2          # multi-tile N (output-column folds)
#            + (ceil(M/128) > 1)              # multi-tile M (ifmap-row folds)
#
# The >1-tile boundary is the same hardware event for all three dims (the array
# processes one tile at a time; folding/streaming begins at tile #2). An earlier
# scheme used a ">=16 tiles" split for N and M, but a threshold sweep showed MAPE is
# monotonic in that threshold -- 16 was an unjustified round number; the data
# prefers the fundamental >1-tile boundary (13.5% vs 18.3% on a region-balanced set;
# a wash on full-space-weighted data but better on the rare/worst regions). See
# scalesim/calibration_pure/RESULTS.md.
#
# Floor B ~= 1.0us = the matmul kernel's own minimum device span (the xprof `fusion`
# event for a 128^3 matmul is 1.22us, matching this). Calibrated DIRECTLY on the
# golden xprof `fusion` time of 1260 region-stratified bf16 GEMMs
# (scalesim/calibration_pure/gemm_fusion_strat.csv, fit_piecewise_pure.py); held-out
# fusion-time MAPE 13.5%. (A = eff. clock us/cyc, B = kernel floor us, BW = bytes/cycle.)
# This is the FUSION linear model: it targets the pure matmul kernel `fusion` span,
# not the loop-method marginal signal (which ran ~1.4x higher). See calibration_pure/RESULTS.md.
TPUV4_REGION_TABLE = {
    0:  (6.791788e-04, 0.7665,  58.953),  # K=1 N=1 M=1tile                       n=59
    1:  (0.000000e+00, 1.1989,  27.204),  # K=1 N=1 1<M<6tile (floor-only)        n=18
    2:  (1.311609e-06, 1.1544,   2.000),  # K=1 N>1 M=1tile                       n=138
    3:  (9.593879e-05, 1.2790,  78.789),  # K=1 N>1 1<M<6tile                     n=44
    4:  (2.874061e-05, 1.0039,  18.479),  # K>1 N=1 M=1tile (deep-K)              n=55
    5:  (1.171307e-06, 1.0530,   2.203),  # K>1 N=1 1<M<6tile                     n=23
    6:  (1.346742e-05, 1.2099,  48.589),  # K>1 N>1 M=1tile (deep-K, wide-N)      n=132
    7:  (5.816404e-05, 1.3492,  58.953),  # K>1 N>1 1<M<6tile (qwen-like M=256-512)n=46
    9:  (1.049768e-06, 1.2062,   2.000),  # K=1 N=1 M>=6tile (tall, thin-K)       n=118
    11: (2.342451e-04, 1.5325, 228.192),  # K=1 N>1 M>=6tile (large thin-K)       n=253
    13: (1.455784e-05, 1.1123,  48.589),  # K>1 N=1 M>=6tile (deep-K, tall)       n=114
    15: (1.218951e-04, 2.1599, 170.744),  # K>1 N>1 M>=6tile (large compute)      n=260
}
_TPUV4_FALLBACK = (1.218951e-04, 2.1599, 170.744)  # if a region id is ever unseen


def _tpuv4_region(M, N, K):
    """Region id: >1-tile bar on K,N,M, plus a 2nd M bar at foldM>=6 (12 regions).
    The 2nd M bar gives small/medium multi-tile M (foldM 2-5, fill-dominated -> lower
    clock) its own slope, separate from large M (foldM>=6) -- the cycle formula's
    additive fill term (3*128) over-counts at small M, so one M slope over-predicts
    it (e.g. qwen M=256-512 GEMMs by ~1.7x). Bar at 6 (M>=768) keeps M=256 and 512 in
    the medium band where they belong."""
    fM = math.ceil(M / 128)
    base = ((math.ceil(K / 128) > 1) * 4 + (math.ceil(N / 128) > 1) * 2 + (fM > 1))
    return base + (fM >= 6) * 8


# --- LEVEL 2: batch latency-reduction ratio (SEPARATE from the cycle model) ---
# SCALE-Sim's cycle model only handles NON-batch GEMMs; a batched matmul (e.g.
# multi-head attention, `batch` independent per-head GEMMs) is mapped to `batch`
# single GEMMs. But running them as one batched kernel is cheaper than `batch x`
# a single GEMM, because the array fill/drain is amortized once over the batch and
# the otherwise-idle array is packed with the other heads. So the whole-op latency is
#
#     batched_latency = batch * single_GEMM_latency * R(batch, M, N, K)
#
# where single_GEMM_latency is the LEVEL-1 cycle->time prediction (tpuv4_linear_model)
# for ONE per-head (M,N,K), and R is this LEVEL-2 reduction. Kept separate from the
# cycle model so the two can be recalibrated independently.
#
#     R = u + (1 - u) / batch          # R=1 at batch=1; R->u as batch->inf
#     u = nt^0.805 / (nt^0.805 + 21)   # nt = ceil(M/128)*ceil(N/128) = output tiles
#
# u is the single-op array utilization (saturating ->1 for large matmuls that
# already fill the array; small for tiny per-head matmuls that leave it idle). The
# floor (3*128 fill) amortizing across the batch is what makes R<1. K-independent.
# Calibrated on a TPU v4 batch x shape sweep (xprof einsum vs batch*single); the
# two-level form matches 70 measured points at 6.3% MAPE. See
# SCALE-Sim_TPU/e2e_work/compensation/ (measure_batch_sweep.py, batch_reduction.csv).
def tpuv4_batch_reduction(batch, M, N, K):
    """Level-2 multiplier on `batch * single_GEMM_latency` for a batched matmul.
    Returns 1.0 for batch<=1 (no batching)."""
    if batch is None or batch <= 1:
        return 1.0
    nt = math.ceil(M / 128) * math.ceil(N / 128)
    u = nt ** 0.805 / (nt ** 0.805 + 21.0)
    return u + (1.0 - u) / batch


def tpuv4_linear_model(cycles, s_row=1, s_col=1, t_time=1, M=None, N=None, K=None):
    """
    TPUv4 linear model: convert SCALE-Sim compute cycles to time (microseconds).

    Model = piecewise (region-selected) G_roof. The (M,N,K) space is partitioned
    into 8 regions by one physical rule per dim -- does K, N, or M span more than one
    128-array tile (fold>1)? -- see TPUV4_REGION_TABLE; each region has its own G_roof:

        cyc_mem = bytes_moved / bytes_per_cycle[r]         # memory roofline term
        time_us = A[r] * max(cycles, cyc_mem) + B[r]       # A=eff. clock, B=overhead

    where bytes_moved = 2*(M*K + K*N + M*N) for bf16 and r = region(M,N,K). The
    target is the GEMM's device compute-kernel time (the matmul `fusion` span; the
    fixed per-program launch overhead is a separate once-per-forward constant, not
    charged per layer). A single global G_roof had a ~20x slope spread across shapes
    (systematic bias on thin-K / large-vocab / GEMV corners); the per-region fit
    removes it while keeping coefficients physical. Calibrated on 10,874 full-space
    bf16 GEMMs (SCALE-Sim_TPU/e2e_work/gemm_pw/); held-out full-space MAPE 14.2%
    (vs 18.8% single), LLM 13.6%, OOD-large 20.1%. Floor B~1.5us = the matmul
    kernel's minimum span (xprof `fusion` for 128^3 is 1.22us, confirming it).

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
        A, B, BYTES_PER_CYCLE = TPUV4_REGION_TABLE.get(_tpuv4_region(M, N, K), _TPUV4_FALLBACK)
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

# v6e 12-region fusion table (CALIBRATION_RUNBOOK step 1, done 2026-06-25).
# Same {region_id: (A, B, BW)} form and same _tpuv4_region tile-fold selector as
# TPUV4_REGION_TABLE (shared 128x128 array); only the coefficients are v6e.
# Calibrated DIRECTLY on the golden xprof `fusion` time of 1260 region-stratified
# bf16 GEMMs measured on a real TPU v6e
# (scalesim/calibration_pure/gemm_pure_master_tpuv6e.csv, fit_piecewise12_v6e.py);
# held-out fusion-time MAPE 11.7% (vs 27.0% single G_roof). Floor B ~= 1.1us = the
# matmul kernel's minimum fusion span on v6e. (A = us/cyc, B = floor us, BW = bytes/cyc.)
TPUV6E_REGION_TABLE = {
    0:  (7.163260e-04, 0.8318,  64.937),  # K=1 N=1 M=1tile                        n=59
    1:  (0.000000e+00, 1.2192,  27.204),  # K=1 N=1 1<M<6tile (floor-only)         n=18
    2:  (1.228766e-06, 1.1583,   2.000),  # K=1 N>1 M=1tile                        n=138
    3:  (4.143272e-06, 1.2872,   4.774),  # K=1 N>1 1<M<6tile                      n=44
    4:  (4.599992e-06, 1.1676,   8.527),  # K>1 N=1 M=1tile (deep-K)               n=55
    5:  (7.711162e-07, 1.1342,   2.000),  # K>1 N=1 1<M<6tile                      n=23
    6:  (1.736788e-05, 1.1318,  71.528),  # K>1 N>1 M=1tile (deep-K, wide-N)       n=132
    7:  (2.402579e-05, 1.5676,  58.953),  # K>1 N>1 1<M<6tile                      n=46
    9:  (1.113757e-05, 1.1858,  22.421),  # K=1 N=1 M>=6tile (tall, thin-K)        n=118
    11: (1.638546e-04, 1.5609, 276.866),  # K=1 N>1 M>=6tile (large thin-K)        n=253
    13: (4.845264e-07, 1.2152,   2.427),  # K>1 N=1 M>=6tile (deep-K, tall)        n=114
    15: (4.264973e-05, 2.3951, 188.075),  # K>1 N>1 M>=6tile (large compute)       n=260
}
_TPUV6E_FALLBACK = (4.264973e-05, 2.3951, 188.075)  # large-compute region, if id unseen


def tpuv6e_batch_reduction(batch, M, N, K):
    """Level-2 batched-matmul reduction for v6e. Until a v6e batch-sweep is fit
    (CALIBRATION_RUNBOOK step 3), reuse the v4 reduction shape (same 128x128 array;
    recalibrate p,c on v6e for accuracy)."""
    return tpuv4_batch_reduction(batch, M, N, K)


def tpuv6e_linear_model(cycles, s_row=1, s_col=1, t_time=1, M=None, N=None, K=None):
    """
    TPUv6e linear model: convert SCALE-Sim compute cycles to time (microseconds).

    Model = piecewise (region-selected) G_roof, exactly like the TPU v4 model
    (12 regions via the shared _tpuv4_region tile-fold selector; 128x128 array),
    only the coefficients are v6e:

        cyc_mem = bytes_moved / bytes_per_cycle[r]         # memory roofline term
        time_us = A[r] * max(cycles, cyc_mem) + B[r]       # A=eff. clock, B=floor

    where bytes_moved = 2*(M*K + K*N + M*N) for bf16 and r = region(M,N,K). The
    target is the GEMM's device compute-kernel time (the matmul `fusion` span; the
    per-program launch overhead is a separate once-per-forward constant handled by
    the compensation layer, never per layer). Calibrated DIRECTLY on the golden
    xprof fusion time of 1260 region-stratified bf16 GEMMs measured on a real TPU
    v6e (scalesim/calibration_pure/gemm_pure_master_tpuv6e.csv, fit_piecewise12_v6e
    .py); held-out fusion-time MAPE 11.7% (vs 27.0% single G_roof). See
    TPUV6E_REGION_TABLE and scalesim/calibration_pure/RESULTS.md.

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
    A0, B0 = 7.488700448879183e-05, 0.8550086502536669  # fallback (no M,N,K)

    if M is not None and N is not None and K is not None:
        A, B, BYTES_PER_CYCLE = TPUV6E_REGION_TABLE.get(_tpuv4_region(M, N, K),
                                                        _TPUV6E_FALLBACK)
        bytes_moved = 2.0 * (M * K + K * N + M * N)       # bf16
        cyc_mem = bytes_moved / BYTES_PER_CYCLE
        return A * max(cycles, cyc_mem) + B
    return A0 * cycles + B0
