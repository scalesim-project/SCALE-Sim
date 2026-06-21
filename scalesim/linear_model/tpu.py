def tpuv4_linear_model(cycles, s_row=1, s_col=1, t_time=1, M=None, N=None, K=None):
    """
    TPUv4 linear model: convert SCALE-Sim compute cycles to time (microseconds).

    Model = G_roof (calibrated on 7118 bf16 GEMMs measured on a real TPU v4,
    single device; see SCALE-Sim_TPU/gemm_calib/). It keeps a SINGLE linear piece
    so the coefficients stay physical:

        cyc_mem = bytes_moved / bytes_per_cycle           # memory roofline term
        time_us = A * max(cycles, cyc_mem) + B            # A=eff. clock, B=overhead

    where bytes_moved = 2*(M*K + K*N + M*N) for bf16. The memory term absorbs the
    memory-bound regime so A stays the effective clock and B the fixed overhead
    EVERYWHERE (no arbitrary thresholds). This replaces the previous 3-segment
    model keyed on s_row/s_col/t_time, whose breakpoints were unmotivated.

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
    A, B, BYTES_PER_CYCLE = 3.4082892257395366e-05, 1.470841789063479, 29.473692968738646
    A0, B0 = 0.00010941119711042656, 1.4237668165026633  # fallback (no M,N,K)

    if M is not None and N is not None and K is not None:
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

def tpuv6e_linear_model(cycles, s_row=1, s_col=1, t_time=1):
    """
    TPUv6e linear model for converting cycles to time in microseconds.
    
    Args:
        cycles: Total compute cycles
        s_row: Spatial dimension rows
        s_col: Spatial dimension columns
        t_time: Temporal dimension
    
    Returns:
        Time in microseconds
    """
    if s_row <=128 and s_col <=128 and t_time <=128:
        return 0.001389 * cycles + 0.604798
    elif s_row <=1024 and s_col <=1024 and t_time <=1024:
        return 0.000068 * cycles + 1.546793
    else:
        return 0.000040 * cycles + 4.384712
    