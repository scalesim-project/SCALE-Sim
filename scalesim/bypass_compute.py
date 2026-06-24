"""
Analytical compute-cycle bypass for SCALE-Sim.

The default flow runs a cycle-accurate systolic-array simulation per layer
(operand-matrix generation + element-by-element mapping + SRAM/DRAM trace
generation). That is expensive for large workloads (minutes for an LLM forward).

When you only need the *compute cycle count* -> time (e.g. driving the TPU v4
time model), the systolic-array math is unnecessary: SCALE-Sim's no-stall
compute-cycle count is reproduced **exactly** by a closed form. This module
computes that closed form for every layer and writes COMPUTE_REPORT.csv and
TIME_REPORT.csv directly, skipping the simulation entirely.

Validation: `closed_form_cycles` matches SCALE-Sim's `Total Cycles` 4096/4096
rows (100%) on `validation/data_matmul` for the 128x128 weight-stationary config
(incl. multi-tile folding). See SCALE-Sim_TPU/GEMM_LINEAR_PLAN.md.

Scope/limits:
  * Models the compute-bound floor (no memory-stall cycles). This matches the
    intended use (a compute-bound config such as configs/tpuv4.cfg, where memory
    effects are meant to be handled analytically by the linear/roofline time
    model, not by SCALE-Sim stall cycles). Stall cycles are reported as 0.
  * No SRAM/DRAM traces or bandwidth report are produced (that is the point).
  * Works for GEMM and CONV: SCALE-Sim lowers conv to a GEMM (im2col) and
    `get_transformed_mnk_dimensions()` returns (M, N, K) for both
    (M = #ofmap pixels, N = #filters, K = window size = R*S*C for conv).
"""
import math
import os

from scalesim.linear_model.tpu import (tpuv4_linear_model, tpuv5e_linear_model,
                                        tpuv6e_linear_model)


def closed_form_cycles(M, N, K, arr_h=128, arr_w=128):
    """No-stall compute cycles for an (M x K) . (K x N) GEMM on an
    arr_h x arr_w weight-stationary systolic array.

    SCALE-Sim loads the filter (K x N) stationary -- K over the array rows
    (ceil(K/arr_h) folds), N over the array columns (ceil(N/arr_w) folds) -- and
    streams the M ifmap rows. Per fold the cost is the array fill/drain/traverse
    plus the streamed rows, `2*arr_h + arr_w + M - 2`, and a single -1 end
    convention applies once over the whole layer:

        cycles = (2*arr_h + arr_w + M - 2) * ceil(N/arr_w) * ceil(K/arr_h) - 1

    Validated == SCALE-Sim 'Total Cycles' on 72/73 gpt2 GEMM layers (exact) and
    4096/4096 data_matmul rows. The streaming dim is ALWAYS M (it is the #ofmap
    pixels): there is no min() over orientations -- that earlier heuristic only
    held for square-ish shapes and is wrong for skewed GEMMs (e.g. M=128,N=2304).

    This is the compute-bound floor; layers whose operands exceed on-chip SRAM
    (e.g. an LM head with a huge vocab) incur memory stalls the real sim adds but
    this omits -- by design they are meant to be captured by the roofline term in
    the linear time model, not by stall cycles here.
    """
    return (2 * arr_h + arr_w + M - 2) * math.ceil(N / arr_w) * math.ceil(K / arr_h) - 1


def mem_cycles(M, N, K, bytes_per_cycle, dtype_bytes=2, sram_bytes=None,
               arr_h=128, arr_w=128):
    """Analytical memory-bound cycle estimate (roofline): bytes moved / bandwidth.

        bytes_moved = dtype_bytes * (M*K + K*N + M*N)        # operand reads + output write

    Optional tiling refetch: if the stationary operand (the K*N filter tile) does
    not fit on-chip SRAM, the streamed operand is re-read once per N-fold:
        bytes_moved += dtype_bytes * M*K * (ceil(N/arr_w) - 1)
    Returns cycles; pair with closed_form_cycles via max() to get total (stalled)
    cycles. `bytes_per_cycle` is calibrated against full SCALE-Sim (see M1.5).
    """
    bytes_moved = dtype_bytes * (M * K + K * N + M * N)
    if sram_bytes is not None and dtype_bytes * K * N > sram_bytes:
        bytes_moved += dtype_bytes * M * K * (math.ceil(N / arr_w) - 1)
    return bytes_moved / bytes_per_cycle


def _layer_time_us(time_model, cycles, s_row, s_col, t_time, M, N, K):
    """Apply the configured linear time model (returns None if model is OFF)."""
    if time_model == 'TPUv4':
        return tpuv4_linear_model(cycles, s_row, s_col, t_time, M, N, K)
    if time_model == 'TPUv5e':
        return tpuv5e_linear_model(cycles, s_row, s_col, t_time)
    if time_model == 'TPUv6e':
        return tpuv6e_linear_model(cycles, s_row, s_col, t_time, M, N, K)
    return None


def _layer_mnk(topo_obj, lid):
    """Return the true (M, N, K) GEMM dims of a layer, for both CONV and GEMM
    topologies.

    `get_transformed_mnk_dimensions()` reports M as the total ofmap element count
    (ofmap_h * ofmap_w * num_filters), i.e. inflated by N. The true GEMM row count
    is the spatial ofmap-pixel count, so we divide that back out:
        M_true = (#ofmap elements) / N = ofmap_h * ofmap_w
    K stays the window size (R*S*C for conv; the contraction dim for gemm) and N
    the filter count. Verified against the cycle-accurate path (see module test).
    """
    M_tot, N, K = topo_obj.get_transformed_mnk_dimensions()[lid]
    M = M_tot // N if N else M_tot
    return int(M), int(N), int(K)


def run_bypass(config_obj, topo_obj, top_path, gemm_mode=False, verbose=True,
               memory_bw=None, sram_bytes=None):
    """Compute every layer's cycles via the closed form and write
    COMPUTE_REPORT.csv + TIME_REPORT.csv under top_path/<run_name>/.

    gemm_mode is accepted for call-site symmetry with scalesim's input_type_gemm
    but is not needed: _layer_mnk() recovers the true dims for both layer types.

    memory_bw (bytes/cycle): if given, adds the analytical roofline stall term so
    total_cycles = max(compute_cycles, mem_cycles) and the Stall Cycles column is
    populated -- reproducing full SCALE-Sim's memory behavior cheaply. If None
    (default), reports the compute-bound floor (stall = 0).
    Returns the report directory path.
    """
    arr_h, arr_w = config_obj.get_array_dims()
    dataflow = config_obj.get_dataflow()
    time_model = config_obj.get_time_linear_model()
    n_layers = topo_obj.get_num_layers()

    report_path = os.path.join(top_path, config_obj.get_run_name())
    os.makedirs(report_path, exist_ok=True)

    compute_report = open(os.path.join(report_path, 'COMPUTE_REPORT.csv'), 'w')
    compute_report.write('LayerID, Total Cycles (incl. prefetch), Total Cycles, '
                         'Stall Cycles, Overall Util %, Mapping Efficiency %, '
                         'Compute Util %,\n')
    time_report = open(os.path.join(report_path, 'TIME_REPORT.csv'), 'w')
    time_report.write('LayerID, Time (us),\n')

    array_size = arr_h * arr_w
    total_us = 0.0
    for lid in range(n_layers):
        M, N, K = _layer_mnk(topo_obj, lid)
        compute_cyc = closed_form_cycles(M, N, K, arr_h, arr_w)

        # analytical memory roofline: total = max(compute, mem); stall = the excess
        if memory_bw:
            mc = mem_cycles(M, N, K, memory_bw, sram_bytes=sram_bytes,
                            arr_h=arr_h, arr_w=arr_w)
            cycles = int(max(compute_cyc, mc))
            stall = max(0, cycles - compute_cyc)
        else:
            cycles = compute_cyc
            stall = 0

        # utilization metrics (analytical): useful MACs / (array * cycles)
        num_mac = M * N * K
        util = (num_mac * 100.0) / (cycles * array_size) if cycles else 0.0
        # mapping efficiency = utilization of the mapped array footprint
        mapped = min(K, arr_h) * min(N, arr_w) if min(K, arr_h) and min(N, arr_w) else 1
        map_eff = (num_mac * 100.0) / (compute_cyc * mapped) if compute_cyc else 0.0
        map_eff = min(map_eff, 100.0)

        compute_report.write(f'{lid}, {cycles}, {cycles}, {stall}, '
                             f'{util:.2f}, {map_eff:.2f}, {util:.2f},\n')

        if time_model in ('TPUv4', 'TPUv5e', 'TPUv6e'):
            s_row, s_col, t_time = topo_obj.get_spatiotemporal_dims(layer_id=lid, df=dataflow)
            time_us = _layer_time_us(time_model, cycles, s_row, s_col, t_time, M, N, K)
        else:
            time_us = cycles  # no model: report cycles as time (matches default path)
        total_us += time_us
        time_report.write(f'{lid}, {time_us},\n')

        if verbose:
            print(f'  [bypass] Layer {lid}: M={M} N={N} K={K}  cycles={cycles}  '
                  f'time={time_us:.3f} us' if time_us is not None else
                  f'  [bypass] Layer {lid}: cycles={cycles}')

    compute_report.close()
    time_report.close()
    if verbose:
        print(f'[bypass] wrote COMPUTE_REPORT.csv + TIME_REPORT.csv to {report_path}')
        print(f'[bypass] total predicted compute time: {total_us:.1f} us')
    return report_path
