#!/usr/bin/env python3
"""Profile saved decode timing and evaluate explicitly analytical architecture proposals.

Does not change simulator defaults or claim new end-to-end simulation results.
Schedule equations are checked against bounded native WS/OS/IS demand matrices.
Projections reuse measured memory waits, so changed layouts, refresh phase,
bank conflicts, DMA overlap and SRAM ports still need a new coupled simulation.
"""
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from config import Architecture
from workload import Graph
from scalesim.scale_config import scale_config
from scalesim.compute.systolic_compute_os import systolic_compute_os
from scalesim.compute.systolic_compute_ws import systolic_compute_ws
from scalesim.compute.systolic_compute_is import systolic_compute_is


def cycles(flow, m, k, n, rows, cols):
    if flow == 'ws':
        return math.ceil(k/rows)*math.ceil(n/cols)*(m+2*rows+cols-2)
    if flow == 'os':
        return math.ceil(m/rows)*math.ceil(n/cols)*(k+rows+cols-2)
    if flow == 'is':
        return math.ceil(k/rows)*math.ceil(m/cols)*(n+2*rows+cols-2)
    raise ValueError(flow)


def staged_cycles(flow, op, rows, cols, buffer_bytes=262144, n_tile=1024):
    """Pessimistically drain every independently staged compute chunk.

    OS/WS: one physical column tile, with K chunks fitting one B buffer.
    IS: one physical K fold and an output-channel tile fitting one B buffer.
    Input reuse and partial sums are assumed resident across these chunks.
    """
    m, k, n = (op[name] for name in ('m', 'k', 'n'))
    if flow == 'is':
        tile = min(n_tile, buffer_bytes//rows)
        return sum(cycles(flow, m, k, min(tile, n-start), rows, cols)
                   for start in range(0, n, tile))
    k_chunk = buffer_bytes//cols
    if flow == 'ws':
        k_chunk = k_chunk//rows*rows
    return sum(cycles(flow, m, min(k_chunk, k-ks), min(cols, n-ns), rows, cols)
               for ns in range(0, n, cols) for ks in range(0, k, k_chunk))


def validate_schedules():
    checks = []
    candidates = [('ws', 256, 256), ('os', 1, 128), ('os', 1, 256),
                  ('os', 1, 512), ('os', 4, 64), ('is', 128, 4), ('is', 256, 4)]
    engines = {'ws': systolic_compute_ws, 'os': systolic_compute_os, 'is': systolic_compute_is}
    for flow, rows, cols in candidates:
        # Actual decode-sized tiles and ragged K/N, without a full-vocabulary trace.
        for m, k, n in [(1, 1152, 256), (1, 128, 128), (4, 256, 513), (4, 515, 256)]:
            cfg = scale_config()
            cfg.valid_conf_flag = True
            cfg.array_rows, cfg.array_cols = rows, cols
            engine = engines[flow]()
            engine.set_params(config_obj=cfg, ifmap_op_mat=np.zeros((m, k)),
                              filter_op_mat=np.zeros((k, n)), ofmap_op_mat=np.zeros((m, n)))
            actual = [len(matrix) for matrix in engine.get_demand_matrices()]
            expected = cycles(flow, m, k, n, rows, cols)
            assert actual == [expected]*3, (flow, rows, cols, m, k, n, actual, expected)
            checks.append(dict(flow=flow, array=[rows, cols], m=m, k=k, n=n,
                               native_cycles=expected, status='passed'))
    return checks


class Sink:
    def write(self, text):
        pass


def main():
    folder = HERE/'build/fast_decode'
    report = json.loads((folder/'result.json').read_text())
    manifest = json.loads((folder/'manifest.json').read_text())
    timing = json.loads((folder/'operations.json').read_text())
    steps = report['decode_steps']
    hz = report['assumed_core_clock_hz']
    ms = 1000/(hz*steps)
    assert len(manifest['ops']) == len(timing)
    profile = defaultdict(Counter)
    groups = defaultdict(Counter)
    gemms = []
    for op, row in zip(manifest['ops'], timing):
        assert op['id'] == int(row['op'])
        duration = int(row['end_core'])-int(row['start_core'])
        values = dict(total_ms=duration*ms, compute_ms=op['compute_cycles']*ms,
                      memory_wait_ms=int(row['memory_wait_cycles'])*ms,
                      physical_bytes=(int(row['reads'])+int(row['writes']))*32/steps,
                      operators=1/steps)
        profile[op['kind']].update(values)
        name = op['name']
        group = ('value_cache_conversion' if 'value_cache' in name else
                 'lm_head_and_argmax' if name.startswith('lm_head') or name == 'greedy_token_selection' else
                 'mlp' if any(part in name for part in ('.gate', '.up', '.down', '.gelu', '.ff_', '.pre_ff', '.post_ff')) else
                 'other')
        groups[group].update(values)
        if op['kind'] == 'gemm':
            gemms.append(op)

    macs = sum(o['m']*o['k']*o['n'] for o in gemms)/steps
    array_cycles = sum(o['native_ws_cycles'] for o in gemms)/steps
    physical_bytes = sum(t['dram_transaction_bytes'] for t in report['tokens'])/steps
    linear_bytes = sum(o['k']*o['n'] for o in gemms if o['m'] == 1)/steps
    measured = dict(
        decode_tps=report['mean_decode_tps'], ms_per_token=report['mean_decode_ms'],
        macs_per_token=macs, linear_weight_bytes_per_token=linear_bytes,
        physical_bytes_per_token=physical_bytes, array_schedule_ms=array_cycles/hz*1000,
        array_schedule_utilization=macs/(65536*array_cycles),
        end_to_end_array_utilization=macs/(65536*report['mean_decode_ms']/1000*hz),
        gemm_memory_phase_GB_s=profile['gemm']['physical_bytes']/profile['gemm']['memory_wait_ms']/1e6,
        effective_end_to_end_GB_s=physical_bytes/report['mean_decode_ms']/1e6,
        profile_by_kind=profile, profile_by_function=groups)

    schedule_checks = validate_schedules()
    schedule_sweep = []
    for flow, rows, cols in [('ws', 256, 256), ('ws', 32, 32),
                             ('os', 1, 64), ('os', 1, 128), ('os', 1, 256), ('os', 1, 512),
                             ('os', 2, 128), ('os', 4, 64), ('os', 16, 16),
                             ('is', 128, 4), ('is', 256, 4), ('is', 384, 4)]:
        full = sum(cycles(flow, o['m'], o['k'], o['n'], rows, cols) for o in gemms)
        staged = sum(staged_cycles(flow, o, rows, cols) for o in gemms)
        schedule_sweep.append(dict(flow=flow, array=[rows, cols], mac_units=rows*cols,
                                   uninterrupted_matrix_ms=full*ms, staged_matrix_ms=staged*ms,
                                   operand_buffer_bytes=262144,
                                   uninterrupted_utilization=macs/(rows*cols*full/steps),
                                   classification='analytical matrix schedules, not full-model TPS'))

    numerical = json.loads((folder/'numerical_result.json').read_text())
    vector_sweep = []
    lane32_ops = None
    for lanes in (16, 32, 64, 128):
        config = replace(Architecture(**manifest['configuration']), vector_lanes=lanes)
        graph = Graph(Sink(), steps=steps, context=manifest['first_context'],
                      input_ids=numerical['decode_input_token_ids'], architecture=config).run()
        assert [(o['name'], o['kind']) for o in graph['ops']] == [(o['name'], o['kind']) for o in manifest['ops']]
        extra = sum(o['compute_cycles']-o.get('native_ws_cycles', 0) for o in graph['ops'])
        value = sum(o['compute_cycles'] for o in graph['ops'] if 'value_cache' in o['name'])
        vector_sweep.append(dict(lanes=lanes, non_array_compute_ms=extra*ms,
                                 value_cache_compute_ms=value*ms, vector_sram_bytes=config.vector_staging_bytes,
                                 classification='compute counts only; memory waits not resimulated'))
        if lanes == 32:
            lane32_ops = graph['ops']

    # Sensitivity models, not a new Ramulator run. The same observed memory waits
    # are deliberately retained to expose compute and overlap opportunities.
    serial = overlap = fused = 0
    for op, row in zip(lane32_ops, timing):
        compute = op['compute_cycles']-op.get('native_ws_cycles', 0)
        wait = int(row['memory_wait_cycles'])
        if op['kind'] == 'gemm':
            matrix = staged_cycles('os', op, 1, 256)
            serial += compute+matrix+wait
            overlap += compute+max(matrix, wait)
            fused += compute+max(matrix, wait)
        else:
            serial += compute+wait
            overlap += compute+wait
            fused += compute+(0 if 'value_cache' in op['name'] else wait)
    # A two-pass V-cache conversion still reads the INT8 values and scales.
    # Preserve its current 32-lane compute cost; only HBM intermediate waits are
    # replaced by this explicitly assumed, nonzero transfer budget.
    value_read_bytes = sum(2*(o['elements']+o['elements']//256*4)
                           for o in manifest['ops'] if o['name'].endswith('.value_cache_dequant'))/steps
    value_stream_ms = value_read_bytes/(100e9)*1000  # assumed 100 GB/s
    value_stream_ms += 0.1  # assumed total per-token startup/queue allowance
    scenarios = [dict(name='OS 1x256, 32 vector lanes, existing serial policy', milliseconds=serial*ms),
                 dict(name='Plus ideal overlap within each GEMM', milliseconds=overlap*ms),
                 dict(name='Plus SRAM V-cache fusion with two reads and explicit transfer allowance',
                      milliseconds=fused*ms+value_stream_ms)]
    for scenario in scenarios:
        scenario.update(tps=1000/scenario['milliseconds'], classification='analytical sensitivity, not measured TPS')

    result = dict(
        scope='Gemma3-1B W8A8/KV8, batch one, contexts 513-515, assumed 1 GHz',
        measured=measured, native_schedule_checks=schedule_checks,
        schedule_sweep=schedule_sweep, vector_sweep=vector_sweep, scenarios=scenarios,
        scenario_assumptions=dict(
            memory='Reuses original per-operator Ramulator waits, which will change under a new layout/schedule.',
            overlap='max(matrix_compute, observed_memory_wait) per GEMM is an optimistic overlap model; no causal DMA pipeline was simulated.',
            value_cache='Keep all existing 32-lane conversion math; avoid HBM FP32/layout intermediates using local SRAM.',
            value_cache_two_pass_bytes=value_read_bytes, value_cache_transfer_ms=value_stream_ms,
            value_cache_transfer_bandwidth_GB_s=100, value_cache_startup_allowance_ms=0.1,
            excluded_from_projection='New SRAM port stalls, resource conflicts, prefill/TTFT, host/PCIe, clock/area feasibility.'),
        sources_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__), HERE/'workload.py', folder/'manifest.json', folder/'timing.csv', folder/'result.json',
             ROOT/'scalesim/compute/systolic_compute_os.py', ROOT/'scalesim/compute/systolic_compute_is.py']})
    (HERE/'decode_architecture_analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(measured=measured, schedule_sweep=schedule_sweep,
                          vector_sweep=vector_sweep, scenarios=scenarios,
                          native_schedule_checks_passed=len(schedule_checks)), indent=2))


if __name__ == '__main__':
    main()
