#!/usr/bin/env python3
"""Independent banked-wrapper differential, ownership, and reset regression.

Expected demand rows come from upstream SCALE-Sim, and all cumulative results
come from NumPy. The bench observes committed core rows and separately checks
the externally drained output packets; it does not reuse the wrapper's arbiter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scalesim.compute.systolic_compute_ws import systolic_compute_ws
from scalesim.scale_config import scale_config


def parse_coverage(output):
    match = re.search(r'PASS buffered integration .* complete=(\d+) abort=(\d+) reject=(\d+) '
                      r'blocked=(\d+) simultaneous=(\d+) faulted=(\d+)', output)
    if match is None:
        raise ValueError('Missing complete buffered integration PASS record')
    completed, aborted, rejected, blocked, simultaneous, faulted = map(int, match.groups())
    return dict(completed=completed, reset_aborts=aborted, rejected=rejected,
                forced_faults=faulted, blocked=blocked, simultaneous=simultaneous)


def pack(path, values, width=32):
    mask = (1 << width) - 1
    with path.open('w') as stream:
        for row in values:
            stream.write(''.join(f'{int(v) & mask:0{width // 4}x}' for v in reversed(row)) + '\n')


def reference(folder, index, rows, cols, a_banks, b_banks, shape, seed):
    m, k, n = shape
    bases = (13 + index * 97, 1003 + index * 113, 3011 + index * 149)
    if rows == cols == 1 and index == 1 and shape == (1, 1, 1):
        bases = (0xffffffff, 0xffffffff, 0xffffffff)
    cfg = scale_config()
    cfg.valid_conf_flag = True
    cfg.array_rows, cfg.array_cols = rows, cols
    engine = systolic_compute_ws()
    engine.set_params(config_obj=cfg,
                      ifmap_op_mat=np.arange(m*k, dtype=np.int64).reshape(m, k) + bases[0],
                      filter_op_mat=np.arange(k*n, dtype=np.int64).reshape(k, n) + bases[1],
                      ofmap_op_mat=np.arange(m*n, dtype=np.int64).reshape(m, n) + bases[2])
    ea, eb, ec = (x.astype(np.int64) for x in engine.get_demand_matrices())
    rng = np.random.default_rng(seed)
    a = rng.integers(-128, 128, (m, k), dtype=np.int16).astype(np.int8)
    b = rng.integers(-128, 128, (k, n), dtype=np.int16).astype(np.int8)
    a.flat[0] = b.flat[0] = -128
    if a.size > 1:
        a.flat[-1] = 127
    if b.size > 1:
        b.flat[-1] = -127
    values = np.zeros_like(ec)
    final = np.zeros((len(ec), 1), dtype=np.int64)
    fold_length, folds = m + 2*rows + cols - 2, (k + rows - 1)//rows
    commit_times, elapsed = [], 0
    for t, requests in enumerate(ec):
        fold = (t // fold_length) % folds
        prefix = min(k, (fold + 1)*rows)
        product = (a[:, :prefix].astype(np.int64) @ b[:prefix].astype(np.int64)).astype(np.int32)
        final[t, 0] = fold == folds - 1
        for lane, address in enumerate(requests):
            if address != -1:
                mi, ni = divmod(int(address - bases[2]), n)
                values[t, lane] = product[mi, ni]
        # Independent latency oracle from the public memory contract: a vector
        # request is accepted one edge after the previous row, each bank serves
        # one lane per edge, then SRAM gathering, wrapper capture, and commit.
        occupancies = []
        for addresses, base, banks in ((ea[t], bases[0], a_banks), (eb[t], bases[1], b_banks)):
            counts = [0]*banks
            for address in addresses:
                if address != -1:
                    counts[(int(address)-base) % banks] += 1
            occupancies.append(max(counts))
        read_occupancy = max(occupancies)
        row_latency = read_occupancy + 4 if read_occupancy else 1
        if fold > 0 and np.any(requests != -1):
            row_latency = max(row_latency, 2)
        elapsed += row_latency
        commit_times.append([elapsed])
    masks = {}
    for name, trace in [('a', ea), ('b', eb), ('c', ec)]:
        masks[name] = np.array([[sum((int(v) != -1) << lane for lane, v in enumerate(row))]
                               for row in trace], dtype=np.int64)
        pack(folder/f'{index}_e{name}.hex', trace)
        pack(folder/f'{index}_v{name}.hex', masks[name])
    packets = np.any(ec != -1, axis=1)
    for name, array, width in [('a', a.reshape(-1, 1), 8), ('b', b.reshape(-1, 1), 8),
                              ('ev', values, 32), ('ef', final, 32),
                              ('et', commit_times, 64),
                              ('pa', ec[packets], 32), ('pv', values[packets], 32),
                              ('pm', masks['c'][packets], 32), ('pf', final[packets], 32)]:
        pack(folder/f'{index}_{name}.hex', array, width)
    return dict(shape=list(shape), bases=list(bases), rows=len(ec), packets=int(packets.sum()),
                a_words=m*k, b_words=k*n, seed=seed, always_ready_core_cycles=elapsed,
                always_ready_transaction_cycles=elapsed + 1)


def bench(parameters, refs):
    cases = []
    for index, ref in enumerate(refs):
        m, k, n = ref['shape']
        ab, bb, cb = ref['bases']
        loads = '\n'.join(f'      $readmemh("{index}_{name}.hex",{name},0,{ref["rows"]-1});'
                          for name in ('ea', 'eb', 'ec', 'va', 'vb', 'vc', 'ev', 'ef', 'et'))
        loads += '\n' + '\n'.join(f'      $readmemh("{index}_{name}.hex",{name},0,{ref["packets"]-1});'
                                   for name in ('pa', 'pv', 'pm', 'pf'))
        cases.append(f'''    {index}: begin
      cmd_m={m}; cmd_k={k}; cmd_n={n}; cmd_a_base={ab}; cmd_b_base={bb}; cmd_c_base={cb};
      expected_rows={ref['rows']}; expected_packets={ref['packets']};
      a_words={ref['a_words']}; b_words={ref['b_words']};
      $readmemh("{index}_a.hex",amem,0,{ref['a_words']-1});
      $readmemh("{index}_b.hex",bmem,0,{ref['b_words']-1});
{loads}
    end''')
    replacements = dict(PARAMETERS=parameters, CASES='\n'.join(cases),
                        MAX_ROWS=max(r['rows'] for r in refs), MAX_PACKETS=max(r['packets'] for r in refs),
                        NEXT_A=refs[1]['a_words'], NEXT_B=refs[1]['b_words'])
    text = Path(__file__).with_name('buffered_top_tb.sv.in').read_text()
    for key, value in replacements.items():
        text = text.replace('@' + key + '@', str(value))
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true', help='Generate vectors/benches without compiling')
    parser.add_argument('--case', type=int, help='Run one parameter configuration')
    args = parser.parse_args()
    configurations = [
        # R,C,A banks,B banks,A capacity,B capacity,FIFO depth,first GEMM,second GEMM
        (4, 3, 4, 1, 64, 64, 1, (3, 5, 7), (2, 3, 5)),
        (3, 4, 1, 2, 64, 64, 2, (3, 7, 5), (2, 3, 3)),
        (1, 1, 1, 1, 16, 16, 1, (2, 3, 2), (1, 1, 1)),
        (32, 32, 32, 1, 128, 128, 2, (1, 33, 2), (2, 3, 2)),
        # Exact capacity and a non-power-of-two output FIFO.
        (2, 2, 2, 2, 8, 8, 3, (2, 4, 2), (1, 2, 3)),
    ]
    build = ROOT/'src_rtl'/'build'/'buffered_top_checks'
    build.mkdir(parents=True, exist_ok=True)
    (build/'results.json').unlink(missing_ok=True)
    reference_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    results = []
    for index, config in enumerate(configurations):
        if args.case is not None and index != args.case:
            continue
        r, c, ab, bb, aw, bw, depth, shape, next_shape = config
        folder = build/f'case_{index}'
        folder.mkdir(parents=True, exist_ok=True)
        refs = [reference(folder, j, r, c, ab, bb, s, 719 + index*17 + j) for j, s in enumerate((shape, next_shape))]
        params = f'R={r},C={c},AB={ab},BB={bb},AW={aw},BW={bw},DEPTH={depth},MAXM={max(shape[0],next_shape[0])}'
        (folder/'tb.sv').write_text(bench(params, refs))
        entry = dict(index=index, array=[r, c], banks=[ab, bb], capacities=[aw, bw], depth=depth,
                     references=refs, reference_commit=reference_commit)
        if not args.prepare_only:
            for tool in ('iverilog', 'vvp'):
                if not shutil.which(tool):
                    raise SystemExit(f'{tool} required')
            sources = sorted((ROOT/'src_rtl'/'rtl').glob('*.sv'))
            compiled = subprocess.run(['iverilog', '-g2012', '-s', 'tb', '-o', str(folder/'sim.vvp'),
                                       *map(str, sources), str(folder/'tb.sv')], text=True, capture_output=True)
            (folder/'compile.log').write_text(compiled.stdout + compiled.stderr)
            if compiled.returncode:
                raise RuntimeError(f'Compilation failed: {folder}/compile.log')
            run = subprocess.run(['vvp', str(folder/'sim.vvp')], cwd=folder, text=True,
                                 capture_output=True, timeout=180)
            (folder/'simulation.log').write_text(run.stdout + run.stderr)
            if run.returncode or 'PASS buffered integration' not in run.stdout:
                raise RuntimeError(f'Integration case {index} failed: {run.stdout}{run.stderr}')
            entry['status'] = 'passed'
            entry['coverage'] = parse_coverage(run.stdout)
            entry['timing_oracle_passed'] = True
            print(f'case {index}: {run.stdout.strip()}', flush=True)
        else:
            entry['status'] = 'prepared'
            print(f'Prepared case {index}: {folder}', flush=True)
        results.append(entry)
    (build/'results.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
