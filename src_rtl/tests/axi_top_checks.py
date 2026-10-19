#!/usr/bin/env python3
"""Independent AXI memory/protocol model plus native WS and NumPy scoreboards."""
from pathlib import Path
import argparse
import json
import re
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scalesim.compute.systolic_compute_ws import systolic_compute_ws
from scalesim.scale_config import scale_config

OUT = ROOT / 'src_rtl/build/axi_top_checks'
CONFIGS = [
    (1, 1, 32, 1, 1, 1, 1024, 0, (3, 5, 3)),
    (4, 3, 64, 4, 4, 1, 1024, 0, (4, 17, 7)),
    (3, 4, 128, 3, 1, 2, 1024, 1, (4, 19, 9)),
    (32, 32, 512, 16, 32, 32, 2048, 0, (4, 33, 35)),
    (2, 2, 32, 256, 2, 2, 4096, 0, (4, 512, 2)),
    (2, 3, 1024, 4, 2, 1, 1024, 0, (4, 5, 7)),
]


def pack(path, rows, bits=32):
    mask = (1 << bits) - 1
    path.write_text(''.join(''.join(f'{int(x) & mask:0{bits//4}x}' for x in reversed(row))
                           + '\n' for row in rows))


def reference(folder, index, rows, cols, shape, intermediates, seed):
    m, k, n = shape
    cfg = scale_config()
    cfg.valid_conf_flag = True
    cfg.array_rows, cfg.array_cols = rows, cols
    native = systolic_compute_ws()
    native.set_params(config_obj=cfg,
                      ifmap_op_mat=np.arange(m*k).reshape(m, k),
                      filter_op_mat=np.arange(k*n).reshape(k, n),
                      ofmap_op_mat=np.arange(m*n).reshape(m, n))
    ea, eb, ec = (v.astype(np.int64) for v in native.get_demand_matrices())
    rng = np.random.default_rng(seed)
    a = rng.integers(-128, 128, (m, k), dtype=np.int16).astype(np.int8)
    b = rng.integers(-128, 128, (k, n), dtype=np.int16).astype(np.int8)
    a.flat[0] = b.flat[0] = -128
    values = np.zeros_like(ec)
    writes = []
    length, folds = m + 2*rows + cols - 2, (k + rows - 1)//rows
    products = [(a[:, :min(k, (f+1)*rows)].astype(np.int64)
                 @ b[:min(k, (f+1)*rows)].astype(np.int64)).astype(np.int32)
                for f in range(folds)]
    for t, addresses in enumerate(ec):
        fold = (t // length) % folds
        for lane, address in enumerate(addresses):
            if address != -1:
                mi, ni = divmod(int(address), n)
                values[t, lane] = products[fold][mi, ni]
                if intermediates or fold == folds - 1:
                    writes.append((int(address), int(values[t, lane])))
    for name, matrix in [('ea', ea), ('eb', eb), ('ec', ec), ('ev', values),
                         ('ewaddr', [[x[0]] for x in writes]),
                         ('ewdata', [[x[1]] for x in writes]),
                         ('final', products[-1].reshape(-1, 1))]:
        pack(folder/f'{index}_{name}.hex', matrix)
    pack(folder/f'{index}_a.hex', a.reshape(-1, 1), 8)
    pack(folder/f'{index}_b.hex', b.reshape(-1, 1), 8)
    return dict(shape=list(shape), demand_rows=len(ea), writes=len(writes), seed=seed)


def bench(config, refs):
    rows, cols, width, burst, ab, bb, words, intermediates, _ = config
    cases = []
    for i, ref in enumerate(refs):
        m, k, n = ref['shape']
        loads = []
        for name in ('ea', 'eb', 'ec', 'ev'):
            loads.append(f'$readmemh("{i}_{name}.hex",{name},0,{ref["demand_rows"]-1});')
        for name in ('ewaddr', 'ewdata'):
            loads.append(f'$readmemh("{i}_{name}.hex",{name},0,{ref["writes"]-1});')
        for name, size in [('a', m*k), ('b', k*n), ('final', m*n)]:
            loads.append(f'$readmemh("{i}_{name}.hex",{name}_mem,0,{size-1});')
        cases.append(f'''{i}: begin
          cmd_m={m}; cmd_k={k}; cmd_n={n};
          expected_rows={ref['demand_rows']}; expected_writes={ref['writes']};
          a_words={m*k}; b_words={k*n}; c_words={m*n};
          {' '.join(loads)}
        end''')
    replacements = dict(PARAMS=f'R={rows},C={cols},DW={width},BURST={burst},AB={ab},BB={bb},'
                               f'WORDS={words},INTERMEDIATES={intermediates}',
                        CASES='\n'.join(cases),
                        DMAX=max(r['demand_rows'] for r in refs),
                        WMAX=max(r['writes'] for r in refs),
                        MMAX=4)
    text = Path(__file__).with_name('axi_top_tb.sv.in').read_text()
    for key, value in replacements.items():
        text = text.replace('@'+key+'@', str(value))
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=int)
    parser.add_argument('--lint-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    report = OUT/('lint_results.json' if args.lint_only else 'results.json')
    report.unlink(missing_ok=True)
    results = []
    for index, config in enumerate(CONFIGS):
        if args.case is not None and args.case != index:
            continue
        folder = OUT/f'case_{index}'
        folder.mkdir(exist_ok=True)
        r, c, width, burst, ab, bb, words, intermediates, shape = config
        sources = sorted((ROOT/'src_rtl/rtl').glob('*.sv'))
        if args.lint_only:
            parameters = dict(ROWS=r, COLS=c, MAX_M=4, A_BANKS=ab, B_BANKS=bb,
                              A_PAGE_WORDS=words, B_PAGE_WORDS=words, AXI_DATA_WIDTH=width,
                              MAX_BURST_BEATS=burst, WRITE_INTERMEDIATES=intermediates)
            lint = subprocess.run(['verilator', '--lint-only', '-Wall', '--top-module',
                                   'scalesim_ws_axi_top',
                                   *(f'-G{k}={v}' for k, v in parameters.items()),
                                   *map(str, sources)], capture_output=True, text=True)
            (folder/'lint.log').write_text(lint.stdout+lint.stderr)
            if lint.returncode:
                raise RuntimeError(f'AXI lint failed: {folder}/lint.log')
            results.append(dict(index=index, parameters=parameters, passed=True))
            print(f'PASS AXI lint {index}: {width}-bit bus / {r}x{c} array', flush=True)
            continue
        refs = [reference(folder, j, r, c, s, intermediates, 8181+index*10+j)
                for j, s in enumerate((shape, (1, 1, 1), (4, 128, 2)))]
        (folder/'tb.sv').write_text(bench(config, refs))
        build = subprocess.run(['iverilog', '-g2012', '-s', 'tb', '-o', str(folder/'sim.vvp'),
                                *map(str, sources), str(folder/'tb.sv')], capture_output=True, text=True)
        (folder/'compile.log').write_text(build.stdout+build.stderr)
        if build.returncode:
            raise RuntimeError(f'Compilation failed: {folder}/compile.log')
        sim = subprocess.run(['vvp', str(folder/'sim.vvp')], cwd=folder, capture_output=True,
                             text=True, timeout=180)
        (folder/'simulation.log').write_text(sim.stdout+sim.stderr)
        if sim.returncode:
            raise RuntimeError(f'AXI case {index} failed:\n{sim.stdout}{sim.stderr}')
        match = re.search(r'PASS AXI complete=(\d+) rejected=(\d+) faults=(\d+) resets=(\d+) '
                          r'aw_first=(\d+) w_first=(\d+) simultaneous=(\d+) boundaries=(\d+) full256=(\d+)', sim.stdout)
        if not match:
            raise RuntimeError(f'Missing AXI PASS record: {folder}/simulation.log')
        coverage = dict(zip(('completed', 'rejected', 'faults', 'reset_aborts', 'aw_first',
                             'w_first', 'simultaneous_aw_w', 'boundary_splits', 'full_256_beat_bursts'), map(int, match.groups())))
        results.append(dict(index=index, array=[r, c], axi_data_width=width, max_burst_beats=burst,
                            write_intermediates=bool(intermediates), references=refs,
                            coverage=coverage, passed=True))
        print(f'case {index}: {match.group(0)}', flush=True)
    report.write_text(json.dumps(dict(cases=results, passed=True,
        reference_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()), indent=2)+'\n')


if __name__ == '__main__':
    main()
