#!/usr/bin/env python3
"""Check generic synthesis and actual clocked SRAM inference in the wrapper.

This deliberately preserves RAM cells; it is not FPGA mapping or a timing run.
The small output packet FIFO may have asynchronous reads. Operand and partial-
sum SRAMs must each infer one clocked read and one clocked write per bank.
"""
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'src_rtl/build/buffered_synthesis_checks'
TOP = 'scalesim_ws_buffered_top'


def check(name, parameters, *, top=TOP, memory_prefix=''):
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    sources = sorted((ROOT / 'src_rtl/rtl').glob('*.sv'))
    netlist = folder / 'netlist.json'
    commands = 'read_verilog -sv ' + ' '.join(f'"{p}"' for p in sources) + '; '
    if parameters:
        commands += 'chparam ' + ' '.join(f'-set {k} {v}' for k, v in parameters.items()) + f' {top}; '
    commands += (f'hierarchy -check -top {top}; proc; opt; memory_dff; '
                 'memory_collect; opt_clean; check -assert; stat; '
                 f'write_json "{netlist}"')
    result = subprocess.run(['yosys', '-Q', '-p', commands], capture_output=True, text=True)
    (folder / 'synthesis.log').write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f'Yosys failed: {folder}/synthesis.log')
    modules = json.loads(netlist.read_text())['modules']
    memories = []

    def visit(module_name, path):
        module = modules[module_name]
        for cell_name, cell in module['cells'].items():
            instance = (path + '/' if path else '') + cell_name
            kind = cell['type']
            assert 'latch' not in kind.lower(), f'Unexpected latch: {instance}'
            if kind in modules:
                visit(kind, instance)
            elif kind == '$mem_v2' and path in tuple(memory_prefix + p for p in ('a_buffer', 'b_buffer', 'core')):
                p, connections = cell['parameters'], cell['connections']
                for field in ('RD_PORTS', 'WR_PORTS', 'RD_CLK_ENABLE', 'WR_CLK_ENABLE'):
                    assert int(p[field], 2) == 1, f'{instance}: {field}={p[field]}'
                clock = module['ports']['clk']['bits']
                assert connections['RD_CLK'] == connections['WR_CLK'] == clock, instance
                assert connections['RD_EN'] != ['1'], f'Missing read enable: {instance}'
                memories.append(dict(instance=instance, width=int(p['WIDTH'], 2),
                                     words=int(p['SIZE'], 2), clocked_read=True,
                                     clocked_write=True))

    visit(top, '')
    return memories


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = OUT / 'results.json'
    report.unlink(missing_ok=True)
    cases = [
        ('default', dict(ROWS=32, COLS=32, MAX_M=512, A_BANKS=32, B_BANKS=32,
                         A_PAGE_WORDS=32768, B_PAGE_WORDS=32768, OUTPUT_DEPTH=2), TOP, ''),
        ('unequal_banks', dict(ROWS=4, COLS=3, MAX_M=8, A_BANKS=4, B_BANKS=1,
                               A_PAGE_WORDS=64, B_PAGE_WORDS=64, OUTPUT_DEPTH=3), TOP, ''),
        ('axi_default', dict(ROWS=32, COLS=32, MAX_M=512, A_BANKS=32, B_BANKS=32,
                             A_PAGE_WORDS=32768, B_PAGE_WORDS=32768, OUTPUT_DEPTH=2,
                             AXI_DATA_WIDTH=512, MAX_BURST_BEATS=16), 'scalesim_ws_axi_top', 'engine/'),
        ('axi_small', dict(ROWS=2, COLS=3, MAX_M=4, A_BANKS=1, B_BANKS=1,
                           A_PAGE_WORDS=64, B_PAGE_WORDS=64, OUTPUT_DEPTH=1,
                           AXI_DATA_WIDTH=32, MAX_BURST_BEATS=256,
                           WRITE_INTERMEDIATES=1), 'scalesim_ws_axi_top', 'engine/'),
    ]
    summaries = []
    for name, parameters, top, prefix in cases:
        memories = check(name, parameters, top=top, memory_prefix=prefix)
        for owner, count, width, words in (
            ('a_buffer', parameters['A_BANKS'], 8, 2 * parameters['A_PAGE_WORDS'] // parameters['A_BANKS']),
            ('b_buffer', parameters['B_BANKS'], 8, 2 * parameters['B_PAGE_WORDS'] // parameters['B_BANKS']),
            ('core', parameters['COLS'], 32, parameters['MAX_M']),
        ):
            banks = [m for m in memories if m['instance'].startswith(prefix + owner + '/')]
            assert len(banks) == count, f'{name}/{owner}: expected {count} SRAMs, got {len(banks)}'
            assert all(m['width'] == width and m['words'] == words for m in banks), banks
        summary = dict(name=name, top=top, parameters=parameters, sram_banks=len(memories),
                       sram_bytes=sum(m['width'] * m['words'] // 8 for m in memories),
                       memories=memories, passed=True)
        summaries.append(summary)
        print(f"PASS buffered synthesis {name}: {summary['sram_banks']} clocked SRAM banks, "
              f"{summary['sram_bytes']} bytes", flush=True)
    report.write_text(json.dumps(dict(cases=summaries, passed=True,
        scope='Generic synthesis and RAM inference only; no device mapping, placement, or timing.'), indent=2) + '\n')


if __name__ == '__main__':
    main()
