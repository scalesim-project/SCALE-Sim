#!/usr/bin/env python3
"""Check clocked partial-sum RAM against native WS demands and NumPy GEMM.

Data is compared only on cycle_fire. Extra physical clocks are checked against
the one-read-wait contract, including waits hidden by external backpressure.
Yosys must infer clocked, enabled read ports for all partial-sum banks.
"""
from pathlib import Path
import json
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scalesim.compute.systolic_compute_ws import systolic_compute_ws
from scalesim.scale_config import scale_config

OUT = ROOT / 'src_rtl/build/sync_psum_checks'
SOURCES = sorted((ROOT / 'src_rtl/rtl').glob('*.sv'))


def pack(path, matrix, width=32):
    mask = (1 << width) - 1
    path.write_text(''.join(''.join(f'{int(v) & mask:0{width//4}x}' for v in reversed(row))
                           + '\n' for row in matrix))


def vectors(folder, index, rows, cols, shape):
    m, k, n = shape
    abase, bbase, cbase = 13 + index*17, 100000 + index*23, 200000 + index*31
    config = scale_config()
    config.valid_conf_flag = True
    config.array_rows, config.array_cols = rows, cols
    ref = systolic_compute_ws()
    ref.set_params(config_obj=config,
                   ifmap_op_mat=np.arange(m*k).reshape(m, k) + abase,
                   filter_op_mat=np.arange(k*n).reshape(k, n) + bbase,
                   ofmap_op_mat=np.arange(m*n).reshape(m, n) + cbase)
    ea, eb, ec = (x.astype(np.int64) for x in ref.get_demand_matrices())
    rng = np.random.default_rng(19381 + index + rows*100 + cols)
    a = rng.integers(-128, 128, (m, k), dtype=np.int16).astype(np.int8)
    b = rng.integers(-128, 128, (k, n), dtype=np.int16).astype(np.int8)
    a.flat[0] = b.flat[0] = -128
    folds = (k + rows - 1) // rows
    length = m + 2*rows + cols - 2
    prefixes = [(a[:, :min(k, (f+1)*rows)].astype(np.int64)
                 @ b[:min(k, (f+1)*rows)].astype(np.int64)).astype(np.int32)
                for f in range(folds)]
    values, final, needs_read = np.zeros_like(ec), [], []
    for t, requests in enumerate(ec):
        fold = (t // length) % folds
        final.append([fold == folds-1])
        needs_read.append([fold > 0 and np.any(requests != -1)])
        for c, addr in enumerate(requests):
            if addr != -1:
                mi, ni = divmod(int(addr-cbase), n)
                values[t, c] = prefixes[fold][mi, ni]
    for name, trace in [('a', ea), ('b', eb), ('c', ec)]:
        pack(folder/f'{index}_e{name}.hex', trace)
        pack(folder/f'{index}_v{name}.hex',
             [[sum((int(v) != -1) << lane for lane, v in enumerate(row))] for row in trace])
    for name, data, bits in [('a', a.reshape(-1, 1), 8), ('b', b.reshape(-1, 1), 8),
                             ('value', values, 32), ('final', final, 32), ('need', needs_read, 32)]:
        pack(folder/f'{index}_{name}.hex', data, bits)
    load = f'''{index}: begin
      cmd_m={m}; cmd_k={k}; cmd_n={n}; expected={len(ea)};
      expected_reads={sum(int(x[0]) for x in needs_read)};
      cmd_a_base={abase}; cmd_b_base={bbase}; cmd_c_base={cbase};
      $readmemh("{index}_a.hex",amem,0,{a.size-1});
      $readmemh("{index}_b.hex",bmem,0,{b.size-1});
    '''
    for name, array in [('ea', 'ea'), ('eb', 'eb'), ('ec', 'ec'), ('va', 'va'),
                         ('vb', 'vb'), ('vc', 'vc'), ('value', 'ev'), ('final', 'ef'), ('need', 'en')]:
        load += f'$readmemh("{index}_{name}.hex",{array},0,{len(ea)-1});\n'
    return load + 'end', dict(gemm=list(shape), demand_cycles=len(ea),
                             read_wait_rows=sum(int(x[0]) for x in needs_read))


BENCH = r'''`timescale 1ns/1ps
module tb;
localparam R=ROWS,C=COLS;
reg clk=0; always #5 clk=~clk;
reg rst=1,cmd_valid=0,cycle_ready=1;
reg [31:0] cmd_m=0,cmd_k=0,cmd_n=0,cmd_a_base=0,cmd_b_base=0,cmd_c_base=0;
reg [31:0] active_a_base,active_b_base;
wire cmd_ready,busy,done,error,c_final,cycle_fire;
wire [63:0] logical_cycle,elapsed_cycles,stall_cycles;
wire [R-1:0] a_valid; wire [C-1:0] b_valid,c_valid;
wire [R*32-1:0] a_addr; wire [C*32-1:0] b_addr,c_addr,c_data;
reg [R*8-1:0] a_data; reg [C*8-1:0] b_data;
reg [7:0] amem[0:AMAX-1],bmem[0:BMAX-1];
reg [R*32-1:0] ea[0:DMAX-1];
reg [C*32-1:0] eb[0:DMAX-1],ec[0:DMAX-1],ev[0:DMAX-1];
reg [31:0] va[0:DMAX-1],vb[0:DMAX-1],vc[0:DMAX-1],ef[0:DMAX-1],en[0:DMAX-1];
integer expected,expected_reads,slot,physical,stalls,read_waits,internal_only;
integer completed=0,aborted=0,r,c,j;
reg have_read,take,need;
scalesim_ws_top #(.ROWS(R),.COLS(C),.MAX_M(MMAX),.SYNC_PSUM(1)) dut(.*);
always @(a_valid or a_addr or b_valid or b_addr) begin
  a_data={R{8'h80}}; b_data={C{8'h7f}};
  for(integer ar=0;ar<R;ar=ar+1)
    if(a_valid[ar]) a_data[ar*8+:8]=amem[a_addr[ar*32+:32]-active_a_base];
  for(integer bc=0;bc<C;bc=bc+1)
    if(b_valid[bc]) b_data[bc*8+:8]=bmem[b_addr[bc*32+:32]-active_b_base];
end
task load_case(input integer index);
  begin
    case(index)
      CASES
    endcase
    active_a_base=cmd_a_base;active_b_base=cmd_b_base;
  end
endtask
task step;
  begin
    #1; slot=logical_cycle;
    if(slot>=expected) $fatal(1,"Too many logical rows");
    if(a_valid!==va[slot][R-1:0] || b_valid!==vb[slot][C-1:0] || c_valid!==vc[slot][C-1:0])
      $fatal(1,"Native demand masks differ slot=%0d",slot);
    for(r=0;r<R;r=r+1)
      if(a_valid[r] && a_addr[r*32+:32]!==ea[slot][r*32+:32]) $fatal(1,"A address differs");
    for(c=0;c<C;c=c+1) begin
      if(b_valid[c] && b_addr[c*32+:32]!==eb[slot][c*32+:32]) $fatal(1,"B address differs");
      if(c_valid[c] && c_addr[c*32+:32]!==ec[slot][c*32+:32]) $fatal(1,"C address differs");
      // Read-data is not required to be ready until the row commits.
      if(cycle_fire && c_valid[c]) begin
        if(c_data[c*32+:32]!==ev[slot][c*32+:32])
          $fatal(1,"C value slot=%0d lane=%0d got=%0d expected=%0d",slot,c,$signed(c_data[c*32+:32]),$signed(ev[slot][c*32+:32]));
        if(c_final!==ef[slot][0]) $fatal(1,"Final fold differs");
      end
    end
    need=en[slot][0];
    if(cycle_fire !== (cycle_ready && (!need || have_read)))
      $fatal(1,"Clocked read handshake differs slot=%0d read=%0d ready=%0d",slot,have_read,cycle_ready);
    take=cycle_fire;
    if(!take) stalls=stalls+1;
    if(!take && cycle_ready) internal_only=internal_only+1;
    if(need && !have_read) read_waits=read_waits+1;
    physical=physical+1;
    @(posedge clk); #1;
    if(logical_cycle != slot+(take?1:0)) $fatal(1,"Logical row advanced incorrectly");
    if(elapsed_cycles!=physical || stall_cycles!=stalls) $fatal(1,"Physical counters differ");
    have_read=take ? 0 : (have_read || need);
    if(physical>expected*8+100) $fatal(1,"Timeout");
    if(busy) @(negedge clk);
  end
endtask
task execute(input integer index,input integer external_stalls,input integer reset_kind);
  begin
    load_case(index);cmd_valid=1;cycle_ready=1;
    @(posedge clk);#1;
    if(!busy || error) $fatal(1,"Command rejected");
    @(negedge clk);cmd_valid=0;
    cmd_m=0;cmd_k=0;cmd_n=0;cmd_a_base=32'hffffffff;cmd_b_base=0;cmd_c_base=0;
    physical=0;stalls=0;read_waits=0;internal_only=0;have_read=0;
    while(busy && !(reset_kind>=0 && en[logical_cycle][0])) begin
      cycle_ready=!external_stalls || (physical%7!=1 && physical%7!=2 && physical%7!=3);
      step();
    end
    if(reset_kind>=0) begin
      if(!busy || !en[logical_cycle][0] || have_read) $fatal(1,"Reset target missing");
      cycle_ready=0;
      // 0: before RAM read; 1: immediately after; 3: held ready under backpressure.
      repeat(reset_kind) step();
      rst=1;cycle_ready=1;#1;
      if(cycle_fire || cmd_ready) $fatal(1,"Reset permits external acceptance");
      @(posedge clk);#1;
      if(busy || done || error || logical_cycle || elapsed_cycles || stall_cycles || a_valid || b_valid || c_valid)
        $fatal(1,"Reset failed to clear transaction state");
      @(negedge clk);rst=0;
      aborted=aborted+1;
    end else begin
      if(!done || error || logical_cycle!=expected || read_waits!=expected_reads)
        $fatal(1,"Completion/read count differs got=%0d expected=%0d",read_waits,expected_reads);
      if(!external_stalls && (stalls!=expected_reads || physical!=expected+expected_reads))
        $fatal(1,"Unstalled synchronous read latency differs");
      completed=completed+1;
      $display("PASS command=%0d R=%0d C=%0d logical=%0d physical=%0d stalls=%0d read_waits=%0d internal_only=%0d",completed,R,C,logical_cycle,physical,stalls,read_waits,internal_only);
      @(negedge clk);
    end
  end
endtask
initial begin
  repeat(2) @(posedge clk);
  @(negedge clk);rst=0;
  execute(0,0,-1);execute(1,0,-1);execute(0,1,-1);execute(1,1,-1);
  execute(0,0,0);execute(1,1,-1);
  execute(0,0,1);execute(0,0,-1);
  execute(0,0,3);execute(1,0,-1);
  $display("PASS synchronous psum: %0d completed, %0d reset aborts",completed,aborted);
  $finish;
end
endmodule
'''


def simulate(index, rows, cols, shapes):
    folder = OUT/f'case_{index}'
    folder.mkdir(parents=True, exist_ok=True)
    loads, metadata = zip(*(vectors(folder, i, rows, cols, s) for i, s in enumerate(shapes)))
    text = BENCH.replace('R=ROWS,C=COLS', f'R={rows},C={cols}')
    replacements = dict(AMAX=max(m*k for m,k,n in shapes),
                        BMAX=max(k*n for m,k,n in shapes), MMAX=max(s[0] for s in shapes),
                        DMAX=max(x['demand_cycles'] for x in metadata), CASES='\n'.join(loads))
    for name, value in replacements.items():
        text = text.replace(name, str(value))
    bench = folder/'tb.sv'
    bench.write_text(text)
    result = subprocess.run(['iverilog', '-g2012', '-s', 'tb', '-o', str(folder/'sim.vvp'),
                             *map(str, SOURCES), str(bench)], capture_output=True, text=True)
    (folder/'compile.log').write_text(result.stdout+result.stderr)
    if result.returncode:
        raise RuntimeError(f'Compilation failed: {folder}/compile.log')
    result = subprocess.run(['vvp', str(folder/'sim.vvp')], cwd=folder, capture_output=True, text=True)
    (folder/'simulation.log').write_text(result.stdout+result.stderr)
    if (result.returncode or 'PASS synchronous psum: 7 completed, 3 reset aborts' not in result.stdout
            or '$finish called at' not in result.stdout):
        raise RuntimeError(f'Simulation failed: {folder}/simulation.log\n{result.stdout}{result.stderr}')
    print(f'PASS synchronous psum {rows}x{cols}: 7 commands, 3 reset aborts', flush=True)
    return dict(array=[rows, cols], cases=metadata, completed_commands=7, reset_aborts=3)


def check_clocked_ports():
    netlist = OUT/'netlist.json'
    commands = ('read_verilog -sv ' + ' '.join(map(str, SOURCES))
                + '; chparam -set ROWS 2 -set COLS 3 -set MAX_M 8 -set SYNC_PSUM 1 scalesim_ws_top; '
                'hierarchy -top scalesim_ws_top; proc; opt; memory_dff; memory_collect; opt_clean; check -assert; '
                f'write_json {netlist}')
    result = subprocess.run(['yosys', '-Q', '-p', commands], text=True, capture_output=True)
    (OUT/'synthesis.log').write_text(result.stdout+result.stderr)
    if result.returncode:
        raise RuntimeError(f'Yosys failed: {OUT}/synthesis.log')
    module = json.loads(netlist.read_text())['modules']['scalesim_ws_top']
    memories = {name: cell for name, cell in module['cells'].items() if cell['type'] == '$mem_v2'}
    assert len(memories) == 3, f'Expected 3 partial-sum RAM banks, got {len(memories)}'
    for name, cell in memories.items():
        p = cell['parameters']
        assert int(p['RD_PORTS'], 2) == 1 and int(p['WR_PORTS'], 2) == 1, name
        assert int(p['RD_CLK_ENABLE'], 2) == 1 and int(p['WR_CLK_ENABLE'], 2) == 1, name
        assert cell['connections']['RD_CLK'] == module['ports']['clk']['bits'], name
        assert cell['connections']['RD_EN'] != ['1'], f'{name}: read enable missing'
    print('PASS Yosys: 3 RAM banks with clocked, enabled read and write ports', flush=True)
    return dict(banks=len(memories), clocked_read_ports=True, enabled_read_ports=True,
                clocked_write_ports=True, scope='generic inference; not FPGA placement/timing')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'review.json').unlink(missing_ok=True)
    suites = [(1, 1, [(1, 4, 1), (3, 2, 3)]),
              (1, 3, [(4, 5, 7), (2, 3, 2)]),
              (3, 1, [(3, 7, 2), (1, 4, 3)]),
              (2, 3, [(4, 7, 5), (2, 3, 4)]),
              (32, 32, [(1, 65, 35), (4, 33, 31)])]
    results = [simulate(i, r, c, shapes) for i, (r, c, shapes) in enumerate(suites)]
    synthesis = check_clocked_ports()
    report = dict(passed=True, completed_commands=35, reset_aborts=15, suites=results,
                  reference_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  synthesis=synthesis)
    (OUT/'review.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'PASS synchronous psum checks: {OUT}/review.json')


if __name__ == '__main__':
    main()
