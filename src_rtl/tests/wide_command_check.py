#!/usr/bin/env python3
"""Check the first real fold of a Gemma-scale command without huge traces.

The DUT receives full dimensions M=512,K=6912,N=262144. The upstream oracle
receives only the first 32x32 operand tile, with the full command's row strides,
so it produces exactly the first fold's lane schedule at bounded memory cost.
This validates acceptance/address width and one fold, not the complete command.
"""
from pathlib import Path
import json
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scalesim.scale_config import scale_config
from scalesim.compute.systolic_compute_ws import systolic_compute_ws
from review_checks import pack


def main():
    folder = ROOT / 'src_rtl' / 'build' / 'wide_command_check'
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'review.json').unlink(missing_ok=True)
    r = c = 32
    m, k, n = 512, 6912, 262144
    a_base, b_base, c_base = 0, 10_000_000, 20_000_000
    cfg = scale_config()
    cfg.valid_conf_flag = True
    cfg.array_rows, cfg.array_cols = r, c
    amap = np.arange(m, dtype=np.int64)[:, None]*k + np.arange(r)[None, :] + a_base
    bmap = np.arange(r, dtype=np.int64)[:, None]*n + np.arange(c)[None, :] + b_base
    cmap = np.arange(m, dtype=np.int64)[:, None]*n + np.arange(c)[None, :] + c_base
    reference = systolic_compute_ws()
    reference.set_params(config_obj=cfg, ifmap_op_mat=amap, filter_op_mat=bmap, ofmap_op_mat=cmap)
    traces = [v.astype(np.int64) for v in reference.get_demand_matrices()]
    cycles = len(traces[0])
    assert cycles == m + 2*r + c - 2
    for name, trace in zip(('a', 'b', 'c'), traces):
        pack(folder / f'{name}.hex', trace, 32)
    tb = r'''`timescale 1ns/1ps
module tb;
localparam R=32,C=32,D=606;
reg clk=0; always #5 clk=~clk;
reg rst=1,cmd_valid=0,cycle_ready=1;
reg [31:0] cmd_m=512,cmd_k=6912,cmd_n=262144;
reg [31:0] cmd_a_base=0,cmd_b_base=10000000,cmd_c_base=20000000;
wire cmd_ready,busy,done,error,c_final,cycle_fire;
wire [63:0] logical_cycle,elapsed_cycles,stall_cycles;
wire [R-1:0] a_valid; wire [C-1:0] b_valid,c_valid;
wire [R*32-1:0] a_addr; wire [C*32-1:0] b_addr,c_addr,c_data;
wire [R*8-1:0] a_data={R{8'h80}};
wire [C*8-1:0] b_data={C{8'h80}};
reg [R*32-1:0] ea[0:D-1];
reg [C*32-1:0] eb[0:D-1],ec[0:D-1];
integer slot,r,c,physical=0,stalls=0,writes=0;
scalesim_ws_top dut(.*);
initial begin
  $readmemh("a.hex",ea); $readmemh("b.hex",eb); $readmemh("c.hex",ec);
  repeat(2) @(posedge clk);
  @(negedge clk); rst=0; cmd_valid=1;
  @(posedge clk); #1;
  if(!busy || error) $fatal(1,"Wide Gemma command rejected");
  @(negedge clk); cmd_valid=0;
  while(logical_cycle<D) begin
    cycle_ready=(physical!=1 && physical!=300);
    #1; slot=logical_cycle;
    if(!busy || done || error) $fatal(1,"Wide command completed during first fold");
    if(cycle_fire!==cycle_ready) $fatal(1,"Acceptance mismatch");
    for(r=0;r<R;r=r+1) begin
      if(a_valid[r] !== (ea[slot][r*32+:32]!=32'hffffffff)) $fatal(1,"A valid slot%0d lane%0d",slot,r);
      if(a_valid[r] && a_addr[r*32+:32]!==ea[slot][r*32+:32]) $fatal(1,"A addr slot%0d lane%0d",slot,r);
    end
    for(c=0;c<C;c=c+1) begin
      if(b_valid[c] !== (eb[slot][c*32+:32]!=32'hffffffff)) $fatal(1,"B valid slot%0d lane%0d",slot,c);
      if(b_valid[c] && b_addr[c*32+:32]!==eb[slot][c*32+:32]) $fatal(1,"B addr slot%0d lane%0d",slot,c);
      if(c_valid[c] !== (ec[slot][c*32+:32]!=32'hffffffff)) $fatal(1,"C valid slot%0d lane%0d",slot,c);
      if(c_valid[c]) begin
        if(c_addr[c*32+:32]!==ec[slot][c*32+:32]) $fatal(1,"C addr slot%0d lane%0d",slot,c);
        if(c_data[c*32+:32]!==32'd524288) $fatal(1,"C arithmetic slot%0d lane%0d",slot,c);
        if(c_final) $fatal(1,"First K fold marked final");
        if(cycle_fire) writes=writes+1;
      end
    end
    if(!cycle_ready) stalls=stalls+1;
    physical=physical+1;
    @(posedge clk); #1;
    @(negedge clk);
  end
  if(!busy || done || error || elapsed_cycles!=D+2 || stall_cycles!=2 || writes!=512*32)
    $fatal(1,"First fold accounting mismatch");
  rst=1; #1;
  if(cycle_fire || cmd_ready) $fatal(1,"Reset must suppress external acceptance while ready is high");
  @(posedge clk); #1;
  if(busy || done || error || a_valid || b_valid || c_valid || logical_cycle || elapsed_cycles || stall_cycles)
    $fatal(1,"Reset did not abort wide command");
  $display("PASS wide Gemma M512 K6912 N262144: first fold %0d accepted cycles, %0d cumulative writes, %0d stalls; reset abort",D,writes,stalls);
  $finish;
end
endmodule
'''
    (folder / 'tb.sv').write_text(tb)
    sources = sorted((ROOT / 'src_rtl' / 'rtl').glob('*.sv'))
    subprocess.run(['iverilog', '-g2012', '-s', 'tb', '-o', str(folder/'sim.vvp'),
                    *map(str, sources), str(folder/'tb.sv')], check=True)
    result = subprocess.run(['vvp', str(folder/'sim.vvp')], cwd=folder, capture_output=True, text=True)
    print(result.stdout, end='')
    if result.returncode or 'PASS wide Gemma M512 K6912 N262144: first fold 606 accepted cycles, 16384 cumulative writes, 2 stalls; reset abort' not in result.stdout or '$finish called at' not in result.stdout:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode or 1)
    (folder/'review.json').write_text(json.dumps({'passed': True,
        'command': {'M': m, 'K': k, 'N': n}, 'array': [r, c],
        'verified_folds': 1, 'accepted_cycles': cycles, 'stall_cycles': 2,
        'checked_partial_writes': m*c, 'scope': 'First K/N fold only; reset aborts remaining command'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
