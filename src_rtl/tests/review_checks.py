#!/usr/bin/env python3
"""Independent reset/reconfiguration checks for the dense WS SRAM interface.

All demand vectors come from the upstream compute class; every output is checked
against an independent NumPy prefix-K matrix multiplication. This suite exercises
reset while stalled, changing shapes/bases without reset, and idle/reject behavior.
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


def pack(path, values, width):
    mask = (1 << width) - 1
    with path.open('w') as handle:
        for row in values:
            handle.write(''.join(f'{int(v) & mask:0{width//4}x}' for v in reversed(row)) + '\n')


def main():
    folder = ROOT / 'src_rtl' / 'build' / 'review_checks'
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'review.json').unlink(missing_ok=True)
    rows, cols, max_m = 2, 3, 6
    shapes = [(4, 7, 5), (3, 5, 4), (1, 1, 1), (6, 3, 7), (2, 8, 3)]
    metadata = []
    for index, (m, k, n) in enumerate(shapes):
        abase, bbase, cbase = 43 + index * 113, 712 + index * 137, 3021 + index * 211
        # Exercise the inclusive 32-bit upper boundary without huge memories.
        if index == 2:
            abase = bbase = cbase = 0xffffffff
        cfg = scale_config()
        cfg.valid_conf_flag = True
        cfg.array_rows, cfg.array_cols = rows, cols
        amap = np.arange(m*k, dtype=np.int64).reshape(m, k) + abase
        bmap = np.arange(k*n, dtype=np.int64).reshape(k, n) + bbase
        cmap = np.arange(m*n, dtype=np.int64).reshape(m, n) + cbase
        reference = systolic_compute_ws()
        reference.set_params(config_obj=cfg, ifmap_op_mat=amap,
                             filter_op_mat=bmap, ofmap_op_mat=cmap)
        a_trace, b_trace, c_trace = [v.astype(np.int64) for v in reference.get_demand_matrices()]
        rng = np.random.default_rng(821 + index)
        a = rng.integers(-128, 128, (m, k), dtype=np.int16).astype(np.int8)
        b = rng.integers(-128, 128, (k, n), dtype=np.int16).astype(np.int8)
        a.flat[0] = -128
        b.flat[0] = -128
        values = np.zeros_like(c_trace)
        final = np.zeros((len(c_trace), 1), dtype=np.int64)
        fold_length = m + 2*rows + cols - 2
        folds = (k + rows - 1) // rows
        for cycle, requests in enumerate(c_trace):
            fold = (cycle // fold_length) % folds
            prefix = min(k, (fold + 1)*rows)
            product = a[:, :prefix].astype(np.int64) @ b[:prefix].astype(np.int64)
            final[cycle, 0] = fold == folds - 1
            for lane, address in enumerate(requests):
                if address != -1:
                    mi, ni = divmod(int(address-cbase), n)
                    values[cycle, lane] = np.int32(product[mi, ni])
        # A valid address may itself be 0xffffffff, so masks are separate files.
        for name, trace, lanes in [('a', a_trace, rows), ('b', b_trace, cols), ('c', c_trace, cols)]:
            pack(folder / f'{index}_e{name}.hex', trace, 32)
            mask = np.array([[sum((int(v) != -1) << lane for lane, v in enumerate(row))]
                             for row in trace], dtype=np.int64)
            pack(folder / f'{index}_v{name}.hex', mask, 32)
        pack(folder / f'{index}_a.hex', a.reshape(-1, 1), 8)
        pack(folder / f'{index}_b.hex', b.reshape(-1, 1), 8)
        pack(folder / f'{index}_value.hex', values, 32)
        pack(folder / f'{index}_final.hex', final, 32)
        metadata.append((m, k, n, len(c_trace), abase, bbase, cbase))

    commands = []
    for index, (m, k, n, cycles, abase, bbase, cbase) in enumerate(metadata):
        commands.append(f'''{index}: begin
          cmd_m={m}; cmd_k={k}; cmd_n={n}; expected={cycles};
          cmd_a_base=32'h{abase:08x}; cmd_b_base=32'h{bbase:08x}; cmd_c_base=32'h{cbase:08x};
          $readmemh("{index}_a.hex",amem,0,{m*k-1}); $readmemh("{index}_b.hex",bmem,0,{k*n-1});
          $readmemh("{index}_ea.hex",ea,0,{cycles-1}); $readmemh("{index}_eb.hex",eb,0,{cycles-1});
          $readmemh("{index}_ec.hex",ec,0,{cycles-1}); $readmemh("{index}_value.hex",ev,0,{cycles-1});
          $readmemh("{index}_va.hex",va,0,{cycles-1}); $readmemh("{index}_vb.hex",vb,0,{cycles-1});
          $readmemh("{index}_vc.hex",vc,0,{cycles-1}); $readmemh("{index}_final.hex",ef,0,{cycles-1});
        end''')
    tb = r'''`timescale 1ns/1ps
module tb;
localparam R=2,C=3;
reg clk=0; always #5 clk=~clk;
reg rst=1,cmd_valid=0,cycle_ready=1;
reg [31:0] cmd_m=0,cmd_k=0,cmd_n=0,cmd_a_base=0,cmd_b_base=0,cmd_c_base=0;
reg [31:0] active_a_base=0,active_b_base=0;
wire cmd_ready,busy,done,error,c_final,cycle_fire;
wire [63:0] logical_cycle,elapsed_cycles,stall_cycles;
wire [R-1:0] a_valid; wire [C-1:0] b_valid,c_valid;
wire [R*32-1:0] a_addr; wire [C*32-1:0] b_addr,c_addr,c_data;
reg [R*8-1:0] a_data; reg [C*8-1:0] b_data;
reg [7:0] amem[0:127], bmem[0:127];
reg [R*32-1:0] ea[0:511];
reg [C*32-1:0] eb[0:511],ec[0:511],ev[0:511];
reg [31:0] va[0:511],vb[0:511],vc[0:511],ef[0:511];
integer expected,slot,physical,stalls,completed=0,aborted=0;
integer r,c,j;
scalesim_ws_top #(.ROWS(R),.COLS(C),.MAX_M(6)) dut(.*);
always @* begin
  // Poison invalid lanes: the datapath must mask them, even during reset/restart.
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
    active_a_base=cmd_a_base; active_b_base=cmd_b_base;
  end
endtask
task check_slot;
  begin
    slot=logical_cycle;
    if(slot>=expected) $fatal(1,"Too many slots");
    if(a_valid!==va[slot][R-1:0] || b_valid!==vb[slot][C-1:0] || c_valid!==vc[slot][C-1:0])
      $fatal(1,"Mask mismatch slot %0d",slot);
    for(r=0;r<R;r=r+1)
      if(a_valid[r] && a_addr[r*32+:32]!==ea[slot][r*32+:32]) $fatal(1,"A addr slot %0d",slot);
    for(c=0;c<C;c=c+1) begin
      if(b_valid[c] && b_addr[c*32+:32]!==eb[slot][c*32+:32]) $fatal(1,"B addr slot %0d",slot);
      if(c_valid[c]) begin
        if(c_addr[c*32+:32]!==ec[slot][c*32+:32]) $fatal(1,"C addr slot %0d",slot);
        if(c_data[c*32+:32]!==ev[slot][c*32+:32]) $fatal(1,"C data slot %0d lane %0d",slot,c);
        if(c_final!==ef[slot][0]) $fatal(1,"C final slot %0d",slot);
      end
    end
  end
endtask
task check_idle;
  begin
    if(busy || cmd_ready!==!rst || cycle_fire || a_valid || b_valid || c_valid)
      $fatal(1,"Idle interface leaks transaction");
  end
endtask
task execute(input integer index,input integer reset_at);
  begin
    load_case(index); cmd_valid=1; cycle_ready=1;
    @(posedge clk); #1;
    if(!busy || error) $fatal(1,"Valid command rejected index %0d",index);
    @(negedge clk); cmd_valid=0;
    // The accepted descriptor must be latched; live command inputs may change.
    cmd_m=0; cmd_k=0; cmd_n=0;
    cmd_a_base=32'hdeadbeef; cmd_b_base=0; cmd_c_base=32'hffffffff;
    physical=0; stalls=0;
    while(busy && (reset_at<0 || logical_cycle<reset_at)) begin
      cycle_ready=(physical%5!=1 && physical%5!=2);
      cmd_valid=(physical==3); // An invalid command presented while busy is ignored.
      #1; check_slot();
      if(cycle_fire!==cycle_ready) $fatal(1,"Acceptance mismatch");
      if(!cycle_ready) stalls=stalls+1;
      physical=physical+1;
      @(posedge clk); #1;
      if(busy) @(negedge clk);
    end
    if(reset_at>=0) begin
      cmd_valid=0;
      // Hold a valid request for three clocks, then reset while still stalled.
      cycle_ready=0;
      repeat(3) begin
        #1; check_slot();
        @(posedge clk); #1;
        if(logical_cycle!=reset_at || cycle_fire) $fatal(1,"Stall advanced state");
        @(negedge clk);
      end
      if(reset_at==10) cycle_ready=1;
      rst=1; #1;
      if(cycle_fire || cmd_ready) $fatal(1,"Reset must suppress external acceptance");
      @(posedge clk); #1;
      check_idle();
      if(done || error || logical_cycle || elapsed_cycles || stall_cycles) $fatal(1,"Reset state mismatch");
      @(negedge clk); rst=0; cycle_ready=1;
      aborted=aborted+1;
    end else begin
      if(!done || error || logical_cycle!=expected || elapsed_cycles!=physical || stall_cycles!=stalls)
        $fatal(1,"Completion counters mismatch");
      completed=completed+1;
      @(negedge clk);
    end
  end
endtask
task reject(input integer kind);
  begin
    load_case(1);
    case(kind)
      0:cmd_m=0;
      1:cmd_k=0;
      2:cmd_n=0;
      3:cmd_m=7;
      4:cmd_a_base=32'hffffffff;
      5:cmd_b_base=32'hffffffff;
      6:cmd_c_base=32'hffffffff;
      7:begin cmd_k=32'hffffffff; cmd_n=32'hffffffff; end
    endcase
    cmd_valid=1;
    @(posedge clk); #1;
    if(!done || !error || busy) $fatal(1,"Invalid command accepted kind %0d",kind);
    check_idle();
    @(negedge clk); cmd_valid=0;
    @(posedge clk); #1;
    if(done || error) $fatal(1,"Done/error must pulse once");
    @(negedge clk);
  end
endtask
initial begin
  repeat(2) @(posedge clk);
  @(negedge clk); rst=0;
  execute(0,0); execute(1,-1);
  execute(0,1); execute(2,-1);
  execute(0,4); execute(3,-1);
  execute(0,6); execute(4,-1);
  execute(0,10); execute(1,-1);
  execute(2,-1); execute(3,-1); execute(4,-1); execute(0,-1);
  repeat(3) begin
    @(posedge clk); #1; check_idle();
    if(done || error) $fatal(1,"Completion pulse persisted");
    @(negedge clk);
  end
  for(j=0;j<8;j=j+1) reject(j);
  execute(3,-1);
  $display("PASS independent review: %0d completed, %0d midflight resets, 8 rejection classes",completed,aborted);
  $finish;
end
endmodule
'''.replace('CASES', '\n'.join(commands))
    (folder / 'tb.sv').write_text(tb)
    sources = sorted((ROOT / 'src_rtl' / 'rtl').glob('*.sv'))
    subprocess.run(['iverilog', '-g2012', '-s', 'tb', '-o', str(folder/'sim.vvp'),
                    *map(str, sources), str(folder/'tb.sv')], check=True)
    result = subprocess.run(['vvp', str(folder/'sim.vvp')], cwd=folder,
                            text=True, capture_output=True)
    print(result.stdout, end='')
    if result.returncode or 'PASS independent review: 10 completed, 5 midflight resets, 8 rejection classes' not in result.stdout or '$finish called at' not in result.stdout:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode or 1)
    (folder/'review.json').write_text(json.dumps({'passed': True, 'completed_commands': 10,
        'midflight_resets': 5, 'invalid_command_classes': 8,
        'coverage': ['different shapes and bases without reset', 'reset while stalled in several phases',
                     'latched descriptor despite live command input changes', 'command ignored while busy',
                     'prefix-K signed values', 'invalid lane poisoning', 'valid address 0xffffffff',
                     'dimension/address rejection', 'done/error single pulses']}, indent=2)+'\n')


if __name__ == '__main__':
    main()
