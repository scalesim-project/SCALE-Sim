#!/usr/bin/env python3
"""Compare actual RTL lane-by-lane against upstream dense WS demand matrices.

Numeric oracle is independently computed NumPy int64 GEMM, cast to signed int32.
No Python timing formula is used to generate expected request matrices.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
RTL_ROOT = ROOT / 'src_rtl'
sys.path.insert(0, str(ROOT))
import numpy as np
from scalesim.compute.systolic_compute_ws import systolic_compute_ws
from scalesim.scale_config import scale_config


def packed_file(path: Path, matrix: np.ndarray, bits: int = 32):
    mask = (1 << bits) - 1
    with path.open('w') as out:
        for row in matrix:
            out.write(''.join(f'{int(value) & mask:0{bits//4}x}' for value in reversed(row)) + '\n')


def bench(rows, cols, m, k, n, demand_cycles, a_base, b_base, c_base, seed, stalls):
    return f'''`timescale 1ns/1ps
module tb;
localparam R={rows}, C={cols}, M={m}, K={k}, N={n}, D={demand_cycles};
reg clk=0; always #5 clk=~clk;
reg rst=1, cmd_valid=0, cycle_ready=1;
wire cmd_ready,busy,done,error,c_final,cycle_fire;
reg [31:0] cmd_m=M,cmd_k=K,cmd_n=N;
reg [31:0] cmd_a_base={a_base},cmd_b_base={b_base},cmd_c_base={c_base};
wire [63:0] logical_cycle,elapsed_cycles,stall_cycles;
wire [R-1:0] a_valid; wire [C-1:0] b_valid,c_valid;
wire [R*32-1:0] a_addr; wire [C*32-1:0] b_addr,c_addr,c_data;
reg [R*8-1:0] a_data; reg [C*8-1:0] b_data;
reg [7:0] amem[0:M*K-1]; reg [7:0] bmem[0:K*N-1];
reg [R*32-1:0] ea[0:D-1];
reg [C*32-1:0] eb[0:D-1],ec[0:D-1],ev[0:D-1];
reg [31:0] rng={seed};
integer r,c,slot,phys,stalled,rep;
scalesim_ws_top #(.ROWS(R),.COLS(C),.MAX_M(M)) dut(.*);
// Operand ROMs are initialized once, before commands. Avoid expanding a
// sensitivity list over every word of a large testbench memory.
always @(a_valid or a_addr or b_valid or b_addr) begin
  a_data=0; b_data=0;
  for(integer ar=0; ar<R; ar=ar+1)
    if(a_valid[ar]) a_data[ar*8 +:8]=amem[a_addr[ar*32 +:32]-cmd_a_base];
  for(integer bc=0; bc<C; bc=bc+1)
    if(b_valid[bc]) b_data[bc*8 +:8]=bmem[b_addr[bc*32 +:32]-cmd_b_base];
end
initial begin
  $readmemh("a.hex",amem); $readmemh("b.hex",bmem);
  $readmemh("ea.hex",ea); $readmemh("eb.hex",eb);
  $readmemh("ec.hex",ec); $readmemh("ev.hex",ev);
  repeat(2) @(posedge clk);
  @(negedge clk); rst=0;
  for(rep=0; rep<2; rep=rep+1) begin
    cmd_valid=1; cycle_ready=1;
    @(posedge clk); #1;
    if(!busy || error) $fatal(1,"Command rejected");
    @(negedge clk); cmd_valid=0;
    phys=0; stalled=0;
    while(busy) begin
      rng=rng ^ (rng<<13); rng=rng ^ (rng>>17); rng=rng ^ (rng<<5);
      cycle_ready=({1 if stalls else 0}==0 || rng[2:0]>2);
      #1; slot=logical_cycle;
      if(slot>=D) $fatal(1,"Too many logical cycles %0d",slot);
      for(r=0; r<R; r=r+1) begin
        if(a_valid[r] !== (ea[slot][r*32+:32] != 32'hffffffff))
          $fatal(1,"A valid slot=%0d lane=%0d",slot,r);
        if(a_valid[r] && a_addr[r*32+:32] !== ea[slot][r*32+:32])
          $fatal(1,"A address slot=%0d lane=%0d",slot,r);
      end
      for(c=0; c<C; c=c+1) begin
        if(b_valid[c] !== (eb[slot][c*32+:32] != 32'hffffffff))
          $fatal(1,"B valid slot=%0d lane=%0d",slot,c);
        if(b_valid[c] && b_addr[c*32+:32] !== eb[slot][c*32+:32])
          $fatal(1,"B address slot=%0d lane=%0d",slot,c);
        if(c_valid[c] !== (ec[slot][c*32+:32] != 32'hffffffff))
          $fatal(1,"C valid slot=%0d lane=%0d",slot,c);
        if(c_valid[c] && c_addr[c*32+:32] !== ec[slot][c*32+:32])
          $fatal(1,"C address slot=%0d lane=%0d",slot,c);
        if(c_valid[c] && c_data[c*32+:32] !== ev[slot][c*32+:32])
          $fatal(1,"C value slot=%0d lane=%0d got=%0d expected=%0d",slot,c,$signed(c_data[c*32+:32]),$signed(ev[slot][c*32+:32]));
        if(c_valid[c] && c_final !== ((slot/(M+2*R+C-2))%((K+R-1)/R)==((K+R-1)/R)-1))
          $fatal(1,"C final slot=%0d",slot);
      end
      if(cycle_fire !== cycle_ready) $fatal(1,"cycle_fire mismatch");
      if(!cycle_ready) stalled=stalled+1;
      phys=phys+1;
      @(posedge clk); #1;
      if(phys>D*10+100) $fatal(1,"Timeout");
      if(busy) @(negedge clk);
    end
    if(!done || error || logical_cycle!=D || elapsed_cycles!=phys || stall_cycles!=stalled)
      $fatal(1,"Completion mismatch logical=%0d elapsed=%0d stalls=%0d",logical_cycle,elapsed_cycles,stall_cycles);
    $display("PASS rep=%0d R=%0d C=%0d M=%0d K=%0d N=%0d logical=%0d elapsed=%0d stalls=%0d",rep,R,C,M,K,N,logical_cycle,elapsed_cycles,stall_cycles);
    @(negedge clk);
  end
  // Rejection must not start any memory transaction.
  cmd_m=0; cmd_valid=1;
  @(posedge clk); #1;
  if(!done || !error || busy) $fatal(1,"Zero shape was not rejected");
  @(negedge clk); cmd_m=M; cmd_a_base=32'hffffffff;
  @(posedge clk); #1;
  if(M*K>1 && (!done || !error || busy)) $fatal(1,"Address overflow was not rejected");
  $finish;
end
endmodule
'''


def run_case(index, shape, stalls, seed, simulator="iverilog"):
    rows, cols, m, k, n = shape
    folder = RTL_ROOT / 'build' / f'case_{index:03}'
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'simulation.log').unlink(missing_ok=True)
    a_base, b_base, c_base = 13, 10000000, 20000000
    conf = scale_config()
    conf.valid_conf_flag = True
    conf.array_rows = rows
    conf.array_cols = cols
    amap = np.arange(m*k, dtype=np.int64).reshape(m,k) + a_base
    bmap = np.arange(k*n, dtype=np.int64).reshape(k,n) + b_base
    cmap = np.arange(m*n, dtype=np.int64).reshape(m,n) + c_base
    reference = systolic_compute_ws()
    reference.set_params(config_obj=conf, ifmap_op_mat=amap, filter_op_mat=bmap, ofmap_op_mat=cmap)
    ea, eb, ec = (v.astype(np.int64) for v in reference.get_demand_matrices())
    assert len(ea) == len(eb) == len(ec)
    rng = np.random.default_rng(seed)
    a = rng.integers(-128, 128, size=(m,k), dtype=np.int16).astype(np.int8)
    b = rng.integers(-128, 128, size=(k,n), dtype=np.int16).astype(np.int8)
    a.flat[0] = -128
    b.flat[0] = -128
    if a.size>1: a.flat[-1] = 127
    if b.size>1: b.flat[-1] = -127
    values = np.zeros_like(ec)
    per_fold = m + 2*rows + cols - 2
    row_folds = (k+rows-1)//rows
    cumulative = [a[:,:min((fr+1)*rows,k)].astype(np.int64) @ b[:min((fr+1)*rows,k)].astype(np.int64)
                  for fr in range(row_folds)]
    for t, requests in enumerate(ec):
        fold = t // per_fold
        partial_gemm = cumulative[fold % row_folds].astype(np.int32)
        for lane, addr in enumerate(requests):
            if addr != -1:
                mi, ni = divmod(int(addr-c_base), n)
                values[t,lane] = partial_gemm[mi,ni]
    packed_file(folder/'a.hex', a.reshape(-1,1), 8)
    packed_file(folder/'b.hex', b.reshape(-1,1), 8)
    for name, matrix in [('ea',ea), ('eb',eb), ('ec',ec), ('ev',values)]:
        packed_file(folder/f'{name}.hex', matrix)
    (folder/'tb.sv').write_text(bench(rows,cols,m,k,n,len(ea),a_base,b_base,c_base,seed,stalls))
    sources = sorted((RTL_ROOT/'rtl').glob('*.sv'))
    if simulator == 'iverilog':
        compile_command = ['iverilog','-g2012','-s','tb','-o',str(folder/'sim.vvp'),*[str(s) for s in sources],str(folder/'tb.sv')]
        run_command = ['vvp',str(folder/'sim.vvp')]
    elif simulator == 'verilator':
        compile_command = ['verilator','--binary','--timing','-Wno-fatal','--top-module','tb',
                           '--Mdir',str(folder/'obj_dir'),'-j','2','-o','sim_exec',
                           *[str(s) for s in sources],str(folder/'tb.sv')]
        run_command = [str(folder/'obj_dir'/'sim_exec')]
    else:
        raise ValueError(f'Unsupported simulator: {simulator}')
    compiled = subprocess.run(compile_command, capture_output=True, text=True)
    (folder/'compile.log').write_text(compiled.stdout + compiled.stderr)
    if compiled.returncode:
        raise RuntimeError(f'Compilation failed: see {folder}/compile.log')
    run = subprocess.run(run_command, cwd=folder, text=True, capture_output=True)
    (folder/'simulation.log').write_text(run.stdout + run.stderr)
    expected_markers = [f'PASS rep={rep} R={rows} C={cols} M={m} K={k} N={n} logical={len(ea)} ' for rep in range(2)]
    normal_finish = '$finish called at' in run.stdout or 'Verilog $finish' in run.stdout
    if run.returncode or not all(marker in run.stdout for marker in expected_markers) or not normal_finish:
        print(run.stdout, run.stderr, file=sys.stderr)
        raise RuntimeError(f'Case {index}: {shape}, stalls={stalls} failed; inspect {folder}')
    print(run.stdout.strip(), flush=True)
    return dict(array=[rows,cols], gemm=[m,k,n], demand_cycles=len(ea), stalls_enabled=stalls, repeated_commands=2, seed=seed, simulator=simulator)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick',action='store_true',help='Run directed small/32x32 cases only')
    parser.add_argument('--seed',type=int,default=20261018)
    args = parser.parse_args()
    (RTL_ROOT/'build'/'validation.json').unlink(missing_ok=True)
    for tool in ('iverilog','vvp'):
        if not shutil.which(tool): raise SystemExit(f'{tool} required')
    shapes = [(1,1,1,1,1),(1,3,2,5,4),(3,1,4,7,2),(2,3,1,1,1),
              (4,4,7,9,10),(3,5,8,4,7),(5,2,1,11,5),(32,32,1,33,35),(32,32,4,65,31)]
    if not args.quick:
        rng = random.Random(args.seed)
        shapes += [(rng.randint(1,7),rng.randint(1,7),rng.randint(1,12),rng.randint(1,20),rng.randint(1,20)) for _ in range(15)]
    results=[]
    for index,shape in enumerate(shapes):
        for stalled in (False,True):
            results.append(run_case(len(results),shape,stalled,args.seed+len(results)))
    report = dict(reference_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  equivalence='dense WS compute-demand lanes and numeric signed INT8/INT32 GEMM; ready stalls are RTL extension',
                  cases=results, passed=len(results), commands=len(results)*2)
    output=RTL_ROOT/'build'/'validation.json'
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(f'PASS: {len(results)} cases / {len(results)*2} GEMM commands; report {output}')

if __name__=='__main__': main()
