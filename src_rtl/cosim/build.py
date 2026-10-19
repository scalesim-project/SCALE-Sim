"""Build a native shared library containing Verilated RTL and Ramulator2."""
import hashlib
import json
import platform
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
VENDOR=ROOT/'submodules/ramulator2'
PIN='c6c7a5de0cef35935d6fad7369e6271406798865'
INPUTS=['clk','rst','cmd_valid','cmd_m','cmd_k','cmd_n','cmd_a_addr','cmd_b_addr','cmd_c_addr','cmd_page',
        'm_axi_arready','m_axi_rvalid','m_axi_rlast','m_axi_rresp','m_axi_awready','m_axi_wready','m_axi_bvalid','m_axi_bresp']
OUTPUTS=['cmd_ready','busy','done','error','error_code','logical_cycle','core_elapsed_cycles','core_stall_cycles',
         'total_cycles','operand_bytes','result_bytes','cycle_fire','m_axi_araddr','m_axi_arlen','m_axi_arsize','m_axi_arburst',
         'm_axi_arvalid','m_axi_rready','m_axi_awaddr','m_axi_awlen','m_axi_awsize','m_axi_awburst','m_axi_awvalid',
         'm_axi_wstrb','m_axi_wlast','m_axi_wvalid','m_axi_bready','trace_a_valid','trace_b_valid','trace_c_valid','trace_c_final']

def ports(rows,cols):
    ins=INPUTS+[f'm_axi_rdata[{i}]' for i in range(8)]
    outs=OUTPUTS+[f'm_axi_wdata[{i}]' for i in range(8)]
    for name,n in [('trace_a_addr',rows),('trace_b_addr',cols),('trace_c_addr',cols),('trace_c_data',cols)]:
        outs += [f'{name}[{i}]' for i in range(n)]
    return ins,outs

def ensure(rows=4,cols=4):
    assert rows in (4,32) and cols in (4,32)
    pin=subprocess.check_output(['git','-C',str(VENDOR),'rev-parse','HEAD'],text=True).strip()
    if pin!=PIN: raise RuntimeError('Ramulator revision differs from validated pin')
    folder=ROOT/f'src_rtl/build/cosim/r{rows}c{cols}';folder.mkdir(parents=True,exist_ok=True)
    sources=sorted((ROOT/'src_rtl/rtl').glob('*.sv'))+[HERE/'native_bridge.cpp',Path(__file__)]
    version=subprocess.check_output(['verilator','--version'],text=True)
    library=VENDOR/('libramulator.dylib' if platform.system()=='Darwin' else 'libramulator.so')
    if not library.exists(): raise RuntimeError('Run experiments/gemma3_1b_int8/hbm2/setup_ramulator2.sh first')
    digest=hashlib.sha256((''.join(hashlib.sha256(p.read_bytes()).hexdigest() for p in sources)+version+str(library.stat().st_mtime_ns)).encode()).hexdigest()
    target=folder/('cosim.dylib' if platform.system()=='Darwin' else 'cosim.so')
    stamp=folder/'build.json'
    if target.exists() and stamp.exists() and json.loads(stamp.read_text())['sha256']==digest:return target
    header=(ROOT/'src_rtl/rtl/scalesim_ws_axi_top.sv').read_text().split(');',1)[0]
    header=header.replace('module scalesim_ws_axi_top','module scalesim_cosim_top')
    extra=''' ,output wire [ROWS-1:0] trace_a_valid,
 output wire [COLS-1:0] trace_b_valid,trace_c_valid,
 output wire [ROWS*32-1:0] trace_a_addr,
 output wire [COLS*32-1:0] trace_b_addr,trace_c_addr,trace_c_data,
 output wire trace_c_final
);
scalesim_ws_axi_top #(.ROWS(ROWS),.COLS(COLS),.MAX_M(MAX_M),.A_BANKS(A_BANKS),.B_BANKS(B_BANKS),
 .A_PAGE_WORDS(A_PAGE_WORDS),.B_PAGE_WORDS(B_PAGE_WORDS),.OUTPUT_DEPTH(OUTPUT_DEPTH),
 .AXI_DATA_WIDTH(AXI_DATA_WIDTH),.MAX_BURST_BEATS(MAX_BURST_BEATS),
 .WRITE_INTERMEDIATES(WRITE_INTERMEDIATES)) dut(.*);
assign trace_a_valid=dut.engine.core_a_valid; assign trace_b_valid=dut.engine.core_b_valid;
assign trace_c_valid=dut.engine.core_c_valid; assign trace_a_addr=dut.engine.core_a_addr;
assign trace_b_addr=dut.engine.core_b_addr; assign trace_c_addr=dut.engine.core_c_addr;
assign trace_c_data=dut.engine.core_c_data; assign trace_c_final=dut.engine.core_c_final;
endmodule
'''
    wrapper=folder/'scalesim_cosim_top.sv';wrapper.write_text(header+extra)
    ins,outs=ports(rows,cols)
    cpp='''#include <cstdint>
#include "Vscalesim_cosim_top.h"
#include "verilated.h"
struct RTL { VerilatedContext ctx; Vscalesim_cosim_top model{&ctx}; };
extern "C" void* rtl_create(){return new RTL;}
extern "C" void rtl_destroy(void* p){auto* r=static_cast<RTL*>(p);r->model.final();delete r;}
extern "C" void rtl_eval(void* p,const std::uint64_t* in,std::uint64_t* out){auto& m=static_cast<RTL*>(p)->model;
'''
    cpp+='\n'.join(f'm.{name}=in[{i}];' for i,name in enumerate(ins))+'\nm.eval();\n'
    cpp+='\n'.join(f'out[{i}]=m.{name};' for i,name in enumerate(outs))+'\n}\n'
    binding=folder/'rtl_binding.cpp';binding.write_text(cpp)
    obj=folder/'obj'
    flags=['-fPIC','-DVL_TIME_CONTEXT']+(['-arch',platform.machine()] if platform.system()=='Darwin' else [])
    commands=[['verilator','--cc','-Wall','--top-module','scalesim_cosim_top','-Mdir',str(obj),
               f'-GROWS={rows}',f'-GCOLS={cols}','-GAXI_DATA_WIDTH=256','-CFLAGS',' '.join(flags),
               *map(str,sorted((ROOT/'src_rtl/rtl').glob('*.sv'))),str(wrapper)],
              ['make','-C',str(obj),'-f','Vscalesim_cosim_top.mk','-j2','CXX=clang++','Vscalesim_cosim_top__ALL.a']]
    verilator_root=re.search(r'VERILATOR_ROOT\s*=\s*(\S+)',subprocess.check_output(['verilator','-V'],text=True)).group(1)
    inc=Path(verilator_root)/'include'
    commands.append(['clang++','-std=c++20','-O2',*flags,'-dynamiclib' if platform.system()=='Darwin' else '-shared',
        str(binding),str(HERE/'native_bridge.cpp'),str(obj/'Vscalesim_cosim_top__ALL.a'),str(inc/'verilated.cpp'),
        str(inc/'verilated_threads.cpp'),'-I'+str(obj),'-I'+str(inc),'-I'+str(VENDOR/'src'),'-I'+str(VENDOR/'ext/fmt/include'),
        '-L'+str(VENDOR),'-lramulator','-Wl,-rpath,'+str(VENDOR),'-pthread','-o',str(target)])
    with (folder/'build.log').open('w') as log:
        for command in commands:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'Build failed; see {folder}/build.log')
    stamp.write_text(json.dumps(dict(sha256=digest,ramulator_pin=pin,verilator=version,rows=rows,cols=cols),indent=2))
    return target

if __name__=='__main__':print(ensure())
