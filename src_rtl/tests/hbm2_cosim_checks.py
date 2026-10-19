#!/usr/bin/env python3
"""Closed-loop Verilated WS/AXI RTL vs independent Python contract, real HBM2.

Run from the repository: env -u PYTHONPATH .venv/bin/python src_rtl/tests/hbm2_cosim_checks.py
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src_rtl.cosim.build import ensure,PIN,VENDOR
from src_rtl.cosim.runtime import Library,RTL,HBM,U
from src_rtl.cosim.reference import Reference
from src_rtl.cosim.native_compare import compare as compare_native
from experiments.gemma3_1b_int8.hbm2.f2_config import make_f2_config


def normalized(out,rows,cols):
    names=['cmd_ready','busy','done','error','error_code','logical_cycle','core_elapsed_cycles','core_stall_cycles',
           'total_cycles','operand_bytes','result_bytes','cycle_fire',
           'm_axi_arvalid','m_axi_rready','m_axi_awvalid','m_axi_wvalid','m_axi_bready']
    result={name:out[name] for name in names}
    for valid,payload in [('m_axi_arvalid',['araddr','arlen','arsize','arburst']),
                          ('m_axi_awvalid',['awaddr','awlen','awsize','awburst']),
                          ('m_axi_wvalid',['wdata','wstrb','wlast'])]:
        if out[valid]:
            for field in payload:result['m_axi_'+field]=out['m_axi_'+field]
    if out['cycle_fire']:
        for name,lanes in [('a',rows),('b',cols),('c',cols)]:
            mask=out[f'trace_{name}_valid'];result[f'trace_{name}_valid']=mask
            for lane in range(lanes):
                if mask>>lane&1:
                    key=f'trace_{name}_addr[{lane}]';result[key]=out[key]
                    if name=='c':
                        key=f'trace_c_data[{lane}]';result[key]=out[key]
            if name=='c' and mask:result['trace_c_final']=out['trace_c_final']
    return result


def audit_transport(bridge):
    """Check causality/order from recorded events, independent of bridge.ready()."""
    accepts={e[1]:e for e in bridge.events if e[0]=='accept'}
    callbacks={e[1]:e for e in bridge.events if e[0]=='callback'}
    assert len(accepts)==len(callbacks)==len(bridge.requests)
    for ident,request in bridge.requests.items():
        accepted=accepts[ident][2];callback=callbacks[ident][2]
        assert accepted*bridge.tck_ps*bridge.core_hz > request['arrival']*10**12
        assert callback>=accepted # this DMA never forwards a read from an outstanding write
        assert callbacks[ident][4]==callback+(bridge.guard if request['write'] else bridge.extra_read_ticks)
    reads=deque(i for i,r in bridge.requests.items() if not r['write'])
    writes=deque(i for i,r in bridge.requests.items() if r['write'])
    aw=w=None;orders=dict(aw_first=0,w_first=0,together=0,read_after_write_bus_bytes=0)
    committed=set();write_address=write_strobe=None
    for event in bridge.bus:
        cycle,kind=event[:2]
        if kind=='R':
            ident=reads.popleft();assert event[2]==bridge.requests[ident]['address']
            orders['read_after_write_bus_bytes']+=sum(address in committed for address in range(event[2],event[2]+32))
            due=callbacks[ident][2]+bridge.extra_read_ticks
            assert cycle*10**12>=due*bridge.tck_ps*bridge.core_hz
        elif kind=='AW':aw=cycle;write_address=event[2]
        elif kind=='W':w=cycle;write_strobe=event[3]
        elif kind=='B':
            ident=writes.popleft();assert aw is not None and w is not None
            assert event[2]==bridge.requests[ident]['address']
            assert bridge.requests[ident]['arrival']==max(aw,w)
            due=callbacks[ident][2]+bridge.guard
            assert cycle*10**12>=due*bridge.tck_ps*bridge.core_hz
            orders['aw_first' if aw<w else 'w_first' if w<aw else 'together']+=1
            committed.update(write_address+b for b in range(32) if write_strobe>>b&1)
            aw=w=None
    assert not reads and not writes and aw is None and w is None
    return orders


def check_stability(held,out,inputs):
    """AXI valid and payload must survive backpressure on all five channels."""
    for channel,source,ready,payload in [
        ('AR',out,inputs['m_axi_arready'],['araddr','arlen','arsize','arburst']),
        ('AW',out,inputs['m_axi_awready'],['awaddr','awlen','awsize','awburst']),
        ('W',out,inputs['m_axi_wready'],['wdata','wstrb','wlast']),
        ('R',inputs,out['m_axi_rready'],['rdata','rlast','rresp']),
        ('B',inputs,out['m_axi_bready'],['bresp'])]:
        valid=source['m_axi_'+channel.lower()+'valid']
        data=tuple(source.get('m_axi_'+field,0) for field in payload)
        if channel in held:assert valid and data==held[channel],f'{channel} changed while stalled'
        if valid and not ready:held[channel]=data
        else:held.pop(channel,None)


def episode(lib,rows,cols,commands,arrays,config,hz,extra,kind,expected=None):
    memory={};expected_memory={};written=set()
    for cmd in commands:
        key=(cmd['cmd_m'],cmd['cmd_k'],cmd['cmd_n']);a,b=arrays[key]
        for base,values in [(cmd['cmd_a_addr'],a),(cmd['cmd_b_addr'],b)]:
            for i,value in enumerate(values.reshape(-1)):memory[base+i]=int(value)&255
        c=(a.astype(np.int64)@b.astype(np.int64)).astype(np.int32)
        for i,value in enumerate(c.reshape(-1)):
            for byte in range(4):
                address=cmd['cmd_c_addr']+i*4+byte
                memory[address]=0xa5;expected_memory[address]=(int(value)>>(8*byte))&255;written.add(address)
        for address in range(cmd['cmd_c_addr']-32,cmd['cmd_c_addr']):memory[address]=0x93
        for address in range(cmd['cmd_c_addr']+c.size*4,cmd['cmd_c_addr']+c.size*4+32):memory[address]=0x79
    # Poison every output after loading inputs, including an operand aliasing a
    # previous command's C. That input becomes valid only through live writes.
    memory.update({address:0xa5 for address in written})
    initial=dict(memory);bridge=HBM(lib,config,hz,memory,extra)
    client=RTL(lib,rows,cols) if kind=='rtl' else Reference(rows,cols,arrays)
    trace=[];completed=[];command=0;send=True;start=0;held={}
    try:
        for cycle in range(1000000):
            bridge.advance(cycle);inputs=bridge.inputs(cycle)
            if command<len(commands):inputs.update(commands[command],cmd_valid=int(send))
            out=client.eval(inputs);check_stability(held,out,inputs);entry=normalized(out,rows,cols)
            if expected is not None:
                if cycle>=len(expected):raise AssertionError(f'{kind} ran longer at {cycle}')
                want=expected[cycle]
                if entry!=want:
                    diffs={key:(want.get(key),entry.get(key)) for key in want.keys()|entry.keys() if want.get(key)!=entry.get(key)}
                    raise AssertionError(f'Cycle {cycle}, expected RTL / got {kind}: {diffs}')
            trace.append(entry)
            if out['done']:
                assert not out['error'];cmd=commands[command]
                assert out['operand_bytes']==cmd['cmd_k']*(cmd['cmd_m']+cmd['cmd_n'])
                assert out['result_bytes']==cmd['cmd_m']*cmd['cmd_n']*4
                completed.append(dict(shape=[cmd['cmd_m'],cmd['cmd_k'],cmd['cmd_n']],
                                      command_cycle=start,done_cycle=cycle,total_cycles=out['total_cycles'],
                                      logical_cycles=out['logical_cycle'],core_cycles=out['core_elapsed_cycles'],
                                      core_stalls=out['core_stall_cycles']))
                command+=1;send=True;start=cycle+1
                if command==len(commands):break
            elif send and out['cmd_ready']:send=False
            bridge.edge(out,inputs,cycle);client.edge(inputs,cycle)
        else:raise AssertionError('Timeout')
        if expected is not None:assert len(trace)==len(expected)
        assert bridge.written==written
        assert all(memory[address]==value for address,value in expected_memory.items())
        assert all(memory[address]==value for address,value in initial.items() if address not in written)
        assert not bridge.pending and not bridge.reads and bridge.write_ids is None
        assert all(r['ready'] is not None for r in bridge.requests.values())
        orders=audit_transport(bridge)
        assert bridge.capacity==len(json.loads(config.read_text())['memory_system']['controllers'])*1024**3
        summary=dict(axi_write_orders=orders,completion_causality_checked=True,channel_stability_checked=True,completed=completed,cycles_checked=len(trace),requests=len(bridge.requests),
                     reads=sum(not r['write'] for r in bridge.requests.values()),
                     writes=sum(r['write'] for r in bridge.requests.values()),admission_retries=bridge.retries,
                     reordered_read_callbacks=bridge.out_of_order,hbm_ticks=bridge.tick,write_guard_ticks=bridge.guard,
                     written_bytes=len(written),ar_bursts=sum(x[1]=='AR' for x in bridge.bus),
                     boundary_splits=sum(x[1]=='AR' and x[2]%4096+x[3]*32==4096 for x in bridge.bus),
                     trace_sha256=hashlib.sha256(json.dumps(trace,sort_keys=True).encode()).hexdigest())
        return trace,bridge.bus,bridge.events,summary
    finally:client.close();bridge.close()


def run_case(index,rows,cols,shapes,hz,channels,extra=0,pressure=False):
    folder=ROOT/f'src_rtl/build/hbm2_cosim_checks/case_{index}';folder.mkdir(parents=True,exist_ok=True)
    (folder/'results.json').unlink(missing_ok=True)
    config,_=make_f2_config(channels)
    if pressure:
        for ctrl in config['memory_system']['controllers']:ctrl['read_buffer_size']=1
    path=folder/'hbm2.json';path.write_text(json.dumps(config,indent=2)+'\n')
    rng=np.random.default_rng(892);arrays={};commands=[]
    for j,(m,k,n) in enumerate(shapes):
        arrays[m,k,n]=(rng.integers(-128,128,(m,k),dtype=np.int16).astype(np.int8),
                       rng.integers(-128,128,(k,n),dtype=np.int16).astype(np.int8))
        base=j*0x100000
        commands.append(dict(cmd_m=m,cmd_k=k,cmd_n=n,cmd_a_addr=base+4091,
                             cmd_b_addr=base+0x1ffe,cmd_c_addr=base+0x100fc,cmd_page=j%2))
        if index==4 and j==3:
            previous=commands[j-1];pa,pb=arrays[previous['cmd_m'],previous['cmd_k'],previous['cmd_n']]
            raw=(pa.astype(np.int64)@pb.astype(np.int64)).astype('<i4').tobytes()
            arrays[m,k,n]=(np.frombuffer(raw,dtype=np.int8,count=m*k).reshape(m,k).copy(),arrays[m,k,n][1])
            commands[-1]['cmd_a_addr']=previous['cmd_c_addr']
    lib=Library(ensure(rows,cols))
    rtl,bus,events,summary=episode(lib,rows,cols,commands,arrays,path,hz,extra,'rtl')
    ref,refbus,refevents,refsummary=episode(lib,rows,cols,commands,arrays,path,hz,extra,'reference',rtl)
    assert bus==refbus and events==refevents and summary==refsummary
    if index in (0,4):
        comparisons=[]
        for completed in summary['completed']:
            commits=[out['core_elapsed_cycles'] for out in rtl[completed['command_cycle']:completed['done_cycle']]
                     if out['cycle_fire']]
            comparisons.append(compare_native(completed['shape'],rows,cols,commits))
        (folder/'native_policy_comparison.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    summary.update(index=index,array=[rows,cols],assumed_core_clock_hz=hz,physical_hbm_channels=channels,
                   extra_read_ticks=extra,pressure=pressure,status='passed',
                   config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),ramulator_pin=PIN,
                   full_system_contract_equivalent=True,native_scalesim_memory_policy_equivalent=False,
                   scope='WS compute + banked SRAM + two-packet output queue + serial AXI DMA + causal Ramulator HBM2')
    (folder/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
    (folder/'bus.json').write_text(json.dumps(bus)+'\n');(folder/'hbm_events.json').write_text(json.dumps(events)+'\n')
    print(f'PASS case {index}: {summary["cycles_checked"]} cycles, {summary["requests"]} HBM requests, {summary["admission_retries"]} retries',flush=True)
    return summary


def bridge_checks():
    folder=ROOT/'src_rtl/build/hbm2_cosim_checks/bridge';folder.mkdir(parents=True,exist_ok=True)
    config,_=make_f2_config(16);path=folder/'valid.json';path.write_text(json.dumps(config))
    library=Library(ensure());checks=[]
    def rejects(name,action):
        try:action()
        except (ValueError,RuntimeError):checks.append(name)
        else:raise AssertionError(f'Expected rejection: {name}')
    rejects('zero core clock',lambda:HBM(library,path,0,{}))
    rejects('negative bridge delay',lambda:HBM(library,path,250000000,{},-1))
    bridge=HBM(library,path,1000000000,{})
    try:
        assert bridge.capacity==16*1024**3 and bridge.guard==24 and bridge.tck_ps==1111
        for cycle in (0,1,2,100,1111,2222):
            bridge.advance(cycle)
            info=(U*6)();library.lib.hbm_info(bridge.handle,info)
            assert bridge.tick==info[4]==cycle*1000//1111
        checks.append('integer core/DRAM phase and native tick agreement')
        rejects('time reversal',lambda:bridge.advance(2221))
        rejects('unaligned native address',lambda:library.check(library.lib.hbm_send(bridge.handle,0,1,0)))
        rejects('out of range native address',lambda:library.check(library.lib.hbm_send(bridge.handle,0,bridge.capacity,0)))
        ident=bridge.submit(bridge.capacity-32,False,2222)
        bridge.advance(3000);assert bridge.ready(ident)
        assert bridge.requests[ident]['accepted'] is not None
        checks.append('last legal 32-byte transaction in 16 GiB')
    finally:bridge.close()
    for name,mutate in [
        ('nonunit DRAM clock ratio',lambda c:c['memory_system'].update(clock_ratio=2)),
        ('heterogeneous channel timing',lambda c:c['memory_system']['controllers'][1]['dram']['timing'].__setitem__(0,1801)),
        ('non-HBM2 controller',lambda c:c['memory_system']['controllers'][0].update(impl='Generic'))]:
        altered=json.loads(path.read_text());mutate(altered)
        invalid=folder/(name.replace(' ','_')+'.json');invalid.write_text(json.dumps(altered))
        rejects(name,lambda:HBM(library,invalid,250000000,{}))
    report=dict(status='passed',checks=checks)
    (folder/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def provenance():
    sources=sorted((ROOT/'src_rtl/rtl').glob('*.sv'))+sorted((ROOT/'src_rtl/cosim').glob('*.py'))
    sources += [ROOT/'src_rtl/cosim/native_bridge.cpp',Path(__file__),ROOT/'experiments/gemma3_1b_int8/hbm2/f2_config.py']
    return dict(source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                ramulator_tracked_patch_sha256=hashlib.sha256(subprocess.check_output(['git','-C',str(VENDOR),'diff','--no-ext-diff','HEAD'])).hexdigest(),
                native_compute_sha256=hashlib.sha256((ROOT/'scalesim/compute/systolic_compute_ws.py').read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',type=int);args=parser.parse_args()
    configs=[(4,4,[(1,1,1),(4,17,7)],250000000,16,0,False),
             (4,4,[(4,17,7)],450000000,1,0,True),
             (4,4,[(4,17,7)],450000000,1,80,True),
             (4,4,[(2,65,11)],1000000000,16,0,False),
             (32,32,[(1,33,35),(1,1152,16),(8,65,33),(2,17,7)],250000000,16,0,False)]
    results=[]
    if args.case is None:
        (ROOT/'src_rtl/build/hbm2_cosim_checks/results.json').unlink(missing_ok=True)
        bridge_report=bridge_checks()
    for index,config in enumerate(configs):
        if args.case is None or args.case==index:results.append(run_case(index,*config))
    if args.case is None:
        assert results[4]['reordered_read_callbacks']>0
        assert results[4]['axi_write_orders']['read_after_write_bus_bytes']>0
        assert all(sum(r['axi_write_orders'][kind] for r in results)>0 for kind in ('aw_first','w_first','together'))
        assert results[1]['completed'][0]['total_cycles']<results[2]['completed'][0]['total_cycles']
        assert results[1]['admission_retries']>0
        out=ROOT/'src_rtl/build/hbm2_cosim_checks/results.json'
        out.write_text(json.dumps(dict(status='passed',cases=results,bridge_checks=bridge_report,provenance=provenance(),
            latency_feedback_checked=True,reference_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()),indent=2)+'\n')

if __name__=='__main__':main()
