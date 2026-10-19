#!/usr/bin/env python3
"""Build and run whole-decode dependency-ordered Ramulator timing from real tokens.

Numerical execution is run_numerical.py. This command requires its successful
report and emits a finite-buffer, serial-load/compute/store accelerator model.
It is not stock SCALE-Sim memory policy or full-transformer RTL equivalence.
"""
import argparse,csv,hashlib,json,platform,subprocess,sys,time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT))
from workload import Graph
from config import Architecture
from experiments.gemma3_1b_int8.hbm2.f2_config import make_f2_config,RAMULATOR_PIN,VENDOR


def build():
    actual=subprocess.check_output(['git','-C',str(VENDOR),'rev-parse','HEAD'],text=True).strip()
    assert actual==RAMULATOR_PIN
    target=HERE/'build/runner';target.parent.mkdir(exist_ok=True)
    flags=['-arch',platform.machine()] if platform.system()=='Darwin' else []
    args=['clang++','-std=c++20','-O3',*flags,'-I'+str(ROOT),'-I'+str(VENDOR/'src'),'-I'+str(VENDOR/'ext/fmt/include'),
          str(HERE/'runner.cpp'),'-L'+str(VENDOR),'-lramulator','-Wl,-rpath,'+str(VENDOR),'-o',str(target)]
    subprocess.run(args,check=True);return target


def verify_trace(numerical,manifest,events):
    steps=numerical['decode_steps'];count=manifest['counts']
    assert count['gemm']==235*steps and count['rmsnorm']==157*steps
    assert count['gelu']==26*steps and count['residual']==52*steps and count['kv_write']==26*steps
    assert count['softmax']==26*steps and count['argmax']==steps
    actual=numerical['operator_counts']
    assert actual['gemm']==391*steps and actual['softmax']==104*steps and actual['rmsnorm']==157*steps
    assert actual['quantize']==count['quantize']+234*steps
    assert actual['dequantize']==count['dequantize']+156*steps
    for name in ('gelu','gating','residual','kv_write','argmax','embedding'):
        assert count[name]==actual[name],(name,count[name],actual[name])
    # Verify every matrix shape against the executed numerical trace, not just totals.
    grouped=defaultdict(list)
    for event in events:
        if event['kind']!='gemm':continue
        token=event['position']-numerical['prompt_tokens']
        name=event['name'].split('.h')[0] if '.qk.h' in event['name'] or '.av.h' in event['name'] else event['name']
        grouped[token,name].append(event)
    for op in manifest['ops']:
        if op['kind']!='gemm':continue
        real=grouped.pop((op['token'],op['name']))
        assert sum(e['m'] for e in real)==op['m']
        assert all(e['k']==op['k'] and e['n']==op['n'] for e in real)
    assert not grouped
    return dict(status='passed',real_checkpoint=numerical['checkpoint_sha256'],
                mapping='Four QK/AV heads fused as M=4; identical Q/K/V and gate/up input quantizations shared. Explicit layouts/dispatch/position/EOS work added.',
                numerical_gemm_calls=actual['gemm'],timing_gemm_calls=count['gemm'])


def summarize(folder,manifest,numerical,architecture,profile,validation):
    hz=architecture.core_hz
    assert manifest['configuration']==architecture.to_dict()
    with (folder/'timing.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==len(manifest['ops']), (len(rows),len(manifest['ops']))
    tokens=defaultdict(lambda:dict(total_cycles=0,compute_cycles=0,reads=0,writes=0,retries=0,stages=defaultdict(int)))
    end=0
    for row,op in zip(rows,manifest['ops']):
        assert int(row['op'])==op['id'] and row['kind']==op['kind'] and int(row['token'])==op['token']
        assert int(row['start_core'])==end;end=int(row['end_core'])
        duration=end-int(row['start_core']);assert duration>=op['compute_cycles']
        assert int(row['commands'])==op['commands']
        t=tokens[op['token']];t['total_cycles']+=duration;t['compute_cycles']+=op['compute_cycles']
        for field in ('reads','writes','retries'):t[field]+=int(row[field])
        t['stages'][op['kind']]+=duration
        row.update(name=op['name'],compute_cycles=op['compute_cycles'],memory_wait_cycles=duration-op['compute_cycles'])
    result=[]
    for index,t in tokens.items():
        result.append(dict(index=index,context=manifest['first_context']+index,**t,
                           milliseconds=t['total_cycles']/hz*1000,tps=hz/t['total_cycles'],
                           memory_wait_cycles=t['total_cycles']-t['compute_cycles'],
                           dram_transaction_bytes=(t['reads']+t['writes'])*32,
                           input_token_id=numerical['decode_input_token_ids'][index],output_token_id=numerical['decode_output_token_ids'][index]))
    sources=[HERE/'runner.cpp',HERE/'workload.py',HERE/'config.py',Path(__file__),ROOT/'src_rtl/cosim/native_bridge.cpp']
    report=dict(status='passed',classification='Real numerical W8A8+KV8 decode paired with dependency-ordered native Ramulator2 memory timing and explicit compute models',
                assumed_core_clock_hz=hz,decode_steps=len(tokens),mean_decode_tps=len(tokens)*hz/end,
                mean_decode_ms=end/len(tokens)/hz*1000,tokens=result,total_cycles=end,operator_coverage=validation,
                numerical_report=str(folder/'numerical_result.json'),generated_text=numerical['generated_text'],
                numerical_quantization=numerical['quantization'],HBM=profile,
                configuration=architecture.to_dict(),architecture=architecture.describe(),
                limitations=['This is not full-transformer RTL execution, nor stock SCALE-Sim CALC/USER memory-policy equivalence.',
                    f'{hz/1e9:g} GHz array/vector frequency, configured array/SRAM resources and SFU costs are assumptions, not achieved FPGA timing or measured non-GEMM hardware.',
                    'Numerical execution and memory timing are separate runs of matched operator graphs, not a lockstep data-carrying RTL simulation.',
                    'HBM2 uses generic HBM12 scheduling, striped addresses and write recovery guard, not AMD proprietary F2 controller/PHY timing.',
                    'Aggregate HBM frontend differs from the existing single-master RTL DMA. No cycle-equivalence claim to that DMA.',
                    'Greedy device-side cached decode only. Host tokenization/PCIe and simulated prefill/TTFT are outside this measurement.',
                    'Quantization quality is checked on one prompt, not a model accuracy benchmark. No trained quantization calibration.',
                    'All activations remain FP32 outside INT8 GEMMs; this model does not assume free non-GEMM operations.'],
                source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                manifest_sha256=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest(),
                command_sha256=hashlib.sha256((folder/'commands.txt').read_bytes()).hexdigest(),
                ramulator_pin=RAMULATOR_PIN,
                hbm_config_sha256=hashlib.sha256((folder/'hbm2.json').read_bytes()).hexdigest(),
                native_binary_sha256=hashlib.sha256((HERE/'build/runner').read_bytes()).hexdigest(),
                native_library_sha256=hashlib.sha256((VENDOR/('libramulator.dylib' if platform.system()=='Darwin' else 'libramulator.so')).read_bytes()).hexdigest(),
                numerical_trace_sha256=hashlib.sha256((folder/'numerical_operator_trace.json').read_bytes()).hexdigest())
    (folder/'operations.json').write_text(json.dumps(rows,indent=2))
    (folder/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='passed',mean_decode_tps=report['mean_decode_tps'],mean_decode_ms=report['mean_decode_ms'],tokens=len(tokens),result=str(folder/'result.json')),indent=2))
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--numerical',type=Path,default=HERE/'build/numerical_512/result.json')
    p.add_argument('--config',type=Path,help='Architecture JSON (default: configs/fast_decode.json)')
    p.add_argument('--out',type=Path);p.add_argument('--core-hz',type=int,help='Override the configured assumed core clock')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--summarize-only',action='store_true');args=p.parse_args()
    if args.prepare_only and args.summarize_only:p.error('Choose either --prepare-only or --summarize-only')
    config_path=args.config or HERE/'configs/fast_decode.json'
    args.out=args.out or HERE/'build'/config_path.stem
    if args.summarize_only:
        execution=json.loads((args.out/'execution.json').read_text())
        assert execution['status']=='completed','Native simulation has not completed'
        architecture=Architecture(**execution['configuration'])
        if args.config:assert Architecture.read(args.config)==architecture,'Configuration differs from the completed run'
        if args.core_hz is not None:assert args.core_hz==architecture.core_hz,'Clock differs from the completed run'
        for name,digest in execution['inputs_sha256'].items():
            assert hashlib.sha256((args.out/name).read_bytes()).hexdigest()==digest,f'Run input changed: {name}'
        assert hashlib.sha256((args.out/'timing.csv').read_bytes()).hexdigest()==execution['timing_sha256'],'Timing output changed'
        args.numerical=args.out/'numerical_result.json'
    else:
        architecture=Architecture.read(config_path)
        if args.core_hz is not None:architecture=replace(architecture,core_hz=args.core_hz)
    args.out.mkdir(parents=True,exist_ok=True)
    numerical=json.loads(args.numerical.read_text());assert numerical['status']=='passed' and numerical['all_logits_finite']
    events=json.loads((args.out/'numerical_operator_trace.json' if args.summarize_only else args.numerical.parent/'operator_trace.json').read_text())
    profile_config,profile=make_f2_config(16)
    # This harness supplies dependency/finite-buffer timing absent from offline replay.
    profile['unmodeled']=['AMD proprietary controller, crossbar, physical mapping and CDC','ECC and PHY/calibration']
    profile['frontend']='Aggregate model: up to 16 native 32B admissions/DRAM tick, max 512 outstanding, blocking dependencies'
    validation={}
    if not args.summarize_only:
        (args.out/'result.json').unlink(missing_ok=True)
        (args.out/'numerical_result.json').write_text(json.dumps(numerical,indent=2))
        (args.out/'numerical_operator_trace.json').write_text(json.dumps(events,indent=2))
        (args.out/'configuration.json').write_text(json.dumps(architecture.to_dict(),indent=2)+'\n')
        with (args.out/'commands.txt').open('w') as stream:
            manifest=Graph(stream,steps=numerical['decode_steps'],context=numerical['first_decode_context'],
                           input_ids=numerical['decode_input_token_ids'],architecture=architecture).run()
        validation=verify_trace(numerical,manifest,events)
        (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2));(args.out/'hbm2.json').write_text(json.dumps(profile_config))
        execution=dict(status='prepared',configuration=architecture.to_dict(),inputs_sha256={
            name:hashlib.sha256((args.out/name).read_bytes()).hexdigest() for name in
            ('configuration.json','manifest.json','commands.txt','hbm2.json','numerical_result.json','numerical_operator_trace.json')})
        (args.out/'execution.json').write_text(json.dumps(execution,indent=2)+'\n')
        print('Prepared',len(manifest['ops']),'operators;',sum(o['commands'] for o in manifest['ops']),'commands',flush=True)
        if args.prepare_only:return
        binary=build();t=time.monotonic()
        with (args.out/'ramulator.log').open('w') as log:
            result=subprocess.run([str(binary),str(args.out/'hbm2.json'),str(args.out/'commands.txt'),str(args.out/'timing.csv'),
                                   str(architecture.core_hz),'16','10000000'],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Native simulation failed: {args.out}/ramulator.log')
        execution.update(status='completed',timing_sha256=hashlib.sha256((args.out/'timing.csv').read_bytes()).hexdigest())
        (args.out/'execution.json').write_text(json.dumps(execution,indent=2)+'\n')
        print(f'Native timing simulation took {time.monotonic()-t:.3f} host seconds',flush=True)
    else:
        manifest=json.loads((args.out/'manifest.json').read_text());validation=verify_trace(numerical,manifest,events)
    summarize(args.out,manifest,numerical,architecture,profile,validation)
if __name__=='__main__':main()
