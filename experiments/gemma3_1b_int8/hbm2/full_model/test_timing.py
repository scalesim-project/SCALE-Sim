"""Check native WS tile schedules, numerical-to-timing coverage, and HBM timing."""
import argparse,csv,json,subprocess,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];sys.path.insert(0,str(ROOT))
from run import verify_trace
from config import Architecture
from scalesim.scale_config import scale_config
from scalesim.compute.systolic_compute_ws import systolic_compute_ws
from scalesim.compute.systolic_compute_os import systolic_compute_os
from experiments.gemma3_1b_int8.study import demand_cycles
from src_rtl.cosim.build import ensure
from src_rtl.cosim.runtime import Library,HBM


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,default=HERE/'build/fast_decode')
    args=parser.parse_args();folder=args.run_dir
    manifest=json.loads((folder/'manifest.json').read_text())
    architecture=Architecture(**manifest.get('configuration',{}))
    shapes=set()
    for op in manifest['ops']:
        if op['kind']!='gemm':continue
        for n0 in range(0,op['n'],architecture.array_cols):
            for k0,kt in architecture.k_tiles(op['k']):
                shapes.add((op['m'],kt,min(architecture.array_cols,op['n']-n0)))
    checks=[]
    for m,k,n in sorted(shapes):
        cfg=scale_config();cfg.valid_conf_flag=True;cfg.array_rows=architecture.array_rows;cfg.array_cols=architecture.array_cols
        engine=(systolic_compute_os() if architecture.dataflow=='os' else systolic_compute_ws());engine.set_params(config_obj=cfg,ifmap_op_mat=np.arange(m*k).reshape(m,k),
                     filter_op_mat=np.arange(k*n).reshape(k,n),ofmap_op_mat=np.arange(m*n).reshape(m,n))
        assert all(len(mat)==architecture.demand_cycles(m,n,k) for mat in engine.get_demand_matrices())
        checks.append(dict(check='native_'+architecture.dataflow.upper()+'_tile',m=m,k=k,n=n,array=[cfg.array_rows,cfg.array_cols]))
    validation=verify_trace(json.loads((folder/'numerical_result.json').read_text()),manifest,
                            json.loads((folder/'numerical_operator_trace.json').read_text()))
    checks.append(dict(check='executed_numerical_matrix_shapes_match_every_timing_GEMM',**validation))
    # Compare a cold read and write completion against the separately tested live
    # AXI/HBM transport. One transaction removes frontend bandwidth differences.
    library=Library(ensure());native=HERE/'build/runner';cfg=folder/'hbm2.json';hz=architecture.core_hz
    check_folder=folder/'checks';check_folder.mkdir(exist_ok=True)
    for kind in ('R','W'):
        bridge=HBM(library,cfg,hz,{})
        ident=bridge.submit(0,kind=='W',0)
        try:
            for cycle in range(10000):
                bridge.advance(cycle)
                if bridge.ready(ident):break
            else:raise AssertionError('Native callback timed out')
        finally:bridge.close()
        commands=check_folder/f'check_{kind}.txt';output=check_folder/f'check_{kind}.csv'
        commands.write_text(f'0 0 memory {kind} 0 32 1 0\n')
        subprocess.run([str(native),str(cfg),str(commands),str(output),str(hz),'16','100'],check=True,stdout=subprocess.DEVNULL)
        with output.open() as f:row=next(csv.DictReader(f))
        assert int(row['end_core'])==cycle,(kind,row,cycle)
        checks.append(dict(check='native_completion_matches_live_bridge',kind=kind,core_cycles=cycle))
    commands=check_folder/'check_compute.txt';output=check_folder/'check_compute.csv'
    commands.write_text('0 0 vector C 17 0 0 0\n0 0 vector C 23 0 0 0\n')
    subprocess.run([str(native),str(cfg),str(commands),str(output),str(hz),'16','100'],check=True,stdout=subprocess.DEVNULL)
    with output.open() as f:row=next(csv.DictReader(f))
    dram_ticks=40*10**12//(hz*1111)
    assert int(row['end_core'])==40 and int(row['end_dram'])==dram_ticks and int(row['reads'])==int(row['writes'])==0
    checks.append(dict(check='compute_time_preserves_native_DRAM_clock_phase',core_cycles=40,dram_ticks=dram_ticks))
    result=dict(status='passed',configuration=architecture.to_dict(),checks=checks)
    (folder/'timing_tests.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS',len(checks),'timing checks')
if __name__=='__main__':main()
