"""Numerical fusion, packed OS tiling, scratch bounds and real DMA dependencies."""
import csv
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import torch
from config import Architecture
from workload import Graph
from numerical import quantize, int_gemm
from fused_value_cache import requantize_value_cache
from tiled_gemm_reference import execute_tiled
from scalesim.scale_config import scale_config
from scalesim.compute.systolic_compute_os import systolic_compute_os

HERE=Path(__file__).resolve().parent

class DecodeArchitectureChecks(unittest.TestCase):
    def test_fusion_bit_exact_with_ragged_context_and_scales(self):
        torch.set_num_threads(2);torch.manual_seed(101)
        for length in (1,31,83,84,511,512,513,515,2049):
            values=torch.randint(-127,128,(length,256),dtype=torch.int8)
            values[:,0]=0
            scales=torch.exp(torch.randn(length,1)*3)
            if length>1:scales[0]=0
            expected,expected_scale=quantize(values.float()*scales,dim=0)
            for chunk in (1,17,83,1024):
                actual,actual_scale=requantize_value_cache(values,scales,chunk)
                self.assertTrue(torch.equal(actual,expected),(length,chunk))
                self.assertTrue(torch.equal(actual_scale,expected_scale))
            probs=torch.randint(0,128,(4,length),dtype=torch.int8)
            self.assertTrue(torch.equal(int_gemm(probs,actual),int_gemm(probs,expected)))

    def test_os_partial_tiles_reconstruct_integer_gemm(self):
        for prefetch,balance in ((False,False),(False,True),(True,False),(True,True)):
            arch=Architecture(array_rows=1,array_cols=5,operand_staging_bytes=64,
                              vector_staging_bytes=256,dataflow='os',gemm_prefetch=prefetch,balance_k_tiles=balance)
            stream=io.StringIO();g=Graph(stream,steps=1,architecture=arch)
            m,k,n=2,17,13
            a=g.tensor('a',m*k,1);b=g.tensor('b',k*n,1);c=g.gemm('ragged',a,m,k,n,b)
            weights=np.arange(k*n,dtype=np.int64).reshape(k,n)-110
            inputs=np.arange(m*k,dtype=np.int64).reshape(m,k)-17
            packed=np.concatenate([weights[:,s:s+5].ravel() for s in range(0,n,5)])
            reads=[];writes=[]
            for line in stream.getvalue().splitlines():
                _,_,_,action,address,size,count,stride=line.split()
                address,size,count,stride=map(int,(address,size,count,stride))
                if (action=='R' or action.startswith('P')) and b.address<=address<b.address+b.size:
                    self.assertEqual(count,1);reads.append(packed[address-b.address:address-b.address+size])
                if action=='W':
                    for row in range(count):writes.extend(range((address-c.address+row*stride)//4,(address-c.address+row*stride+size)//4))
            result=np.zeros((m,n),dtype=np.int64);i=0
            for n0 in range(0,n,5):
                nt=min(5,n-n0)
                for k0,kt in arch.k_tiles(k):
                    result[:,n0:n0+nt]+=inputs[:,k0:k0+kt]@reads[i].reshape(kt,nt);i+=1
            np.testing.assert_array_equal(result,inputs@weights)
            self.assertEqual(sorted(writes),list(range(m*n)))

    def test_all_layouts_four_heads_and_partial_k_tiles(self):
        arch=Architecture(array_rows=1,array_cols=5,operand_staging_bytes=64,
                          vector_staging_bytes=8192,dataflow='os',gemm_prefetch=True,balance_k_tiles=True)
        torch.manual_seed(2026)
        for layout,m,k,n in [('packed',4,33,17),('key_transpose',4,256,19),('row_major',4,515,256)]:
            a=torch.randint(-127,128,(m,k),dtype=torch.int8)
            b=torch.randint(-127,128,(k,n),dtype=torch.int8)
            got=execute_tiled(a,b,arch,layout)
            # Independent INT64 arithmetic avoids using the tiled INT8 kernel
            # as its own oracle and catches accumulation/overflow errors.
            np.testing.assert_array_equal(got.numpy(),a.numpy().astype(np.int64)@b.numpy().astype(np.int64))

    def test_native_os_wavefront_produces_exact_integer_outputs(self):
        rng=np.random.default_rng(47)
        for m,k,n in [(1,576,257),(4,256,513),(4,515,256)]:
            a=rng.integers(-127,128,(m,k),dtype=np.int64)
            b=rng.integers(-127,128,(k,n),dtype=np.int64)
            expected=(a@b).ravel();seen=np.zeros(m*n,dtype=np.int32)
            cfg=scale_config();cfg.valid_conf_flag=True;cfg.array_rows=1;cfg.array_cols=256
            engine=systolic_compute_os();engine.set_params(config_obj=cfg,
                ifmap_op_mat=np.arange(m*k).reshape(m,k),filter_op_mat=np.arange(k*n).reshape(k,n),
                ofmap_op_mat=np.arange(m*n).reshape(m,n))
            ad,bd,cd=(x.astype(np.int64) for x in engine.get_demand_matrices())
            shifted=np.full(256,-1,dtype=np.int64);acc=np.zeros(256,dtype=np.int64)
            for time in range(len(ad)):
                shifted[1:]=shifted[:-1];shifted[0]=ad[time,0]
                valid=(shifted>=0)&(bd[time]>=0)
                acc[valid]+=a.ravel()[shifted[valid]]*b.ravel()[bd[time,valid]]
                output=cd[time]>=0
                np.testing.assert_array_equal(acc[output],expected[cd[time,output]])
                seen[cd[time,output]]+=1;acc[output]=0
            self.assertTrue(np.all(seen==1))

    def test_fused_scratch_is_bounded_at_long_context(self):
        for size in (8192,131072):
            arch=Architecture(fuse_value_cache=True,vector_staging_bytes=size)
            g=Graph(io.StringIO(),steps=1,architecture=arch)
            g.fused_value_cache('L',g.tensor('v',16385*256,1),g.tensor('s',16385),16385)
            self.assertLessEqual(g.op['scratch_live_bytes'],size)
            self.assertEqual(g.op['read_bytes'],2*16385*(256+4))
            self.assertEqual(g.op['write_bytes'],16385*256+256*4)

    def run_native(self,commands,success=True):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);inp=folder/'commands';out=folder/'timing.csv'
            inp.write_text(''.join(f'0 0 test {line}\n' for line in commands))
            result=subprocess.run([str(HERE/'build/runner'),str(HERE/'build/decode_100tps/hbm2.json'),
                                   str(inp),str(out),'1000000000','16','1000'],capture_output=True,text=True)
            if success:
                self.assertEqual(result.returncode,0,result.stderr)
                with out.open() as f:row=next(csv.DictReader(f))
                return {k:int(v) for k,v in row.items() if k!='kind'}
            self.assertNotEqual(result.returncode,0)
            return result.stderr

    def test_native_overlap_uses_real_completions(self):
        serial=self.run_native(['R 0 262144 1 0','C 10000 0 0 0','R 262144 262144 1 0','C 10000 0 0 0'])
        overlap=self.run_native(['P0 0 262144 1 0','J 0 0 0 0','P1 262144 262144 1 0',
                                 'C 10000 0 0 0','F 0 0 0 0','J 1 0 0 0','C 10000 0 0 0','F 1 0 0 0'])
        self.assertEqual(overlap['reads'],serial['reads'])
        self.assertEqual(overlap['reads'],16384)
        self.assertGreater(overlap['end_core'],20000)
        self.assertLess(overlap['end_core'],serial['end_core'])
        # Without useful compute, waiting must still pay full callback latency.
        waited=self.run_native(['P0 0 32 1 0','J 0 0 0 0','F 0 0 0 0'])
        cold=self.run_native(['R 0 32 1 0'])
        self.assertEqual(waited['end_core'],cold['end_core'])

    def test_recycled_banks_and_foreground_write_dependencies(self):
        commands=[]
        count=17
        commands.append('P0 0 4096 1 0')
        for i in range(count):
            commands.append(f'J {i%2} 0 0 0')
            if i+1<count:commands.append(f'P{(i+1)%2} {(i+1)*8192} 4096 1 0')
            commands.extend(['C 17 0 0 0',f'F {i%2} 0 0 0',f'W {1048576+i*32} 32 1 0'])
        result=self.run_native(commands)
        self.assertEqual(result['reads'],count*128)
        self.assertEqual(result['writes'],count)
        self.assertGreater(result['end_core'],count*17)
        # A completed but held bank selects the asynchronous foreground path.
        prefix=['P0 0 32 1 0','J 0 0 0 0']
        async_write=self.run_native(prefix+['W 8192 32 1 0','F 0 0 0 0'])
        serial_write=self.run_native(prefix+['F 0 0 0 0','W 8192 32 1 0'])
        self.assertEqual(async_write['end_core'],serial_write['end_core'])

    def test_native_rejects_unsafe_bank_lifetimes(self):
        for commands in (['P0 0 32 1 0','P0 64 32 1 0'],
                         ['P0 0 32 1 0','F 0 0 0 0'],['J 0 0 0 0'],
                         ['P0 0 32 1 0','J 0 0 0 0']):
            self.run_native(commands,False)

if __name__=='__main__':unittest.main()
