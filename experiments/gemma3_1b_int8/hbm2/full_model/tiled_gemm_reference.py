"""Data-carrying oracle for the addresses emitted by the OS workload generator.

This checks tiling/packing and partial sums separately from Ramulator timing.
It does not claim that the timing runner executes a hardware datapath.
"""
import io
import numpy as np
import torch
from workload import Graph


def execute_tiled(a,b,architecture,layout='packed'):
    assert a.dtype==b.dtype==torch.int8
    m,k=a.shape;kb,n=b.shape;assert kb==k and k*127*127<2**31
    stream=io.StringIO();graph=Graph(stream,steps=1,architecture=architecture)
    ta=graph.tensor('oracle_a',m*k,1);tb=graph.tensor('oracle_b',k*n,1)
    tc=graph.gemm('oracle',ta,m,k,n,tb,layout)
    source=b.numpy()
    if layout=='packed':
        storage=np.concatenate([source[:,start:start+architecture.array_cols].ravel()
                                for start in range(0,n,architecture.array_cols)])
    elif layout=='key_transpose':storage=source.T.copy().ravel()
    elif layout=='row_major':storage=source.copy().ravel()
    else:raise ValueError(layout)
    payloads=[];coverage=np.zeros(m*n,dtype=np.int32);activation_coverage=np.zeros(m*k,dtype=np.int32)
    for line in stream.getvalue().splitlines():
        _,_,_,action,address,size,count,stride=line.split()
        address,size,count,stride=map(int,(address,size,count,stride))
        if action in ('R','P0','P1') and tb.address<=address<tb.address+tb.size:
            offset=address-tb.address
            rows=[storage[offset+row*stride:offset+row*stride+size] for row in range(count)]
            payload=np.concatenate(rows);assert len(payload)==size*count
            payloads.append(payload)
        elif action=='R' and ta.address<=address<ta.address+ta.size:
            assert count==1;activation_coverage[address-ta.address:address-ta.address+size]+=1
        elif action=='W':
            for row in range(count):
                start=(address-tc.address+row*stride)//4
                assert address%4==0 and size%4==0
                coverage[start:start+size//4]+=1
    assert np.all(coverage==1) and np.all(activation_coverage==1)
    result=torch.zeros((m,n),dtype=torch.int32);index=0
    for n0 in range(0,n,architecture.array_cols):
        nt=min(architecture.array_cols,n-n0)
        for k0,kt in architecture.k_tiles(k):
            payload=payloads[index];index+=1
            tile=(payload.reshape(nt,kt).T.copy() if layout=='key_transpose'
                  else payload.reshape(kt,nt).copy())
            # This independent address oracle catches transposed, skipped,
            # duplicated, or mis-strided reads even if arithmetic is symmetric.
            np.testing.assert_array_equal(tile,source[k0:k0+kt,n0:n0+nt])
            result[:,n0:n0+nt]+=torch._int_mm(a[:,k0:k0+kt].contiguous(),torch.from_numpy(tile))
    assert index==len(payloads)
    return result
