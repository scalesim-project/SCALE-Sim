#!/usr/bin/env python3
"""
How does a batched matmul (multi-head attention) actually cost on TPU?
Compare, via xprof FUSION kernel time, three ways to view a (batch,M,N,K) batched
matmul [b,M,K]@[b,K,N]->[b,M,N]:

  einsum   : one batched dot_general (what the model really runs)
  b*single : batch x the time of ONE M,N,K matmul (sequential per-head view)
  dense    : a single (b*M)x(b*N)xK matmul  (the converter's WRONG flatten)

Tells us the right cycle mapping for batched dot_general:
  is einsum ~= b*single (batch the count), or ~= dense (flatten -> 12x overcount)?
"""
import os, glob, gzip, json, time, shutil
os.environ.setdefault("PJRT_DEVICE","TPU"); os.environ.setdefault("XLA_USE_SPMD","0")
import jax, jax.numpy as jnp
import torch  # noqa  (unused; keep env parity)
ITERS=20

def dev_pid(ev):
    for e in ev:
        if e.get("ph")=="M" and e.get("name")=="process_name" and "device:TPU:0" in e.get("args",{}).get("name",""):
            return e["pid"]
def kernel_us(folder):
    fs=[]
    for r,_,f in os.walk(folder): fs+=[os.path.join(r,x) for x in f if x.endswith(".trace.json.gz")]
    if not fs: return None
    ev=json.load(gzip.open(max(fs,key=os.path.getmtime),"rt"))["traceEvents"]
    pid=dev_pid(ev); inner=0.0
    for e in ev:
        if e.get("pid")==pid and e.get("ph")=="X" and "dur" in e and not e["name"].startswith("jit"):
            inner+=e["dur"]
    return inner/ITERS

def measure(fn, *args):
    c=jax.jit(fn).lower(*args).compile()
    for _ in range(6): c(*args).block_until_ready()
    d=f"/tmp/bt/{abs(hash((args[0].shape,args[-1].shape)))%99999}"; shutil.rmtree(d,ignore_errors=True); os.makedirs(d,exist_ok=True)
    with jax.profiler.trace(d):
        for _ in range(ITERS): c(*args).block_until_ready()
    k=kernel_us(d); shutil.rmtree(d,ignore_errors=True); return k

def run(b,M,N,K):
    A=jnp.ones((b,M,K),jnp.bfloat16); B=jnp.ones((b,K,N),jnp.bfloat16)
    a1=jnp.ones((M,K),jnp.bfloat16); b1=jnp.ones((K,N),jnp.bfloat16)
    Ad=jnp.ones((b*M,K),jnp.bfloat16); Bd=jnp.ones((K,b*N),jnp.bfloat16)
    for x in (A,B,a1,b1,Ad,Bd): x.block_until_ready()
    e=measure(lambda x,y: jnp.einsum('bmk,bkn->bmn',x,y), A,B)
    s=measure(lambda x,y: x@y, a1,b1)
    d=measure(lambda x,y: x@y, Ad,Bd)
    return e, s*b, d

SHAPES=[(1,128,128,64),(4,128,128,64),(12,128,128,64),(32,128,128,64),(64,128,128,64),
        (12,512,512,64),(12,256,256,128),(8,1024,1024,128),(32,512,512,128)]
print(f"{'b,M,N,K':>18}{'einsum_us':>11}{'b*single_us':>13}{'dense_us':>10}{'einsum/b*1':>11}{'dense/einsum':>13}")
for (b,M,N,K) in SHAPES:
    e,sb,d=run(b,M,N,K)
    print(f"{f'{b},{M},{N},{K}':>18}{e:>11.2f}{sb:>13.2f}{d:>10.2f}{e/sb:>11.2f}{d/e:>13.2f}")
