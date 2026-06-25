#!/usr/bin/env python3
"""
Level-2 calibration data: batch latency-reduction ratio R for batched matmul.
R = einsum_time / (batch * single_matmul_time), i.e. how much cheaper one batched
dot_general is than running the `batch` per-head GEMMs separately. Level 1 (the
cycle->latency GEMM model) predicts the single per-head GEMM; this R is the
SEPARATE second-level factor applied on top. Sweeps batch x per-head shape.
Outputs batch_reduction.csv: b,M,N,K, einsum_us, single_us, b_single_us, R.
"""
import os, glob, gzip, json, shutil, csv, itertools
os.environ.setdefault("PJRT_DEVICE","TPU"); os.environ.setdefault("XLA_USE_SPMD","0")
import jax, jax.numpy as jnp
ITERS=15

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
def measure(fn,*args):
    c=jax.jit(fn).lower(*args).compile()
    for _ in range(6): c(*args).block_until_ready()
    d=f"/tmp/bsw/{abs(hash(tuple(a.shape for a in args)))%999999}"; shutil.rmtree(d,ignore_errors=True); os.makedirs(d,exist_ok=True)
    with jax.profiler.trace(d):
        for _ in range(ITERS): c(*args).block_until_ready()
    k=kernel_us(d); shutil.rmtree(d,ignore_errors=True); return k

BATCHES=[1,2,4,8,16,32,64]
SHAPES=[(128,128,64),(256,256,64),(512,512,64),(1024,1024,64),
        (128,128,128),(256,256,128),(512,512,128),
        (256,128,64),(512,128,128),(128,512,64)]   # +a few non-square (M!=N)
out=os.path.join(os.path.dirname(__file__),"batch_reduction.csv")
fh=open(out,"w",newline=""); w=csv.writer(fh)
w.writerow(["b","M","N","K","einsum_us","single_us","b_single_us","R"])
# cache single-matmul time per shape (independent of b)
single={}
for (M,N,K) in SHAPES:
    a=jnp.ones((M,K),jnp.bfloat16); b_=jnp.ones((K,N),jnp.bfloat16); a.block_until_ready(); b_.block_until_ready()
    single[(M,N,K)]=measure(lambda x,y:x@y, a,b_)
n=0
for (M,N,K) in SHAPES:
    s=single[(M,N,K)]
    for b in BATCHES:
        A=jnp.ones((b,M,K),jnp.bfloat16); B=jnp.ones((b,K,N),jnp.bfloat16); A.block_until_ready(); B.block_until_ready()
        e=measure(lambda x,y: jnp.einsum('bmk,bkn->bmn',x,y), A,B)
        bs=s*b; R=e/bs
        w.writerow([b,M,N,K,f"{e:.4f}",f"{s:.4f}",f"{bs:.4f}",f"{R:.5f}"]); n+=1
    fh.flush()
fh.close()
print(f"wrote {out}: {n} rows ({len(SHAPES)} shapes x {len(BATCHES)} batches)")
