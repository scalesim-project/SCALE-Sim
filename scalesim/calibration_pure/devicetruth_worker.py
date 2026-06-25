import os,sys
os.environ["PJRT_DEVICE"]="TPU"; os.environ["XLA_USE_SPMD"]="0"
import torch, time
import torch_xla.core.xla_model as xm
from transformers import AutoModelForCausalLM
MODELS={"gpt2":"gpt2","qwen2.5-0.5b":"Qwen/Qwen2.5-0.5B","smollm2-135m":"HuggingFaceTB/SmolLM2-135M"}
name,seq,batch=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
mid=MODELS[name]
m=AutoModelForCausalLM.from_pretrained(mid,dtype=torch.bfloat16,attn_implementation="eager").eval()
vocab=m.config.vocab_size
if batch*seq*vocab*2>6e9: print(f"{name},{seq},{batch},{vocab},,,skip_oom"); sys.exit()
dev=xm.xla_device(); m=m.to(dev)
cm=torch.compile(m,backend="openxla")
ids=torch.arange(batch*seq,dtype=torch.long).remainder(vocab).reshape(batch,seq).to(dev)
def issue():
    with torch.no_grad(): cm(input_ids=ids,use_cache=False)
    xm.mark_step()
def full(): issue(); xm.wait_device_ops()
for _ in range(6): full()
def med(fn,K=20):
    ts=[]
    for _ in range(K): t0=time.perf_counter(); fn(); ts.append(time.perf_counter()-t0)
    ts.sort(); return ts[len(ts)//2]*1e3
tf=med(full)
ts=[]
for _ in range(20): t0=time.perf_counter(); issue(); ts.append(time.perf_counter()-t0); xm.wait_device_ops()
ts.sort(); ti=ts[len(ts)//2]*1e3
print(f"{name},{seq},{batch},{vocab},{tf:.4f},{max(tf-ti,0.0):.4f},ok")
