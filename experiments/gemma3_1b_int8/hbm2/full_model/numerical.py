"""Numerical Gemma3 decoder: exact INT8 GEMMs, FP32 vector ops, INT8 KV.

Checkpoint tensors use Hugging Face names. No synthetic weights are substituted.
W8: symmetric per-output-channel. A8: symmetric per-row. KV8: per-token/head.
AV requantizes dequantized cached V per feature to make an INT8 dot product
with one output scale valid; omitting this step would be numerically incorrect.
"""
from pathlib import Path
import hashlib,json,math,time
import torch
from safetensors import safe_open


def quantize(x,dim=-1):
    maximum=x.abs().amax(dim=dim,keepdim=True)
    scale=torch.where(maximum>0,maximum/127,torch.ones_like(maximum))
    return torch.round(x/scale).clamp(-127,127).to(torch.int8),scale


def int_gemm(a,b):
    assert a.dtype==b.dtype==torch.int8 and a.ndim==b.ndim==2
    assert a.shape[1]==b.shape[0] and a.shape[1]*127*127<2**31
    return torch._int_mm(a.contiguous(),b.contiguous())


def rmsnorm(x,weight,eps):
    return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+eps)*(1+weight)


def rope(x,positions,theta):
    d=x.shape[-1];inv=1.0/(theta**(torch.arange(0,d,2,dtype=torch.float32)/d))
    angle=positions.float()[:,None]*inv[None,:]
    angle=torch.cat((angle,angle),dim=-1)[:,None,:]
    rotated=torch.cat((-x[...,d//2:],x[...,:d//2]),dim=-1)
    return x*angle.cos()+rotated*angle.sin()


class Decoder:
    def __init__(self,config,weights,quantized=True,record=True):
        self.config=config;self.quantized=quantized;self.record=record;self.events=[]
        self.weights={};self.linear_weights={};self.cache={};self.position=0
        self.embed_key='model.embed_tokens.weight'
        for name,value in weights.items():
            value=value.detach().float().cpu()
            if quantized and value.ndim==2:
                self.linear_weights[name]=quantize(value)
            else:self.weights[name]=value
        self.layers=config['num_hidden_layers'];self.h=config['hidden_size'];self.d=config['head_dim']
        self.nq=config['num_attention_heads'];self.nkv=config['num_key_value_heads']
        assert self.nkv==1,'This executor currently targets Gemma3-1B GQA'
        assert config.get('attn_logit_softcapping') is None and config.get('final_logit_softcapping') is None
        assert config.get('hidden_activation','gelu_pytorch_tanh')=='gelu_pytorch_tanh'
    @classmethod
    def load(cls,path,quantized=True,record=True):
        path=Path(path);config=json.loads((path/'config.json').read_text())
        files=sorted(path.glob('*.safetensors'));assert files,'No safetensors checkpoint'
        # Yield one tensor at a time: BF16 checkpoint remains memory-mapped.
        class Tensors:
            def items(self):
                for file in files:
                    with safe_open(file,framework='pt',device='cpu') as f:
                        for key in f.keys():yield key,f.get_tensor(key)
        return cls(config,Tensors(),quantized,record)
    def event(self,name,kind,output,**details):
        if self.record:
            self.events.append(dict(name=name,kind=kind,position=self.position,shape=list(output.shape),dtype=str(output.dtype),**details))
        return output
    def norm(self,name,x,weight):
        return self.event(name,'rmsnorm',rmsnorm(x,self.weights[weight],self.config.get('rms_norm_eps',1e-6)))
    def linear(self,name,x,key):
        if self.quantized:
            a,sa=quantize(x);b,sb=self.linear_weights[key]
            self.event(name+'.input_quant','quantize',a)
            acc=int_gemm(a,b.T);self.event(name,'gemm',acc,m=x.shape[0],k=x.shape[1],n=b.shape[0])
            return self.event(name+'.dequant','dequantize',acc.float()*sa*sb.T)
        result=x@self.weights[key].T
        return self.event(name,'gemm',result,m=x.shape[0],k=x.shape[1],n=result.shape[1])
    def attention(self,p,q,k,v,positions,layer):
        length=q.shape[0];head_dim=self.d
        if self.quantized:
            ki,ks=quantize(k);vi,vs=quantize(v)
            self.event(p+'.key_quant','quantize',ki);self.event(p+'.value_quant','quantize',vi)
            old=self.cache.get(layer)
            current=(ki,ks,vi,vs)
            if old:current=tuple(torch.cat((a,b),0) for a,b in zip(old,current))
            self.cache[layer]=current
            ki,ks,vi,vs=current
            self.event(p+'.cache_update','kv_write',ki[-length:])
        else:
            old=self.cache.get(layer)
            if old:k=torch.cat((old[0],k),0);v=torch.cat((old[1],v),0)
            self.cache[layer]=(k,v)
        total=self.position+length
        layer_types=self.config.get('layer_types')
        local=(layer_types[layer]=='sliding_attention') if layer_types else (layer+1)%self.config.get('sliding_window_pattern',6)!=0
        # Decode crops old local keys. Prefill retains them until the causal/window mask.
        start=max(0,total-self.config['sliding_window']) if local and length==1 else 0
        key_positions=torch.arange(start,total)
        allowed=key_positions[None,:]<=positions[:,None]
        if local:allowed &= key_positions[None,:] > positions[:,None]-self.config['sliding_window']
        if self.quantized:
            ki,ks,vi,vs=(x[start:,0,:] for x in (ki,ks,vi,vs))
            # V scales vary along the reduction axis. Recover V and requantize
            # per output feature before a single INT8 AV matrix product.
            v_float=vi.float()*vs
            self.event(p+'.value_cache_dequant','dequantize',v_float)
            vq,vscale=quantize(v_float,dim=0)
            self.event(p+'.value_cache_requant','quantize',vq)
        else:k=k[start:,0,:];v=v[start:,0,:]
        outputs=[]
        for head in range(self.nq):
            query=q[:,head,:]
            if self.quantized:
                qi,qs=quantize(query);self.event(p+f'.query_quant.h{head}','quantize',qi)
                score_acc=int_gemm(qi,ki.T)
                self.event(p+f'.qk.h{head}','gemm',score_acc,m=length,k=head_dim,n=total-start)
                scores=score_acc.float()*qs*ks.T
                self.event(p+f'.qk.dequant.h{head}','dequantize',scores)
            else:scores=query@k.T
            scores=scores*self.config.get('query_pre_attn_scalar',head_dim)**-0.5
            scores=scores.masked_fill(~allowed,float('-inf'))
            self.event(p+f'.mask.h{head}','attention_mask',scores)
            probs=torch.softmax(scores,dim=-1);self.event(p+f'.softmax.h{head}','softmax',probs)
            if self.quantized:
                pi,ps=quantize(probs);self.event(p+f'.probability_quant.h{head}','quantize',pi)
                acc=int_gemm(pi,vq);self.event(p+f'.av.h{head}','gemm',acc,m=length,k=total-start,n=head_dim)
                output=acc.float()*ps*vscale;self.event(p+f'.av.dequant.h{head}','dequantize',output)
            else:output=probs@v
            outputs.append(output)
        return torch.stack(outputs,dim=1).reshape(length,-1)
    @torch.inference_mode()
    def forward(self,tokens):
        ids=torch.tensor(tokens,dtype=torch.long);length=len(tokens);positions=torch.arange(self.position,self.position+length)
        if self.quantized:
            q,s=self.linear_weights[self.embed_key];x=q[ids].float()*s[ids]
        else:x=self.weights[self.embed_key][ids]
        x=x*math.sqrt(self.h);self.event('embedding_lookup_and_scale','embedding',x)
        for layer in range(self.layers):
            p=f'L{layer:02d}';prefix=f'model.layers.{layer}.';residual=x
            x=self.norm(p+'.input_norm',x,prefix+'input_layernorm.weight')
            q=self.linear(p+'.q',x,prefix+'self_attn.q_proj.weight').reshape(length,self.nq,self.d)
            k=self.linear(p+'.k',x,prefix+'self_attn.k_proj.weight').reshape(length,1,self.d)
            v=self.linear(p+'.v',x,prefix+'self_attn.v_proj.weight').reshape(length,1,self.d)
            q=self.norm(p+'.q_norm',q,prefix+'self_attn.q_norm.weight')
            k=self.norm(p+'.k_norm',k,prefix+'self_attn.k_norm.weight')
            types=self.config.get('layer_types');local=(types[layer]=='sliding_attention') if types else (layer+1)%self.config.get('sliding_window_pattern',6)!=0
            theta=self.config.get('rope_local_base_freq',10000) if local else self.config.get('rope_theta',1000000)
            # Transformers v5 stores these in rope_parameters.
            rp=self.config.get('rope_parameters') or {};kind='sliding_attention' if local else 'full_attention'
            theta=rp.get(kind,{}).get('rope_theta',theta)
            q=rope(q,positions,theta);k=rope(k,positions,theta)
            self.event(p+'.rope_q','rope_apply',q);self.event(p+'.rope_k','rope_apply',k)
            attn=self.attention(p,q,k,v,positions,layer)
            x=self.linear(p+'.o',attn,prefix+'self_attn.o_proj.weight')
            x=self.norm(p+'.post_attn_norm',x,prefix+'post_attention_layernorm.weight')
            x=self.event(p+'.attention_residual','residual',x+residual);residual=x
            x=self.norm(p+'.pre_ff_norm',x,prefix+'pre_feedforward_layernorm.weight')
            gate=self.linear(p+'.gate',x,prefix+'mlp.gate_proj.weight')
            up=self.linear(p+'.up',x,prefix+'mlp.up_proj.weight')
            gate=self.event(p+'.gelu','gelu',torch.nn.functional.gelu(gate,approximate='tanh'))
            gated=self.event(p+'.gating','gating',gate*up)
            x=self.linear(p+'.down',gated,prefix+'mlp.down_proj.weight')
            x=self.norm(p+'.post_ff_norm',x,prefix+'post_feedforward_layernorm.weight')
            x=self.event(p+'.ff_residual','residual',x+residual)
        x=self.norm('final_norm',x,'model.norm.weight')
        head='lm_head.weight' if ('lm_head.weight' in self.linear_weights or 'lm_head.weight' in self.weights) else self.embed_key
        logits=self.linear('lm_head',x[-1:],head)[0]
        self.position+=length
        assert torch.isfinite(logits).all()
        return logits
