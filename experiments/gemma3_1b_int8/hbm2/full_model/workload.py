"""Complete Gemma3-1B cached-decode operator graph for performance simulation.

This is a timing/traffic model, not numerical inference. Matrix schedules use
SCALE-Sim WS; vector/SFU timing is an explicit architecture assumption. All
inter-operator tensors live in HBM; array and finite staging sizes are configured.
"""
from dataclasses import dataclass
from collections import Counter
from pathlib import Path
import json, math, sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from experiments.gemma3_1b_int8.study import MODEL,demand_cycles
from config import Architecture

@dataclass
class Tensor:
    address:int
    size:int
    width:int
    name:str
    def view(self,start,size):
        assert 0<=start<=self.size and 0<=size<=self.size-start,(self.name,start,size,self.size)
        return Tensor(self.address+start*self.width,size,self.width,self.name)

class Graph:
    def __init__(self,stream,steps=3,context=513,input_ids=None,architecture=None):
        self.architecture=architecture or Architecture()
        self.stream=stream;self.steps=steps;self.context=context
        self.lanes=self.architecture.vector_lanes;self.sfu_lanes=self.architecture.sfu_lanes
        self.rows=self.architecture.array_rows;self.cols=self.architecture.array_cols
        self.operand_bytes=self.architecture.operand_staging_bytes
        self.vector_bytes=self.architecture.vector_staging_bytes
        self.input_ids=input_ids or list(range(1,steps+1))
        assert len(self.input_ids)==steps and all(0<=x<262144 for x in self.input_ids)
        self.next_address=0;self.ops=[];self.tensors=[];self.op=None;self.token=0
        self.weights={};self.scales={};self.norm_weights={};self.cache={};self.kv_scales={}
        h=1152;f=6912
        for layer in range(26):
            for name,k,n in [('q',h,1024),('k',h,256),('v',h,256),('o',1024,h),('gate',h,f),('up',h,f),('down',f,h)]:
                self.weights[layer,name]=self.tensor(f'L{layer}.{name}.packed_weights',k*n,1)
                self.scales[layer,name]=self.tensor(f'L{layer}.{name}.weight_scales',n,4)
            for name,n in [('input_norm',h),('post_attn_norm',h),('pre_ff_norm',h),('post_ff_norm',h),('q_norm',256),('k_norm',256)]:
                self.norm_weights[layer,name]=self.tensor(f'L{layer}.{name}.weight',n,4)
            for name in ('k','v'):
                self.cache[layer,name]=self.tensor(f'L{layer}.{name}.cache',(context+steps-1)*256,1)
                self.kv_scales[layer,name]=self.tensor(f'L{layer}.{name}.cache_scales',context+steps-1,4)
        self.weights['head']=self.tensor('tied_embedding_and_packed_lm_head',1152*262144,1)
        self.scales['head']=self.tensor('head.weight_scales',262144,4)
        self.norm_weights['final']=self.tensor('final_norm.weight',1152,4)
        self.inv_freq={name:self.tensor(name+'.inv_frequency',128,4) for name in ('local','global')}
        self.parameters_end=self.next_address
    def tensor(self,name,size,width=4):
        assert size>0
        address=(self.next_address+31)//32*32;self.next_address=address+size*width
        t=Tensor(address,size,width,name);self.tensors.append(t.__dict__);return t
    def begin(self,name,kind,**info):
        self.op=dict(id=len(self.ops),token=self.token,name=name,kind=kind,compute_cycles=0,read_bytes=0,write_bytes=0,commands=0,**info)
        self.ops.append(self.op);self.compute(4) # explicit descriptor dispatch cost
    def command(self,action,a,b=0,c=0,d=0):
        self.stream.write(f'{self.token} {self.op["id"]} {self.op["kind"]} {action} {a} {b} {c} {d}\n');self.op['commands']+=1
    def compute(self,cycles):
        assert isinstance(cycles,int) and cycles>=0
        self.command('C',cycles);self.op['compute_cycles']+=cycles
    def memory(self,t,write=False,count=1,stride=0):
        assert t.size*t.width*count<=max(self.operand_bytes,self.vector_bytes),'Transfer exceeds staging capacity'
        self.command('W' if write else 'R',t.address,t.size*t.width,count,stride)
        self.op['write_bytes' if write else 'read_bytes']+=t.size*t.width*count
    def bulk(self,t,write=False,capacity=None):
        chunk=(capacity or self.vector_bytes)//t.width
        for i in range(0,t.size,chunk):self.memory(t.view(i,min(chunk,t.size-i)),write)
    def vector(self,count,inputs,output=None,alu=1,sfu=0,latency=0,reduction=False):
        # Reserve space for all live input/output chunks in the vector buffer.
        widths=sum(t.width for t in inputs)+(output.width if output and not reduction else 0)
        chunk=max(1,self.vector_bytes//max(1,widths))
        for start in range(0,count,chunk):
            n=min(chunk,count-start)
            for t in inputs:
                self.memory(t if t.size==1 else t.view(start,n))
            self.compute(math.ceil(n/self.lanes)*alu + (math.ceil(n/self.sfu_lanes)*sfu+latency if sfu else 0) + (math.ceil(math.log2(self.lanes)) if reduction else 0))
            if output and not reduction:self.memory(output.view(start,n),True)
        if reduction:
            self.compute(math.ceil(math.log2(max(1,math.ceil(count/chunk))))+1)
            if output:self.memory(output,True)
    def norm(self,name,x,weight,groups=1):
        self.begin(name,'rmsnorm',elements=x.size,groups=groups)
        out=self.tensor(name+'.output',x.size);width=x.size//groups
        for row in range(groups):
            inp=x.view(row*width,width);inv=self.tensor(name+f'.inv{row}',1)
            self.vector(width,[inp],alu=2,reduction=True)
            self.compute(2+16) # mean, epsilon, reciprocal square root
            self.memory(inv,True)
            self.vector(width,[inp,weight,inv],out.view(row*width,width),alu=3)
        return out
    def quant(self,name,x,groups=1,fixed=None):
        self.begin(name,'quantize',elements=x.size,groups=groups,scale='fixed_per_layer' if fixed else 'dynamic_per_row')
        out=self.tensor(name+'.int8',x.size,1);scales=fixed or self.tensor(name+'.scales',groups)
        width=x.size//groups
        for row in range(groups):
            inp=x.view(row*width,width);scale=scales if fixed else scales.view(row,1)
            if not fixed:self.vector(width,[inp],alu=2,reduction=True)
            self.compute(8) # scalar scale calculation/reciprocal
            if not fixed:self.memory(scale,True)
            self.vector(width,[inp,scale],out.view(row*width,width),alu=3) # multiply, round, clamp
        return out,scales
    def dequant(self,name,x,rows,cols,sa,sb):
        self.begin(name,'dequantize',elements=x.size)
        out=self.tensor(name+'.fp32',x.size)
        for row in range(rows):
            a=sa if sa.size==1 else sa.view(row,1)
            self.vector(cols,[x.view(row*cols,cols),a,sb],out.view(row*cols,cols),alu=2)
        return out
    def gemm(self,name,a,m,k,n,b,layout='packed'):
        self.begin(name,'gemm',m=m,n=n,k=k,layout=layout,native_ws_cycles=demand_cycles(m,n,k,self.rows,self.cols))
        assert a.size==m*k and a.width==1 and a.size<=self.operand_bytes
        c=self.tensor(name+'.int32',m*n)
        self.bulk(a,capacity=self.operand_bytes)
        cycles=0
        for n0 in range(0,n,self.cols):
            nt=min(self.cols,n-n0)
            for k0 in range(0,k,self.architecture.k_chunk):
                kt=min(self.architecture.k_chunk,k-k0)
                if layout=='packed':self.memory(Tensor(b.address+n0*k+k0*nt,kt*nt,1,b.name))
                elif layout=='key_transpose':
                    assert k==256
                    if kt==k:self.memory(Tensor(b.address+n0*k,nt*k,1,b.name))
                    else:self.memory(Tensor(b.address+n0*k+k0,kt,1,b.name),count=nt,stride=k)
                    self.compute(math.ceil(nt*kt/self.lanes)) # explicit SRAM transpose/packing
                elif layout=='row_major':
                    self.memory(Tensor(b.address+k0*n+n0,nt,1,b.name),count=kt,stride=n)
                else:raise ValueError(layout)
                duration=demand_cycles(m,nt,kt,self.rows,self.cols);cycles+=duration;self.compute(duration)
            self.memory(Tensor(c.address+n0*4,nt,4,c.name),True,count=m,stride=n*4)
        assert cycles==self.op['native_ws_cycles']
        return c
    def linear(self,name,x,scale,layer,which,k,n):
        w=self.weights['head'] if which=='head' else self.weights[layer,which]
        ws=self.scales['head'] if which=='head' else self.scales[layer,which]
        acc=self.gemm(name,x,1,k,n,w)
        return self.dequant(name+'.dequant',acc,1,n,scale,ws)
    def element(self,name,kind,inputs,alu=1,sfu=0,latency=0):
        self.begin(name,kind,elements=inputs[0].size);out=self.tensor(name+'.output',inputs[0].size)
        self.vector(out.size,inputs,out,alu=alu,sfu=sfu,latency=latency);return out
    def softmax(self,name,x,length):
        self.begin(name,'softmax',elements=x.size,groups=4)
        out=self.tensor(name+'.probabilities',x.size)
        for row in range(4):
            inp=x.view(row*length,length);maximum=self.tensor(name+'.max',1)
            exp=self.tensor(name+'.exp',length);total=self.tensor(name+'.sum',1)
            self.vector(length,[inp],maximum,reduction=True)
            self.vector(length,[inp,maximum],exp,alu=1,sfu=1,latency=12)
            self.vector(length,[exp],reduction=True);self.compute(8);self.memory(total,True)
            self.vector(length,[exp,total],out.view(row*length,length),alu=1)
        return out
    def transpose(self,name,x,rows,cols):
        self.begin(name,'layout',elements=x.size)
        out=self.tensor(name+'.output',x.size,x.width)
        for i in range(0,rows,32):
            nr=min(32,rows-i)
            for j in range(0,cols,32):
                nc=min(32,cols-j)
                self.memory(Tensor(x.address+(i*cols+j)*x.width,nc,x.width,x.name),count=nr,stride=cols*x.width)
                self.compute(math.ceil(nr*nc/self.lanes))
                self.memory(Tensor(out.address+(j*rows+i)*x.width,nr,x.width,out.name),True,count=nc,stride=rows*x.width)
        return out
    def run(self):
        for token in range(self.steps):
            self.token=token;context=self.context+token
            self.begin('position_update','control');self.compute(1)
            rope={}
            for typ in ('local','global'):
                self.begin(typ+'.rope_angles','rope_generation',elements=256)
                rope[typ]=self.tensor(typ+'.cos_sin',512)
                self.bulk(self.inv_freq[typ]);self.compute(math.ceil(128/self.lanes))
                self.compute(math.ceil(256/self.sfu_lanes)+16);self.bulk(rope[typ],True)
            self.begin('embedding_lookup_and_scale','embedding',token_id=self.input_ids[token])
            x=self.tensor('embedded_token',1152)
            # LM-head uses column tiles: a tied embedding row is a strided gather.
            token_id=self.input_ids[token]
            tile_start=token_id//self.cols*self.cols;tile_width=min(self.cols,262144-tile_start)
            base=self.weights['head'].address+tile_start*1152+token_id-tile_start
            self.memory(Tensor(base,1,1,'embedding_gather'),count=1152,stride=tile_width)
            self.memory(self.scales['head'].view(token_id,1));self.compute(math.ceil(1152/self.lanes)*2);self.bulk(x,True)
            for layer in range(26):
                p=f'L{layer:02d}';residual=x
                x=self.norm(p+'.input_norm',x,self.norm_weights[layer,'input_norm'])
                aq,ascale=self.quant(p+'.input_quant',x)
                q=self.linear(p+'.q',aq,ascale,layer,'q',1152,1024)
                k=self.linear(p+'.k',aq,ascale,layer,'k',1152,256)
                v=self.linear(p+'.v',aq,ascale,layer,'v',1152,256)
                q=self.norm(p+'.q_norm',q,self.norm_weights[layer,'q_norm'],4)
                k=self.norm(p+'.k_norm',k,self.norm_weights[layer,'k_norm'])
                typ='global' if (layer+1)%6==0 else 'local'
                self.begin(p+'.rope_apply','rope_apply',elements=1280)
                self.bulk(rope[typ]);qr=self.tensor(p+'.rotated_q',1024);kr=self.tensor(p+'.rotated_k',256)
                self.vector(1024,[q,q],qr,alu=4);self.vector(256,[k,k],kr,alu=4)
                qi,qs=self.quant(p+'.query_quant',qr,4)
                ki,ks=self.quant(p+'.key_quant',kr)
                vi,vs=self.quant(p+'.value_quant',v)
                self.begin(p+'.cache_update','kv_write',elements=512)
                for name,t,scale in [('k',ki,ks),('v',vi,vs)]:
                    self.bulk(t);self.bulk(self.cache[layer,name].view((context-1)*256,256),True)
                    self.memory(scale);self.memory(self.kv_scales[layer,name].view(context-1,1),True)
                length=context if typ=='global' else min(512,context);offset=(context-length)*256
                kc=self.cache[layer,'k'].view(offset,length*256);vc=self.cache[layer,'v'].view(offset,length*256)
                ks=self.kv_scales[layer,'k'].view(context-length,length)
                vs=self.kv_scales[layer,'v'].view(context-length,length)
                self.begin(p+'.value_cache_dequant','dequantize',elements=length*256)
                vf=self.tensor(p+'.value_cache_fp32',length*256)
                for row in range(length):self.vector(256,[vc.view(row*256,256),vs.view(row,1)],vf.view(row*256,256),alu=1)
                vt=self.transpose(p+'.value_cache_transpose',vf,length,256)
                vqt,vs=self.quant(p+'.value_cache_requant',vt,256)
                vc=self.transpose(p+'.value_cache_untranspose',vqt,256,length)
                scores=self.gemm(p+'.qk',qi,4,256,length,kc,'key_transpose')
                scores=self.dequant(p+'.qk.dequant',scores,4,length,qs,ks)
                scores=self.element(p+'.scale_and_causal_window_mask','attention_mask',[scores],alu=3)
                probs=self.softmax(p+'.softmax',scores,length)
                pi,ps=self.quant(p+'.probability_quant',probs,4)
                attn=self.gemm(p+'.av',pi,4,length,256,vc,'row_major')
                attn=self.dequant(p+'.av.dequant',attn,4,256,ps,vs)
                # Four adjacent head rows already form the O-projection vector.
                oi,os=self.quant(p+'.attention_output_quant',attn)
                x=self.linear(p+'.o',oi,os,layer,'o',1024,1152)
                x=self.norm(p+'.post_attn_norm',x,self.norm_weights[layer,'post_attn_norm'])
                x=self.element(p+'.attention_residual','residual',[x,residual])
                residual=x;x=self.norm(p+'.pre_ff_norm',x,self.norm_weights[layer,'pre_ff_norm'])
                fi,fs=self.quant(p+'.ff_input_quant',x)
                gate=self.linear(p+'.gate',fi,fs,layer,'gate',1152,6912)
                up=self.linear(p+'.up',fi,fs,layer,'up',1152,6912)
                gate=self.element(p+'.gelu_tanh','gelu',[gate],alu=8,sfu=1,latency=12)
                gated=self.element(p+'.gate_multiply','gating',[gate,up])
                di,ds=self.quant(p+'.down_input_quant',gated)
                x=self.linear(p+'.down',di,ds,layer,'down',6912,1152)
                x=self.norm(p+'.post_ff_norm',x,self.norm_weights[layer,'post_ff_norm'])
                x=self.element(p+'.ff_residual','residual',[x,residual])
            x=self.norm('final_norm',x,self.norm_weights['final'])
            hi,hs=self.quant('head_input_quant',x)
            logits=self.linear('lm_head',hi,hs,None,'head',1152,262144)
            self.begin('greedy_token_selection','argmax',elements=262144)
            selected=self.tensor('selected_token',1)
            self.vector(logits.size,[logits],selected,reduction=True)
            self.begin('eos_check_and_token_bookkeeping','control');self.memory(selected);self.compute(4)
        assert self.next_address<16*1024**3
        return dict(model=MODEL,steps=self.steps,first_context=self.context,last_context=self.context+self.steps-1,
                    configuration=self.architecture.to_dict(),architecture=self.architecture.describe(),
                    vector_lanes=self.lanes,sfu_lanes=self.sfu_lanes,ops=self.ops,
                    counts=dict(Counter(o['kind'] for o in self.ops)),address_space_bytes=self.next_address,
                    preallocated_parameter_and_cache_bytes=self.parameters_end,tensors=self.tensors,
                    zero_cost_or_inapplicable={'reshape_and_head_concat':'Views over contiguous head rows; no copy.',
                        'GQA_repeat':'Four Q rows share K/V in fused M=4 GEMMs; no physical replication.',
                        'softcapping':'Both attention and final-logit softcaps are None in Gemma3_1B.',
                        'dropout':'Disabled during inference.', 'training_loss':'Not part of inference.',
                        'sampling_softmax_topk_topp_rng':'Greedy argmax selected; stochastic sampling is not enabled.',
                        'vision':'Gemma3-1B is text-only.', 'tokenizer_and_host_io':'Device-side cached decode; input IDs and KV are preloaded. Not included in device TPS.'})
