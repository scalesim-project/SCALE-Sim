"""Independent executable contract for WS + banked SRAM + serial AXI DMA.

Demand rows are produced by upstream SCALE-Sim; products by NumPy. The model
uses bank occupancy latency and a queue, not RTL state or PE implementation.
This models the implemented hardware policy, NOT native CALC/USER memory policy.
"""
from collections import deque
import numpy as np
from scalesim.compute.systolic_compute_ws import systolic_compute_ws
from scalesim.scale_config import scale_config


def demands(a, b, rows, cols):
    m,k=a.shape; n=b.shape[1]
    cfg=scale_config();cfg.valid_conf_flag=True;cfg.array_rows=rows;cfg.array_cols=cols
    engine=systolic_compute_ws()
    engine.set_params(config_obj=cfg,ifmap_op_mat=np.arange(m*k).reshape(m,k),
                      filter_op_mat=np.arange(k*n).reshape(k,n),ofmap_op_mat=np.arange(m*n).reshape(m,n))
    ea,eb,ec=(v.astype(np.int64) for v in engine.get_demand_matrices())
    length=m+2*rows+cols-2; folds=(k+rows-1)//rows
    result=[];products={}
    for t,(aa,bb,cc) in enumerate(zip(ea,eb,ec)):
        fold=(t//length)%folds
        if fold not in products:
            prefix=min(k,(fold+1)*rows)
            products[fold]=(a[:,:prefix].astype(np.int64)@b[:prefix].astype(np.int64)).astype(np.int32).reshape(-1)
        values=[int(products[fold][addr])&0xffffffff if addr>=0 else 0 for addr in cc]
        counts=[]
        for addresses,banks in ((aa,rows),(bb,cols)):
            counts.append(max(np.bincount(addresses[addresses>=0]%banks,minlength=banks)))
        occupancy=int(max(counts));latency=occupancy+4 if occupancy else 1
        if fold and np.any(cc>=0):latency=max(latency,2)
        trace={}
        for name,addresses in [('a',aa),('b',bb),('c',cc)]:
            trace[f'trace_{name}_valid']=sum((int(x)>=0)<<i for i,x in enumerate(addresses))
            for i,x in enumerate(addresses):trace[f'trace_{name}_addr[{i}]']=max(0,int(x))
        trace['trace_c_final']=int(fold==folds-1)
        for i,x in enumerate(values):trace[f'trace_c_data[{i}]']=x
        packet=[(int(addr),value) for addr,value in zip(cc,values) if addr>=0]
        result.append(dict(trace=trace,latency=latency,packet=packet,final=bool(trace['trace_c_final'])))
    return result


class Engine:
    def __init__(self,trace):
        self.trace=trace;self.index=0;self.elapsed=0;self.stalls=0
        self.timer=trace[0]['latency'];self.queue=deque();self.last=None;self.wrapper_done=None
    def fire(self,pop):
        return self.index<len(self.trace) and self.timer==1 and (not self.trace[self.index]['packet'] or len(self.queue)<2 or pop)
    def step(self,pop,cycle):
        fire=self.fire(pop)
        if pop:self.queue.popleft()
        if self.index<len(self.trace):
            self.elapsed+=1
            if fire:
                item=self.trace[self.index]
                if item['packet']:self.queue.append((item['final'],item['packet']))
                self.index+=1
                if self.index==len(self.trace):self.last=cycle
                else:self.timer=self.trace[self.index]['latency']
            else:
                self.stalls+=1
                if self.timer>1:self.timer-=1
        if self.last is not None and cycle>self.last and not self.queue and self.wrapper_done is None:
            self.wrapper_done=cycle


class Reference:
    def __init__(self,rows,cols,workloads):
        self.rows=rows;self.cols=cols;self.workloads=workloads
        self.engine=None;self.finished=False;self.total=0;self.operands=0;self.results=0
        self.done=0;self.command=None;self.program=None;self.signals={};self.start=False
    def program_for(self,cmd):
        # Each yield is one clock interval. Incoming values are sampled at its edge.
        for address,size in ((cmd['cmd_a_addr'],cmd['cmd_m']*cmd['cmd_k']),
                             (cmd['cmd_b_addr'],cmd['cmd_k']*cmd['cmd_n'])):
            remaining=size;skip=address%32;address-=skip;beats=(skip+size+31)//32
            while beats:
                burst=min(beats,16,(4096-address%4096)//32)
                while True:
                    inputs=yield dict(m_axi_arvalid=1,m_axi_araddr=address,m_axi_arlen=burst-1,m_axi_arsize=5,m_axi_arburst=1)
                    if inputs['m_axi_arready']:break
                for _ in range(burst):
                    while True:
                        inputs=yield dict(m_axi_rready=1)
                        if inputs['m_axi_rvalid']:break
                    take=min(32-skip,remaining)
                    for _ in range(take):
                        yield {}
                        self.operands+=1
                    remaining-=take;skip=0;address+=32;beats-=1
        yield {} # command acceptance by banked engine
        self.start=True
        while True:
            queue=self.engine.queue if self.engine else ()
            if queue:
                final,packet=queue[0]
                yield {'_pop':True}
                if not final:continue
                for logical,data in packet:
                    yield {} # select a C word
                    address=cmd['cmd_c_addr']+logical*4;offset=address%32
                    aw=w=False
                    while not (aw and w):
                        inputs=yield dict(m_axi_awvalid=int(not aw),m_axi_awaddr=address-offset,m_axi_awlen=0,
                                          m_axi_awsize=5,m_axi_awburst=1,m_axi_wvalid=int(not w),
                                          m_axi_wdata=data<<(offset*8),m_axi_wstrb=15<<offset,m_axi_wlast=1)
                        aw |= bool(inputs['m_axi_awready']);w |= bool(inputs['m_axi_wready'])
                    while True:
                        inputs=yield dict(m_axi_bready=1)
                        if inputs['m_axi_bvalid']:break
                    self.results+=4
                yield {} # empty packet selection
            elif self.finished:
                yield {'_complete':True}
                return
            else:yield {}
    def eval(self,inputs,clock=0):
        pop=bool(self.signals.get('_pop'))
        out=dict(cmd_ready=int(self.program is None),busy=int(self.program is not None),done=self.done,error=0,error_code=0,
                 logical_cycle=self.engine.index if self.engine else 0,core_elapsed_cycles=self.engine.elapsed if self.engine else 0,
                 core_stall_cycles=self.engine.stalls if self.engine else 0,total_cycles=self.total,operand_bytes=self.operands,
                 result_bytes=self.results,cycle_fire=int(self.engine.fire(pop)) if self.engine else 0)
        for name in ['m_axi_arvalid','m_axi_rready','m_axi_awvalid','m_axi_wvalid','m_axi_bready']:out[name]=0
        out.update(self.signals)
        if out['cycle_fire']:out.update(self.engine.trace[self.engine.index]['trace'])
        return out
    def edge(self,inputs,cycle):
        self.done=0
        if self.engine:
            # DMA sees the wrapper's completion on the following edge; its RUN
            # decision uses the previously latched finished flag.
            completion=self.engine.wrapper_done is not None and cycle>self.engine.wrapper_done
            self.engine.step(bool(self.signals.get('_pop')),cycle)
        else:completion=False
        if self.program is not None:
            self.total+=1
            try:self.signals=self.program.send(inputs)
            except StopIteration:self.program=None;self.signals={};self.done=1
        elif inputs.get('cmd_valid'):
            cmd=dict(inputs);key=(cmd['cmd_m'],cmd['cmd_k'],cmd['cmd_n'])
            self.command=cmd;self.trace=demands(*self.workloads[key],self.rows,self.cols)
            self.engine=None;self.finished=False;self.total=self.operands=self.results=0
            self.program=self.program_for(cmd);self.signals=next(self.program)
            completion=False
        if self.start:
            self.engine=Engine(self.trace);self.start=False
        if completion:self.finished=True
    def close(self):pass
