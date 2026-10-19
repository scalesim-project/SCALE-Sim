"""Live, causal AXI-to-HBM2 transport. Time conversion uses integer arithmetic."""
import ctypes as C
from collections import deque
from .build import ports

U=C.c_uint64

class Library:
    def __init__(self,path):
        self.lib=C.CDLL(str(path))
        for name,args,result in [
            ('bridge_error',[],C.c_char_p),('hbm_create',[C.c_char_p],C.c_void_p),
            ('hbm_destroy',[C.c_void_p],None),('hbm_info',[C.c_void_p,C.POINTER(U)],None),
            ('hbm_send',[C.c_void_p,U,U,C.c_int],C.c_int),('hbm_tick',[C.c_void_p],C.c_int),
            ('hbm_pop',[C.c_void_p,C.POINTER(U)],C.c_int),('rtl_create',[],C.c_void_p),
            ('rtl_destroy',[C.c_void_p],None),('rtl_eval',[C.c_void_p,C.POINTER(U),C.POINTER(U)],None)]:
            f=getattr(self.lib,name);f.argtypes=args;f.restype=result
    def check(self,value):
        if value<0:raise RuntimeError(self.lib.bridge_error().decode())
        return value

class RTL:
    def __init__(self,lib,rows,cols):
        self.lib=lib.lib;self.handle=self.lib.rtl_create()
        self.inputs,self.outputs=ports(rows,cols)
        self.i=(U*len(self.inputs))();self.o=(U*len(self.outputs))()
        for _ in range(2):self.eval({'rst':1},0);self.eval({'rst':1},1)
    def eval(self,inputs,clock=0):
        values=dict(inputs,clk=clock)
        data=values.get('m_axi_rdata',0)
        for j in range(8):values[f'm_axi_rdata[{j}]']=(data>>(32*j))&0xffffffff
        for j,name in enumerate(self.inputs):self.i[j]=values.get(name,0)
        self.lib.rtl_eval(self.handle,self.i,self.o)
        output=dict(zip(self.outputs,map(int,self.o)))
        output['m_axi_wdata']=sum(output[f'm_axi_wdata[{j}]']<<(32*j) for j in range(8))
        return output
    def edge(self,inputs,cycle):self.eval(inputs,1)
    def close(self):self.lib.rtl_destroy(self.handle)

class HBM:
    def __init__(self,library,config,core_hz,memory,extra_read_ticks=0):
        if not isinstance(core_hz,int) or core_hz<=0:raise ValueError('core_hz must be a positive integer')
        if not isinstance(extra_read_ticks,int) or extra_read_ticks<0:raise ValueError('extra_read_ticks must be nonnegative')
        self.last_cycle=-1
        self.library=library;self.lib=library.lib
        self.handle=self.lib.hbm_create(str(config).encode())
        if not self.handle:raise RuntimeError(self.lib.bridge_error().decode())
        info=(U*6)();self.lib.hbm_info(self.handle,info)
        self.tx,self.tck_ps,self.guard,self.capacity=map(int,info[:4])
        self.core_hz=core_hz;self.memory=memory;self.extra_read_ticks=extra_read_ticks
        self.tick=0;self.next_id=0;self.pending=deque();self.requests={};self.reads=deque()
        self.aw=None;self.w=None;self.write_ids=None;self.write_data=None;self.write_done=False
        self.events=[];self.bus=[];self.retries=0;self.out_of_order=0;self.last_read_callback=-1
        self.callback=(U*3)();self.written=set()
    def submit(self,address,write,cycle):
        if address%32 or not 0<=address<=self.capacity-32:raise ValueError('Bad native HBM transaction')
        ident=self.next_id;self.next_id+=1
        self.requests[ident]=dict(address=address,write=write,arrival=cycle,ready=None,accepted=None)
        self.pending.append(ident)
        return ident
    def advance(self,cycle):
        # DRAM ticks at/before this core edge finish before the AXI sampling edge.
        if not isinstance(cycle,int) or cycle<self.last_cycle:raise ValueError('Core time must be monotonic integer cycles')
        self.last_cycle=cycle
        target=cycle*10**12//(self.core_hz*self.tck_ps)
        while self.tick<target:
            # One admission per tick: explicit single-port bridge budget.
            if self.pending:
                ident=self.pending[0];request=self.requests[ident]
                accepted=self.library.check(self.lib.hbm_send(self.handle,ident,request['address'],request['write']))
                if accepted:
                    self.pending.popleft();request['accepted']=self.tick+1
                    self.events.append(('accept',ident,self.tick+1,request['address'],request['write']))
                else:self.retries+=1
            self.library.check(self.lib.hbm_tick(self.handle));self.tick+=1
            while self.lib.hbm_pop(self.handle,self.callback):
                ident,tick,command=map(int,self.callback);request=self.requests[ident]
                if request['ready'] is not None:raise AssertionError('Duplicate callback')
                if request['write'] and command==(1<<64)-1:
                    raise AssertionError('Coalesced write: issue-based per-request guard would be unsafe')
                request['ready']=tick+(self.guard if request['write'] else self.extra_read_ticks)
                self.events.append(('callback',ident,tick,command,request['ready']))
                if not request['write']:
                    if ident<self.last_read_callback:self.out_of_order+=1
                    self.last_read_callback=ident
        if self.write_ids is not None and not self.write_done and all(self.ready(i) for i in self.write_ids):
            address,data,strobe=self.write_data
            for byte in range(32):
                if strobe>>byte&1:self.memory[address+byte]=(data>>(8*byte))&255;self.written.add(address+byte)
            self.write_done=True
    def ready(self,ident):
        ready=self.requests[ident]['ready'];return ready is not None and ready<=self.tick
    def inputs(self,cycle):
        # Deterministic independent AW/W/AR stalls expose all channel orders.
        result=dict(m_axi_arready=int(not self.reads and cycle%5!=1),
                    m_axi_awready=int(self.aw is None and self.write_ids is None and cycle%7 not in (1,2)),
                    m_axi_wready=int(self.w is None and self.write_ids is None and cycle%11 not in (0,1,2,3)),
                    m_axi_bvalid=int(self.write_done),m_axi_bresp=0,m_axi_rvalid=0,m_axi_rlast=0,m_axi_rresp=0)
        if self.reads and self.ready(self.reads[0][0]):
            ident,address,last=self.reads[0]
            result.update(m_axi_rvalid=1,m_axi_rlast=int(last),
                          m_axi_rdata=sum(self.memory.get(address+b,0)<<(8*b) for b in range(32)))
        return result
    def edge(self,out,inputs,cycle):
        if out['m_axi_arvalid'] and inputs['m_axi_arready']:
            address,length=out['m_axi_araddr'],out['m_axi_arlen']+1
            assert address%32==0 and address%4096+length*32<=4096
            assert out['m_axi_arsize']==5 and out['m_axi_arburst']==1 and not self.reads
            self.bus.append((cycle,'AR',address,length))
            for i in range(length):
                addr=address+i*32;ident=self.submit(addr,False,cycle)
                self.reads.append((ident,addr,i==length-1))
        if inputs['m_axi_rvalid'] and out['m_axi_rready']:
            ident,address,last=self.reads.popleft();assert self.ready(ident)
            self.bus.append((cycle,'R',address,inputs['m_axi_rdata'],int(last)))
        if out['m_axi_awvalid'] and inputs['m_axi_awready']:
            assert self.aw is None and out['m_axi_awaddr']%32==0 and out['m_axi_awlen']==0
            assert out['m_axi_awsize']==5 and out['m_axi_awburst']==1
            self.aw=out['m_axi_awaddr'];self.bus.append((cycle,'AW',self.aw))
        if out['m_axi_wvalid'] and inputs['m_axi_wready']:
            assert self.w is None and out['m_axi_wlast']==1
            self.w=(out['m_axi_wdata'],out['m_axi_wstrb']);self.bus.append((cycle,'W',*self.w))
        if self.aw is not None and self.w is not None and self.write_ids is None:
            self.write_data=(self.aw,*self.w)
            self.write_ids=[self.submit(self.aw,True,cycle)] if self.w[1] else []
        if inputs['m_axi_bvalid'] and out['m_axi_bready']:
            assert self.write_done and all(self.ready(i) for i in self.write_ids)
            self.bus.append((cycle,'B',self.aw));self.aw=None;self.w=None
            self.write_ids=None;self.write_data=None;self.write_done=False
    def close(self):self.lib.hbm_destroy(self.handle)
