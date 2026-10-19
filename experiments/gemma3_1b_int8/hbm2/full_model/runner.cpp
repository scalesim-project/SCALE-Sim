// Full operator-graph timing driver. Every DRAM tick calls the real Ramulator.
#include "src_rtl/cosim/native_bridge.cpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>
#include <algorithm>

struct Simulator {
    HBM h;
    U core_hz, core=0, reads=0, writes=0, retries=0, last_write_ready=0, admitted=0;
    U pending=0, callbacks=0, injection;
    explicit Simulator(const char* config,U hz,U cap):h(config),core_hz(hz),injection(cap){}
    U core_to_tick(U c)const{return U((__uint128_t(c)*1000000000000ULL)/(core_hz*h.spec->get_timing_value("tCK_ps")));}
    U tick_to_core(U tick)const{return U((__uint128_t(tick)*core_hz*h.spec->get_timing_value("tCK_ps")+999999999999ULL)/1000000000000ULL);}
    void tick(){++h.tick;h.memory->tick();}
    void compute(U cycles){core+=cycles;U end=core_to_tick(core);while(h.tick<end)tick();}
    void transfer(U address,U bytes,U count,U stride,bool write){
        if(!bytes||!count)throw std::runtime_error("Empty transfer");
        // Requests for rows are admitted concurrently, up to the finite limit.
        // Row starts need not be aligned. Byte enables are assumed on writes.
        U row=0,cur=address&~U(31),end=(address+bytes+31)&~U(31);
        auto advance=[&](){cur+=32;if(cur==end){++row;if(row<count){U a=address+row*stride;cur=a&~U(31);end=(a+bytes+31)&~U(31);}}};
        while(row<count||pending||h.tick<last_write_ready){
            U budget=injection;
            while(row<count && pending<512 && budget){
                U a=cur;if(a>h.capacity-32)throw std::runtime_error("Address exceeds HBM capacity");
                Ramulator::Request request(long(a),write?Ramulator::Request::Type::Write:Ramulator::Request::Type::Read,0,
                    [this,write](Ramulator::Request& r){
                        if(!pending)throw std::runtime_error("Duplicate completion");--pending;++callbacks;
                        if(write){if(r.command<0)throw std::runtime_error("Unexpected write merging");last_write_ready=std::max(last_write_ready,h.tick+h.guard);}
                    });
                request.size_bytes=32;++pending;
                if(!h.memory->send(request)){--pending;++retries;break;}
                ++admitted;if(write)++writes;else++reads;--budget;advance();
            }
            tick();
        }
        core=std::max(core,tick_to_core(h.tick));
        // Align the dependency release to a core edge; retain DRAM refresh phase.
        U end_tick=core_to_tick(core);while(h.tick<end_tick)tick();
    }
};
int main(int argc,char**argv){try{
    if(argc!=7)throw std::runtime_error("runner CONFIG COMMANDS OUT CORE_HZ INJECTION MAX_COMMANDS");
    Simulator sim(argv[1],std::stoull(argv[4]),std::stoull(argv[5]));
    if(!sim.core_hz||!sim.injection||sim.injection>16)throw std::runtime_error("Invalid clock/injection");
    std::ifstream input(argv[2]);std::ofstream output(argv[3]);if(!input||!output)throw std::runtime_error("Cannot open command/report path");
    output<<"token,op,kind,start_core,end_core,start_dram,end_dram,reads,writes,retries,commands\n";
    U max_commands=std::stoull(argv[6]),commands=0;
    long token,op,last_token=-1,last_op=-1;std::string kind,last_kind,action;U a,b,c,d;
    U begin=0,begin_tick=0,begin_reads=0,begin_writes=0,begin_retries=0,begin_commands=0;
    auto flush=[&](){if(last_op>=0)output<<last_token<<','<<last_op<<','<<last_kind<<','<<begin<<','<<sim.core<<','<<begin_tick<<','<<sim.h.tick<<','<<sim.reads-begin_reads<<','<<sim.writes-begin_writes<<','<<sim.retries-begin_retries<<','<<commands-begin_commands<<'\n';};
    auto wall=std::chrono::steady_clock::now();
    while(input>>token>>op>>kind>>action>>a>>b>>c>>d){
        if(commands>=max_commands)throw std::runtime_error("Command limit exceeded");
        if(token!=last_token||op!=last_op){
            flush();
            if(token!=last_token||op%100==0){output.flush();std::cout<<"token="<<token<<" op="<<op<<" core="<<sim.core<<" dram="<<sim.h.tick<<" wall_s="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-wall).count()<<std::endl;}
            last_token=token;last_op=op;last_kind=kind;begin=sim.core;begin_tick=sim.h.tick;begin_reads=sim.reads;begin_writes=sim.writes;begin_retries=sim.retries;begin_commands=commands;
        }
        if(action=="C")sim.compute(a);
        else if(action=="R"||action=="W")sim.transfer(a,b,c,d,action=="W");
        else throw std::runtime_error("Unknown command");
        ++commands;
    }
    if(!input.eof())throw std::runtime_error("Malformed command stream");flush();output.flush();
    if(sim.pending||sim.admitted!=sim.callbacks)throw std::runtime_error("Undrained requests");
    std::cout<<"PASS commands="<<commands<<" callbacks="<<sim.callbacks<<" core="<<sim.core<<" dram="<<sim.h.tick<<" wall_s="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-wall).count()<<std::endl;
    return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
