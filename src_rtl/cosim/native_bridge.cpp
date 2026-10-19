// Native Ramulator lifecycle and callback boundary, shared by live RTL and reference runs.
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <stdexcept>
#include <limits>
#include <cmath>
#include "ramulator/base/config.h"
#include "ramulator/base/factory.h"
#include "ramulator/dram/dram_spec.h"
#include "ramulator/frontend/i_frontend.h"
#include "ramulator/memory_system/i_memory_system.h"
using U = std::uint64_t;
static thread_local std::string last_error;
struct Event { U id, tick, command; };
struct HBM {
    Ramulator::ConfigNode config;
    std::unique_ptr<Ramulator::DRAMSpec> spec;
    std::unique_ptr<Ramulator::IFrontEnd> frontend;
    std::unique_ptr<Ramulator::IMemorySystem> memory;
    std::deque<Event> events;
    U tick=0, capacity=0, guard=0, outstanding=0;
    explicit HBM(const char* path) : config(Ramulator::Config::parse_config_file(path)) {
        auto controllers=config["memory_system"]["controllers"].seq();
        const auto mem=config["memory_system"];
        if(controllers.empty() || controllers.size()>16 || (controllers.size() & (controllers.size()-1)) ||
           mem["impl"].as<std::string>()!="GenericDRAM" || mem["clock_ratio"].as<int>()!=1 ||
           mem["channel_mapper"]["impl"].as<std::string>()!="CacheLineInterleave" ||
           mem["channel_mapper"]["interleave_bits"].as<int>(0)!=0 ||
           config["frontend"]["impl"].as<std::string>()!="External" ||
           config["frontend"]["clock_ratio"].as<int>()!=1)
            throw std::runtime_error("Requires External/GenericDRAM clocks 1:1 and power-of-two striped HBM2 channels <=16");
        for(const auto& ctrl:controllers) {
            if(ctrl["impl"].as<std::string>()!="HBM12" || ctrl["dram"]["impl"].as<std::string>()!="HBM2" ||
               ctrl["addr_mapper"]["impl"].as<std::string>()!="RoBaRaCoCh")
                throw std::runtime_error("Requires HBM12/HBM2/RoBaRaCoCh");
            auto current=Ramulator::DRAMSpec::create("HBM2",ctrl);
            for(int count:current->organization.level_sizes)
                if(count<=0 || (count & (count-1))) throw std::runtime_error("Unresolved or non-power-of-two organization");
            for(int timing:current->timing_vals)
                if(timing<0) throw std::runtime_error("Unresolved timing");
            if(current->get_tx_bytes()!=32 || current->get_level_size("Channel")!=1 ||
               current->get_level_size("PseudoChannel")!=2 || current->organization.dq!=64 ||
               current->channel_width!=128 || current->get_timing_value("tCK_ps")<=0)
                throw std::runtime_error("Expected 32-byte HBM2 transactions and two 64-bit pseudochannels");
            if(spec && (spec->organization.level_sizes!=current->organization.level_sizes || spec->timing_vals!=current->timing_vals))
                throw std::runtime_error("Channels must have matching geometry/timing");
            if(!spec) spec=std::move(current);
        }
        capacity=controllers.size()*U(spec->organization.dq/8);
        for(auto n:spec->organization.level_sizes) {
            if(capacity>U(std::numeric_limits<long>::max())/U(n)) throw std::runtime_error("Capacity overflows native address");
            capacity*=U(n);
        }
        guard=spec->get_timing_value("nCWL")+spec->get_timing_value("nBL")+spec->get_timing_value("nWR");
        frontend.reset(Ramulator::Factory::create_frontend(config));
        memory.reset(Ramulator::Factory::create_memory_system(config));
        frontend->connect_memory_system(memory.get()); memory->connect_frontend(frontend.get());
        if(memory->get_tx_bytes()!=32 || std::abs(memory->get_tCK()-double(spec->get_timing_value("tCK_ps"))/1000.0)>1e-6)
            throw std::runtime_error("Native memory clock or transfer size differs from configured HBM2");
    }
};
extern "C" {
const char* bridge_error(){ return last_error.c_str(); }
void* hbm_create(const char* path){try{return new HBM(path);}catch(const std::exception& e){last_error=e.what();return nullptr;}}
void hbm_destroy(void* p){delete static_cast<HBM*>(p);}
void hbm_info(void* p,U* out){auto& h=*static_cast<HBM*>(p);out[0]=32;out[1]=h.spec->get_timing_value("tCK_ps");out[2]=h.guard;out[3]=h.capacity;out[4]=h.tick;out[5]=h.outstanding;}
int hbm_send(void* p,U id,U address,int write){try{
    auto& h=*static_cast<HBM*>(p);
    if(address%32 || address>h.capacity-32) throw std::runtime_error("HBM address out of range or unaligned");
    Ramulator::Request req(long(address),write?Ramulator::Request::Type::Write:Ramulator::Request::Type::Read,0,
        [&h,id](Ramulator::Request& r){--h.outstanding;h.events.push_back({id,h.tick,U(std::int64_t(r.command))});});
    req.size_bytes=32; ++h.outstanding;
    if(!h.memory->send(req)){--h.outstanding;return 0;} return 1;
}catch(const std::exception& e){last_error=e.what();return -1;}}
int hbm_tick(void* p){try{auto& h=*static_cast<HBM*>(p);++h.tick;h.memory->tick();return 0;}catch(const std::exception& e){last_error=e.what();return -1;}}
int hbm_pop(void* p,U* out){auto& h=*static_cast<HBM*>(p);if(h.events.empty())return 0;auto e=h.events.front();h.events.pop_front();out[0]=e.id;out[1]=e.tick;out[2]=e.command;return 1;}
}
