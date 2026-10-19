// Timestamp-preserving HBM replay against pinned Ramulator 1 (214f635845...).
// Usage: ramulator_hbm config input.csv output.csv stats.txt core_hz
//                       max_dram_cycles [max_accepted_per_dram_tick=64]
// Input: id,arrival_core_cycle,address_bytes,operation; integer IDs/addresses,
// sorted nonnegative arrivals, R/W, one aligned DRAM transaction per row.
// Output is event ordered. W completion=-1 because vendor Controller has no
// write callback. R completion is the actual callback tick, including reads
// forwarded from the write queue (whose nominal latency can be one tick).
// All streams share one Memory and one controller per configured HBM channel.
// Per-channel FIFO admission avoids a full channel blocking all other channels.
// After all requests drain, nCWL+nBL+nWR extra ticks conservatively cover the
// last write's transfer/recovery; this is NOT a per-write completion timestamp.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "Config.h"
#include "HBM.h"
#include "Memory.h"
#include "Controller.h"
#include "Statistics.h"

bool ramulator::warmup_complete = true;
using U64 = std::uint64_t;
using U128 = __uint128_t;

static std::string trim(const std::string& x) {
    const auto a = x.find_first_not_of(" \t\r\n");
    return a == std::string::npos ? "" : x.substr(a, x.find_last_not_of(" \t\r\n") - a + 1);
}

static U64 integer(const std::string& x, const std::string& what) {
    if (x.empty() || x.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error(what + " must be a nonnegative decimal integer");
    std::size_t used = 0;
    const auto value = std::stoull(x, &used);
    if (used != x.size()) throw std::runtime_error("Invalid " + what);
    return value;
}

static std::vector<std::string> csv(const std::string& line) {
    std::vector<std::string> fields;
    std::size_t start = 0;
    while (true) {
        const auto end = line.find(',', start);
        fields.push_back(trim(line.substr(start, end == std::string::npos ? end : end - start)));
        if (end == std::string::npos) return fields;
        start = end + 1;
    }
}

struct Entry {
    U64 id, arrival_core, due_dram, address;
    char operation;
};

class Input {
    std::ifstream file;
    U64 previous = 0, line_number = 1, core_hz, tck_fs, capacity, transaction;
    std::unordered_set<U64> ids;
public:
    U64 records = 0;
    Input(const std::string& path, U64 hz, U64 fs, U64 cap, U64 tx)
        : file(path), core_hz(hz), tck_fs(fs), capacity(cap), transaction(tx) {
        if (!file) throw std::runtime_error("Cannot open input CSV: " + path);
        std::string line;
        if (!std::getline(file, line) || csv(line) != std::vector<std::string>{
                "id", "arrival_core_cycle", "address_bytes", "operation"})
            throw std::runtime_error("Unexpected input CSV header");
    }
    bool next(Entry& e) {
        std::string line;
        while (std::getline(file, line)) {
            ++line_number;
            if (trim(line).empty()) continue;
            const auto f = csv(line);
            if (f.size() != 4) throw std::runtime_error("Expected four fields at input line " + std::to_string(line_number));
            e.id = integer(f[0], "id");
            e.arrival_core = integer(f[1], "arrival_core_cycle");
            e.address = integer(f[2], "address_bytes");
            if (f[3] != "R" && f[3] != "W") throw std::runtime_error("Operation must be R or W");
            e.operation = f[3][0];
            if (!ids.insert(e.id).second) throw std::runtime_error("Duplicate request ID: " + f[0]);
            if (records && e.arrival_core < previous) throw std::runtime_error("Input arrivals are not sorted");
            if (e.address % transaction) throw std::runtime_error("Unaligned transaction address: " + f[2]);
            if (e.address > capacity - transaction || e.address > U64(std::numeric_limits<long>::max()))
                throw std::runtime_error("Transaction address exceeds configured HBM capacity: " + f[2]);
            // Exact rational clock conversion at femtosecond tCK precision:
            // ceil(arrival_core * 1e15 / (core_hz * tCK_femtoseconds)).
            const U128 numerator = U128(e.arrival_core) * U64(1000000000000000ULL);
            const U128 denominator = U128(core_hz) * tck_fs;
            const U128 due = (numerator + denominator - 1) / denominator;
            if (due > U64(std::numeric_limits<long>::max())) throw std::runtime_error("Arrival overflows DRAM cycle counter");
            e.due_dram = U64(due);
            previous = e.arrival_core;
            ++records;
            return true;
        }
        if (!file.eof()) throw std::runtime_error("Error reading input CSV");
        return false;
    }
};

static int run(int argc, char** argv) {
    if (argc != 7 && argc != 8) throw std::runtime_error(
        "Usage: ramulator_hbm config input.csv output.csv stats.txt core_hz max_dram_cycles [max_accepted_per_dram_tick=64]");
    if (std::string(argv[2]) == argv[3] || std::string(argv[2]) == argv[4] || std::string(argv[3]) == argv[4])
        throw std::runtime_error("Input, output and stats paths must differ");
    std::size_t parsed = 0;
    const long double frequency = std::stold(argv[5], &parsed);
    if (parsed != std::string(argv[5]).size() || !std::isfinite(frequency) || frequency <= 0 ||
        frequency != std::floor(frequency) || frequency > U64(1000000000000000ULL))
        throw std::runtime_error("core_hz must be a positive whole-Hz frequency <= 1e15");
    const U64 core_hz = U64(frequency);
    const U64 max_cycles = integer(argv[6], "max_dram_cycles");
    const U64 max_injection = argc == 8 ? integer(argv[7], "max_accepted_per_dram_tick") : 64;
    if (!max_cycles || max_cycles > U64(std::numeric_limits<long>::max()) || !max_injection)
        throw std::runtime_error("Cycle limit and injection limit must be positive and cycles must fit signed long");
    if (!std::ifstream(argv[1])) throw std::runtime_error("Cannot open HBM config");
    ramulator::Config config(argv[1]);
    if (config["standard"] != "HBM") throw std::runtime_error("Only standard=HBM is supported");
    if (!ramulator::HBM::org_map.count(config["org"]) || !ramulator::HBM::speed_map.count(config["speed"]))
        throw std::runtime_error("Unsupported HBM org or speed (this pin supports HBM_1Gbps, not HBM2)");
    const U64 channels = integer(config["channels"], "channels");
    const U64 ranks = integer(config["ranks"], "ranks");
    if (!channels || !ranks || (channels & (channels - 1)) || (ranks & (ranks - 1)) || channels > 1024 || ranks > 1024)
        throw std::runtime_error("Channels/ranks must be positive powers of two <= 1024");
    if (config.contains("translation") && config["translation"] != "None")
        throw std::runtime_error("Only untranslated physical addresses (translation=None) are supported");
    config.set_core_num(1);
    auto* spec = new ramulator::HBM(config["org"], config["speed"]);
    spec->set_channel_number(int(channels));
    spec->set_rank_number(int(ranks));
    std::vector<ramulator::Controller<ramulator::HBM>*> controllers;
    for (U64 ch = 0; ch < channels; ++ch) {
        auto* channel = new ramulator::DRAM<ramulator::HBM>(spec, ramulator::HBM::Level::Channel);
        channel->id = int(ch);
        channel->regStats("");
        controllers.push_back(new ramulator::Controller<ramulator::HBM>(config, channel));
    }
    ramulator::Memory<ramulator::HBM, ramulator::Controller> memory(config, controllers);
    const double tck_ns = memory.clk_ns();
    const U64 tck_fs = U64(std::llround(tck_ns * 1e6));
    if (!tck_fs || std::abs(tck_ns - double(tck_fs) / 1e6) > 1e-12)
        throw std::runtime_error("tCK cannot be represented exactly at femtosecond precision");
    const U64 transaction = spec->prefetch_size * spec->channel_width / 8;
    U128 capacity128 = spec->channel_width / 8;
    for (int i = 0; i < int(ramulator::HBM::Level::MAX); ++i) capacity128 *= spec->org_entry.count[i];
    if (capacity128 > U64(std::numeric_limits<long>::max())) throw std::runtime_error("Configured capacity exceeds address range");
    const U64 capacity = U64(capacity128);
    Input input(argv[2], core_hz, tck_fs, capacity, transaction);
    std::ofstream output(argv[3]);
    if (!output) throw std::runtime_error("Cannot open output CSV");
    output << "id,arrival_core_cycle,accepted_dram_cycle,completed_dram_cycle,operation,address_bytes\n";
    Stats::statlist.output(argv[4]);
    std::vector<std::deque<Entry>> ready(channels);
    U64 dram_cycle = 0, accepted_reads = 0, accepted_writes = 0, completed_reads = 0;
    U64 waiting = 0, retries = 0, accepted_total = 0, channel_cursor = 0;
    Entry upcoming{};
    bool has_upcoming = input.next(upcoming);
    auto emit = [&](const Entry& e, U64 accepted, long completed) {
        output << e.id << ',' << e.arrival_core << ',' << accepted << ',' << completed << ','
               << e.operation << ',' << e.address << '\n';
    };
    auto tick = [&]() {
        if (dram_cycle >= max_cycles) throw std::runtime_error("HBM replay exceeded max_dram_cycles before drain");
        ++dram_cycle; // callbacks during tick observe the actual controller time
        memory.tick();
        ++Stats::curTick;
    };
    while (has_upcoming || waiting || memory.pending_requests()) {
        if (dram_cycle >= max_cycles) throw std::runtime_error("HBM replay exceeded max_dram_cycles");
        while (has_upcoming && upcoming.due_dram <= dram_cycle) {
            const U64 channel = (upcoming.address >> memory.tx_bits) & (channels - 1);
            ready[channel].push_back(upcoming);
            ++waiting;
            has_upcoming = input.next(upcoming);
        }
        U64 injected = 0;
        while (injected < max_injection && waiting) {
            bool progress = false;
            for (U64 offset = 0; offset < channels && injected < max_injection; ++offset) {
                const U64 ch = (channel_cursor + offset) % channels;
                if (ready[ch].empty()) continue;
                const Entry e = ready[ch].front();
                const U64 accepted = dram_cycle;
                ramulator::Request request(long(e.address), e.operation == 'R' ? ramulator::Request::Type::READ : ramulator::Request::Type::WRITE,
                    [&, e, accepted](ramulator::Request&) {
                        if (e.operation != 'R') throw std::runtime_error("Unexpected vendor write callback");
                        ++completed_reads;
                        emit(e, accepted, long(dram_cycle));
                    });
                if (!memory.send(request)) { ++retries; continue; }
                if (e.operation == 'R') ++accepted_reads;
                else { ++accepted_writes; emit(e, accepted, -1); }
                ready[ch].pop_front();
                --waiting;
                ++injected;
                ++accepted_total;
                progress = true;
            }
            channel_cursor = (channel_cursor + 1) % channels;
            if (!progress) break;
        }
        if (!has_upcoming && !waiting) memory.set_high_writeq_watermark(0.0f);
        tick();
    }
    const U64 queue_drain_cycle = dram_cycle;
    const U64 write_guard = accepted_writes ? spec->speed_entry.nCWL + spec->speed_entry.nBL + spec->speed_entry.nWR : 0;
    for (U64 i = 0; i < write_guard; ++i) tick();
    while (memory.pending_requests()) tick(); // finish any refresh queued during guard
    if (!dram_cycle) tick(); // avoid zero-duration native statistics on an empty trace
    if (accepted_total != input.records || completed_reads != accepted_reads)
        throw std::runtime_error("Request ID/count conservation failed");
    output.flush();
    if (!output) throw std::runtime_error("Error writing response CSV");
    memory.finish();
    Stats::statlist.printall();
    std::ostringstream summary;
    summary << std::setprecision(17)
        << "{\n  \"status\": \"complete\",\n  \"tck_ns\": " << tck_ns
        << ",\n  \"transaction_bytes\": " << transaction << ",\n  \"channels\": " << channels
        << ",\n  \"ranks\": " << ranks << ",\n  \"capacity_bytes\": " << capacity
        << ",\n  \"core_hz\": " << core_hz << ",\n  \"dram_cycles\": " << dram_cycle
        << ",\n  \"queue_drain_cycle\": " << queue_drain_cycle << ",\n  \"post_queue_write_guard_ticks\": " << write_guard
        << ",\n  \"accepted_reads\": " << accepted_reads << ",\n  \"completed_reads\": " << completed_reads
        << ",\n  \"accepted_writes\": " << accepted_writes << ",\n  \"input_requests\": " << input.records
        << ",\n  \"admission_retry_attempts\": " << retries
        << ",\n  \"max_accepted_per_dram_tick\": " << max_injection
        << ",\n  \"peak_GB_s\": " << spec->speed_entry.rate * double(spec->channel_width) * channels / 8000.0
        << ",\n  \"read_completion\": \"actual callback DRAM tick, including write-forwarded reads\""
        << ",\n  \"write_completion\": \"unobserved: completed_dram_cycle=-1; global guard covers nCWL+nBL+nWR after queue drain\""
        << ",\n  \"address_mapping\": \"RoBaRaCoCh; lowest transaction-address bits select channel\""
        << ",\n  \"response_order\": \"event order, not input order\"\n}\n";
    std::ofstream metadata(std::string(argv[4]) + ".summary.json");
    if (!metadata || !(metadata << summary.str())) throw std::runtime_error("Cannot write metadata JSON");
    std::cout << summary.str();
    return 0;
}

int main(int argc, char** argv) {
    try { return run(argc, argv); }
    catch (const std::exception& e) { std::cerr << "ramulator_hbm: " << e.what() << '\n'; return 2; }
}
