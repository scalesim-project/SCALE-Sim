// Timestamp-preserving replay with the real HBM2 model in pinned Ramulator 2.1.
// CSV input: id,arrival_core_cycle,address_bytes,operation (R/W), sorted arrivals.
// Each row is exactly one aligned native transaction. All streams share memory.
// Write callback means command issue or coalescing, NOT physical write completion.
// Thus completed_dram_cycle=-1 for W; callback fields preserve observable events.
// Application callbacks drain before a conservative nCWL+nBL+nWR write guard.
// This models the open-source controller, not the proprietary AWS/AMD controller.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include "ramulator/base/config.h"
#include "ramulator/base/factory.h"
#include "ramulator/dram/dram_spec.h"
#include "ramulator/frontend/i_frontend.h"
#include "ramulator/memory_system/i_memory_system.h"
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
        "Usage: ramulator_hbm2 config.json input.csv output.csv stats.txt core_hz max_dram_cycles [max_accepted_per_dram_tick=64]");
    if (std::string(argv[2]) == argv[3] || std::string(argv[2]) == argv[4] || std::string(argv[3]) == argv[4])
        throw std::runtime_error("Input, output and stats paths must differ");
    std::size_t parsed = 0;
    const long double frequency = std::stold(argv[5], &parsed);
    if (parsed != std::string(argv[5]).size() || !std::isfinite(frequency) || frequency <= 0 ||
        frequency != std::floor(frequency) || frequency > 1000000000000000ULL)
        throw std::runtime_error("core_hz must be a positive whole-Hz frequency <= 1e15");
    const U64 core_hz = U64(frequency);
    const U64 max_cycles = integer(argv[6], "max_dram_cycles");
    const U64 max_injection = argc == 8 ? integer(argv[7], "max_accepted_per_dram_tick") : 64;
    if (!max_cycles || max_cycles > U64(std::numeric_limits<long>::max()) || !max_injection)
        throw std::runtime_error("Cycle and injection limits must be positive; cycles must fit signed long");

    auto config = Ramulator::Config::parse_config_file(argv[1]);
    const auto mem_config = config["memory_system"];
    const auto ctrl_configs = mem_config["controllers"].seq();
    const U64 channels = ctrl_configs.size();
    if (!channels || (channels & (channels - 1)) || channels > 1024)
        throw std::runtime_error("Physical channel count must be a positive power of two <= 1024");
    if (mem_config["impl"].as<std::string>() != "GenericDRAM" ||
        mem_config["channel_mapper"]["impl"].as<std::string>() != "CacheLineInterleave" ||
        mem_config["channel_mapper"]["interleave_bits"].as<int>(0) != 0 ||
        mem_config["clock_ratio"].as<int>() != 1 ||
        config["frontend"]["impl"].as<std::string>() != "External")
        throw std::runtime_error("Requires External frontend and GenericDRAM clock_ratio=1 with CacheLineInterleave(interleave_bits=0)");
    std::unique_ptr<Ramulator::DRAMSpec> spec;
    for (const auto& ctrl : ctrl_configs) {
        if (ctrl["impl"].as<std::string>() != "HBM12" ||
            ctrl["dram"]["impl"].as<std::string>() != "HBM2" ||
            ctrl["addr_mapper"]["impl"].as<std::string>() != "RoBaRaCoCh")
            throw std::runtime_error("Every controller must use HBM12, HBM2 and RoBaRaCoCh");
        const auto dram = ctrl["dram"];
        if (dram["org"]["count"].size() != 7 || dram["timing"].size() != 26)
            throw std::runtime_error("HBM2 requires 7 organization levels and 26 resolved timing values");
        auto current = Ramulator::DRAMSpec::create("HBM2", ctrl);
        for (const int count : current->organization.level_sizes)
            if (count <= 0 || (count & (count - 1))) throw std::runtime_error("Organization counts must be positive powers of two");
        for (const int timing : current->timing_vals)
            if (timing < 0) throw std::runtime_error("Unresolved HBM2 timing");
        if (current->get_level_size("Channel") != 1 || current->get_level_size("PseudoChannel") != 2 ||
            current->organization.dq != 64 || current->channel_width != 128 || current->get_tx_bytes() != 32)
            throw std::runtime_error("Expected HBM2 physical channel with two 64-bit PCs and 32-byte transactions");
        if (spec && (spec->organization.level_sizes != current->organization.level_sizes ||
                     spec->timing_vals != current->timing_vals))
            throw std::runtime_error("All physical channels must have identical geometry and timing");
        if (!spec) spec = std::move(current);
    }
    const U64 tck_ps = spec->get_timing_value("tCK_ps");
    if (!tck_ps) throw std::runtime_error("tCK_ps must be positive");
    const double tck_ns = double(tck_ps) / 1000.0;
    const U64 transaction = spec->get_tx_bytes();
    U64 capacity = channels * U64(spec->organization.dq / 8);
    for (const int count : spec->organization.level_sizes) {
        if (capacity > U64(std::numeric_limits<long>::max()) / U64(count))
            throw std::runtime_error("Capacity exceeds signed address range");
        capacity *= U64(count);
    }
    auto frontend = std::unique_ptr<Ramulator::IFrontEnd>(Ramulator::Factory::create_frontend(config));
    auto memory = std::unique_ptr<Ramulator::IMemorySystem>(Ramulator::Factory::create_memory_system(config));
    frontend->connect_memory_system(memory.get());
    memory->connect_frontend(frontend.get());
    if (memory->get_tx_bytes() != int(transaction) || std::abs(memory->get_tCK() - tck_ns) > 1e-6)
        throw std::runtime_error("Memory model disagrees with configured transaction or clock");
    Input input(argv[2], core_hz, tck_ps * 1000, capacity, transaction);
    std::ofstream output(argv[3]), stats(argv[4]);
    if (!output || !stats) throw std::runtime_error("Cannot open response or statistics output");
    output << "id,arrival_core_cycle,accepted_dram_cycle,completed_dram_cycle,operation,address_bytes,callback_dram_cycle,callback_kind\n";
    std::vector<std::deque<Entry>> ready(channels);
    std::unordered_set<U64> returned_ids;
    U64 dram_cycle = 0, accepted_reads = 0, accepted_writes = 0, completed_reads = 0;
    U64 write_callbacks = 0, issued_writes = 0, coalesced_writes = 0, forwarded_reads = 0;
    U64 outstanding = 0, waiting = 0, retries = 0, accepted_total = 0, channel_cursor = 0;
    Entry upcoming{};
    bool has_upcoming = input.next(upcoming);
    auto tick = [&]() {
        if (dram_cycle >= max_cycles) throw std::runtime_error("HBM2 replay exceeded max_dram_cycles before drain");
        ++dram_cycle;
        memory->tick();
    };
    while (has_upcoming || waiting || outstanding) {
        if (dram_cycle >= max_cycles) throw std::runtime_error("HBM2 replay exceeded max_dram_cycles");
        while (has_upcoming && upcoming.due_dram <= dram_cycle) {
            const U64 channel = (upcoming.address / transaction) & (channels - 1);
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
                Ramulator::Request request(long(e.address), e.operation == 'R' ? Ramulator::Request::Type::Read : Ramulator::Request::Type::Write,
                    0, [&, e, accepted](Ramulator::Request& req) {
                        if (!returned_ids.insert(e.id).second || !outstanding)
                            throw std::runtime_error("Duplicate or unexpected callback");
                        --outstanding;
                        std::string kind;
                        if (e.operation == 'R') {
                            ++completed_reads;
                            if (req.command < 0) { kind = "read_forwarded"; ++forwarded_reads; }
                            else kind = "read_completed";
                        } else {
                            ++write_callbacks;
                            if (req.command < 0) { kind = "write_coalesced"; ++coalesced_writes; }
                            else { kind = "write_command_issued"; ++issued_writes; }
                        }
                        output << e.id << ',' << e.arrival_core << ',' << accepted << ','
                               << (e.operation == 'R' ? long(dram_cycle) : -1L) << ',' << e.operation << ','
                               << e.address << ',' << dram_cycle << ',' << kind << '\n';
                    });
                request.size_bytes = int(transaction);
                // A coalesced write callback may execute synchronously inside send().
                ++outstanding;
                if (!memory->send(request)) { --outstanding; ++retries; continue; }
                if (e.operation == 'R') ++accepted_reads;
                else ++accepted_writes;
                ready[ch].pop_front();
                --waiting;
                ++injected;
                ++accepted_total;
                progress = true;
            }
            channel_cursor = (channel_cursor + 1) % channels;
            if (!progress) break;
        }
        tick();
    }
    const U64 callback_drain_cycle = dram_cycle;
    const U64 write_guard = accepted_writes ? spec->get_timing_value("nCWL") + spec->get_timing_value("nBL") + spec->get_timing_value("nWR") : 0;
    for (U64 i = 0; i < write_guard; ++i) tick();
    if (!dram_cycle) tick();
    if (accepted_total != input.records || completed_reads != accepted_reads || write_callbacks != accepted_writes ||
        returned_ids.size() != input.records || outstanding)
        throw std::runtime_error("Request ID/count conservation failed");
    output.flush();
    if (!output) throw std::runtime_error("Error writing response CSV");
    memory->finalize();
    memory->print_stats(stats);
    stats.flush();
    if (!stats) throw std::runtime_error("Error writing native statistics");
    std::ostringstream summary;
    summary << std::setprecision(17)
        << "{\n  \"status\": \"complete\",\n  \"standard\": \"HBM2\",\n  \"ramulator_version\": \"2.1\",\n"
        << "  \"ramulator_revision\": \"c6c7a5de0cef35935d6fad7369e6271406798865\",\n"
        << "  \"tck_ns\": " << tck_ns << ",\n  \"tck_ps\": " << tck_ps
        << ",\n  \"transaction_bytes\": " << transaction << ",\n  \"channels\": " << channels
        << ",\n  \"pseudochannels\": " << channels * spec->get_level_size("PseudoChannel")
        << ",\n  \"capacity_bytes\": " << capacity << ",\n  \"data_rate_mbps\": " << spec->get_timing_value("rate")
        << ",\n  \"effective_data_rate_mbps\": " << 2e6 / tck_ps
        << ",\n  \"core_hz\": " << core_hz << ",\n  \"dram_cycles\": " << dram_cycle
        << ",\n  \"queue_drain_cycle\": " << callback_drain_cycle
        << ",\n  \"application_callback_drain_cycle\": " << callback_drain_cycle
        << ",\n  \"post_queue_write_guard_ticks\": " << write_guard
        << ",\n  \"accepted_reads\": " << accepted_reads << ",\n  \"completed_reads\": " << completed_reads
        << ",\n  \"forwarded_reads\": " << forwarded_reads << ",\n  \"accepted_writes\": " << accepted_writes
        << ",\n  \"write_callbacks\": " << write_callbacks << ",\n  \"write_commands_issued\": " << issued_writes
        << ",\n  \"writes_coalesced\": " << coalesced_writes << ",\n  \"input_requests\": " << input.records
        << ",\n  \"read_commands_issued\": " << completed_reads - forwarded_reads
        << ",\n  \"command_data_bytes\": " << (completed_reads - forwarded_reads + issued_writes) * transaction
        << ",\n  \"admission_retry_attempts\": " << retries << ",\n  \"max_accepted_per_dram_tick\": " << max_injection
        << ",\n  \"peak_GB_s\": " << spec->channel_width * double(channels) * 2e3 / (8 * tck_ps)
        << ",\n  \"nominal_peak_GB_s\": " << spec->channel_width * double(channels) * spec->get_timing_value("rate") / 8000.0
        << ",\n  \"read_completion\": \"actual callback tick, including forwarded reads\""
        << ",\n  \"write_completion\": \"unobserved: completed_dram_cycle=-1; callback reports WR command issue or buffered-write coalescing; global guard nCWL+nBL+nWR\""
        << ",\n  \"drain_semantics\": \"all application requests callback, then write guard; autonomous refresh is not a completion event\""
        << ",\n  \"address_mapping\": \"CacheLineInterleave(0) across physical channels, then RoBaRaCoCh inside each channel\""
        << ",\n  \"controller\": \"Ramulator HBM12 with configured scheduler; not the proprietary AWS/AMD controller\""
        << ",\n  \"response_order\": \"callback event order, not input order\"\n}\n";
    std::ofstream metadata(std::string(argv[4]) + ".summary.json");
    if (!metadata || !(metadata << summary.str())) throw std::runtime_error("Cannot write summary JSON");
    std::cout << summary.str();
    return 0;
}

int main(int argc, char** argv) {
    try { return run(argc, argv); }
    catch (const std::exception& e) { std::cerr << "ramulator_hbm2: " << e.what() << '\n'; return 2; }
}
