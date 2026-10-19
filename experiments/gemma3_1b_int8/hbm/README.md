# HBM memory replay for SCALE-Sim

This experiment runs a bounded native SCALE-Sim GEMM, captures its actual DRAM
port requests, and replays the three streams through **one shared Ramulator HBM
memory system**. It produces request IDs, read completions, admission delays,
per-stream statistics, and Ramulator statistics.

**This is offline memory replay.** Memory backpressure is modeled within
Ramulator, but is not fed back to change the previously captured compute
schedule. It therefore does **not** establish new Gemma decode TPS/TTFT or full
RTL/SCALE-Sim memory-cycle equivalence. The existing ideal matrix estimates
remain unchanged.

## Reproduce

From the SCALE-Sim repository root, with the study's Python dependencies installed:

```sh
git submodule update --init submodules/ramulator
env -u PYTHONPATH .venv/bin/python experiments/gemma3_1b_int8/hbm/run_hbm.py
env -u PYTHONPATH .venv/bin/python experiments/gemma3_1b_int8/hbm/test_hbm.py
```

The runner builds a small C++ frontend against the pinned Ramulator sources.
It requires a C++ compiler. No SCALE-Sim Ramulator patches are copied into the
submodule. Generated binaries, traces, logs, and reports stay under `build/`.

The default GEMM is `M=512, N=1, K=1152` on a 64x32 WS array. This represents one
512-output-channel tile of a transposed Gemma decode projection:
`C^T = B^T A^T`. It is not the entire model. The core clock assumption is 1 GHz.
The physical scratchpad capacities represented are 64 KiB per operand: inputs
are INT8, outputs are INT32. Because SCALE-Sim allocates abstract one-byte
elements, its OFMAP size is set to 16 Ki elements to represent 64 KiB of INT32s.

Run a smaller smoke case or compare channel counts:

```sh
env -u PYTHONPATH .venv/bin/python experiments/gemma3_1b_int8/hbm/run_hbm.py --m 19 --n 4 --k 67 --out experiments/gemma3_1b_int8/hbm/build/smoke
env -u PYTHONPATH .venv/bin/python experiments/gemma3_1b_int8/hbm/run_hbm.py --channels 1 --out experiments/gemma3_1b_int8/hbm/build/gemma_tile_1ch
```

Inspect `build/gemma_tile_8ch/report.json`, `ramulator.stats`, `requests.csv`,
`responses.csv`, and `rows.json`. The latter maps source stream/row IDs to burst
IDs and read completion times. The `native/` reports describe the initial CALC
run only; they are **not** updated HBM accelerator performance reports.

## Validation result

All 19 adapter/native-capture/frontend tests pass. The default Gemma-shaped
tile produced 79,314 transactions: 78,706 reads with observed completions and
608 writes with observed admission. Replaying the same trace took 23,801 DRAM
ticks (47.602 microseconds) with eight channels and 243,607 ticks (487.214
microseconds) with one channel, including the 14-tick global write guard.
These are memory replay durations, not accelerator execution times or TPS.

## Memory model and adapter contract

The pinned submodule is Ramulator 1 commit
`214f635845214adf030367939655d172ef0fed5f`. Its HBM model supports:

- `standard=HBM`, `org=HBM_4Gb`, `speed=HBM_1Gbps`, one rank.
- Eight 128-bit channels by default, 2 ns DRAM clock period, 64-byte transactions.
- 512 MiB per channel, 4 GiB total with eight channels.
- 16 GB/s peak per channel, **128 GB/s theoretical aggregate peak** for eight.
  Sustained bandwidth depends on requests, scheduling, bank conflicts and refresh.

This is the older HBM model, not HBM2/HBM3. Those standards require a separately
validated Ramulator 2 backend and configuration; renaming the speed is insufficient.

The adapter:

1. Captures INPUT arguments to native `read_port.service_reads` and
   `write_port.service_writes` during a CALC run. It copies arrays immediately.
   Exported DRAM CSV files contain response timestamps, so they are not used as
   request timestamps here.
2. Sorts all streams by timestamp and shifts their shared earliest request to
   time zero. Negative prefetch time and all relative idle gaps are retained.
3. Converts source word addresses into disjoint physical byte regions: IFMAP
   at 0, FILTER at 16 MiB, OFMAP at 32 MiB. Each region reserves 16 MiB; larger
   tensors are rejected. Word widths are respectively 1, 1, and 4 bytes.
4. Drops `-1` padding and merges accesses to the same 64-byte burst **within one
   source row**. Repeated bursts on later rows remain separate requests; there
   is no cross-row line cache. Partial stores assume byte enables, charge the
   whole burst, and do not add ECC read-modify-write traffic.
5. Converts core-clock arrivals to memory-clock time, retries queue-full
   requests, and allows up to 64 accepted transactions per memory tick. This
   frontend admission limit is explicit; HBM command/data timing is enforced
   by Ramulator. The stock DRAM trace runner admits only one request per tick,
   which would otherwise impose an artificial 32 GB/s limit with 64-byte
   requests and a 2 ns tick.
6. Records actual read callbacks. Upstream does not provide write completion
   callbacks, so write `completed_dram_cycle=-1` means **unavailable**. Queue
   drain is not a physical write-completion measurement. The frontend adds a
   conservative global `nCWL+nBL+nWR` guard after the queues drain when writes
   are present, and reports that guard separately.

All streams contend for the same controllers. A separate run starts fresh
memory state. A full-model evaluation must preserve state across layers and
tokens, and account for scale metadata, KV layout, output conversion, and all
other traffic absent from this bounded example.

## Why the existing replay scripts are not used

The bundled `scripts/dram_sim.py` hardcodes DDR4 and discards original request
timestamps. It also treats each abstract word as an entire DRAM transaction.
`dram_latency.py` assigns latencies using fixed address ranges and groups of ten
requests; the memory ports have missing-latency fallbacks and do not convert
DRAM ticks to core cycles. These assumptions are unsuitable for claiming
cycle-accurate HBM behavior.

For a future closed-loop implementation, use an event-driven memory adapter
that stalls the array/SRAM controller on actual completions and queue capacity,
preserves memory state across work, and drives the RTL's readiness from the same
events. Comparing that implementation against SCALE-Sim requires an explicit
shared SRAM/DMA policy as well as this DRAM model. This experiment supplies no
physical HBM controller/PHY RTL or board-specific memory interface.

Upstream references: [Ramulator HBM example configuration](https://github.com/CMU-SAFARI/ramulator/blob/master/configs/HBM-config.cfg),
[pinned HBM model](https://github.com/CMU-SAFARI/ramulator/blob/214f635845214adf030367939655d172ef0fed5f/src/HBM.h).
