# AWS F2 HBM2 profile for SCALE-Sim memory replay

This experiment uses **Ramulator 2.1's actual HBM2 model** and HBM12 controller.
The default profile follows the public AWS F2 VU47P HBM IP configuration:

| Property | Profile |
| --- | --- |
| HBM generation | HBM2 |
| Capacity per FPGA | 16 GiB across two 8 GiB stacks |
| Physical channels | 16, with two pseudochannels each |
| Pseudochannels / AXI ports | 32 |
| DRAM clock / pin rate | Nominal 900 MHz / 1800 Mbit/s |
| AXI interface | 256 bits per port, up to 450 MHz |
| Aggregate peak bandwidth | Nominal 460.8 GB/s, not sustained throughput |
| Native transaction | 32 bytes per pseudochannel |
| Geometry within a pseudochannel | 2 SID x 4 bank groups x 4 banks x 16384 rows |

This matches published memory organization and selected timing settings. It
does **not** reproduce AMD's proprietary controller, AXI crossbar, PHY or exact
physical address map. It remains **offline replay**, so memory delays are not
fed back to stall the SCALE-Sim/RTL array and it does not establish new Gemma
TPS/TTFT.

A separate [live RTL co-simulation](../../../src_rtl/README.md#live-hbm2-co-simulation-and-system-timing)
now feeds Ramulator HBM2 responses back into the AXI DMA and compute stalls:
`make -C src_rtl test-hbm2`. Its reports cover the implemented WS/SRAM/DMA
policy and bounded GEMMs; the offline results below remain unchanged. Neither
path establishes full Gemma TPS/TTFT or AMD-controller equivalence.

A third [full-model numerical and timing path](full_model/README.md) executes
real quantized Gemma weights and measures a matching complete operator graph,
including non-GEMM operations, with dependent native HBM2 callbacks. Its saved
32x32 WS / assumed 1 GHz baseline reports **8.9865 decode tokens/s**. Its new
default configuration reaches **41.6419 decode tokens/s** with a 256x256 WS
array, 256 KiB per operand buffer, 128 FP32 vector lanes and 128 KiB vector
staging at the same assumed 1 GHz. Both results cover three steps after a
512-token prompt with the same HBM2 profile. The faster configuration uses
65,536 MAC units versus 1,024 in the baseline. This uses an aggregate memory
frontend and assumed vector/SFU timings; it does not establish FPGA fit,
full-model RTL cycle equivalence, simulated TTFT, or AMD-controller equivalence.

## Run

From the SCALE-Sim repository root:

```sh
bash experiments/gemma3_1b_int8/hbm2/setup_ramulator2.sh
env -u PYTHONPATH .venv/bin/python experiments/gemma3_1b_int8/hbm2/run_hbm2.py
env -u PYTHONPATH .venv/bin/python experiments/gemma3_1b_int8/hbm2/test_hbm2.py
```

Setup pins Ramulator2 to `c6c7a5de0cef35935d6fad7369e6271406798865`, builds the
C++ library, and records a minimal C++ template-keyword portability patch under
`patches/`. The patch changes no timing/model behavior. Existing Ramulator1
and its HBM1 experiment remain available separately.

The default workload is a single `M=512, N=1, K=1152` WS GEMM on a 64x32 array,
representing a transposed Gemma decode output-channel tile. The assumed core
clock is **250 MHz**, matching the F2 example's shell reference; it is not a
timing-closure result. The memory clock and AXI clock are separate. Use
`--core-hz` for a different compute-clock assumption.

```sh
env -u PYTHONPATH .venv/bin/python experiments/gemma3_1b_int8/hbm2/run_hbm2.py --m 19 --n 4 --k 67 --out experiments/gemma3_1b_int8/hbm2/build/smoke
```

Results are in `build/gemma_tile_f2/`: `report.json`, `F2-profile.json`, resolved
`F2-HBM2.json`, `requests.csv`, `responses.csv`, `rows.json`, `ramulator.stats`
and `ramulator.stats.summary.json`. Its `native/` reports describe the source
CALC simulation, not execution with HBM stalls.

`f2_config.py` regenerates the standalone resolved profile/config files:

```sh
env -u PYTHONPATH .venv/bin/python experiments/gemma3_1b_int8/hbm2/f2_config.py
```

## Timing and modeling boundaries

The selected AWS IP snapshot is
`aws/aws-fpga-resources@6d32be972e6da854e61a8d3d6ec0466ab491c1b3`.
`f2_provenance.json` records the source evidence. Most cycle timings come from
its `cl_hbm.xci`; CCD timings also use its supplied AMD model. All timing
parameters are passed explicitly, so the generic 2000 Mbps constructor seed
does not silently determine the resulting model timings.

Ramulator stores integer picoseconds. The profile represents the nominal
`10000/9 ps` period as **1111 ps**, about 100 ppm faster. Reports expose nominal
and modeled frequency/bandwidth separately. This is a documented quantization,
not a claim that F2's physical clock differs from 900 MHz.

Remaining modeling choices include:

- HBM12/FRFCFS, open-page policy, and all-bank refresh replace AMD's proprietary
  queue scheduling and refresh lookahead/single-bank policy.
- `nRTW=21` uses the upstream HBM2 timing expression with the published RL/WL;
  `nREFIpb=109` is derived for 32 banks and is unused by all-bank refresh.
- XCI `USER_RL=22` and `USER_WL=7` are interpreted as decoded cycle latencies
  (`nCL=22`, `nCWL=7`). The relation between these XCI settings and raw mode
  register encoding has not been independently verified, so this is an
  explicit modeling assumption.
- Logical buffers are striped across physical channels using 32-byte blocks,
  then mapped by native `RoBaRaCoCh`. F2's actual global map assigns contiguous
  512 MiB regions to pseudochannels. The striped mapping represents a software
  data-layout choice; the emitted byte addresses are **not literal F2 addresses**.
- The default host admission cap is 16 transactions per DRAM tick, approximating
  the aggregate 32-port, 450 MHz AXI data rate. This is not a per-port AXI
  handshake/burst/interconnect model. `--max-injection` makes that assumption
  explicit.
- Input capture, word-to-byte conversion, source-row IDs and within-row burst
  coalescing reuse the verified HBM adapter. INT8 input/weight and INT32 output
  widths are explicit. Separate rows can fetch the same burst again; a real
  DMA line buffer can change the traffic substantially.
- Read completions use actual callbacks. Write callbacks indicate command
  issue or merging with a buffered write, not persistence. Writes keep
  `completed_dram_cycle=-1`; separate callback fields record those events. The
  final drain adds a conservative `nCWL+nBL+nWR` guard.
- Stream `requested_transaction_bytes` counts submitted payload. The frontend
  also reports `command_data_bytes` because write merging and read forwarding
  can reduce actual DRAM command traffic.

All streams share one memory system. A new replay starts fresh state; this is
one bounded GEMM, not full inference. Full F2 performance requires a persistent
memory/compute simulation, actual DMA/SRAM policies, non-GEMM operations, and
an achieved FPGA clock.

## Validation result

The native HBM2 frontend and Ramulator 2.1 library built successfully on macOS
arm64. All 12 HBM2 adapter tests pass, including clock conversion, channel
scaling, 16 GiB address bounds, read forwarding and write coalescing. All 19
existing HBM adapter tests also pass.

The default native Gemma-shaped tile replay completed with 88,005 requests:
86,821 read completions and 1,184 write callbacks (722 issued writes and 462
merged writes). The replay took 55,122 DRAM ticks, or 61.240542 microseconds,
including its source timing, initial prefetch and final write guard. This is
**not a token latency**, and no compute-stall feedback was applied. Its detailed
result is `build/gemma_tile_f2/report.json`.

## Decode TPS projections

`decode_estimate.py` separately combines native-validated WS schedule equations
with a minimum streaming-traffic model. It does **not** derive TPS from the
bounded Ramulator replay. Regenerate its 26 operating points and 13 bounded
native demand checks with:

```sh
env -u PYTHONPATH .venv/bin/python experiments/gemma3_1b_int8/hbm2/decode_estimate.py
```

The workload is Gemma 3 1B IT, batch one, W8A8 with INT8 KV, a 512-token prompt
and 127 decode steps (contexts 513 through 639). It assumes each matrix weight
and required KV value is read once per token: **1,006,698,496 bytes/token** on
average. This takes 2.185 ms at the nominal 460.8 GB/s peak, or 5.033 ms at an
**illustrative, unmeasured 200 GB/s** sustained rate.

| Array / mapping | Assumed core clock | Matrix-only ceiling | Matrix + traffic serialized at 200 GB/s |
| --- | ---: | ---: | ---: |
| Current 32x32 normal WS | 250 MHz | 2.68 TPS | 2.64 TPS |
| Proposed 64x32 transposed WS | 250 MHz | 11.83 TPS | 11.16 TPS |
| Proposed 64x32 transposed WS | 450 MHz | 21.29 TPS | 19.23 TPS |
| Proposed 128x16 transposed WS | 450 MHz | 36.05 TPS | 30.51 TPS |
| Proposed 576x8 transposed WS | 250 MHz | 40.27 TPS | 33.49 TPS |

Proposed mappings compute `C.T = B.T @ A.T`, with original output channels
tiled into groups of 512. They require new operand/output layouts and DMA
integration; they are not the current default RTL configuration. The 450 MHz
cases are hypothetical compute clocks, not a consequence of F2's 450 MHz AXI
clock. Neither clock has been validated by FPGA timing closure for these arrays.

The serialized column is a scenario, **not a guaranteed lower bound**: it still
excludes non-GEMM work, scale metadata, layout conversions, KV writes, repeated
SRAM refills, host overhead and sampling. Perfect compute/memory overlap gives
the matrix-only ceiling for these points. The 576x8/250 MHz scenario has only
3.47 ms/token left for omitted costs before falling below 30 TPS. Its 4,608 PEs
alone do not establish FPGA resource fit or routability.

`decode_estimate.json` includes all assumptions, per-point prefill matrix times,
200/300/460.8 GB/s scenarios, source hashes and native validation results.

## Sources

- [AWS F2 memory performance example](https://github.com/aws/aws-fpga/blob/f2/hdk/cl/examples/cl_mem_perf/README.md)
- [Pinned AWS HBM IP settings](https://github.com/aws/aws-fpga-resources/blob/6d32be972e6da854e61a8d3d6ec0466ab491c1b3/cl_ip/cl_ip.srcs/sources_1/ip/cl_hbm/cl_hbm.xci)
- [AMD HBM address map](https://docs.amd.com/r/en-US/pg276-axi-hbm/HBM-Address-Map-and-Protocol-Considerations)
- [Pinned Ramulator HBM2 specification](https://github.com/CMU-SAFARI/ramulator2/blob/c6c7a5de0cef35935d6fad7369e6271406798865/python/ramulator/dram/hbm2.py)
