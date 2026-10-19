# Dense weight-stationary accelerator RTL

This directory implements a usable dense signed-INT8 GEMM engine, with an actual
systolic MAC array, command controller, SRAM request interfaces, and banked
INT32 accumulation storage. A buffered integration top adds synchronous operand
SRAM, two software-managed pages, bank-conflict handling and an output FIFO.
An AXI4 integration top loads operands and stores results through a DMA controller.
The default array is 32 rows × 32 columns.

**Equivalence scope:** every valid/address lane in every accepted compute-demand
cycle matches the dense weight-stationary scheduler in SCALE-Sim v3 at commit
`9f98c4371055a54c75209c2e02b640b897550532`. This is a complete engine for that
specified GEMM contract, **not a complete RTL implementation of all SCALE-Sim v3
features, an OpenTPU-compatible ISA, or a ready-to-program FPGA/ASIC design**.
SCALE-Sim generates operand addresses and timing, not numerical outputs;
signed INT8 × signed INT8 with modulo-2^32 accumulation is the numerical contract
chosen here. Its arithmetic is checked independently with NumPy GEMM.

## Build and verify

From this directory:

```sh
make check-all  # differential/control/memory/AXI/Gemma tests, lint, synthesis
make test       # upstream address/valid comparison + independent numeric GEMM
make test-memory # clocked partial sums, banked SRAM, buffered integration
make test-axi    # AXI memory/protocol model, errors, reset and numeric GEMM
make quick      # directed edge cases, including the default 32×32 array
make gemma      # full-K Gemma hidden-width chunks for M=1 and M=512
make lint       # Verilator, all warnings fatal; all tops and AXI parameter cases
make synth      # Yosys generic synthesis/check, SRAMs preserved
make synth-memory # buffered/AXI synthesis; verify actual clocked RAM ports
```

Tests require the repository virtual environment with NumPy and SCALE-Sim's
Python dependencies, plus Icarus Verilog (`iverilog`, `vvp`). Override `PYTHON`
if needed. Verilator and Yosys are required for lint, synthesis and clocked-memory
checks. Results and generated vectors live in `build/`, which is ignored.
`build/validation.json` records the reference commit, shapes and random seeds.
Sources were written for this implementation; no OpenTPU RTL was copied.

## Top-level contract

`rtl/scalesim_ws_top.sv` exposes the compute-demand interface. `ROWS`, `COLS`,
`MAX_M` and `SYNC_PSUM` are elaboration parameters. Dimensions must be positive;
`MAX_M` defaults to 512. Its accumulator memory uses `COLS × MAX_M × 32` bits,
64 KiB by default. This is a physical INT32 partial-sum memory, independent
of SCALE-Sim's abstract OFMAP buffer and configured traffic word size; equal
byte capacities do not establish memory-system equivalence. `SYNC_PSUM=0`
(default) uses one asynchronous read and one clocked write per column bank.
`SYNC_PSUM=1` uses clocked, enabled reads and writes. Each later-K demand row
with any valid C lane first reads its partial sums, then becomes eligible to
commit on the next clock. This read wait overlaps external backpressure.
First-K rows initialize storage without reading it. Yosys inference is checked
for both the standalone synchronous core and the buffered top; vendor BRAM
mapping, frequency, resource budget and PPA have not been established.

A command computes `C[M,N] = A[M,K] × B[K,N]` for row-major dense matrices.
All command dimensions are unsigned 32-bit values, so the Gemma 3 vocabulary
width `N=262144` is representable. Validity also requires `M <= MAX_M` and each
matrix's final word address to fit in 32 bits. Commands with zero dimensions,
excessive M or an overflowing address range generate `done=error=1` for one
clock without starting. The dimension tuple `M=512,K=6912,N=262144` is
representable with suitable bases, although full traces at that size are
expensive to materialize and are not exercised by the default regression.

| Port | Meaning |
| --- | --- |
| `clk`, `rst` | Rising-edge clock, synchronous active-high reset. Reset aborts a command. |
| `cmd_valid`, `cmd_ready` | Command sampled on their simultaneous rising edge, only when idle; invalid commands complete with an error. |
| `cmd_m`, `cmd_k`, `cmd_n` | GEMM dimensions. |
| `cmd_a_base`, `cmd_b_base`, `cmd_c_base` | Independent unsigned 32-bit **word** bases, as in SCALE-Sim operand matrices. A/B words are 8 bits; C words are 32 bits. They are separate logical spaces. |
| `busy`, `done`, `error` | Busy during execution; one-cycle done on completion/rejection, error only on rejection. |
| `a_valid`, `a_addr`, `a_data` | ROWS parallel SRAM read request/data lanes, one activation byte per lane. |
| `b_valid`, `b_addr`, `b_data` | COLS parallel SRAM read request/data lanes, one weight byte per lane. |
| `c_valid`, `c_addr`, `c_data` | COLS parallel SRAM write proposals, signed 32-bit cumulative sums. |
| `c_final` | Valid alongside C writes: this is the final K fold for the corresponding output. |
| `cycle_ready`, `cycle_fire` | Entire current demand row commits only on `cycle_fire`. This requires busy, external readiness, no reset, and completed partial-sum reads in synchronous mode. |
| `logical_cycle` | Number of committed demand rows, 64 bits. |
| `elapsed_cycles`, `stall_cycles` | Clocks while busy and clocks without a commit, respectively; 64 bits. Includes internal partial-sum waits in synchronous mode. |

Packed lane zero occupies the least significant slice, e.g.
`a_addr[32*r +: 32]`. A/B data must correspond to the current proposed address
before the commit edge: this interface assumes combinational SRAM reads or an
adapter that holds `cycle_ready=0` until all reads and write sinks are ready.
Invalid input lanes are masked to zero internally.

**A valid lane alone is not a transfer.** The memory system must commit all
valid reads/writes only on `cycle_fire`. Demand position and PE state freeze whenever
`cycle_fire=0`; a pending synchronous partial-sum read can still complete.
Addresses and masks remain stable; C data settles after input data and the
partial-sum read are available. Do not capture C data solely from `c_valid`.
There are no independent lane handshakes or transaction queues in this compute
interface. The buffered top below supplies those memory adapters. Avoid a
combinational ready loop in an external adapter.

The engine stores intermediate sums internally. Every K fold still emits a C
write at the exact address and accepted logical cycle requested by SCALE-Sim; later folds emit the
**cumulative** sum, and `c_final` marks final results. Consumers interested only
in final results may ignore earlier writes, but the complete demand comparison
includes them. Accumulator banks are overwritten on the first K fold, so they
need no bulk reset. They are reused for each N tile. A/B/C bases need not remain
stable after command acceptance.

## Buffered memory integration

`rtl/scalesim_ws_buffered_top.sv` instantiates the core with `SYNC_PSUM=1`, two
`scalesim_banked_operand_buffer` instances, and a C packet FIFO. Its command
dimensions and logical word bases have the same meaning as the compute core.
`cmd_page` selects page 0 or 1 in both operand memories. Each command's complete
A and B matrices must fit in one page; larger matrices need software tiling.
Capacity errors and the core's invalid-command cases return `done=error=1`
without reading operands or producing outputs.

| Interface | Contract |
| --- | --- |
| `a_load_valid/ready`, `a_load_page`, `a_load_addr`, `a_load_data` | One byte written per handshake to a page-local A word offset. B has the corresponding `b_load_*` ports. |
| `cmd_valid/ready`, `cmd_page` | Accepted command owns the chosen page until its final queued C packet drains. A command takes priority over a coincident write to that page. |
| `page_busy[1:0]` | One-hot ownership of page 0 or 1 across both operand memories. Host writes to an owned page are backpressured. The other page can be loaded concurrently. |
| `out_valid/ready`, `out_mask`, `out_addr`, `out_data`, `out_final` | FIFO packets of COLS C lanes, using the core's logical addresses and cumulative INT32 values. Each packet stays stable until consumed. All K-fold writes are retained; `out_final` identifies final-K packets. |
| `busy`, `done`, `error` | Busy includes compute and output drain. Done pulses only after every output packet is consumed, or after command rejection/abort. |
| `cycle_fire`, `logical_cycle`, `core_elapsed_cycles`, `core_stall_cycles` | Core commit event and counters. Logical demand rows match the native WS generator; core stalls include operand waits, partial-sum waits and full output FIFO. |
| `transaction_cycles` | Clocks after command acceptance through wrapper completion, including final FIFO drain. Command acceptance itself is excluded. |

Load every required operand byte before issuing its command. A matrix is packed
row-major starting at offset zero within its page; the wrapper subtracts the
accepted logical base from each core request before accessing SRAM. There is
no loaded-byte bitmap. Reset cancels in-flight commands and clears queues and
ownership, while preserving SRAM contents. A defensive operand-response error
also cancels the transaction and discards queued packets, including any packet
currently stalled at `out_valid`. Previously consumed output packets remain
consumed; a consumer must discard incomplete commands after reset/error. The
ordinary stable-until-consumed output contract excludes these cancellation events.

`A_BANKS` and `B_BANKS` default to ROWS and COLS. Each SRAM has two pages of
`A_PAGE_WORDS` or `B_PAGE_WORDS` bytes (32768 per page by default). Page sizes
must be positive multiples of their bank counts. The default 32×32 design has
128 KiB of operand SRAM plus 64 KiB of partial sums. `OUTPUT_DEPTH` defaults to
2 C packets and must be positive; non-power-of-two depths are supported.

The operand buffer uses `bank = local_address % BANKS` and
`row = local_address / BANKS`. Each bank serves its lowest pending lane on each
clock with a synchronous read, followed by a gather register. Duplicate
addresses are read separately. For a request accepted at edge `n`, let `L` be
the largest number of valid requested lanes in any bank. Its response becomes
valid just after edge `n + L + 1`; an empty or entirely invalid request responds
at `n + 1`. Active out-of-range lanes return zero with `rsp_error=1`. A standalone
buffer permits concurrent reads/writes with read-first collision behavior;
the wrapper prevents host writes to the active page.

The wrapper gathers one native demand row at a time and holds operands until
the core commits. With no output backpressure, a row takes
`max(1, L + 4 if operand reads exist, 2 if a partial-sum read is needed)` clocks,
where `L` is the larger A/B bank occupancy. Final wrapper completion takes one
additional clock. This conservative schedule is checked against a separate
bank-occupancy timing oracle. It is not a throughput-optimized prefetcher.

This is a concrete software-managed memory policy. SCALE-Sim's CALC/USER
prefetch, active-window and layout-dependent bank models use different timing
rules. Accepted compute rows remain 1:1; **physical memory and full-transaction
cycles are not claimed to match SCALE-Sim's memory reports**. The AXI top below
provides DMA integration. HBM device control and Ramulator coupling remain
unfinished.

## AXI4 DMA integration

`rtl/scalesim_ws_axi_top.sv` adds a descriptor-driven AXI4 master around the
buffered engine. A descriptor supplies `cmd_m/k/n`, 64-bit **byte** addresses
`cmd_a_addr/b_addr/c_addr`, and `cmd_page`. A/B are packed row-major signed INT8;
C is packed row-major little-endian INT32 with a four-byte-aligned base. The
entire A and B tiles must fit their SRAM pages, and `M <= MAX_M`. The DMA rejects
zero dimensions, capacity violations, overflowing 64-bit byte ranges, a C
matrix larger than the core's 32-bit word address space, and unaligned C bases
before issuing any AXI request. Descriptor inputs can change after acceptance.

Each command loads A, then B, starts the core, and writes results. It accepts
the next descriptor only after all writes have returned B responses. The
compute engine uses zero-based logical operand addresses; physical AXI byte
addresses remain separate. `WRITE_INTERMEDIATES=0` (default) writes only final-K
C values. Setting it to 1 also stores every cumulative K-fold write, in native
packet order and ascending lane order within each packet. Both modes leave
the same final C matrix in memory.

`AXI_DATA_WIDTH` supports powers of two from 32 through 1024 bits and defaults
to 512. `MAX_BURST_BEATS` is 1–256, default 16. Reads use aligned full-width INCR
bursts split at the configured limit and at 4 KiB boundaries. Unaligned A/B
starts and partial tails are unpacked into the correct local SRAM offsets.
Reads can overfetch within their first/last full bus beat, so those entire
aligned beats must be mapped and readable. This interface is intended for
memory, not read-sensitive MMIO.

Writes are aligned full-width single-beat transactions with exactly four byte
strobes selecting one INT32 C word; other bytes are preserved. AW and W progress
independently, and their address/data/strobes stay stable under backpressure.
The design permits one outstanding transaction and one implicit ID. At a
platform boundary, tie transaction IDs to zero and supply the platform's
constant protection/cache/QoS sidebands; these sidebands are not ports of this
module. The generated burst boundaries, strobes and channel stability follow
the [AMD AXI protocol checks](https://docs.amd.com/r/en-US/pg101-axi-protocol-checker/AXI-Protocol-Checks-and-Descriptions).

| Status | Meaning |
| --- | --- |
| `busy`, `done`, `error` | Busy covers loading, compute and acknowledged stores. Done/error pulse for one clock. |
| `error_code` | Sticky until the next descriptor/reset: 0 success, 1 invalid descriptor, 2 RRESP error, 3 RLAST/count mismatch, 4 BRESP error, 5 core error. Only OKAY responses are accepted as success; exclusive transactions are not used. |
| `total_cycles` | Clocks after descriptor acceptance through completion, including DMA and write responses. |
| `operand_bytes` | Useful bytes actually loaded into SRAM, excluding AXI overfetch. |
| `result_bytes` | Four bytes per successful write response, including intermediate stores when enabled. An error response can still have changed external memory. |
| `logical_cycle`, `core_elapsed_cycles`, `core_stall_cycles`, `cycle_fire` | The core's committed native demand rows and physical compute counters; descriptor acceptance clears their previous values. |

On a read response error, the DMA drains the outstanding burst's requested
beat count, issues no further requests, resets the local engine and completes
with an error. A write error stops after the outstanding B response. A core
error completes any already-started AW/W/B transaction before resetting the
engine. Treat C as incomplete after an error. A detected RLAST mismatch is a
protocol violation; recovery assumes the target still supplies the requested
number of beats. There is no timeout for a target that stops responding.

`rst` must be shared with the attached AXI fabric/target when cancelling an
outstanding transaction. It clears master state and immediately suppresses
handshakes. Internal error recovery instead drains legal outstanding AXI
transactions before local reset. SRAM contents survive either reset, but each
new DMA command reloads both operands.

This implementation is conservative: it unpacks one operand byte per clock,
stores one C word per AXI transaction, and executes one descriptor at a time.
It has no automatic tiling, transfer/compute overlap across descriptors, MMIO
control block, or PCIe shell. No HBM throughput, decode TPS or FPGA frequency is
established by these tests. `make test-hbm2` now attaches a live Ramulator2 HBM2
model at the AXI boundary and checks closed-loop timing, as described below.
An F2 hardware shell/controller integration remains separate work.

## Cycle correspondence

Let `R=ROWS`, `S=COLS`, `Fr=ceil(K/R)`, and `Fc=ceil(N/S)`. Folds execute with
N tiles outermost, K tiles innermost, exactly as upstream. Each fold has

```
L = M + 2*R + S - 2
D = Fr * Fc * L
```

accepted cycles. Number the first demand cycle after command acceptance as
logical cycle zero. In fold `(fr,fc)`, at local cycle `t`:

- Weight lane `c`, `0 <= t < R`, loads `B[fr*R+R-1-t, fc*S+c]`.
- Activation lane `r` reads `A[m, fr*R+r]` at `t=R+r+m`.
- Output lane `c` writes `C[m, fc*S+c]` at `t=2*R-1+c+m`.

Out-of-range rows/columns have invalid lanes. Weight values shift down the
physical PE columns in reverse order; activations travel right and partial sums
travel down. The combinational bottom-row MAC is captured at the requested C
write edge. No precomputed trace, fake completion timer, or behavioral matrix
multiplication is used in synthesizable RTL.

Completion occurs on acceptance of demand row `D-1`; the hardware count is then
`logical_cycle=D`. SCALE-Sim's no-stall compute CSV reports the zero-based last
cycle `D-1`, so compare that report with `logical_cycle-1`. Command acceptance
itself is outside this compute-cycle interval. Including both external and
internal read waits, core elapsed clocks are `D+stall_cycles`. The buffered
top separately counts output drain and completion in `transaction_cycles`.

## What is and is not verified

The Python harness calls the repository's **actual**
`scalesim.compute.systolic_compute_ws.get_demand_matrices()` for expected lane
masks/addresses, then simulates the RTL and checks every proposal. Its numeric
oracle independently computes every cumulative K-prefix GEMM with int64 NumPy
arithmetic, cast to int32. The suite covers 1×1, rectangular arrays, ragged K/N,
signed extreme operands, consecutive commands, the default 32×32 array, and
random all-interface stalls. Both intermediate and final outputs are checked. A separate reviewer-written
`tests/review_checks.py` additionally tests changes of shape/base without reset,
reset aborts while stalled in several pipeline phases, poisoned invalid data
lanes, rejected commands, and valid word address `0xffffffff`. `make test` runs
both suites plus a first-fold wide-command check: `(M,K,N)=(512,6912,262144)`
with all 606 first-fold demand cycles and 16,384 numeric outputs. This wide
command check deliberately resets before the remaining folds and does not
claim a full wide GEMM run. `make gemma` separately checks GEMMs
`(M,K,N)=(1,1152,32)` and `(512,1152,32)` on the actual 32×32 array,
including every K fold; their no-stall demand lengths are 3420 and 21816
cycles, respectively. The longer prefill case uses Verilator compiled simulation; the decode case
uses Icarus Verilog. These are bounded column chunks, not full inference.
The 64-bit counters wrap only on their natural unsigned overflow.

`make test-memory` adds three layers of verification:

- `tests/sync_psum_checks.py` checks native demand rows, NumPy cumulative sums,
  exact internal read waits, overlap with external stalls, and reset recovery
  for clocked partial sums, including the 32×32 array.
- `tests/banked_buffer_checks.py` checks exact synchronous read latency, data,
  lowest-lane arbitration, bank conflicts, duplicates, both pages, invalid and
  empty requests, stalled responses, read-first collisions and reset without
  clearing SRAM. Bank counts include 1, 2, 3, 4 and 32.
- `tests/buffered_top_checks.py` checks every committed native row and every
  drained output packet against independent NumPy results, page ownership,
  overlapped host loading, changed command inputs, page reuse, invalid commands,
  output FIFO pressure, reset/abort recovery and physical transaction timing.

Reports are in `build/sync_psum_checks/review.json`,
`build/banked_buffer_checks/results.json` and
`build/buffered_top_checks/results.json`. `make synth-memory` additionally
requires clocked read/write ports on every operand and partial-sum bank,
expected storage capacities, and no inferred latches. Its report is
`build/buffered_synthesis_checks/results.json`. These checks preserve generic
RAMs; they do not substitute for a vendor implementation or timing closure.

`make test-axi` checks six configurations with 32-, 64-, 128-, 512- and 1024-bit
AXI buses, burst limits 1/3/4/16/256, and both final-only and intermediate stores.
An independent slave model randomizes channel timing, accepts AW/W in either
order and together, delays read and write responses, and checks full-width
alignment, 4 KiB boundaries and every write strobe. Every committed core row
matches the actual native WS generator, and all intermediate values and final
memory contents match NumPy. Guard bytes across the mapped memory are checked
for unintended writes. The suite also checks page flips, descriptor changes,
high addresses through byte `0xffffffffffffffff`, invalid descriptors,
RRESP/BRESP errors, RLAST/count mismatches with the requested beat count still
delivered, a core failure with an outstanding write, and shared reset
during reads, split AW/W transfers and delayed B responses. The report is
`build/axi_top_checks/results.json`; parameter lint results are in
`build/axi_top_checks/lint_results.json`. Generic synthesis additionally checks
both the default and a small AXI top without losing clocked SRAM inference.

## Live HBM2 co-simulation and system timing

```sh
# From the SCALE-Sim repository root; native dependencies are needed once.
bash experiments/gemma3_1b_int8/hbm2/setup_ramulator2.sh
make -C src_rtl test-hbm2
# Run one bounded configuration:
env -u PYTHONPATH .venv/bin/python src_rtl/tests/hbm2_cosim_checks.py --case 4
```

`cosim/build.py` compiles the actual AXI RTL with Verilator, plus a small C ABI
around pinned Ramulator2. `cosim/runtime.py` advances both models on a common
integer-picosecond timeline. A request enters Ramulator only after its AXI
handshake. Controller rejection retains the request for retry. Native read
callbacks gate RVALID; a reorder queue preserves AXI burst order. AW and W can
arrive independently. Byte strobes update functional memory, and write responses
feed back through the output FIFO to stall the compute array. The simulated
memory and DRAM row/refresh state persist across consecutive descriptors.

This is a **single 256-bit AXI master**, with one admission per DRAM tick,
one read burst or write outstanding in the current DMA, and a maximum 16-beat
read burst. Read beats within a burst can be outstanding concurrently. It uses
the existing F2-derived 16 GiB / 16-channel HBM2 profile (or a reduced one-channel
stress configuration), with 32-byte channel striping. This is not the aggregate
32-port F2 interface or its literal address map. The AXI master and core share
the selected clock; no FPGA clock-domain crossing is modeled. The 1 GHz case
is a clock-conversion stress test, not a hardware frequency claim.

Ramulator HBM12 **write callbacks indicate command issue**, so BVALID waits an
additional `nCWL+nBL+nWR = 24` DRAM ticks. This is an explicit conservative bridge
completion contract, not a measured AMD controller response latency or a claim
of persistent storage. Coalesced writes are rejected by this bridge: its serial
write/response policy must prevent that ambiguity. Partial writes assume byte
enables without ECC read-modify-write traffic. Read callbacks are completion
callbacks. No fixed-latency replacement or pre-recorded memory trace is used.

`cosim/reference.py` is an independent Python executable contract. It takes
native SCALE-Sim WS demand matrices and NumPy cumulative products, then models
bank occupancy latency, synchronous partial sums, the two-packet output queue,
DMA loading and AXI handshakes. It does not read RTL internal state. Each test
runs this reference against a fresh real Ramulator instance and compares every
clock's valid bus payloads, control/counters, every accepted native compute row,
all AXI events, HBM admissions/callbacks, final C bytes and untouched guard bytes.
The two runs share the transport implementation; independent event audits check
response causality, write recovery, admission timing, ordering and channel
stability. This is differential testing of bounded workloads, not a formal proof.

The suite includes 4x4 and 32x32 arrays, ragged dimensions, page flips and
consecutive commands (including an input read from a preceding command's C),
4 KiB splits, unaligned inputs, output backpressure,
controller queue saturation, reordered callbacks, different clock ratios and
an explicit read-delay sensitivity check. The default-array workloads include
`(M,K,N)=(1,1152,16)` and `(8,65,33)`. These are GEMM tiles, not token execution.
Bridge checks also cover malformed configurations and the last legal 32-byte
transaction in 16 GiB. Reports are under `build/hbm2_cosim_checks/`, with resolved
HBM configs, bus/native event logs, source/config hashes and per-command cycles.

The validated suite completes nine GEMMs in five configurations, comparing
54,546 clocks and 1,238 HBM requests, plus ten bridge checks. The `(1,1152,16)`
tile takes 34,029 total clocks, or 136.116 microseconds at the assumed 250 MHz;
13,293 are compute-active clocks, including 9,873 stalls. The one-channel
read-delay sensitivity case increases from 1,306 to 1,467 total core clocks
when adding 80 DRAM ticks to read delivery. These measurements describe the
current conservative DMA and selected workloads; they are not decode TPS/TTFT.

**Equivalence scope:** the implemented WS/SRAM/FIFO/DMA/HBM system matches its
independent hardware-policy reference cycle by cycle on these tests. Stock
SCALE-Sim CALC/USER memory timing is still **not equivalent**. The reports set
`native_scalesim_memory_policy_equivalent=false`; `native_policy_comparison.json`
runs the unmodified native memory model and records the first differing compute
commit edge, with initial RTL DMA loading excluded. Native prefetch/active
windows, nominal SRAM-hit subtraction and ofmap buffering are different from
this implementation. A match to those stock policies requires implementing
those policies in hardware or explicitly selecting a hardware-policy backend
in the simulator; changing a completion counter would not establish equivalence.

External ready stalls preserve logical demand equivalence. They **do not claim
to reproduce** SCALE-Sim's SRAM double buffering, prefetch cycles, CALC/USER
bandwidth model, native bank/layout policies, or report cycles including
prefetch. The live HBM2 suite separately verifies Ramulator feedback and complete
system timing for the implemented hardware policy. Matching the stock simulator's
policies still requires additional memory logic and differential tests.

Other remaining extensions for broader v3/OpenTPU-like functionality are OS/IS
schedulers and datapaths, sparse formats/mapping, address generation for native
convolution (or a host im2col adapter), multicore scheduling, optimized DMA,
PCIe/platform integration, an instruction/runtime stack, and transformer operations such as
normalization, RoPE, softmax, activation functions and token selection. This
engine alone does not run Gemma inference or establish decode TPS/TTFT.
