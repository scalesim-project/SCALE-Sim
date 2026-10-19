"""Resolved HBM2 configuration based on the published AWS F2 HBM IP snapshot.

HBM geometry/timing are matched to public parameters. The generic Ramulator
controller and striped address layout are NOT AMD's proprietary controller,
AXI crossbar, PHY, or F2 software memory map.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
VENDOR = ROOT / "submodules" / "ramulator2"
RAMULATOR_PIN = "c6c7a5de0cef35935d6fad7369e6271406798865"
AWS_HDK_PIN = "c34f88e97eaf8a9b6ea7a3516ab65ee64690ab82"
AWS_IP_PIN = "6d32be972e6da854e61a8d3d6ec0466ab491c1b3"
AWS_XCI_URL = f"https://raw.githubusercontent.com/aws/aws-fpga-resources/{AWS_IP_PIN}/cl_ip/cl_ip.srcs/sources_1/ip/cl_hbm/cl_hbm.xci"

# These are memory-clock counts, not nanoseconds. See f2_provenance.json for
# the selected raw USER_* values and their source hash.
XCI_TIMINGS = dict(nCL=22, nCWL=7, nRC=43, nRAS=30, nRCDRD=13, nRCDWR=9,
                   nRRDL=4, nRRDS=4, nFAW=15, nRTPL=5, nRP=13, nWR=15,
                   nWTRL=8, nWTRS=4, nRFC=315, nRFCpb=144, nRREFD=8,
                   nREFI=3510)

# Parameters absent from the selected XCI fields use documented model choices.
# nRTW follows the upstream HBM2 expression with nCL=22,nBL=2,nCWL=7.
# Either neighboring 1600/2000 Mbps tDQSQ bound gives ceil(term)=4 at 900MHz.
# nCCDR=3 comes from the AMD model shipped with the same AWS IP snapshot.
MODEL_TIMINGS = dict(rate=1800, tCK_ps=1111, nBL=2, nCCDS=2, nCCDL=4,
                    nCCDR=3, nRTW=21, nREFIpb=3510 // 32)


def make_f2_config(channels=16):
    if channels not in (1, 2, 4, 8, 16):
        raise ValueError("physical channel count must be 1, 2, 4, 8 or 16")
    if not (VENDOR / "python" / "ramulator").is_dir():
        raise RuntimeError("Run hbm2/setup_ramulator2.sh to install the pinned backend")
    vendor_python = str(VENDOR / "python")
    if vendor_python not in sys.path:
        sys.path.insert(0, vendor_python)
    import ramulator

    # Override EVERY timing parameter; the base preset is only the API's
    # required constructor seed. No 2000Mbps-derived timing survives here.
    timings = dict(XCI_TIMINGS, **MODEL_TIMINGS)
    expected = set(ramulator.dram.HBM2.timing_params)
    if set(timings) != expected:
        raise ValueError(f"HBM2 timing schema changed: {set(timings) ^ expected}")
    dram = ramulator.dram.HBM2(org_preset="HBM2_4Gb", timing_preset="HBM2_2000Mbps",
                              sid=2, row=16384, die_density=8192, stack_height=8,
                              **timings)
    controller = ramulator.controller.HBM12(
        dram=dram, scheduler=ramulator.scheduler.FRFCFS(),
        refresh_manager=ramulator.refresh_manager.AllBank(),
        row_policy=ramulator.row_policy.Open(),
        addr_mapper=ramulator.addr_mapper.RoBaRaCoCh())
    memory = ramulator.memory_system.GenericDRAM(
        clock_ratio=1, controllers=[controller] * channels,
        channel_mapper=ramulator.channel_mapper.CacheLineInterleave(interleave_bits=0))
    config = dict(frontend=ramulator.frontend.External(clock_ratio=1).to_config(),
                  memory_system=memory.to_config())
    org, resolved = dram.resolve()
    if resolved != timings:
        raise ValueError("Requested F2 timings were not preserved by the serializer")
    profile = dict(
        name="AWS F2 published HBM2 geometry/timing with generic Ramulator controller",
        full_f2_memory=channels == 16, standard="HBM2", physical_channels=channels,
        pseudochannels=channels * 2, capacity_bytes=channels * 1024**3,
        stacks=2 if channels == 16 else None,
        nominal_dram_clock_hz=900000000, nominal_data_rate_Mbps=1800,
        axi_ports=32 if channels == 16 else channels * 2, axi_data_bits=256,
        nominal_axi_clock_hz=450000000, nominal_peak_GB_s=channels * 28.8,
        transaction_bytes=32, timing_cycles=copy.deepcopy(timings), organization=org,
        modeled_dram_tck_ns=1.111, modeled_dram_clock_hz=1e12 / 1111,
        clock_quantization="Ramulator stores integer picoseconds: 1111ps vs nominal 10000/9ps (about 100ppm faster)",
        sources=[AWS_XCI_URL,
                 "https://github.com/aws/aws-fpga/blob/f2/hdk/cl/examples/cl_mem_perf/README.md",
                 "https://docs.amd.com/r/en-US/pg276-axi-hbm/HBM-Address-Map-and-Protocol-Considerations"],
        unmodeled=["AMD proprietary controller scheduling, physical address mapping, AXI crossbar and CDC delays",
                   "Finite FPGA DMA resources and AXI port backpressure",
                   "ECC/data parity behavior and physical HBM PHY/calibration",
                   "Closed-loop accelerator stalls and non-GEMM execution"],
        assumptions=dict(controller="HBM12 with FRFCFS, open-page, all-bank refresh; 32-entry read/write buffers",
                         latency_mapping="USER_RL=22 -> nCL=22 and USER_WL=7 -> nCWL=7 are interpreted as decoded cycle latencies; the XCI-to-mode-register encoding has not been independently verified",
                         address_layout="32-byte striping across physical channels; native RoBaRaCoCh within channel",
                         model_timing_derivations=copy.deepcopy(MODEL_TIMINGS),
                         bank_geometry="2 SID x 4 bank groups x 4 banks x 16384 rows per pseudochannel"))
    return config, profile


if __name__ == "__main__":
    native_config, profile = make_f2_config()
    (HERE / "F2-HBM2.json").write_text(json.dumps(native_config, indent=2) + "\n")
    (HERE / "F2-profile.json").write_text(json.dumps(profile, indent=2) + "\n")
