#!/usr/bin/env python3
"""Replay captured SCALE-Sim memory requests using F2-shaped HBM2 in Ramulator2.

Offline memory replay, not a new full-model TPS/TTFT prediction or exact model
of AMD's proprietary HBM controller. Defaults to a 250MHz core reference clock
and the published F2 16GiB/32-pseudochannel memory organization.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess

from f2_config import make_f2_config, RAMULATOR_PIN, VENDOR

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location("hbm_shared", HERE.parent / "hbm" / "run_hbm.py")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--k", type=int, default=1152)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--cols", type=int, default=32)
    parser.add_argument("--channels", type=int, default=16, choices=(1, 2, 4, 8, 16),
                        help="Physical channels; 16 channels / 32 PCs is the full F2 profile")
    parser.add_argument("--core-hz", type=int, default=250000000)
    parser.add_argument("--max-dram-cycles", type=int, default=100000000)
    parser.add_argument("--max-injection", type=int, default=16,
                        help="Aggregate admitted 32-byte requests per DRAM tick; 16 approximates the full F2 AXI data-rate limit")
    parser.add_argument("--out", type=Path, default=HERE / "build" / "gemma_tile_f2")
    args = parser.parse_args()
    if min(args.m, args.n, args.k, args.rows, args.cols, args.core_hz, args.max_dram_cycles, args.max_injection) <= 0:
        parser.error("Dimensions, clock frequency and cycle limit must be positive")
    demand_rows = math.ceil(args.k / args.rows) * math.ceil(args.n / args.cols) * (args.m + 2 * args.rows + args.cols - 2)
    if demand_rows * (args.rows + 2 * args.cols) > 10000000:
        parser.error("Native demand matrices exceed this bounded experiment's allocation limit")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    native_config, profile = make_f2_config(args.channels)
    actual_pin = subprocess.check_output(["git", "-C", str(VENDOR), "rev-parse", "HEAD"], text=True).strip()
    if actual_pin != RAMULATOR_PIN:
        raise RuntimeError(f"Unexpected Ramulator2 revision {actual_pin}")
    shared.save_json(args.out / "F2-HBM2.json", native_config)
    shared.save_json(args.out / "F2-profile.json", profile)
    captured, native_report = shared.capture_native(args, args.out)
    elements = dict(ifmap=args.m * args.k, filter=args.k * args.n, ofmap=args.m * args.n)
    requests, manifest, origin = shared.convert_rows(captured, elements, burst_bytes=32)
    request_path = args.out / "requests.csv"
    with request_path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=("id", "arrival_core_cycle", "address_bytes", "operation"), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(requests)
    subprocess.run(["bash", str(HERE / "build_ramulator_hbm2.sh")], cwd=ROOT, check=True)
    command = [str(HERE / "build" / "ramulator_hbm2"), str(args.out / "F2-HBM2.json"),
               str(request_path), str(args.out / "responses.csv"), str(args.out / "ramulator.stats"),
               str(args.core_hz), str(args.max_dram_cycles), str(args.max_injection)]
    completed = subprocess.run(command, cwd=args.out, text=True, capture_output=True)
    (args.out / "ramulator.log").write_text(completed.stdout + completed.stderr)
    completed.check_returncode()
    frontend = json.loads((args.out / "ramulator.stats.summary.json").read_text())
    if (frontend["transaction_bytes"] != 32 or frontend["channels"] != args.channels or
            frontend["capacity_bytes"] != profile["capacity_bytes"] or
            not math.isclose(frontend["tck_ns"], 1.111, abs_tol=1e-9)):
        raise ValueError("Unexpected HBM2 frontend units or geometry")
    streams = shared.analyze_responses(requests, manifest, args.out / "responses.csv",
                                       frontend["tck_ns"], args.core_hz, burst_bytes=32)
    for values in streams.values():
        # Ramulator2 can merge writes and forward reads. Submitted payload is
        # distinct from actual DRAM commands (reported by the C++ frontend).
        values["requested_transaction_bytes"] = values.pop("bus_bytes")
    shared.save_json(args.out / "rows.json", manifest)
    vendor_diff = subprocess.check_output(["git", "-C", str(VENDOR), "diff", "--no-ext-diff", "HEAD"], text=True)
    report = dict(mode="offline timestamped HBM2 replay; not closed-loop F2 accelerator simulation",
                  workload=dict(m=args.m, n=args.n, k=args.k, rows=args.rows, cols=args.cols,
                                description="Default is one transposed Gemma decode output-channel tile"),
                  source=dict(clock_hz=args.core_hz, source_cycle_origin=origin,
                              max_admitted_requests_per_dram_tick=args.max_injection,
                              capture="Actual native CALC memory-port input requests",
                              original_native_compute_report=native_report,
                              physical_sram_capacity_bytes=dict(ifmap=65536, filter=65536, ofmap=65536)),
                  hbm2=profile, frontend=frontend, streams=streams,
                  mapping=dict(source_word_bases=shared.SOURCE_BASES,
                               physical_byte_bases=shared.PHYSICAL_BASES, word_bytes=shared.WORD_BYTES,
                               coalescing="32-byte bursts within each source row; no cross-row cache"),
                  limitations=profile["unmodeled"] + [
                      "No updated Gemma TPS/TTFT is inferred from this one-GEMM memory replay.",
                      "250MHz core is an F2 reference setting, not measured RTL timing closure; AXI and DRAM clocks are separate.",
                      "Repeated bursts in distinct rows are charged again; DMA line buffering/layout can change traffic substantially.",
                      "Write callbacks report issue/merge, not data persistence; completed_dram_cycle stays -1.",
                      "Partial writes assume byte enables, without ECC read-modify-write traffic."],
                  provenance=dict(ramulator2_commit=actual_pin,
                                  ramulator2_tracked_patch_sha256=hashlib.sha256(vendor_diff.encode()).hexdigest(),
                                  ramulator2_has_tracked_patch=bool(vendor_diff),
                                  scalesim_commit=subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                                  requests_sha256=hashlib.sha256(request_path.read_bytes()).hexdigest(),
                                  config_sha256=hashlib.sha256((args.out / "F2-HBM2.json").read_bytes()).hexdigest()))
    shared.save_json(args.out / "report.json", report)
    print(json.dumps(dict(report=str(args.out / "report.json"), transactions=len(requests),
                          core_hz=args.core_hz, physical_channels=args.channels,
                          pseudochannels=profile["pseudochannels"], capacity_bytes=profile["capacity_bytes"],
                          nominal_peak_GB_s=profile["nominal_peak_GB_s"],
                          dram_cycles=frontend["dram_cycles"], streams=streams), indent=2))


if __name__ == "__main__":
    main()
