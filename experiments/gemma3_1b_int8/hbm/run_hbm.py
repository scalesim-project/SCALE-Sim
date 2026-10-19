#!/usr/bin/env python3
"""Capture a bounded native SCALE-Sim GEMM and replay its traffic through HBM.

This is timestamp-preserving OFFLINE replay, not closed-loop co-simulation.
Ramulator backpressure changes memory acceptance times, not SCALE-Sim's
previously captured compute schedule. Consequently this tool reports memory
behavior and does not calculate Gemma TPS or end-to-end execution time.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import configparser
from contextlib import redirect_stdout, redirect_stderr
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BURST_BYTES = 64
REGION_BYTES = 16 * 1024 * 1024
STREAMS = ("ifmap", "filter", "ofmap")
SOURCE_BASES = dict(ifmap=0, filter=10000000, ofmap=20000000)
PHYSICAL_BASES = dict(ifmap=0, filter=REGION_BYTES, ofmap=2 * REGION_BYTES)
WORD_BYTES = dict(ifmap=1, filter=1, ofmap=4)


def exact_integer(value, label):
    result = int(value)
    if result != value:
        raise ValueError(f"{label} must be integral, got {value}")
    return result


def convert_rows(captured, elements, burst_bytes=BURST_BYTES):
    """Map abstract words to bytes and coalesce only within each source row.

    Repeated bursts on different rows remain separate transactions. Sub-burst
    writes assume byte enables, with the full burst charged to the memory bus;
    no read-modify-write/ECC operation is modeled.
    """
    if burst_bytes <= 0 or burst_bytes & (burst_bytes - 1):
        raise ValueError("burst_bytes must be a positive power of two")
    manifest, requests = [], []
    for stream in STREAMS:
        if elements[stream] * WORD_BYTES[stream] > REGION_BYTES:
            raise ValueError(f"{stream} exceeds its {REGION_BYTES}-byte physical region")
        for row_no, (cycle, addresses) in enumerate(captured[stream]):
            cycle = exact_integer(cycle, "source cycle")
            bursts = set()
            useful = set()
            words = 0
            for raw in addresses:
                address = exact_integer(raw, "word address")
                if address == -1:
                    continue
                relative = address - SOURCE_BASES[stream]
                if not 0 <= relative < elements[stream]:
                    raise ValueError(f"{stream} word {address} outside its logical tensor")
                words += 1
                first = PHYSICAL_BASES[stream] + relative * WORD_BYTES[stream]
                for byte in range(first, first + WORD_BYTES[stream]):
                    useful.add(byte)
                    bursts.add(byte // burst_bytes * burst_bytes)
            row = dict(stream=stream, source_row=row_no, source_cycle=cycle,
                       requested_words=words, unique_requested_bytes=len(useful),
                       request_ids=[])
            row_index = len(manifest)
            manifest.append(row)
            for address in sorted(bursts):
                requests.append(dict(source_cycle=cycle, address_bytes=address,
                                     operation="W" if stream == "ofmap" else "R",
                                     row_index=row_index))
    # Keep idle gaps, and use ONE origin for all three streams.
    origin = min((r["source_cycle"] for r in requests), default=0)
    requests.sort(key=lambda r: (r["source_cycle"], r["row_index"], r["address_bytes"]))
    for request_id, request in enumerate(requests):
        request["id"] = request_id
        request["arrival_core_cycle"] = request["source_cycle"] - origin
        manifest[request["row_index"]]["request_ids"].append(request_id)
    return requests, manifest, origin


def capture_native(args, out):
    """Intercept actual port requests, not the exported response-time CSVs."""
    import numpy as np
    from scalesim.memory.read_port import read_port
    from scalesim.memory.write_port import write_port
    from scalesim.scale_sim import scalesim

    cfg = configparser.ConfigParser()
    cfg.read(HERE.parent / "architecture.cfg")
    cfg["general"]["run_name"] = "hbm_capture"
    for key, value in dict(arrayheight=args.rows, arraywidth=args.cols,
                           ifmapsramszkb=64, filtersramszkb=64,
                           # SCALE counts abstract one-byte words. 16K output
                           # words represent 64 KiB of physical INT32 storage.
                           ofmapsramszkb=16).items():
        cfg["architecture_presets"][key] = str(value)
    cfg["run_presets"]["interfacebandwidth"] = "CALC"
    cfg["run_presets"]["useramulatortrace"] = "False"
    config_path = out / "scale.cfg"
    with config_path.open("w") as f:
        cfg.write(f)
    (out / "topology.csv").write_text(f"Layer name,M,N,K,\nhbm_tile,{args.m},{args.n},{args.k},\n")
    (out / "layout.csv").write_text("")
    by_port = defaultdict(list)

    def record(port, addresses, cycles):
        addresses = np.asarray(addresses)
        cycles = np.asarray(cycles).reshape(-1)
        if addresses.ndim != 2 or addresses.shape[0] != len(cycles):
            raise ValueError("Unexpected native port request dimensions")
        # tolist copies immediately, before upstream can reuse any array.
        by_port[id(port)].extend(zip(cycles.tolist(), addresses.tolist()))

    original_read = read_port.service_reads
    original_write = write_port.service_writes

    def read_wrapper(port, incoming_requests_arr_np, incoming_cycles_arr):
        record(port, incoming_requests_arr_np, incoming_cycles_arr)
        return original_read(port, incoming_requests_arr_np, incoming_cycles_arr)

    def write_wrapper(port, incoming_requests_arr_np, incoming_cycles_arr_np):
        record(port, incoming_requests_arr_np, incoming_cycles_arr_np)
        return original_write(port, incoming_requests_arr_np, incoming_cycles_arr_np)

    with (out / "scale.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
        native = scalesim(save_disk_space=True, verbose=False, config=str(config_path),
                          topology=str(out / "topology.csv"), layout=str(out / "layout.csv"),
                          input_type_gemm=True)
        with patch.object(read_port, "service_reads", read_wrapper), \
             patch.object(write_port, "service_writes", write_wrapper):
            native.run_scale(top_path=str(out / "native"))
    layer = native.runner.single_layer_sim_object_list[0]
    ports = layer.memory_system
    captured = {s: by_port[id(getattr(ports, f"{s}_port"))] for s in STREAMS}
    if not all(captured.values()):
        raise RuntimeError("Expected all three native memory streams")
    return captured, [float(x) for x in layer.get_compute_report_items()]


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def analyze_responses(requests, manifest, response_path, tck_ns, core_hz, burst_bytes=BURST_BYTES):
    with response_path.open() as f:
        responses = list(csv.DictReader(f))
    by_id = {int(r["id"]): r for r in responses}
    if len(by_id) != len(responses) or set(by_id) != {r["id"] for r in requests}:
        raise ValueError("Ramulator response IDs do not exactly match submitted requests")
    by_request = {r["id"]: r for r in requests}
    stats = {}
    for stream in STREAMS:
        request_ids = [i for row in manifest if row["stream"] == stream for i in row["request_ids"]]
        latencies, queue_delays = [], []
        for request_id in request_ids:
            r, q = by_id[request_id], by_request[request_id]
            accepted = int(r["accepted_dram_cycle"])
            completed = int(r["completed_dram_cycle"])
            arrival_ns = q["arrival_core_cycle"] * 1e9 / core_hz
            if (int(r["arrival_core_cycle"]) != q["arrival_core_cycle"] or
                    r["operation"] != q["operation"] or
                    int(r["address_bytes"]) != q["address_bytes"]):
                raise ValueError("Response metadata mismatch")
            if accepted * tck_ns + 1e-8 < arrival_ns:
                raise ValueError("Ramulator accepted a request before its arrival")
            queue_delays.append(accepted * tck_ns - arrival_ns)
            if q["operation"] == "R":
                if completed < accepted:
                    raise ValueError("Missing or premature read completion")
                latencies.append(completed * tck_ns - arrival_ns)
            elif completed != -1:
                raise ValueError("Unexpected write completion semantics; review the frontend")
        stats[stream] = dict(transactions=len(request_ids), bus_bytes=len(request_ids) * burst_bytes,
                            mean_admission_delay_ns=sum(queue_delays) / len(queue_delays) if queue_delays else 0,
                            max_admission_delay_ns=max(queue_delays, default=0))
        if latencies:
            stats[stream].update(mean_read_latency_ns=sum(latencies) / len(latencies),
                                 max_read_latency_ns=max(latencies))
    # Source rows have exact transaction IDs; padded rows do not consume memory
    # transactions. Read row completion is the maximum of its burst completions.
    for row in manifest:
        if row["stream"] != "ofmap" and row["request_ids"]:
            done = max(int(by_id[i]["completed_dram_cycle"]) for i in row["request_ids"])
            row["read_completed_core_cycle_from_origin"] = math.ceil(done * tck_ns * core_hz / 1e9)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--k", type=int, default=1152)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--cols", type=int, default=32)
    parser.add_argument("--channels", type=int, default=8, choices=(1, 2, 4, 8))
    parser.add_argument("--core-hz", type=float, default=1e9)
    parser.add_argument("--max-dram-cycles", type=int, default=100000000)
    parser.add_argument("--out", type=Path, default=HERE / "build" / "gemma_tile_8ch")
    args = parser.parse_args()
    if min(args.m, args.n, args.k, args.rows, args.cols) <= 0 or not math.isfinite(args.core_hz) or args.core_hz <= 0:
        parser.error("Dimensions and core frequency must be positive")
    # This native runner deliberately bounds demand-matrix allocation.
    demand_rows = math.ceil(args.k / args.rows) * math.ceil(args.n / args.cols) * (args.m + 2 * args.rows + args.cols - 2)
    if demand_rows * (args.rows + 2 * args.cols) > 10000000:
        parser.error("Native demand matrices exceed this bounded experiment's allocation limit")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    captured, native_report = capture_native(args, args.out)
    elements = dict(ifmap=args.m * args.k, filter=args.k * args.n, ofmap=args.m * args.n)
    requests, manifest, origin = convert_rows(captured, elements)
    fields = ("id", "arrival_core_cycle", "address_bytes", "operation")
    request_path = args.out / "requests.csv"
    with request_path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(requests)
    config_path = args.out / "HBM.cfg"
    config_path.write_text((HERE / "HBM-8ch.cfg").read_text().replace("channels = 8", f"channels = {args.channels}"))
    subprocess.run(["bash", str(HERE / "build_ramulator_hbm.sh")], cwd=ROOT, check=True)
    command = [str(HERE / "build" / "ramulator_hbm"), str(config_path), str(request_path),
               str(args.out / "responses.csv"), str(args.out / "ramulator.stats"),
               str(args.core_hz), str(args.max_dram_cycles), "64"]
    result = subprocess.run(command, cwd=args.out, text=True, capture_output=True)
    (args.out / "ramulator.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    frontend = json.loads((args.out / "ramulator.stats.summary.json").read_text())
    # Fail rather than interpreting an incompatible frontend/model's units.
    if (frontend["tck_ns"] != 2.0 or frontend["transaction_bytes"] != BURST_BYTES or
            frontend["channels"] != args.channels):
        raise ValueError("HBM frontend units/configuration differ from the adapter contract")
    memory_stats = analyze_responses(requests, manifest, args.out / "responses.csv", frontend["tck_ns"], args.core_hz)
    save_json(args.out / "rows.json", manifest)
    report = dict(mode="offline timestamped shared-memory replay; not closed-loop co-simulation",
                  workload=dict(m=args.m, n=args.n, k=args.k, rows=args.rows, cols=args.cols,
                                description="Default is a transposed Gemma decode projection output tile"),
                  source=dict(capture="Actual native read/write port INPUT requests, copied during CALC simulation",
                              source_cycle_origin=origin, clock_hz=args.core_hz,
                              original_native_compute_report=native_report,
                              physical_sram_capacity_bytes=dict(ifmap=65536, filter=65536, ofmap=65536)),
                  hbm=dict(standard="HBM", speed="HBM_1Gbps", org="HBM_4Gb", channels=args.channels,
                           ranks=1, dram_tck_ns=2.0, transaction_bytes=BURST_BYTES,
                           theoretical_peak_GB_s=16 * args.channels,
                           frontend_max_accepted_per_dram_tick=64),
                  mapping=dict(source_word_bases=SOURCE_BASES, physical_byte_bases=PHYSICAL_BASES,
                               word_bytes=WORD_BYTES, coalescing="Within each source row only; no cross-row cache"),
                  streams=memory_stats,
                  frontend=frontend,
                  limitations=["Memory delay is not fed back to regenerate compute requests; no updated TPS/TTFT is claimed.",
                               "Only one bounded GEMM, not full Gemma inference. Initial prefetch is included in the memory timeline.",
                               "INT32 outputs are assumed; quantization/scales/KV handling and DMA layout conversion are not modeled.",
                               "Repeated bursts on distinct rows are charged again; a real DMA line buffer may reduce this traffic.",
                               "Partial writes assume byte enables without ECC read-modify-write.",
                               "Write completion is unavailable in upstream callbacks; -1 denotes unavailable, not zero latency.",
                               "HBM1 model, not HBM2/HBM3. Peak bandwidth is not achieved bandwidth."],
                  provenance=dict(ramulator_commit=subprocess.check_output(["git", "-C", str(ROOT / "submodules/ramulator"), "rev-parse", "HEAD"], text=True).strip(),
                                  scalesim_commit=subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                                  requests_sha256=hashlib.sha256(request_path.read_bytes()).hexdigest()))
    save_json(args.out / "report.json", report)
    print(json.dumps(dict(report=str(args.out / "report.json"), transactions=len(requests),
                          hbm=report["hbm"], streams=memory_stats), indent=2))


if __name__ == "__main__":
    main()
