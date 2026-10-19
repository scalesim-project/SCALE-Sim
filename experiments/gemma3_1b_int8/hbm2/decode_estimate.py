#!/usr/bin/env python3
"""Offline matrix/traffic ceilings for F2-class HBM; not a coupled HBM replay.

Uses the previously validated dense WS schedule with C.T=B.T@A.T and a
512-row limit on the transposed GEMM. Clocks are illustrative assumptions,
not achieved FPGA Fmax. 200/300 GB/s are scenarios, not measured bandwidth.
"""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[1]
sys.path.insert(0, str(STUDY))
import study

SHAPES = [(64, 32), (64, 16), (128, 8), (128, 16), (128, 32), (256, 8), (256, 16),
          (384, 4), (384, 16), (576, 4), (576, 8), (576, 16)]
POINTS = [("normal_ws_baseline", 32, 32)] + [("transposed_ws_tile512", r, c) for r, c in SHAPES]
CLOCKS_HZ = [250_000_000, 450_000_000]
BANDWIDTHS_GB_S = [200.0, 300.0, 460.8]


def transposed_cycles(op, rows, cols):
    return sum(study.demand_cycles(min(512, op.n - start), op.m, op.k, rows, cols)
               for start in range(0, op.n, 512))


def validate():
    import numpy as np
    from scalesim.scale_config import scale_config
    from scalesim.compute.systolic_compute_ws import systolic_compute_ws
    checks = []
    for mapping, rows, cols in POINTS:
        cfg = scale_config()
        cfg.read_conf_file(str(ROOT / "configs/scale.cfg"))
        cfg.array_rows, cfg.array_cols = rows, cols
        k, n = rows + 7, 4
        measured = [0, 0, 0]
        expected = 0
        shapes = [(512, n, k), (7, n, k)] if mapping == "transposed_ws_tile512" else [(4, 519, k)]
        for m, n, k in shapes:
            engine = systolic_compute_ws()
            engine.set_params(cfg, np.zeros((m, k)), np.zeros((m, n)), np.zeros((k, n)))
            measured = [a + x.shape[0] for a, x in zip(measured, engine.get_demand_matrices())]
            expected += study.demand_cycles(m, n, k, rows, cols)
        assert measured == [expected] * 3, (rows, cols, measured, expected)
        checks.append(dict(mapping=mapping, rows=rows, columns=cols, original_mnk=[4, 519, k],
                           native_demand_rows=measured, expected=expected, passed=True))
    return checks


def main():
    base = json.loads((STUDY / "results.json").read_text())
    traffic = base["separate_traffic_roofline"][0]["assumed_bytes_per_token"]
    assert traffic == 1_006_698_496
    prefill = study.workload(512)
    decode = [op for context in range(513, 640) for op in study.workload(512, context)]
    points = []
    for mapping, rows, cols in POINTS:
        decode_cycles = sum(transposed_cycles(op, rows, cols) if mapping == "transposed_ws_tile512"
                            else study.demand_cycles(*op.shape, rows, cols) for op in decode)
        prefill_cycles = sum(study.demand_cycles(*op.shape, rows, cols) for op in prefill)
        transposed_prefill_cycles = sum(transposed_cycles(op, rows, cols) for op in prefill)
        for clock in CLOCKS_HZ:
            matrix_seconds = decode_cycles / 127 / clock
            scenarios = []
            for bandwidth in BANDWIDTHS_GB_S:
                memory_seconds = traffic / (bandwidth * 1e9)
                scenarios.append(dict(
                    bandwidth_GB_s=bandwidth,
                    kind="nominal F2 aggregate peak" if bandwidth == 460.8 else "illustrative sustained bandwidth, not measured",
                    traffic_only_TPS=1 / memory_seconds,
                    traffic_ms=1000 * memory_seconds,
                    perfect_overlap_ceiling_TPS=1 / max(matrix_seconds, memory_seconds),
                    serialized_matrix_plus_traffic_TPS=1 / (matrix_seconds + memory_seconds),
                    serialized_margin_ms_to_30_TPS=1000 / 30 - 1000 * (matrix_seconds + memory_seconds)))
            points.append(dict(mapping=mapping, implementation_status="current default array/dataflow" if mapping == "normal_ws_baseline"
                               else "proposed array and operand/output remapping; not implemented here",
                rows=rows, columns=cols, processing_elements=rows * cols,
                one_MAC_per_DSP_count_within_9024=rows * cols <= 9024,
                assumed_core_hz=clock, ideal_INT8_TOPS=2 * rows * cols * clock / 1e12,
                decode_127_steps_demand_cycles=decode_cycles,
                matrix_decode_mean_ms=matrix_seconds * 1000,
                matrix_decode_TPS=1 / matrix_seconds,
                normal_WS_matrix_TTFT_ms=1000 * prefill_cycles / clock,
                transposed_WS_matrix_TTFT_ms=1000 * transposed_prefill_cycles / clock,
                traffic_scenarios=scenarios))
    with redirect_stdout(io.StringIO()):
        checks = validate()
    result = dict(
        classification="offline analytical ceilings; not measured LLM inference and not a closed-loop HBM simulation",
        model=base["model"], workload=base["workload"], arithmetic="W8A8, INT8 KV",
        decode_mapping="Baseline32x32 uses normal WS. Proposed points use C.T=B.T@A.T with original N tiled to512; WS demand cycles ceil(K/R)*ceil(N/C)*(M+2R+C-2)",
        mean_streamed_weight_plus_unique_KV_bytes_per_token=traffic,
        minimum_bandwidth_GB_s_for_30_TPS=traffic * 30 / 1e9,
        clocks="250MHz shell reference and illustrative450MHz accelerator clock; neither proves this array's achievable Fmax",
        memory="F2 16GiB,32PC,nominal460.8GB/s; 200/300GB/s are illustrative operating points, not measurements",
        limitations=["Excluded: non-GEMM work, quantization, activation/output layout conversions, metadata, KV writes, other activations, host and sampling.",
                     "The weight/KV traffic model does not account for repeated SRAM refills or actual HBM scheduling; use measured native replay separately.",
                     "Perfect overlap is optimistic; serialized matrix+traffic also excludes other overhead and is not a guaranteed lower bound.",
                     "Normal WS prefill with transposed decode needs both weight layouts or a conversion path; all-transposed prefill figures assume one prepacked layout.",
                     "9024-DSP comparison counts only one DSP per MAC as an assumption; it does not establish total resource fit, packing, routability, or achievable clock.",
                     "No synthesized or routed FPGA design and no complete Gemma execution has been measured."],
        points=points, native_demand_validation=checks,
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [STUDY / "study.py", HERE / "decode_estimate.py", ROOT / "scalesim/compute/systolic_compute_ws.py"]},
        reproduce="env -u PYTHONPATH .venv/bin/python experiments/gemma3_1b_int8/hbm2/decode_estimate.py")
    (HERE / "decode_estimate.json").write_text(json.dumps(result, indent=2) + "\n")
    for p in points:
        print(f"{p['mapping']} {p['rows']}x{p['columns']} {p['assumed_core_hz']/1e6:.0f}MHz: "
              f"{p['matrix_decode_TPS']:.9f}TPS matrix; TTFT normal/transposed "
              f"{p['normal_WS_matrix_TTFT_ms']:.6f}/{p['transposed_WS_matrix_TTFT_ms']:.6f}ms; "
              f"serialized200GB/s={p['traffic_scenarios'][0]['serialized_matrix_plus_traffic_TPS']:.9f}TPS")
    print(f"Native demand checks: {len(checks)} passed; traffic bytes/token={traffic}")


if __name__ == "__main__":
    main()
