#!/usr/bin/env python3
"""Reproduce a bounded-memory Gemma 3 1B INT8 matrix-schedule estimate.

From the repository root:
    env -u PYTHONPATH .venv/bin/python experiments/gemma3_1b_int8/study.py

This is an analytical evaluation of the checked-out SCALE-Sim dense WS schedule,
validated against its real demand generator and small complete simulations. It is
not a full native simulation of Gemma, a numerical model evaluation, or measured
end-to-end inference. The LM-head demand matrices alone would exceed 40 GiB.

For C columns, R rows and A[M,K] @ B[K,N], dense WS uses F=ceil(K/R)*ceil(N/C)
folds. Its three demand matrices have D=F*(M+2*R+C-2) rows. D clock periods are
used here. A no-stall native report's 'Total Cycles' is the zero-indexed last
service cycle, D-1; 'Total Cycles (incl. prefetch)' is a different quantity.
CALC calculates bandwidth needed by the schedule; Bandwidth=10 is not a cap.

All generated files stay in this script's directory. --label NAME makes an
independent subdirectory for parameter sweeps. Outputs include complete native
MNK CSVs (with required trailing commas), deduplicated shape counts, a per-stage
breakdown, per-token timings, provenance, and real native validation reports.
Only the small validation topologies are passed to the full native simulator.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import configparser
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL = dict(name="google/gemma-3-1b-it", layers=26, hidden=1152,
             intermediate=6912, query_heads=4, kv_heads=1, head_dim=256,
             vocabulary=262144, sliding_window=512, global_every=6,
             max_context=32768, tied_embeddings=True)
SOURCES = [
    {"url": "https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/_gemma.py",
     "evidence": "Gemma3_1B configuration, 26 layers, five-local/one-global attention pattern."},
    {"url": "https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/modeling_gemma3.py",
     "evidence": "Q/K/V/O and gated MLP dimensions; tied LM head; logits_to_keep slices before head."},
    {"url": "https://ai.google.dev/gemma/docs/core/model_card_3",
     "evidence": "1B text model context limit is 32768 tokens."},
    {"url": "https://huggingface.co/google/gemma-3-1b-it/resolve/main/config.json",
     "evidence": "Original config returned HTTP 401; no gated weights downloaded. Google code supplies dimensions."},
]


@dataclass(frozen=True)
class Gemm:
    name: str
    stage: str
    m: int
    n: int
    k: int

    @property
    def shape(self):
        return self.m, self.n, self.k

    @property
    def macs(self):
        return self.m * self.n * self.k


def demand_cycles(m, n, k, rows, cols):
    return ((k + rows - 1) // rows) * ((n + cols - 1) // cols) * (m + 2 * rows + cols - 2)


def workload(prompt, context=None, fuse_decode_heads=True):
    """Dense masked prefill; local windows bound decode only; batch is one."""
    h, f, d, q, v = (MODEL[x] for x in ("hidden", "intermediate", "head_dim", "query_heads", "vocabulary"))
    tokens = prompt if context is None else 1
    result = []
    for layer in range(MODEL["layers"]):
        prefix = f"layer{layer:02d}"
        def add(stage, m, n, k, suffix=""):
            result.append(Gemm(f"{prefix}_{stage}{suffix}", stage, m, n, k))
        add("q_projection", tokens, q * d, h)
        add("k_projection", tokens, d, h)
        add("v_projection", tokens, d, h)
        if context is None:
            # Dense rectangular GEMMs include masked future/out-of-window entries.
            for head in range(q):
                add("attention_qk", prompt, prompt, d, f"_h{head}")
                add("attention_av", prompt, d, prompt, f"_h{head}")
        else:
            global_layer = (layer + 1) % MODEL["global_every"] == 0
            attended = context if global_layer else min(context, MODEL["sliding_window"])
            if fuse_decode_heads:
                # Four queries share the same single K/V head: reshape to M=4.
                add("attention_qk", q, attended, d)
                add("attention_av", q, d, attended)
            else:
                for head in range(q):
                    add("attention_qk", 1, attended, d, f"_h{head}")
                    add("attention_av", 1, d, attended, f"_h{head}")
        add("o_projection", tokens, h, q * d)
        add("mlp_gate", tokens, f, h)
        add("mlp_up", tokens, f, h)
        add("mlp_down", tokens, h, f)
    result.append(Gemm("lm_head_last_position", "lm_head", 1, v, h))
    return result


def write_topology(path, ops):
    with path.open("w") as out:
        out.write("Layer name,M,N,K,\n")
        for op in ops:
            out.write(f"{op.name},{op.m},{op.n},{op.k},\n")


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def summarize(ops, args):
    stages = defaultdict(lambda: dict(gemm_calls=0, macs=0, demand_cycles=0))
    for op in ops:
        s = stages[op.stage]
        s["gemm_calls"] += 1
        s["macs"] += op.macs
        s["demand_cycles"] += demand_cycles(*op.shape, args.rows, args.cols)
    cycles = sum(s["demand_cycles"] for s in stages.values())
    macs = sum(s["macs"] for s in stages.values())
    for s in stages.values():
        s["seconds"] = s["demand_cycles"] / args.clock_hz
        s["cycle_share_percent"] = 100 * s["demand_cycles"] / cycles
    return dict(demand_cycles=cycles, seconds=cycles / args.clock_hz, macs=macs,
                operations=2 * macs, gemm_calls=len(ops),
                effective_mac_utilization_percent=100 * macs / (cycles * args.rows * args.cols),
                stage_breakdown=dict(stages))


def validate(out, config_path):
    """Use upstream implementations without patching them; keep allocation small."""
    import numpy as np
    from scalesim.scale_config import scale_config
    from scalesim.compute.systolic_compute_ws import systolic_compute_ws
    from scalesim.scale_sim import scalesim
    report = {"demand_generator_cases": [], "complete_native_cases": []}
    cases = [(32, 32, 1, 1, 1), (32, 32, 1, 32, 32), (32, 32, 4, 65, 33),
             (32, 32, 33, 65, 64), (32, 32, 512, 64, 64), (32, 32, 4, 513, 256),
             (8, 16, 5, 19, 11), (16, 8, 9, 35, 33)]
    for rows, cols, m, n, k in cases:
        cfg = scale_config()
        cfg.read_conf_file(str(config_path))
        cfg.array_rows, cfg.array_cols = rows, cols
        engine = systolic_compute_ws()
        engine.set_params(cfg, np.zeros((m, k)), np.zeros((m, n)), np.zeros((k, n)))
        matrices = engine.get_demand_matrices()
        expected = demand_cycles(m, n, k, rows, cols)
        lengths = [int(x.shape[0]) for x in matrices]
        assert lengths == [expected] * 3, (rows, cols, m, n, k, expected, lengths)
        report["demand_generator_cases"].append(dict(array=[rows, cols], mnk=[m, n, k],
                                                     formula=expected, native_lengths=lengths, passed=True))

    # Native CALC run includes compute and double-buffer memory models. Its first
    # report column includes prefetch and must NOT be confused with Total Cycles.
    ops = [Gemm(f"validation_{i}", "validation", *s) for i, s in enumerate(
        [(1, 32, 32), (4, 65, 33), (16, 64, 64), (33, 65, 64),
         # Actual Gemma K and M, one physical output-column tile: this preserves
         # all K folds without allocating the complete vocabulary-head demand.
         (1, 32, 1152), (512, 32, 1152)])]
    topo = out / "validation_topology.csv"
    write_topology(topo, ops)
    native = scalesim(save_disk_space=True, verbose=False, config=str(config_path),
                      topology=str(topo), layout=str(out / "empty_layout.csv"), input_type_gemm=True)
    native.run_scale(top_path=str(out / "native_validation"))
    r, c = native.config.get_array_dims()
    for op, layer in zip(ops, native.runner.single_layer_sim_object_list, strict=True):
        items = layer.get_compute_report_items()
        expected = demand_cycles(*op.shape, r, c)
        assert items[2] == 0, (op.name, "unexpected memory stalls", items)
        assert int(items[1]) == expected - 1, (op.name, expected, items)
        report["complete_native_cases"].append(dict(mnk=op.shape, formula_duration=expected,
            native_total_cycles=int(items[1]), native_total_cycles_including_prefetch=float(items[0]),
            native_stall_cycles=int(items[2]), passed=True))
    report["all_passed"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=int, default=512)
    parser.add_argument("--generated", type=int, default=128)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--cols", type=int, default=32)
    parser.add_argument("--clock-hz", type=float, default=1e9)
    parser.add_argument("--bandwidth-gbps", type=float, nargs="+", default=[10.0, 17.1, 100.0])
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    if min(args.prompt, args.rows, args.cols) < 1 or args.generated < 2 or args.clock_hz <= 0:
        parser.error("Positive dimensions/clock and at least two generated tokens are required")
    if args.prompt + args.generated > MODEL["max_context"]:
        parser.error("Requested workload exceeds the model context limit")
    if any(b <= 0 for b in args.bandwidth_gbps):
        parser.error("Bandwidths must be positive")
    if args.label and (Path(args.label).name != args.label or args.label in (".", "..")):
        parser.error("--label must be a simple directory name")
    out = HERE / args.label if args.label else HERE
    out.mkdir(exist_ok=True)
    topo_dir = out / "topologies"
    topo_dir.mkdir(exist_ok=True)
    cfg = configparser.ConfigParser()
    cfg.read(ROOT / "configs/scale.cfg")
    cfg["general"]["run_name"] = "gemma3_validation"
    cfg["architecture_presets"]["arrayheight"] = str(args.rows)
    cfg["architecture_presets"]["arraywidth"] = str(args.cols)
    config_path = out / "architecture.cfg"
    with config_path.open("w") as f:
        cfg.write(f)
    (out / "empty_layout.csv").write_text("Layer name,\n")

    pref_ops = workload(args.prompt)
    prefill = summarize(pref_ops, args)
    write_topology(topo_dir / "prefill.csv", pref_ops)
    shape_counts = Counter(("prefill", *op.shape) for op in pref_ops)
    decode = []
    unfused_cycles = 0
    aggregate = defaultdict(lambda: dict(gemm_calls=0, macs=0, demand_cycles=0))
    weights = sum(op.macs for op in workload(1) if not op.stage.startswith("attention_"))
    for context in range(args.prompt + 1, args.prompt + args.generated):
        ops = workload(args.prompt, context)
        one = summarize(ops, args)
        one["context_including_current_token"] = context
        one["token_number_in_generated_sequence"] = context - args.prompt + 1
        one["unique_attended_kv_bytes_int8"] = 2 * MODEL["head_dim"] * MODEL["kv_heads"] * (
            22 * min(context, MODEL["sliding_window"]) + 4 * context)
        one["linear_weight_bytes_int8"] = weights
        decode.append(one)
        write_topology(topo_dir / f"decode_context_{context}.csv", ops)
        shape_counts.update(("decode", *op.shape) for op in ops)
        unfused_cycles += summarize(workload(args.prompt, context, False), args)["demand_cycles"]
        for stage, values in one["stage_breakdown"].items():
            for key in aggregate[stage]:
                aggregate[stage][key] += values[key]
    total = sum(d["demand_cycles"] for d in decode)
    mean_s = total / len(decode) / args.clock_hz
    for values in aggregate.values():
        values["total_seconds"] = values["demand_cycles"] / args.clock_hz
        values["mean_seconds_per_decode_token"] = values["total_seconds"] / len(decode)
        values["cycle_share_percent"] = 100 * values["demand_cycles"] / total
    avg_bytes = sum(d["linear_weight_bytes_int8"] + d["unique_attended_kv_bytes_int8"] for d in decode) / len(decode)
    rooflines = [dict(bandwidth_GB_per_s=b, assumed_bytes_per_token=avg_bytes,
                      traffic_seconds_per_token=avg_bytes / (b * 1e9),
                      bandwidth_only_tokens_per_second=b * 1e9 / avg_bytes,
                      optimistic_compute_and_traffic_ceiling_tokens_per_second=
                      1 / max(mean_s, avg_bytes / (b * 1e9))) for b in args.bandwidth_gbps]
    caveats = [
        "Analytical dense WS matrix schedule, validated on bounded native cases; the full Gemma topology was not native-simulated.",
        "The clock is a study assumption, not a demonstrated synthesis/place-and-route frequency or measured hardware clock.",
        "W8A8 with INT8 KV is an assumption; no quantization accuracy or output tokens were evaluated.",
        "CALC bandwidth mode imposes no fixed DRAM bandwidth cap; Bandwidth=10 in the config is not a finite-bandwidth experiment.",
        "Durations count demand rows D; native zero-stall Total Cycles reports D-1, distinct from its including-prefetch column.",
        "No initial DRAM prefetch/weight transfer, memory stalls, inter-operation dispatch, or cache-persistence simulation.",
        "Excludes RMS/QK normalization, RoPE, GELU/gating, softmax/masks, quantize/dequantize, embedding lookup, sampling and host overhead.",
        "Prefill attention is dense causal-masked GEMM, including masked entries; not a FlashAttention/causal-skipping implementation.",
        "Prefill computes all prompt hidden states but only the final position's vocabulary logits.",
        "Decode fuses four query heads sharing the same single KV head into M=4; projections remain separate.",
        "Rooflines are separate traffic arithmetic, not native finite-bandwidth simulation: one read per weight and attended KV element; scales, KV writes, activation/output traffic and protocol inefficiency excluded.",
    ]
    with (out / "validation.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
        validation = validate(out, config_path)
    provenance = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), sources=SOURCES,
        scalesim_git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_sha256={str(p): hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in (
            "scalesim/compute/systolic_compute_ws.py", "scalesim/single_layer_sim.py", "configs/scale.cfg")},
        python=sys.version, packages={p: importlib.metadata.version(p) for p in ("scalesim", "numpy")},
        reproduction_command="env -u PYTHONPATH .venv/bin/python experiments/gemma3_1b_int8/study.py " + " ".join(sys.argv[1:]))
    summary = dict(model=MODEL, hardware=dict(array_rows=args.rows, array_columns=args.cols,
        dataflow="ws", cores=1, clock_hz=args.clock_hz, memory_mode="CALC",
        ideal_INT8_TOPS=2 * args.rows * args.cols * args.clock_hz / 1e12),
        workload=dict(batch=1, prompt_tokens=args.prompt, generated_tokens=args.generated,
                      cached_decode_steps=len(decode), first_decode_context=args.prompt + 1,
                      last_decode_context=args.prompt + args.generated - 1),
        matrix_path_TTFT_seconds=prefill["seconds"],
        decode_mean_tokens_per_second=1 / mean_s, decode_mean_seconds_per_token=mean_s,
        decode_total_cycles=total, decode_total_seconds=total / args.clock_hz,
        entire_generation_matrix_path_seconds=prefill["seconds"] + total / args.clock_hz,
        unfused_decode_attention_tokens_per_second=len(decode) * args.clock_hz / unfused_cycles,
        linear_weights_INT8_bytes=weights, prefill=prefill,
        decode_stage_breakdown=dict(aggregate), separate_traffic_roofline=rooflines,
        limitations=caveats, validation=validation, provenance=provenance)
    dump_json(out / "results.json", summary)
    dump_json(out / "decode_tokens.json", decode)
    dump_json(out / "shape_multiplicities.json", [dict(phase=p, m=m, n=n, k=k, calls=count,
        demand_cycles_per_call=demand_cycles(m, n, k, args.rows, args.cols))
        for (p, m, n, k), count in sorted(shape_counts.items())])
    with (out / "decode_tokens.csv").open("w") as f:
        f.write("generated_token,context,demand_cycles,seconds,macs\n")
        for d in decode:
            f.write(f"{d['token_number_in_generated_sequence']},{d['context_including_current_token']},{d['demand_cycles']},{d['seconds']},{d['macs']}\n")
    print(json.dumps({k: summary[k] for k in ("matrix_path_TTFT_seconds", "decode_mean_tokens_per_second",
        "decode_mean_seconds_per_token", "decode_total_cycles", "linear_weights_INT8_bytes",
        "unfused_decode_attention_tokens_per_second")}, indent=2))
    print(f"Native validation passed: {len(validation['demand_generator_cases'])} demand cases, "
          f"{len(validation['complete_native_cases'])} complete runs. Results: {out / 'results.json'}")


if __name__ == "__main__":
    main()
