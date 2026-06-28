#!/usr/bin/env python3
"""
Whole-model DEVICE-BUSY ground truth on a TPU — the target the compensation fit
(fit_compensation_pure.py) matches. Two stages:

  llm    sweep the 3 reference LLMs x seq{128,256,512,1024} x batch{1,8,32}
         -> e2e_device_truth_<gen>.csv  (model,seq,batch,vocab,wall_ms,device_ms,status)
  tiny   the tiny_transformer small-model anchor -> prints device-busy us
         (pass to fit_compensation_pure.py --tiny-truth)

device_busy = median(full = run + wait_device_ops) - median(issue = run, no wait),
via torch.compile(backend="openxla"). Each LLM combo runs in a FRESH SUBPROCESS:
loading a model per combo in one process leaks torch_xla HBM and collapses the
measurements after ~8 loads.

Run on a TPU VM:
  PJRT_DEVICE=TPU python3 measure_truth.py llm  --gen tpu_v4
  PJRT_DEVICE=TPU python3 measure_truth.py tiny
"""
import argparse, csv, os, subprocess, sys
os.environ.setdefault("PJRT_DEVICE", "TPU"); os.environ.setdefault("XLA_USE_SPMD", "0")

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {"gpt2": "gpt2", "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
          "smollm2-135m": "HuggingFaceTB/SmolLM2-135M"}
SEQS = [128, 256, 512, 1024]; BATCHES = [1, 8, 32]
MAX_LOGITS_BYTES = 6e9   # skip combos whose logits would OOM (batch*seq*vocab*2)


def _median_ms(fn, K):
    import time
    ts = []
    for _ in range(K):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    ts.sort(); return ts[len(ts) // 2] * 1e3


def _llm_worker(name, seq, batch, K=20, warm=6):
    """Measure ONE (model,seq,batch); print a CSV line. Invoked as a subprocess."""
    import torch
    import torch_xla.core.xla_model as xm
    from transformers import AutoModelForCausalLM
    dev = xm.xla_device()
    m = AutoModelForCausalLM.from_pretrained(
        MODELS[name], dtype=torch.bfloat16, attn_implementation="eager").eval()
    vocab = m.config.vocab_size
    if batch * seq * vocab * 2 > MAX_LOGITS_BYTES:
        print(f"{name},{seq},{batch},{vocab},,,skip_oom"); return
    m = m.to(dev)
    cm = torch.compile(m, backend="openxla")
    ids = torch.arange(batch * seq, dtype=torch.long).remainder(vocab).reshape(batch, seq).to(dev)

    def issue():
        with torch.no_grad():
            cm(input_ids=ids, use_cache=False)
        xm.mark_step()

    def full():
        issue(); xm.wait_device_ops()

    for _ in range(warm):
        full()
    tf = _median_ms(full, K)                              # host issue + device
    import time                                           # host-issue only
    ts = []
    for _ in range(K):
        t0 = time.perf_counter(); issue(); ts.append(time.perf_counter() - t0); xm.wait_device_ops()
    ts.sort(); ti = ts[len(ts) // 2] * 1e3
    print(f"{name},{seq},{batch},{vocab},{tf:.4f},{max(tf - ti, 0.0):.4f},ok")


def cmd_llm(args):
    out = open(os.path.join(HERE, f"e2e_device_truth_{args.gen}.csv"), "w", newline="")
    w = csv.writer(out)
    w.writerow(["model", "seq", "batch", "vocab", "wall_ms", "device_ms", "status"]); out.flush()
    for name in MODELS:
        for seq in SEQS:
            for batch in BATCHES:
                r = subprocess.run([sys.executable, __file__, "_worker", name, str(seq), str(batch)],
                                   capture_output=True, text=True)
                line = next((ln for ln in r.stdout.splitlines()
                             if ln.startswith(f"{name},{seq},{batch},")), "")
                if line:
                    out.write(line + "\n"); print(line, flush=True)
                else:
                    err = (r.stderr.strip().splitlines() or ["no-output"])[-1][:60]
                    out.write(f"{name},{seq},{batch},,,,err\n"); print(f"{name} s{seq} b{batch}: ERR {err}")
                out.flush()
    out.close(); print("DONE")


def cmd_tiny(args):
    import torch
    import torch_xla.core.xla_model as xm
    sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "topologies", "stablehlo", "llm"))
    from export_tiny_transformer_pytorch import TinyTransformer, SEQ, VOCAB
    dev = xm.xla_device()
    cm = torch.compile(TinyTransformer().eval().to(dev), backend="openxla")
    ids = torch.arange(SEQ, dtype=torch.long).remainder(VOCAB).reshape(1, SEQ).to(dev)

    def issue():
        with torch.no_grad():
            cm(ids)
        xm.mark_step()

    def full():
        issue(); xm.wait_device_ops()

    for _ in range(8):
        full()
    device_us = max(_median_ms(full, 30) - _median_ms(issue, 30), 0.0) * 1e3
    print(f"device={xm.xla_device_hw(dev)}  tiny_transformer device-busy = {device_us:.1f} us")
    print(f"-> fit_compensation_pure.py --tiny-truth {device_us:.1f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)
    a = sub.add_parser("llm", help="sweep the 3 LLMs -> e2e_device_truth_<gen>.csv")
    a.add_argument("--gen", default="tpu_v4")
    a.set_defaults(func=cmd_llm)
    t = sub.add_parser("tiny", help="tiny_transformer device-busy anchor")
    t.set_defaults(func=cmd_tiny)
    wk = sub.add_parser("_worker")                       # internal (one LLM combo)
    wk.add_argument("name"); wk.add_argument("seq", type=int); wk.add_argument("batch", type=int)
    wk.set_defaults(func=lambda args: _llm_worker(args.name, args.seq, args.batch))
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
