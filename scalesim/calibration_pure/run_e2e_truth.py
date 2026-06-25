"""
End-to-end device-busy ground truth: 3 LLMs x seq{128,256,512,1024} x batch{1,8,32},
torch.compile (openxla), batch-1..32. Writes e2e_device_truth.csv with wall_ms and
device_ms (= wall - host-issue).

IMPORTANT: each (model,seq,batch) is measured in a FRESH SUBPROCESS. Loading a new
model onto the TPU every combo in ONE process leaks HBM (del m,cm does not free
torch_xla device memory), and after ~8 loads the runtime thrashes -> measurements
collapse to fixed ceilings / device~=0. A subprocess per combo gives clean HBM each
time (this is how devicetruth_worker.py was always invoked).

Run:  venv_xla/bin/python run_e2e_truth.py            # driver (spawns workers)
      venv_xla/bin/python run_e2e_truth.py M SEQ BATCH  # worker (one combo)
"""
import os, sys, csv, subprocess
os.environ["PJRT_DEVICE"] = "TPU"; os.environ["XLA_USE_SPMD"] = "0"

MODELS = {"gpt2": "gpt2", "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
          "smollm2-135m": "HuggingFaceTB/SmolLM2-135M"}
SEQS = [128, 256, 512, 1024]; BATCHES = [1, 8, 32]
MAX_LOGITS_BYTES = 6e9   # skip combos whose logits would OOM (batch*seq*vocab*2)


def worker(name, seq, batch, K=20, warm=6):
    import torch, time
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

    def med(fn):
        ts = []
        for _ in range(K):
            t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
        ts.sort(); return ts[len(ts) // 2] * 1e3

    tf = med(full)                                  # full wall-clock (host issue + device)
    ts = []                                         # host-issue only (stop timer before wait)
    for _ in range(K):
        t0 = time.perf_counter(); issue(); ts.append(time.perf_counter() - t0); xm.wait_device_ops()
    ts.sort(); ti = ts[len(ts) // 2] * 1e3
    print(f"{name},{seq},{batch},{vocab},{tf:.4f},{max(tf - ti, 0.0):.4f},ok")


def driver():
    out = open("e2e_device_truth.csv", "w", newline=""); w = csv.writer(out)
    w.writerow(["model", "seq", "batch", "vocab", "wall_ms", "device_ms", "status"]); out.flush()
    for name in MODELS:
        for seq in SEQS:
            for batch in BATCHES:
                r = subprocess.run([sys.executable, __file__, name, str(seq), str(batch)],
                                   capture_output=True, text=True)
                line = ""
                for ln in r.stdout.splitlines():
                    if ln.startswith(f"{name},{seq},{batch},"):
                        line = ln; break
                if line:
                    out.write(line + "\n"); out.flush()
                    print(line, flush=True)
                else:
                    err = (r.stderr.strip().splitlines() or ["no-output"])[-1][:60]
                    out.write(f"{name},{seq},{batch},,,,err\n"); out.flush()
                    print(f"{name} s{seq} b{batch}: ERR {err}", flush=True)
    out.close(); print("DONE")


if __name__ == "__main__":
    if len(sys.argv) == 4:
        worker(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    else:
        driver()
