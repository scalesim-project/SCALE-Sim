# ISCA Tutorial: SCALE-Sim + Accelergy Energy/Power Estimation

This guide sets up the **SCALE-Sim + Accelergy** flow end-to-end and runs the
tutorial energy sweep
([`rundir-accelergy/tutorial_vit_sweep.py`](rundir-accelergy/tutorial_vit_sweep.py)),
which simulates ViT workloads on a systolic array and estimates per-component
energy with Accelergy.

The tutorial driver needs **three** things installed:

1. **SCALE-Sim** (this repo, `isca-tutorial` branch) — cycle-accurate simulation
   and action-count extraction.
2. **Accelergy** — the energy-estimation framework.
3. **accelergy-aladdin-plug-in** — table-based estimator for MAC / regfile /
   SRAM primitives. **Required** (without an estimator plug-in, Accelergy
   crashes with `Can not find an energy estimator for intmac`).

---

## 0. Prerequisites

- Python 3.7+ (a conda/virtualenv is recommended)
- `git`
- The commands below use SSH GitHub URLs (`git@github.com:...`). Switch to
  `https://github.com/...` if you do not have SSH keys configured.

Pick one working directory for all repos, e.g.:

```bash
export TUTORIAL_ROOT=$HOME/isca_tutorial_scalesim
mkdir -p "$TUTORIAL_ROOT" && cd "$TUTORIAL_ROOT"
```

---

## 1. Install SCALE-Sim from source (`isca-tutorial` branch)

```bash
cd "$TUTORIAL_ROOT"
git clone git@github.com:scalesim-project/SCALE-Sim.git
cd SCALE-Sim
git checkout isca-tutorial

# install SCALE-Sim and its Python dependencies
pip3 install -r requirements.txt
pip3 install -e .
```

This brings in `numpy`, `pandas`, `tqdm`, `matplotlib`, `numba`, etc. (the
tutorial plotting needs `matplotlib` + `numpy`, both included).

---

## 2. Install Accelergy

```bash
cd "$TUTORIAL_ROOT"
git clone git@github.com:Accelergy-Project/accelergy.git
cd accelergy
pip3 install .
```

Verify the command is available (it prints a banner and exits):

```bash
accelergy
```

> The first run creates `~/.config/accelergy/accelergy_config.yaml`. See
> [Step 4](#4-verify-the-accelergy-config-important) — this config must point at
> the **same Python environment** where you install the plug-in below.

---

## 3. Install the Aladdin plug-in (required)

```bash
cd "$TUTORIAL_ROOT"
git clone git@github.com:Accelergy-Project/accelergy-aladdin-plug-in.git
cd accelergy-aladdin-plug-in
pip3 install .
```

Aladdin provides table-based energy for `intmac`, `regfile`, `SRAM`, etc. — the
primitives the tutorial architecture uses.

---

## 4. Verify the Accelergy config (important)

Accelergy discovers plug-ins through `~/.config/accelergy/accelergy_config.yaml`.
If you have used Accelergy before (e.g. in another conda env), this file may
point at a **different** environment and Accelergy will fail to find the
plug-in you just installed (`Can not find an energy estimator ...`).

Check that the paths match your active environment:

```bash
cat ~/.config/accelergy/accelergy_config.yaml
```

The `estimator_plug_ins` / `table_plug_ins` paths should live under the same
prefix as `which accelergy` (e.g. `.../envs/<your_env>/share/accelergy/...`).
If they point somewhere stale, regenerate the config:

```bash
mv ~/.config/accelergy/accelergy_config.yaml ~/.config/accelergy/accelergy_config.yaml.bak
accelergy   # regenerates a default config for the active environment
```

---

## 5. Run the tutorial energy sweep

```bash
cd "$TUTORIAL_ROOT"/SCALE-Sim/rundir-accelergy
python3 tutorial_vit_sweep.py
```

This runs the **WS (weight-stationary)** dataflow for three ViT workloads
(`vit_s`, `vit_b`, `vit_l`) across array sizes `32x32`, `64x64`, `128x128`,
then writes results to `rundir-accelergy/tutorial_results/`:

- `workload_energy_ws.png` — a single plot of the **MAC vs SRAM** energy
  distribution. Bar color encodes the component (MAC vs SRAM); the hatch pattern
  encodes the systolic-array size.
- `workload_energy_ws.csv` — MAC / SRAM / Other / total energy (uJ) and the
  MAC/SRAM fractions for each (workload, array size).

Useful flags:

```bash
# different workloads / sizes / dataflow
python3 tutorial_vit_sweep.py --workloads vit_s vit_b --array-sizes 64 128 --dataflow ws

# re-plot from existing results without re-simulating
python3 tutorial_vit_sweep.py --skip-run

# force re-running cached points
python3 tutorial_vit_sweep.py --force
```

> For the underlying single-run flow (`run_all.sh`), config-file parameters, and
> output formats, see [README_accelergy.md](README_accelergy.md).

---

## 6. TPU LLM latency prediction (StableHLO → SCALE-Sim)

A second flow in this repo ([`topologies/stablehlo/llm/`](topologies/stablehlo/llm))
predicts a **real LLM's TPU device latency** from its StableHLO graph: SCALE-Sim
simulates the matmuls, per-op models cover the non-compute ops, and a whole-model
compensation produces a device-time estimate that is validated against a real TPU.

Three stages: **(1)** measure real latency on a TPU (ground truth), **(2)** export the
model to StableHLO MLIR, **(3)** run SCALE-Sim on the MLIR. For the big LLMs, stages 1–2
are **shown in the talk for reference** — stage 1 needs a TPU VM, and the full sim on a
real LLM is slow. In the hands-on you run the **tiny transformer**, which needs no TPU
and finishes in a couple of minutes. All paths below are relative to the repo root.

### A. Demo (presenter / reference — needs a TPU VM and is slow)

```bash
# 1. ground truth: measure gpt2's real device-busy latency on a TPU VM
PJRT_DEVICE=TPU python3 topologies/stablehlo/llm/profile_model_on_tpu.py --models gpt2 --gen tpu_v4
#    -> topologies/stablehlo/llm/measured_tpu_v4.json   (e.g. gpt2 ~500 us)

# 2. export gpt2's forward graph to StableHLO MLIR (CPU; the committed
#    gpt2.stablehlo.mlir is already provided, so this step is optional)
python3 topologies/stablehlo/llm/export_LLM.py --models gpt2 --seq-len 128

# 3. predict with SCALE-Sim  (full cycle-accurate sim -- LENGTHY for a real LLM)
python3 -m scalesim.scale -c configs/tpuv4.cfg \
    -t topologies/stablehlo/llm/gpt2.stablehlo.mlir -p results/gpt2
```

The predicted device latency is the `tuned_us` TOTAL of
`results/gpt2/<run>/TIME_REPORT.csv` (compare it to the measured value from step 1).
The full cycle-accurate gpt2 run takes **~40 minutes** (and step 1 needs a TPU).

### B. Hands-on: the tiny transformer (no TPU, ~couple of minutes)

```bash
# (optional) regenerate the MLIR -- a committed copy is already in the repo:
python3 topologies/stablehlo/llm/export_tiny_transformer_pytorch.py

# run SCALE-Sim on it
python3 -m scalesim.scale -c configs/tpuv4.cfg \
    -t topologies/stablehlo/llm/tiny_transformer_pytorch.stablehlo.mlir -p results/tiny
```

(Use `-c configs/tpuv6e.cfg` for a TPU v6e estimate.)

### C. Build / modify the model that generates the MLIR

The export scripts in [`topologies/stablehlo/llm/`](topologies/stablehlo/llm) are small,
self-contained examples of converting a full **PyTorch** or **JAX** model to StableHLO
MLIR. You can edit any part — the model definition, its dimensions, or which checkpoint
is loaded — and re-run to emit a new `.mlir`. (`export_tiny_transformer_*.py` define a
small transformer inline, easy to tweak; `export_LLM.py` wraps a HuggingFace
`AutoModelForCausalLM`.) They run on CPU — no TPU needed.

The conversion itself is just a couple of lines:

```python
# PyTorch  (export_tiny_transformer_pytorch.py, export_LLM.py)
from torch_xla.stablehlo import exported_program_to_stablehlo
ep   = torch.export.export(model.eval(), (example_inputs,))       # trace the model
text = exported_program_to_stablehlo(ep).get_stablehlo_text()     # -> StableHLO MLIR

# JAX  (export_tiny_transformer_jax.py)
text = jax.jit(forward).lower(params, example_inputs).as_text()   # -> StableHLO MLIR
```

Write `text` to a `.mlir` file and feed it to SCALE-Sim as in **B**.

### D. Reading `TIME_REPORT.csv`

One row per op (in program order) plus a final `TOTAL` row:

| column | meaning |
|--------|---------|
| `OpID` | program-order index of the op |
| `single_op_us` | the op's **standalone** predicted latency (GEMM from the cycle model; non-compute from the per-op models) |
| `tuned_us` | the op's **compensated** contribution to fused whole-model device time |
| `layer` | compute-layer id for GEMMs; `N/A` for non-compute ops |
| `stablehlo` | short op signature (shapes) |

The **`TOTAL` row's `tuned_us` is the whole-model predicted device latency** — the number
to compare against the TPU measurement. Its last cell notes the once-per-forward overhead
`C_forward` and the GEMM-kernel count. (Summed `single_op_us` is far larger: that is the
naive pre-fusion sum, *before* the whole-model compensation.)

> Full details of these graphs, the scripts, and the latency model:
> [`topologies/stablehlo/llm/README.md`](topologies/stablehlo/llm/README.md).

---

## Troubleshooting

- **`Can not find an energy estimator for intmac ...`** — no estimator plug-in
  is visible. Install the Aladdin plug-in (Step 3) and verify the Accelergy
  config points at the right environment (Step 4).
- **`accelergy: command not found`** — the Accelergy install dir is not on
  `$PATH`, or you are in a different environment than where you `pip3 install`ed.
- **`configparser ... No section: 'layout'`** — you ran SCALE-Sim with a config
  missing the `[layout]` section. Use `rundir-accelergy/scale.cfg` (or one
  derived from it) as the template; the tutorial does this automatically.
- **`KeyError: 'SRAM_row_size'`** — the config is missing the
  `SRAM_row_size` / `DRAM_row_size` parameters required by `preprocess.py`. Again,
  use `rundir-accelergy/scale.cfg` as the base.
- **`40nm` / "Failed to evaluate" warnings** — harmless deprecation warnings
  from Accelergy 0.4 treating the unquoted technology node as a string.
