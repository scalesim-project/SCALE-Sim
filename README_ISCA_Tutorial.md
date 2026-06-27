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
