# ARGUS Simulation Framework

This repository contains the hardware modeling and simulation framework for the paper **"ARGUS: Enabling Efficient MLLM Inference through Hierarchical Compression and Algorithm-Hardware-Datatype Co-design"** (Submitted to **DAC 2026**).

Built upon [SCALE-Sim](https://github.com/scalesim-project/SCALE-Sim), this framework is designed to evaluate the decoding stage throughput performance of Unified Multimodal Large Language Models (MLLMs). It supports modeling our proposed **ARGUS** architecture as well as several state-of-the-art baseline accelerators (S-DMA, FlightVGM, FIGNA, AxCore).

## Supported Models & Tasks

The framework supports simulation for the following MLLMs and tasks:

*   **Bagel**:
    *   **GenEdit**: Image Editing
    *   **GenEval**: Image Generation
    *   **MM**: Image Understanding (Multimodal)
*   **Janus**:
    *   **GenEval**: Image Generation
    *   **MM**: Image Understanding

## Project Structure

The core logic is encapsulated within `simulation_core`, while external scripts handle argument parsing and execution.

```text
SCALE-Sim/
├── run_bagel.py              # Entry point for Bagel model simulation
├── run_janus.py              # Entry point for Janus model simulation
├── simulation_core/          # Core simulation logic
│   ├── bagel_sim.py          # Bagel simulation class
│   └── janus_sim.py          # Janus simulation class
├── topologies/               # Simulation flow configurations (JSON)
│   ├── bagel/                # Configs for Bagel tasks
│   └── janus/                # Configs for Janus tasks
├── configs/                  # Hardware architecture specifications (CFG)
│   └── bagel/                # Hardware specs (Array size, SRAM, Bandwidth)
└── results/                  # Simulation logs and outputs
```

## Usage

To run a simulation, use the provided entry scripts (`run_bagel.py` or `run_janus.py`). You need to specify the hardware architecture, the task type, and the path to the simulation configuration file.

### Arguments

*   `--hw`: The hardware architecture to simulate. Options: `ours`, `ours_balence`, `sdma`, `flightvgm`, `figna`, `base`.
*   `--task`: The task to evaluate. Options: `GenEdit`, `GenEval`, `MM`.
*   `--config`: Path to the specific JSON configuration file for the simulation run.

### Example Command

To simulate the **ARGUS (Ours)** hardware running the **GenEdit** task on the **Bagel** model:

```bash
python run_bagel.py --hw ours --task GenEdit --config ./topologies/bagel/config_ours.json
```

To simulate the Janus model on FlightVGM hardware for the **MM** task:

```bash
python run_janus.py --hw flightvgm --task MM --config ./topologies/janus/config_flightvgm.json
```
## Configuration System
The framework uses a two-level configuration system to provide flexibility in modeling both the workload flow and the hardware specifications.

1. Simulation Configuration (`.json`)
Located in `bagel/` or `janus/` directories.
These JSON files control the simulation flow, dataset parameters, and link to specific hardware configs. When creating a custom config, ensure the following keys are defined:

- **KV Cache Settings**: `kv_cache_size` (initial size), `num_heads`, `head_dim`
- **Hardware Config Paths**: Keys like `config_fp16`, `config_int8`, `config_base` point to the `.cfg` files described below.
- **Hardware Specifics**: Parameters like `sparsity`, `activate_rate`, `tile_size`, etc.
- **Output**: `result_path` defines where logs and logs are saved.

Example snippet (`config_ours.json`):
```json
{
    "kv_cache_init": 3254,
    "gen_text_len": 229,
    "config_fp16": "./configs/bagel/ours_fp16.cfg",
    "config_int8": "./configs/bagel/ours_int8.cfg",
    "sparsity_cross_attn": 0.3819
}
``` 
2. Hardware Configuration (`.cfg`)
Located in `bagel/` directory.
These are standard SCALE-Sim configuration files that define the physical hardware constraints.

Architecture: `ArrayHeight, ArrayWidth` (Systolic Array dimensions).
Memory: `IfmapSramSzkB, FilterSramSzkB, OfmapSramSzkB` (On-chip buffer sizes).
System: `Bandwidth` (DRAM bandwidth), `Dataflow`.
Example snippet (`ours_fp16.cfg`):
```
[architecture_presets]
ArrayHeight:    32
ArrayWidth:     32
IfmapSramSzkB:  32
Bandwidth :     8
```
Acknowledgement
This project is based on [SCALE-Sim](https://github.com/scalesim-project/SCALE-Sim) (Systolic CNN Accelerator Simulator).