# SCALE-Sim v3 Sparsity Analysis Scripts - Compute and Memory Plots

This repository contains scripts to analyze and visualize compute cycles and memory usage for SCALE-Sim v3 with different sparsity configurations. The analysis focuses on ResNet-18 architecture with varying memory sizes and sparsity ratios.

## Setup and Configuration

### Memory Configurations
The analysis covers four on-chip memory sizes:
- 16 kB
- 64 kB
- 256 kB
- 1 MB

### Sparsity Ratios
Three sparsity configurations are analyzed:
- 1:4 (25% density)
- 2:4 (50% density)
- 4:4 (100% density / dense)

### Configuration File
Create a configuration file `sparsity.cfg` with the following settings:

```ini
[general]
run_name = scalesim_sparsity

[architecture_presets]
ArrayHeight = 128
ArrayWidth = 128
IfmapSramSzkB = <memory_size>    # Set to: 16, 64, 256, or 1024
FilterSramSzkB = <memory_size>    # Set to: 16, 64, 256, or 1024
OfmapSramSzkB = <memory_size>     # Set to: 16, 64, 256, or 1024
IfmapOffset = 0
FilterOffset = 100
OfmapOffset = 200
Bandwidth = 50
Dataflow = ws
MemoryBanks = 1

[sparsity]
SparsitySupport = true            # Set to false for dense (4:4)
SparseRep = ellpack_block
OptimizedMapping = false
BlockSize = 8
RandomNumberGeneratorSeed = 40

[run_presets]
InterfaceBandwidth = USER
```

### Topology Configuration
Modify the ResNet-18 topology file (`conv.csv`) to set the desired sparsity ratio. Example format:

```csv
Layer name, IFMAP Height, IFMAP Width, Filter Height, Filter Width, Channels, Num Filter, Strides, Sparsity
Conv1,224,224,7,7,3,64,2,1:4,
Conv2_1a,56,56,3,3,64,64,1,1:4,
...
```

## Running the Analysis

### 1. Generate SCALE-Sim Results
Run SCALE-Sim for each combination of memory size and sparsity:

```bash
python scalesim/scale.py -c configs/sparsity.cfg -t topologies/sparsity/conv.csv -p sparsity_results
```

A sample collection of these results are provided in the ```sparsity_results_128x128``` folder. The subfolders inside this folder have the following the naming convention:
`<model_name>_<memory_size>_<sparsity_ratio>_<array_size>`. This format is used for plotting the graphs. 

### 2. Generate Compute Cycles Plot
```bash
python computecyclesplot.py
```
This script:
- Processes results from all configurations
- Groups compute cycles by convolution layers
- Generates a log-scale plot comparing performance across memory sizes and sparsity ratios
- Saves output as `computecycles.png`


## Notes
- Ensure all SCALE-Sim v3 result folders are present in the expected directory structure.
- For dense (4:4) configurations, disable `SparsitySupport` in the config file.
- All memory sizes should use consistent array dimensions (128x128 in this case).