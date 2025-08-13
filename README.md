# Systolic CNN AcceLErator Simulator (SCALE-Sim) 

<!-- [![Documentation Status](https://readthedocs.org/projects/scale-sim-project/badge/?version=latest)](https://scale-sim-project.readthedocs.io/en/latest/?badge=latest) -->

SCALE Sim is a simulator for systolic array based accelerators for Convolution, Feed Forward, and any layer that uses GEMMs.
This is a refreshed version of the simulator with feature enhancements, restructured code to aid feature additions, and ease of distribution.

![scalesim v3 overview](https://github.com/scalesim-project/SCALE-Sim/blob/main/documentation/resources/v3_overview.png "scalesim v3 overview")

<!-- The previous version of the simulator can be found [here](https://github.com/scalesim-project/scale-sim-v2). -->

## Features

SCALE-Sim v3 includes several advanced features:

1. **Sparsity Support**: Layer-wise and row-wise sparsity support for efficient neural network execution
2. **Ramulator Integration**: Detailed memory model integration for evaluating DRAM performance
3. **Accelergy Integration**: Energy and power estimation capabilities
4. **Layout Support**: Advanced memory layout configurations
5. **Multi-core Support**: Support for multi-core simulations

## Getting started in 30 seconds

### *Installing the package*

Getting started is simple! SCALE-Sim is completely written in python and could be installed from source.

You can install SCALE-Sim in your environment using the following command

```$ pip3 install <path-to-scalesim-v3-folder>```

If you are a developer that will modify scale-sim during your usage, please install it with `-e` flag, which will create a symbolic link instead of replicating scalesim in your environment, thus modification of scale-sim code can be syncronized simultaneously

```$ pip3 install -e <path-to-scalesim-v3-folder>```

### *Launching a run*

After installing SCALE-Sim, it can be run by using the ```scalesim.scale``` and providing the paths to the architecture configuration, and the topology descriptor csv file.

```$ python3 -m scalesim.scale -c <path_to_config_file> -t <path_to_topology_file> -p <path_to_output_log_dir>```

### *Running from source*

The above method uses the installed package for running the simulator.
In cases where you would like to run directly from the source, the following command should be used instead.

```$ PYTHONPATH=$PYTHONPATH:<scale_sim_repo_root> python3 <scale_sim_repo_root>/scalesim/scale.py -c <path_to_config_file> -t <path_to_topology_file>```

If you are running from sources for the first time and do not have all the dependencies installed, please install them first  using the following command.

```$ pip3 install -r <scale_sim_repo_root>/requirements.txt```

## Tool inputs

SCALE-Sim uses two input files to run, a configuration file and a topology file.

### Configuration file

The configuration file is used to specify the architecture and run parameters for the simulations.
The following shows a sample config file:

![sample config](https://github.com/scalesim-project/scale-sim-v2/blob/main/documentation/resources/config-file-example.png "sample config")

The config file has three sections. The "*general*" section specifies the run name, which is user specific. The "*architecture_presets*" section describes the parameter of the systolic array hardware to simulate.
The "*run_preset*" section specifies if the simulator should run with user specified bandwidth, or should it calculate the optimal bandwidth for stall free execution.

The detailed documentation for the config file could be found **here (TBD)**

### Topology file

The topology file is a *CSV* file which decribes the layers of the workload topology. The layers are typically described as convolution layer parameters as shown in the example below.

![sample topo](https://github.com/scalesim-project/scale-sim-v2/blob/main/documentation/resources/topo-file-example.png "sample topo")

For other layer types, SCALE-Sim also accepts the workload desciption in M, N, K format of the equivalent GEMM operation as shown in the example below.

![sample mnk topo](https://github.com/scalesim-project/scale-sim-v2/blob/doc/anand/readme/documentation/resources/topo-mnk-file-example.png "sample mnk topo")

The tool however expects the inputs to be in the convolution format by default. When using the mnk format for input, please specify using the  ```-i gemm``` switch, as shown in the example below.

```$ python3 <scale sim repo root>/scalesim/scale.py -c <path_to_config_file> -t <path_to_mnk_topology_file> -i gemm```

### Output

Here is an example output dumped to stdout when running Yolo Tiny (whose configuration is in yolo_tiny.csv):
![screen_out](https://github.com/scalesim-project/scale-sim-v2/blob/doc/anand/readme/documentation/resources/output.png "std_out")

Also, the simulator generates read write traces and summary logs at ```<run_dir>/../scalesim_outputs/```. The user can also provide a custom location using ```-p <custom_output_directory>``` when using `scalesim.py` file.
There are three summary logs:

* COMPUTE_REPORT.csv: Layer wise logs for compute cycles, stalls, utilization percentages etc.
* BANDWIDTH_REPORT.csv: Layer wise information about average and maximum bandwidths for each operand when accessing SRAM and DRAM
* DETAILED_ACCESS_REPORT.csv: Layer wise information about number of accesses and access cycles for each operand for SRAM and DRAM.

In addition cycle accurate SRAM/DRAM access logs are also dumped and could be accesses at ```<outputs_dir>/<run_name>/``` eg `<run_dir>/../scalesim_outputs/<run_name>`

## Advanced Features

### *Using Multi-core feature*

SCALE-Sim v3 introduces **multi-core simulation capabilities** to address the limitations of its predecessor, SCALE-Sim v2, which could only model single-core systolic arrays. This feature allows comprehensive modeling of modern AI accelerators equipped with multiple tensor cores, enabling researchers to simulate advanced workloads and optimize performance. For detailed setup and usage instructions, refer to the ```multi-core/README.md``` file.

### *Using Sparsity feature*

SCALE-Sim v3 introduces advanced support for layer-wise and row-wise sparsity. For detailed information about sparsity features and usage, refer to the ```README_Sparsity.md``` file.

Key features include:
- Layer-wise sparsity with customizable configurations
- Row-wise sparsity with N:M ratio support
- Support for different sparse representations (CSR, CSC, Blocked ELLPACK)
- Detailed sparsity reports and metrics

### *Using Ramulator feature*

SCALE-sim v3 integrates a detailed memory model with the systolic array computation. Users can evaluate:
- Stall cycles due to data load from memory
- Bank conflicts
- Different memory types (DDR3, DDR4, etc.)
- Various memory configurations (channels, rows, etc.)

For detailed setup and usage instructions, refer to the ```README_ramulator.md``` file.

### *Using Accelergy feature*

SCALE-Sim v3 integrates with Accelergy for energy and power estimation. This feature allows:
- Energy estimation of systolic array architectures
- Power analysis
- Integration with CACTI and Aladdin plugins for accurate estimation

For setup and usage instructions, refer to the ```README_accelergy.md``` file.

### *Using Layout feature*

SCALE-Sim v3 supports advanced memory layout configurations for on-chip buffers. The layout feature enables:

- **Custom Data Organization**: Specify different data layouts for ifmap, filter, and ofmap tensors
- **Bank Conflict Evaluation**: Model realistic memory access patterns and bank conflicts
- **Multi-bank Support**: Configure number of memory banks and ports per bank
- **Layout Specification**: Define layouts through three key parameters:
  - `intraline_factor`: Specifies elements per line for each dimension
  - `intraline_order`: Controls dimension ordering within a line
  - `interline_order`: Controls dimension ordering across lines

Layout configurations can be specified in the architecture configuration file using parameters like:
- `OnChipMemoryBanks`: Total number of on-chip memory banks
- `OnChipMemoryBankPorts`: Number of ports per bank
- `IfmapCustomLayout`/`FilterCustomLayout`: Enable custom layouts for tensors

For detailed information about layout features and usage, refer to the documentation in the ```README_layout.md``` file.

## Detailed Documentation

Detailed documentation about the tool can be found **here (TBD)**. You can refer to the SCALE-Sim v3 paper (to be presented at ISPASS'25):

Raj, R., Banerjee, S., Chandra, N., Wan, Z., Tong, J., Samajdhar, A., & Krishna, T.; **"SCALE-Sim v3: A modular cycle-accurate systolic accelerator simulator for end-to-end system analysis."** arXiv preprint arXiv:2504.15377 (2025) [\[pdf\]](https://arxiv.org/abs/2504.15377)

We also recommend referring to the following papers for insights on SCALE-Sim's potential.

[1] Samajdar, A., Zhu, Y., Whatmough, P., Mattina, M., & Krishna, T.;  **"Scale-sim: Systolic cnn accelerator simulator."** arXiv preprint arXiv:1811.02883 (2018). [\[pdf\]](https://arxiv.org/abs/1811.02883)

[2] Samajdar, A., Joseph, J. M., Zhu, Y., Whatmough, P., Mattina, M., & Krishna, T.; **"A systematic methodology for characterizing scalability of DNN accelerators using SCALE-sim"**. In 2020 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS). [\[pdf\]](https://cpb-us-w2.wpmucdn.com/sites.gatech.edu/dist/c/332/files/2020/03/scalesim_ispass2020.pdf)

## Citing this work

If you found this tool useful, please use the following bibtex to cite us

```
@inproceedings{raj2025scale,
  title={SCALE-Sim v3: A modular cycle-accurate systolic accelerator simulator for end-to-end system analysis},
  author={Raj, Ritik and Banerjee, Sarbartha and Chandra, Nikhil and Wan, Zishen and Tong, Jianming and Samajdhar, Ananda and Krishna, Tushar},
  booktitle={2025 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)},
  pages={186--200},
  year={2025},
  organization={IEEE}
}

```

## Contributing to the project

We are happy for your contributions and would love to merge new features into our stable codebase. To ensure continuity within the project, please consider the following workflow.

When contributing to this repository, please first discuss the change you wish to make via issue,
email, or any other method with the owners of this repository before making a change.

### Pull Request Process

1. Ensure any install or build dependencies are removed before the end of the layer when doing a
   build. Please do not commit temporary files to the repo.
2. Update the documentation in the documentation/-folder with details of changes to the interface, this includes new environment
   variables, exposed ports, useful file locations and container parameters.
3. Add a tutorial how to use your new feature in form of a jupyter notebook to the documentation, as well. This makes sure that others can use your code!
4. Add test cases to our unit test system for your contribution.
5. Increase the version numbers in any example's files and the README.md to the new version that this
   Pull Request would represent. The versioning scheme we use is [SemVer](http://semver.org/). Add your changes to the CHANGELOG.md. Address the issue numbers that you are solving.
6. You may merge the Pull Request in once you have the sign-off of two other developers, or if you
   do not have permission to do that, you may request the second reviewer to merge it for you.


## Developers

Dev and maintainers:
* Ritik Raj - Lead developer (@ritikraj7)
* Sarbartha Banerjee - Ramulator feature (@iamsarbartha)
* Nikhil Chandra - Sparsity feature (@NikhilChandraNcbs)
* Zishen Wan - Accelergy feature (@zishenwan)
* Jianming Tong - SRAM Layout feature (@JianmingTONG)

Advisors
* Ananda Samajdar (@AnandS09)
* Tushar Krishna

Previous Contributers/Collaborators
* Jan Moritz Joseph (@jmjos)
* Yuhao Zhu
* Paul Whatmough
* Vineet Nadella
* Sachit Kuhar


