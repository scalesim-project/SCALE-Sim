# Systolic CNN AcceLErator Simulator (SCALE Sim) v2

[![Documentation Status](https://readthedocs.org/projects/scale-sim-project/badge/?version=latest)](https://scale-sim-project.readthedocs.io/en/latest/?badge=latest)

SCALE Sim is a simulator for systolic array based accelerators for Convolution, Feed Forward, and any layer that uses GEMMs.
This is a refreshed version of the simulator with feature enhancements, restructured code to aid feature additions, and ease of distribution.

![scalesim overview](https://github.com/scalesim-project/scale-sim-v2/blob/doc/anand/readme/documentation/resources/scalesim-overview.png "scalesim overview")

The previous version of the simulator can be found [here](https://github.com/ARM-software/SCALE-Sim).

## Getting started in 30 seconds

### *Installing the package*

Getting started is simple! SCALE-Sim is completely written in python and is available both as a package and could be run from source.

You can install SCALE-Sim in your environment using the following command

```$ pip3 install scalesim```

Alternatively you can install the package from the source as well

```$ python3 setup.py install```

### *Launching a run*

SCALE-Sim can be run by using the ```scale.py``` script from the repository and providing the paths to the architecture configuration, and the topology descriptor csv file.

```$ python3 scale.py -c <path_to_config_file> -t <path_to_topology_file> -p <path_to_output_log_dir>```

Try it now in this jupyter [notebook](https://github.com/scalesim-project/scalesim-tutorial-materials/blob/main/scaledemo.ipynb).

### *Running from source*

The above method uses the installed package for running the simulator.
In cases where you would like to run directly from the source, the following command should be used instead.

```$ python3 <scale_sim_repo_root>/scalesim/scale.py -c <path_to_config_file> -t <path_to_topology_file>```

If you are running from sources for the first time and do not have all the dependencies installed, please install them first  using the following command.

```$ pip3 install -r <scale_sim_repo_root>/requirements.txt```

### *Using Sparsity in SCALE-Sim*

Sparsity refers to the presence of many zero or empty values in a dataset, matrix, or model, making it computationally efficient. For a deeper dive into sparsity and its usage, refer to the ```README_Sparsity.md``` file.

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

## Detailed Documentation

Detailed documentation about the tool can be found [here](https://scale-sim-project.readthedocs.io/en/latest/).

We also recommend referring to the following papers for insights on SCALE-Sim's potential.

[1] Samajdar, A., Zhu, Y., Whatmough, P., Mattina, M., & Krishna, T.;  **"Scale-sim: Systolic cnn accelerator simulator."** arXiv preprint arXiv:1811.02883 (2018). [\[pdf\]](https://arxiv.org/abs/1811.02883)

[2] Samajdar, A., Joseph, J. M., Zhu, Y., Whatmough, P., Mattina, M., & Krishna, T.; **"A systematic methodology for characterizing scalability of DNN accelerators using SCALE-sim"**. In 2020 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS). [\[pdf\]](https://cpb-us-w2.wpmucdn.com/sites.gatech.edu/dist/c/332/files/2020/03/scalesim_ispass2020.pdf)

## Citing this work

If you found this tool useful, please use the following bibtex to cite us

```
@article{samajdar2018scale,
  title={SCALE-Sim: Systolic CNN Accelerator Simulator},
  author={Samajdar, Ananda and Zhu, Yuhao and Whatmough, Paul and Mattina, Matthew and Krishna, Tushar},
  journal={arXiv preprint arXiv:1811.02883},
  year={2018}
}

@inproceedings{samajdar2020systematic,
  title={A systematic methodology for characterizing scalability of DNN accelerators using SCALE-sim},
  author={Samajdar, Ananda and Joseph, Jan Moritz and Zhu, Yuhao and Whatmough, Paul and Mattina, Matthew and Krishna, Tushar},
  booktitle={2020 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)},
  pages={58--68},
  year={2020},
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
5. Increase the version numbers in any example’s files and the README.md to the new version that this
   Pull Request would represent. The versioning scheme we use is [SemVer](http://semver.org/). Add your changes to the CHANGELOG.md. Address the issue numbers that you are solving.
4. You may merge the Pull Request in once you have the sign-off of two other developers, or if you
   do not have permission to do that, you may request the second reviewer to merge it for you.


## Developers

Main devs:
* Ananda Samajdar (@AnandS09)
* Jan Moritz Joseph (@jmjos)

Contributers:
* Ritik Raj (@ritikraj7)

Maintainers and Advisors
* Yuhao Zhu
* Paul Whatmough
* Tushar Krishna

Past contributors
* Vineet Nadella
* Sachit Kuhar
* Nikhil Chandra
