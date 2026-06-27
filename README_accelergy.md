# SCALE-Sim + Accerlegy Integration for Enabling Energy and Power Estimation of Systolic CNN-Accelerator

## Requirement

### Install SCALE-Sim

Please install SCALE-Sim following the instructions in the main branch: https://github.com/scalesim-project/SCALE-Sim/tree/main 

### Install Accelergy

Please install Accelergy following the instructions in their github repo
https://github.com/Accelergy-Project/accelergy 

For more accurate estimation, please install accelergy-plug-ins for 3rd party, technology-based estimators
CACTI - https://github.com/Accelergy-Project/accelergy-cacti-plug-in.git 
Aladdin - https://github.com/Accelergy-Project/accelergy-aladdin-plug-in.git

## Run

Run the following command:

```$ cd rundir-accelergy ```

```$ ./run_all.sh -c <path_to_config_file> -t <path_to_topology_file> -p <path_to_scale-sim_log_dir> -o <path_to_output_dir> -i <topology_type: conv|gemm> ```

For Example: 

```$ ./run_all.sh -c ./scale.cfg -t ../topologies/conv_nets/test.csv -p ../test_runs/ -o ./output -i conv ```

## Tool Inputs
### Configuration File
The configuration file is used to specify the architecture and run parameters for the simulations. 

Built based upon SCALE-Sim, we add additional parameters for action count extraction.
* SRAM_row_size: the size of the row buffer (block) that each SRAM access loads
* SRAM_bank_size: temporal capacity for the memory to keep previously-accessed data (with more than one row buffer per bank)
* DRAM_row_size, _bank_size: same concept as SRAM

The rest of the parameters are the same as (https://github.com/scalesim-project/SCALE-Sim).


### Additional compound components for Accelergy
If any other kinds fo compound componetns are to be included, please add them to ```./accelergy_input/components```
For more details, please refer to (http://accelergy.mit.edu/)

## Result
### Files
In the output directory,
* scale_sim_out_<run_name> contains performance results in .csv format. Summary in ```COMPUTE_REPORT.csv```
* accelergy_out_<run_name> contains energy estimation results in .yaml format. Summary in  ```energy_estimation.yaml```
More details can be found from each framework's github link.

### Visualization
```$ jupyter-notebook gen_plot.ipynb``` 
will also help to anaylize the result. Currently set to the example case (see the Single Run section).

