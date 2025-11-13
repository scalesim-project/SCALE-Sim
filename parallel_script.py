import itertools
import os
import subprocess
import multiprocessing

array_heights = range(3, 11)  # [3, 10]
array_widths = range(3, 11)
ifmap_sram_sizes = range(1, 5)  # [1, 4]
filter_sram_sizes = range(1, 5)
ofmap_sram_sizes = range(1, 5)
dataflows = ["ws", "os", "is"]
root_output_dir = "./lab2A/result_ah910"



base_config = """[general]
run_name = {run_name}

[architecture_presets]
ArrayHeight : {array_height}
ArrayWidth : {array_width}
IfmapSramSzkB: {ifmap_sram_size}
FilterSramSzkB: {filter_sram_size}
OfmapSramSzkB: {ofmap_sram_size}
Dataflow : {dataflow}

IfmapOffset: 0
FilterOffset: 10000000
OfmapOffset: 20000000
Bandwidth : 50

[run_presets]
InterfaceBandwidth: USER
"""


os.makedirs(root_output_dir, exist_ok=True)

configurations = []
for params in itertools.product(array_heights, array_widths, ifmap_sram_sizes, filter_sram_sizes, ofmap_sram_sizes, dataflows):
    configurations.append(params)
print(len(configurations))

def run_experiment(params):
    array_height, array_width, ifmap_sram, filter_sram, ofmap_sram, dataflow = params
    
    # Unique folder name for this run
    run_name = f"run_AH{array_height}_AW{array_width}_IF{ifmap_sram}_FL{filter_sram}_OF{ofmap_sram}_DF{dataflow}"
    run_dir = os.path.join(root_output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Create the configuration file inside the run directory
    config_path_conv = os.path.join(run_dir, "config_conv.cfg")
    with open(config_path_conv, "w") as f:
        f.write(base_config.format(
            run_name="lenet_DSE_conv_run",
            array_height=array_height,
            array_width=array_width,
            ifmap_sram_size=ifmap_sram,
            filter_sram_size=filter_sram,
            ofmap_sram_size=ofmap_sram,
            dataflow=dataflow
        ))

    config_path_gemm = os.path.join(run_dir, "config_gemm.cfg")
    with open(config_path_gemm, "w") as f:
        f.write(base_config.format(
            run_name="lenet_DSE_gemm_run",
            array_height=array_height,
            array_width=array_width,
            ifmap_sram_size=ifmap_sram,
            filter_sram_size=filter_sram,
            ofmap_sram_size=ofmap_sram,
            dataflow=dataflow
        ))

    # Output directory for this run
    # output_dir = os.path.join(run_dir, "output")
    # os.makedirs(output_dir, exist_ok=True)

    # Command to execute
    command_conv = f"python3 ./scale-sim-v2/scalesim/scale.py -c {config_path_conv} -t lab2A/lenet_conv.csv -p {run_dir}"
    command_gemm = f"python3 ./scale-sim-v2/scalesim/scale.py -c {config_path_gemm} -t lab2A/lenet_gemm.csv -p {run_dir} -i gemm"
    # print(f"Running: {command}")

    # Run the command and capture output
    # result = subprocess.run(command, shell=True, capture_output=True, text=True)
    subprocess.run(command_conv, shell=True, capture_output=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(command_gemm, shell=True, capture_output=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Save log file
    # log_file = os.path.join(run_dir, "log.txt")
    # with open(log_file, "w") as log:
    #     log.write(result.stdout)
    #     log.write(result.stderr)


if __name__ == "__main__":
    num_workers = 64  # Adjust based on CPU cores
    print(f"Using {num_workers} parallel workers.")

    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(run_experiment, configurations)

    print("All experiments completed.")