# Author: Nikhil Chandra
# Github: @NikhilChandraNcbs
# File: Scatter plot for compute cycles

import os
import subprocess
import csv
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse

# Define paths
cfg_path = "configs/new"
csv_path = "topologies/sparsity/new"
results_path = "sparsity_results/new"
scalesim_command_template = "python scalesim/scale.py -c {cfg_file} -t {csv_file} -p {results_path} -i gemm"

# Ensure directories exist
os.makedirs(cfg_path, exist_ok=True)
os.makedirs(csv_path, exist_ok=True)
os.makedirs(results_path, exist_ok=True)

# Model configurations
MODEL_CONFIGS = {
    "vits": {"dims": "L4,196,384,1536", "label": "ViT-S"},
    "vitb": {"dims": "L4,196,768,3072", "label": "ViT-B"},
    "vitl": {"dims": "L4,196,4096,1024", "label": "ViT-L"},
    "vith": {"dims": "L4,256,1280,5120", "label": "ViT-H"}
}

# Function to generate .cfg content
def generate_cfg_content(array_height, array_width):
    return f"""[general]
run_name = cfg_{array_height}x{array_width}

[architecture_presets]
ArrayHeight : {array_height}
ArrayWidth :  {array_width}
IfmapSramSzkB:   256
FilterSramSzkB:  256
OfmapSramSzkB:   256
IfmapOffset:    0
FilterOffset:   10000000
OfmapOffset:    20000000
Bandwidth : 50
Dataflow : ws
MemoryBanks:   1

[sparsity]
SparsitySupport : true
SparseRep : ellpack_block
OptimizedMapping : false
BlockSize : {array_height}
RandomNumberGeneratorSeed : 40

[run_presets]
InterfaceBandwidth: USER"""

# Function to generate .csv content
def generate_csv_content(array_height, sparsity, model):
    dimensions = MODEL_CONFIGS[model]["dims"]
    return f"L,M,N,K,Sparsity,\n{dimensions},{sparsity},"

# Main script
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run graph simulations for ViT models')
    parser.add_argument('--stage', choices=['1a', '1b', '2'], required=True, help='Execution stage')
    parser.add_argument('--model', choices=list(MODEL_CONFIGS.keys()), help='ViT model type (required for stages 1a, 1b)')
    args = parser.parse_args()
    
    # Set flags based on stage
    if args.stage in ['1a', '1b']:
        if not args.model:
            parser.error("--model is required for stages 1a and 1b")
        
        generate_cfg = True
        generate_csv = True
        execute_commands = True
        generate_graphs = False
        
        # Set output paths based on stage and model
        if args.stage == '1a':
            output_csv_path = f"s1a_output_results_{args.model}.csv"
            use_fixed_cfg = False
        else:  # stage 1b
            output_csv_path = f"s1b_output_results_{args.model}.csv"
            use_fixed_cfg = True
            
    else:  # stage 2
        generate_cfg = False
        generate_csv = False
        execute_commands = False
        generate_graphs = True
        output_csv_path = ""  # Not used in stage 2
        output_plot_path = "sparsity/vit_compute_cycles_scatterplot.pdf"
        use_fixed_cfg = False  # Not relevant in stage 2
        
    array_sizes = [(4, 4), (8, 8), (16, 16), (32, 32)]  # Possible ArrayHeight and ArrayWidth values
    commands = []  # To store all generated commands
    results = []  # To store results for plotting

    if args.stage in ['1a', '1b']:
        for ah, aw in array_sizes:
            # Generate .cfg file if enabled
            cfg_filename = f"config_{ah}x{aw}.cfg"
            cfg_filepath = os.path.join(cfg_path, cfg_filename)
            if generate_cfg:
                with open(cfg_filepath, "w") as cfg_file:
                    cfg_file.write(generate_cfg_content(ah, aw))
                print(f"Generated CFG: {cfg_filepath}")
            
            # Generate .csv files for each sparsity ratio (1:M to M:M)
            for n in range(1, ah + 1):  # N varies from 1 to M where M=ArrayHeight
                sparsity_ratio = f"{n}:{ah}"
                csv_filename = f"{args.model}_{ah}x{aw}_{n}s{ah}.csv"
                csv_filepath = os.path.join(csv_path, csv_filename)
                
                if generate_csv:
                    with open(csv_filepath, "w") as csv_file:
                        csv_file.write(generate_csv_content(ah, sparsity_ratio, args.model))
                    print(f"Generated CSV: {csv_filepath}")
                
                # Generate the simulation command based on stage
                if use_fixed_cfg:
                    cfg_file = os.path.join(cfg_path, f"config_32x32.cfg")
                else:
                    cfg_file = cfg_filepath
                
                command = scalesim_command_template.format(
                    cfg_file=cfg_file,
                    csv_file=csv_filepath,
                    results_path=results_path
                )
                commands.append((command, f"{ah}x{aw}", sparsity_ratio))  # Store command with context

        # Execute commands if enabled
        if execute_commands:
            print("\nExecuting Commands...\n")
            for command, array_size, sparsity_ratio in commands:
                print(f"Executing: {command}")
                try:
                    result = subprocess.run(command, shell=True, capture_output=True, text=True)
                    output = result.stdout
                    compute_cycles = None
                    
                    # Extract COMPUTE CYCLES value
                    if "Compute cycles: " in output:
                        compute_cycles = output.split("Compute cycles: ")[1].strip().split()[0]
                    
                    if compute_cycles:
                        results.append([f"config_{array_size}", array_size, sparsity_ratio, int(compute_cycles)])
                        if use_fixed_cfg:
                            print(f"Extracted Compute Cycles: {compute_cycles} for 32x32, {sparsity_ratio}")
                        else:
                            print(f"Extracted Compute Cycles: {compute_cycles} for {array_size}, {sparsity_ratio}")

                    else:
                        print(f"Failed to extract compute cycles for: {command}")
                except Exception as e:
                    print(f"Error executing command: {e}")
        
            # Write results to output CSV
            with open(output_csv_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Config File", "Array Size", "Sparsity Ratio", "Compute Cycles"])
                writer.writerows(results)
        
            print(f"\nResults saved to {output_csv_path}")

    # Generate graphs for stage 2
    if generate_graphs:
        plot_results_combined_scatter(
            "s1a_output_results_vits.csv", 
            "s1b_output_results_vits.csv", 
            "s1a_output_results_vitb.csv", 
            "s1b_output_results_vitb.csv", 
            "s1a_output_results_vitl.csv", 
            "s1b_output_results_vitl.csv", 
            "s1a_output_results_vith.csv", 
            "s1b_output_results_vith.csv", 
            output_plot_path)

def plot_results_combined_scatter(csv_file1_s1a, csv_file1_s1b, csv_file2_s1a, csv_file2_s1b, csv_file3_s1a, csv_file3_s1b, csv_file4_s1a, csv_file4_s1b, output_filename):
    def read_csv(file):
        results = []
        with open(file, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                results.append(row)
        return results

    def organize_data(results):
        data = defaultdict(lambda: defaultdict(int))  # {array_size: {sparsity_ratio: compute_cycles}}
        for row in results:
            array_size = row["Array Size"]
            sparsity_ratio = row["Sparsity Ratio"]
            compute_cycles = int(row["Compute Cycles"])
            data[array_size][sparsity_ratio] = compute_cycles
        return data

    # Read and organize data
    results1_s1a = read_csv(csv_file1_s1a)
    results1_s1b = read_csv(csv_file1_s1b)
    results2_s1a = read_csv(csv_file2_s1a)
    results2_s1b = read_csv(csv_file2_s1b)
    results3_s1a = read_csv(csv_file3_s1a)
    results3_s1b = read_csv(csv_file3_s1b)
    results4_s1a = read_csv(csv_file4_s1a)
    results4_s1b = read_csv(csv_file4_s1b)

    data1_s1a = organize_data(results1_s1a)
    data1_s1b = organize_data(results1_s1b)
    data2_s1a = organize_data(results2_s1a)
    data2_s1b = organize_data(results2_s1b)
    data3_s1a = organize_data(results3_s1a)
    data3_s1b = organize_data(results3_s1b)
    data4_s1a = organize_data(results4_s1a)
    data4_s1b = organize_data(results4_s1b)

    # Sort array sizes
    sorted_array_sizes = sorted(data1_s1a.keys(), key=lambda x: int(x.split('x')[0]))
    x_positions = []  # X positions of scatter points
    sparsity_labels = []
    array_labels_positions = []
    current_x = 0

    plt.figure(figsize=(18, 10))

    # Generate scatter points for both files
    for array_size in sorted_array_sizes:
        sparsity_ratios = sorted(data1_s1a[array_size].keys(), key=lambda x: int(x.split(':')[0]))  # Sort N:M
        start_x = current_x  # Start position for this array size group

        for sr in sparsity_ratios:
            # Scatter for output_results1.csv
            plt.scatter(
                current_x, data1_s1a[array_size][sr],
                color='tab:blue', alpha=1, label="File 1 S1A" if current_x == 0 else "", s=50, marker='^'
            )
            # Scatter for output_results2.csv
            plt.scatter(
                current_x, data1_s1b[array_size][sr],
                color='tab:blue', alpha=1, label="File 1 S1B" if current_x == 0 else "", s=50, marker='v'
            )
            # Scatter for output_results1.csv
            plt.scatter(
                current_x, data2_s1a[array_size][sr],
                color='tab:orange', alpha=1, label="File 2 S1A" if current_x == 0 else "", s=50, marker='^'
            )
            # Scatter for output_results2.csv
            plt.scatter(
                current_x, data2_s1b[array_size][sr],
                color='tab:orange', alpha=1, label="File 2 S1B" if current_x == 0 else "", s=50, marker='v'
            )
            # Scatter for output_results1.csv
            plt.scatter(
                current_x, data3_s1a[array_size][sr],
                color='tab:green', alpha=1, label="File 3 S1A" if current_x == 0 else "", s=50, marker='^'
            )
            # Scatter for output_results2.csv
            plt.scatter(
                current_x, data3_s1b[array_size][sr],
                color='tab:green', alpha=1, label="File 3 S1B" if current_x == 0 else "", s=50, marker='v'
            )
            # Scatter for output_results1.csv
            plt.scatter(
                current_x, data4_s1a[array_size][sr],
                color='tab:red', alpha=1, label="File 4 S1A" if current_x == 0 else "", s=50, marker='^'
            )
            # Scatter for output_results2.csv
            plt.scatter(
                current_x, data4_s1b[array_size][sr],
                color='tab:red', alpha=1, label="File 4 S1B" if current_x == 0 else "", s=50, marker='v'
            )

            x_positions.append(current_x)
            sparsity_labels.append(sr)
            current_x += 1  # Move to next position

        # Calculate center position for array size labels
        end_x = current_x - 1
        group_mid = (start_x + end_x) / 2
        array_labels_positions.append((group_mid, array_size))

        # Add space after each array size group
        current_x += 1

    # Formatting x-axis
    plt.xticks(x_positions, sparsity_labels, rotation=90, fontsize=20)
    plt.yticks(fontsize=26)
    plt.xlabel("Sparsity Ratios", fontsize=26, labelpad=45)
    plt.ylabel("Compute Cycles", fontsize=26)
    plt.yscale("log")  # Logarithmic Y-axis

    # Increase bottom margin to accommodate array size labels
    plt.subplots_adjust(bottom=0.3)

    # Print array size labels centered below each group
    for position, label in array_labels_positions:
        plt.text(position, min(min(d.values()) for d in (data1_s1a.values())) * 0.18, label,
                 ha='center', va='top', fontsize=26, color="#000000")
    for position, label in array_labels_positions:
        plt.text(position, min(min(d.values()) for d in (data1_s1a.values())) * 0.12, "32x32",
                 ha='center', va='top', fontsize=26, color="#000000", alpha=0.7)

    # Add legend
    plt.legend(["ViT-S Variable PE array with blocksize = PE array dimension", "ViT-S Fixed 32x32 with variable block sizes = 4,8,16,32",
                "ViT-B Variable PE array with blocksize = PE array dimension", "ViT-B Fixed 32x32 with variable block sizes = 4,8,16,32",
                "ViT-L Variable PE array with blocksize = PE array dimension", "ViT-L Fixed 32x32 with variable block sizes = 4,8,16,32",
                "ViT-H Variable PE array with blocksize = PE array dimension", "ViT-H Fixed 32x32 with variable block sizes = 4,8,16,32"], 
                fontsize=18)

    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Scatter plot saved to {output_filename}")
    plt.show()

if __name__ == "__main__":
    main()