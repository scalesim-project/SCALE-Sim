# Author: Nikhil Chandra
# Github: @NikhilChandraNcbs
# File: Compute plot

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to read compute cycles from the COMPUTE_REPORT.csv
def read_compute_cycles(model, size, sparsity):
    file_path = f'sparsity/sparsity_results_128x128/{model}_{size}_{sparsity}_128x128/COMPUTE_REPORT.csv'
    
    if os.path.exists(file_path):
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Extract LayerID and Total Cycles
        return df[['LayerID', ' Total Cycles']]
    else:
        print(f"File {file_path} not found.")
        return None

# Function to group layers and calculate total cycles for each group (Conv1, Conv2, etc.)
def group_layers_by_conv(layer_cycles):
    # Define which layers belong to which convolution groups
    conv_groups = {
        "Conv1": [0],  # Conv1 corresponds to Layer 0
        "Conv2": [1, 2, 3, 4],  # Conv2 layers
        "Conv3": [5, 6, 7, 8, 9],  # Conv3 layers
        "Conv4": [10, 11, 12, 13, 14],  # Conv4 layers
        "Conv5": [15, 16, 17, 18, 19],  # Conv5 layers
        "FC"   : [20] # FC layer
    }
    
    grouped_cycles = {}
    
    # Group total cycles by convolution groups
    for group, layer_ids in conv_groups.items():
        total_cycles = layer_cycles[layer_cycles['LayerID'].isin(layer_ids)][' Total Cycles'].sum()
        grouped_cycles[group] = total_cycles
    
    return grouped_cycles

# Function for compute cycles plot
def plot_cycles_alpha(on_chip_memory_sizes, cycles_dict, sizes):
    # Set up figure
    plt.figure(figsize=(11, 7))

    # Predefined line styles and markers
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', 'D', '^', 'v', 'p', 'P', '*', 'x', '+']

    # Coordinates for annotating the points: Valid only for Resnet 1s4, 2s4, 4s4 custom runs; fontsize=10.
    # Comment the annotating code for other runs
    x_offsets_annotate = {
        "Resnet18_1s4": {  
            '16kb': 27,   
            '64kb': 10,
            '256kb': 56,   
            '1mb': -30     
        },
        "Resnet18_2s4": {  
            '16kb': 30,
            '64kb': 10,
            '256kb': 60,   
            '1mb': -30
        },
        "Resnet18_4s4": {  
            '16kb': 60,
            '64kb': 60,   
            '256kb': 60,
            '1mb': -33    
        }
    }
    
    y_offsets_annotate = {
        "Resnet18_1s4": {  
            '16kb': -18,   
            '64kb': -18,
            '256kb': -18,  
            '1mb': -18     
        },
        "Resnet18_2s4": {  
            '16kb': -18,
            '64kb': 10,
            '256kb': 10,   
            '1mb': 10
        },
        "Resnet18_4s4": {  
            '16kb': 10,
            '64kb': 10,    
            '256kb': 10,
            '1mb': 10     
        }
    }

    # Loop over each model and sparsity combination and plot
    for i, (model_size_sparsity, cycles_by_memory) in enumerate(cycles_dict.items()):
        # Extract the means and standard deviations for each memory size
        means = [np.mean(list(cycles.values())) for cycles in cycles_by_memory]
        positive_errors = [max(list(cycles.values())) - np.mean(list(cycles.values())) for cycles in cycles_by_memory]
        negative_errors = [np.mean(list(cycles.values())) - min(list(cycles.values())) for cycles in cycles_by_memory]
        asymmetric_errors = [negative_errors, positive_errors]

        # Use a unique color, line style, and marker for each model
        color = f"C{i % 10}"  # Use default color cycle
        line_style = line_styles[i % len(line_styles)]
        marker = markers[i % len(markers)]

        # Add a slight horizontal offset to avoid overlap (only for visualization)
        x_offsets = np.linspace(-0.02, 0.02, len(cycles_dict))  # Small offset range
        x_values = [x + x_offsets[i] for x in on_chip_memory_sizes]

        # Plot fully opaque line connecting markers
        plt.plot(
            x_values, 
            means, 
            linestyle=line_style, 
            marker=marker, 
            label=model_size_sparsity, 
            linewidth=2, 
            markersize=6, 
            alpha=1.0, 
            color=color
        )

        # Plot error bars
        errbar = plt.errorbar(
            x_values, 
            means, 
            yerr=asymmetric_errors, 
            fmt='none', 
            capsize=10, 
            mew=1.5,
            elinewidth=1.5, 
            color=color
        )

        # Separate control over vertical lines and caps
        for line in errbar[2]:  # Vertical error bar lines
            line.set_alpha(0.15)  # Set alpha for vertical lines

        for cap in errbar[1]:  # Horizontal cap lines
            cap.set_alpha(1)  # Set alpha for horizontal caps

        # Annotate each point with its coordinates
        # Set the "if" condition  to "False" for other runs
        if True:
            for idx, (x, y) in enumerate(zip(x_values, means)):
                offset_x = x_offsets_annotate[model_size_sparsity][sizes[idx]]
                offset_y = y_offsets_annotate[model_size_sparsity][sizes[idx]]
                
                if (i == 1 and x > 760 and x < 770) or (i == 2 and x > 3000): # for only the 2 points
                    plt.annotate(
                        f"({x:.2f}, {y:.2f})", (x, y),
                        textcoords="offset points",
                        xytext=(offset_x, offset_y),
                        ha="center",
                        fontsize=10,
                        color=color,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.7)
                    )

    # Set x-axis to log scale
    plt.yscale('log')
    plt.xlabel("On-Chip Memory (kB)", fontsize=18, labelpad=10)
    plt.ylabel("Total Compute Cycles", fontsize=18, labelpad=10)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.savefig('sparsity/resnet18_compute_cycles.pdf', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    models = ['Resnet18'] # ['resnet18', 'alexnet']
    sizes = ['16kb', '64kb', '256kb', '1mb'] # ['1kb', '16kb', '64kb', '256kb', '1mb', '5mb', '10mb', '50mb', '100mb']
    sparsities = ['1s4', '2s4', '4s4']
    on_chip_memory_sizes = [16, 64, 256, 1024] # [1, 16, 64, 256, 1024, 5120, 10240, 51200, 102400]
    on_chip_memory_sizes = [x*3 for x in on_chip_memory_sizes]
    
    cycles_dict = {}

    # Loop over models, sizes, and sparsities
    for model in models:
        for sparsity in sparsities:
            # Store cycles for each combination of model, size, and sparsity
            cycles_by_memory = []

            for size in sizes:
                # Read compute cycles for each model, size, and sparsity
                layer_cycles = read_compute_cycles(model, size, sparsity)
                
                if layer_cycles is not None:
                    # Group layers by conv type and get total cycles for each conv group
                    grouped_cycles = group_layers_by_conv(layer_cycles)
                    cycles_by_memory.append(grouped_cycles)
            
            # Add the total cycles for plotting later
            cycles_dict[f'{model}_{sparsity}'] = cycles_by_memory

    # Plot the results
    plot_cycles_alpha(on_chip_memory_sizes, cycles_dict, sizes)
