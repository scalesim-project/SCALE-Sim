# Author: Nikhil Chandra
# Github: @NikhilChandraNcbs
# File: Memory plot

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Function to read SPARSE_REPORT.csv and extract necessary columns
def read_sparse_report(model, size, sparsity):
    # file_path = f'sparsity_results/{model}_{size}_{sparsity}/SPARSE_REPORT.csv'
    file_path = f'sparsity/sparsity_results_128x128/{model}_{size}_{sparsity}_128x128/SPARSE_REPORT.csv'
    
    if os.path.exists(file_path):
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Extract the relevant columns
        dense_storage = df[' Original Filter Storage'].values
        sparse_filter = df[' New Storage (Filter+Metadata)'].values - df[' Filter Metadata Storage'].values
        sparse_metadata = df[' Filter Metadata Storage'].values
        
        return dense_storage, sparse_filter, sparse_metadata
    else:
        print(f"File {file_path} not found.")
        raise SystemExit    

# Function to plot the stacked bar chart
def plot_combined_storage_style1(layers, dense_storage, sparse_filter_1s4, sparse_metadata_1s4, sparse_filter_2s4, sparse_metadata_2s4, sparse_filter_3s4, sparse_metadata_3s4, title_prefix):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Set the positions and width for the bars
    bar_width = 0.2
    index = np.arange(len(layers))  # X-axis: Layers

    # Bottom bars (lighter)
    color_dense = '#3DB2FF'  # Blue
    color_sparse_filter_1s4 = '#EB3324'  # red-orange
    color_sparse_filter_2s4 = '#008000'  # Green
    color_sparse_filter_3s4 = '#EDB525'  # Green

    # Top bars (darker)
    color_sparse_metadata_1s4 = '#FA8072'  # Orange
    color_sparse_metadata_2s4 = '#00CC00'  # grass
    color_sparse_metadata_3s4 = '#5B3E0F'  # grass
    
    # Plot dense storage
    ax.bar(index, dense_storage, bar_width, label='Dense Storage', color=color_dense, edgecolor='black')

    # Plot stacked bars for sparse 1:4 (filter + metadata)
    ax.bar(index + bar_width, sparse_filter_1s4, bar_width, label='Sparse Filter (1:4)', color=color_sparse_filter_1s4, edgecolor='black')
    ax.bar(index + bar_width, sparse_metadata_1s4, bar_width, bottom=sparse_filter_1s4, label='Sparse Metadata (1:4)', color=color_sparse_metadata_1s4, edgecolor='black')

    # Plot stacked bars for sparse 2:4 (filter + metadata)
    ax.bar(index + 2 * bar_width, sparse_filter_2s4, bar_width, label='Sparse Filter (2:4)', color=color_sparse_filter_2s4, edgecolor='black')
    ax.bar(index + 2 * bar_width, sparse_metadata_2s4, bar_width, bottom=sparse_filter_2s4, label='Sparse Metadata (2:4)', color=color_sparse_metadata_2s4, edgecolor='black')

    # Plot stacked bars for sparse 3:4 (filter + metadata)
    ax.bar(index + 3 * bar_width, sparse_filter_3s4, bar_width, label='Sparse Filter (3:4)', color=color_sparse_filter_3s4, edgecolor='black')
    ax.bar(index + 3 * bar_width, sparse_metadata_3s4, bar_width, bottom=sparse_filter_3s4, label='Sparse Metadata (3:4)', color=color_sparse_metadata_3s4, edgecolor='black')

    # Labels, title, legend
    ax.set_xlabel('Layers', fontsize=20)
    ax.set_ylabel('Memory Size (Words)', fontsize=20)
    # ax.set_title(f'{title_prefix}: Memory Storage Comparison: Dense vs 1:4 vs 2:4 vs 3:4', fontsize=18)

    # Set tick labels with the desired font size and rotation
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(layers, rotation=45, ha="right", fontsize=18)
    ax.tick_params(axis='y', labelsize=18)  # Y-axis tick labels

    # Set legend font size
    ax.legend(fontsize=18)

    # Adjust layout, save and show plot
    plt.tight_layout()
    plt.savefig('sparsity/memoryplot.pdf', dpi=300, bbox_inches='tight')
    plt.show()

# Function to plot the stacked bar chart with better colors and hatching
def plot_combined_storage_style2(layers, dense_storage, sparse_filter_1s4, sparse_metadata_1s4, sparse_filter_2s4, sparse_metadata_2s4, sparse_filter_3s4, sparse_metadata_3s4, title_prefix):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Set the positions and width for the bars
    bar_width = 0.2
    index = np.arange(len(layers))  # X-axis: Layers
    
    # Use a Seaborn color palette for better visuals
    palette = sns.color_palette("Set1", 7)
    
    # Plot dense storage
    ax.bar(index, dense_storage, bar_width, label='Dense Storage', color=palette[0], hatch='//')

    # Plot stacked bars for sparse 1:4 (filter + metadata) with hatching
    ax.bar(index + bar_width, sparse_filter_1s4, bar_width, label='Sparse Filter (1:4)', color=palette[1], hatch='\\')
    ax.bar(index + bar_width, sparse_metadata_1s4, bar_width, bottom=sparse_filter_1s4, label='Sparse Metadata (1:4)', color=palette[2], hatch='..')

    # Plot stacked bars for sparse 2:4 (filter + metadata) with hatching
    ax.bar(index + 2 * bar_width, sparse_filter_2s4, bar_width, label='Sparse Filter (2:4)', color=palette[3], hatch='xx')
    ax.bar(index + 2 * bar_width, sparse_metadata_2s4, bar_width, bottom=sparse_filter_2s4, label='Sparse Metadata (2:4)', color=palette[4], hatch='++')

    # Plot stacked bars for sparse 3:4 (filter + metadata)
    ax.bar(index + 3 * bar_width, sparse_filter_3s4, bar_width, label='Sparse Filter (3:4)', color=palette[5], hatch='**')
    ax.bar(index + 3 * bar_width, sparse_metadata_3s4, bar_width, bottom=sparse_filter_3s4, label='Sparse Metadata (3:4)', color=palette[6], hatch='oo')

    # Labels, title, legend
    ax.set_xlabel('Layers')
    ax.set_ylabel('Memory Size (Words)')
    ax.set_title(f'{title_prefix}: Memory Storage Comparison: Dense vs 1:4 vs 2:4')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(layers, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.savefig('memoryplot2_rebuttal.png', dpi=300, bbox_inches='tight', format='png')
    plt.show()


model = 'Resnet18'
size = '100mb'
sparsity_1s4 = '1s4'
sparsity_2s4 = '2s4'
sparsity_3s4 = '3s4'
layers = ['Conv1', 'Conv2_1a', 'Conv2_1b', 'Conv2_2a', 'Conv2_2b', 'Conv3_1a', 'Conv3_1b', 'Conv3_s', 'Conv3_2a', 'Conv3_2b',
          'Conv4_1a', 'Conv4_1b', 'Conv4_s', 'Conv4_2a', 'Conv4_2b', 'Conv5_1a', 'Conv5_1b', 'Conv5_s', 'Conv5_2a', 'Conv5_2b', 'FC']

# Read the storage values for 1:4 and 2:4 sparsities
dense_storage, sparse_filter_1s4, sparse_metadata_1s4 = read_sparse_report(model, size, sparsity_1s4)
_, sparse_filter_2s4, sparse_metadata_2s4 = read_sparse_report(model, size, sparsity_2s4)
_, sparse_filter_3s4, sparse_metadata_3s4 = read_sparse_report(model, size, sparsity_3s4)

# Plot the storage comparison with stacked bars
plot_combined_storage_style1(layers, dense_storage, sparse_filter_1s4, sparse_metadata_1s4, sparse_filter_2s4, sparse_metadata_2s4, sparse_filter_3s4, sparse_metadata_3s4, model)

