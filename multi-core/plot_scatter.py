import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('results.csv')

# Function to create plot
def create_plot(plot_type):
    plt.figure(figsize=(15, 10))
    
    # Increase font sizes globally
    plt.rcParams.update({'font.size': 22})
    
    # Sample every nth row to reduce density
    step = 5
    df_sparse = df.iloc[::step]
    
    # For each row in the dataframe
    for index, row in df_sparse.iterrows():
        if plot_type == 'memory':
            # Get points for memory-optimized case
            points_x = [row['best_memory_spatial'], row['best_memory_spatio-temporal1'], row['best_memory_spatio-temporal2']]
            points_y = [row['best_memory_compute_spatial'], row['best_memory_compute_spatio-temporal1'], row['best_memory_compute_spatio-temporal2']]
            
            # Find point with minimum compute cycles
            min_compute_idx = np.argmin(points_y)
            
            # Connect memory-optimized points with thinner, more transparent lines
            plt.plot(points_x, points_y, 
                    color='blue', alpha=0.2, linewidth=0.5,
                    label='Same configuration' if index == df_sparse.index[0] else "")
            
            # Plot memory-optimized scatter points
            plt.scatter(row['best_memory_spatial'], 
                      row['best_memory_compute_spatial'],
                      color='red', marker='s', label='Spatial' if index == df_sparse.index[0] else "", alpha=0.6, s=120)
            
            plt.scatter(row['best_memory_spatio-temporal1'], 
                      row['best_memory_compute_spatio-temporal1'],
                      color='purple', marker='^', label='Spatiotemporal1' if index == df_sparse.index[0] else "", alpha=0.6, s=120)
            
            plt.scatter(row['best_memory_spatio-temporal2'], 
                      row['best_memory_compute_spatio-temporal2'],
                      color='pink', marker='+', label='Spatiotemporal2' if index == df_sparse.index[0] else "", alpha=0.6, s=120)
            
            # Highlight the best point with a black edge
            plt.scatter(points_x[min_compute_idx], 
                      points_y[min_compute_idx],
                      color='none', edgecolor='black', s=200, linewidth=2, 
                      label='Best Partition' if index == df_sparse.index[0] else "")
            
            title_prefix = "Memory-Optimized"
            
        else:  # compute plot
            points_x = [row['best_compute_spatial'], row['best_compute_spatio-temporal1'], row['best_compute_spatio-temporal2']]
            points_y = [row['best_compute_memory_spatial'], row['best_compute_memory_spatio-temporal1'], row['best_compute_memory_spatio-temporal2']]
            
            min_memory_idx = np.argmin(points_y)
            
            plt.plot(points_x, points_y, 
                    color='red', alpha=0.2, linewidth=0.5,
                    label='Same configuration' if index == df_sparse.index[0] else "")
            
            plt.scatter(row['best_compute_spatial'], 
                      row['best_compute_memory_spatial'],
                      color='red', marker='s', label='Spatial' if index == df_sparse.index[0] else "", alpha=0.6, s=120)
            
            plt.scatter(row['best_compute_spatio-temporal1'], 
                      row['best_compute_memory_spatio-temporal1'],
                      color='purple', marker='^', label='Spatiotemporal1' if index == df_sparse.index[0] else "", alpha=0.6, s=120)
            
            plt.scatter(row['best_compute_spatio-temporal2'], 
                      row['best_compute_memory_spatio-temporal2'],
                      color='pink', marker='+', label='Spatiotemporal2' if index == df_sparse.index[0] else "", alpha=0.6, s=120)
            
            plt.scatter(points_x[min_memory_idx], 
                      points_y[min_memory_idx],
                      color='none', edgecolor='black', s=200, linewidth=2,
                      label=f'Best Partition' if index == df_sparse.index[0] else "")
            
            title_prefix = "Compute-Optimized"

    # Add labels and title
    plt.xlabel('Compute cycles', fontsize=24)
    plt.ylabel('Memory footprint (words)', fontsize=24)

    # Add legend inside the plot
    plt.legend(loc='upper left', fontsize=22, markerscale=2)

    # Add grid with reduced opacity
    plt.grid(True, linestyle='--', alpha=0.3)

    # Use logarithmic scale for both axes
    plt.xscale('log')
    plt.yscale('log')

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'multi-core/{plot_type}_optimized_tradeoff.pdf', bbox_inches='tight', dpi=300)
    plt.close()

# Create both plots
create_plot('memory')
create_plot('compute') 