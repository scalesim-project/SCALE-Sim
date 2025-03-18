import math
import pandas as pd

def find_factors(n):
    """
    Find all factor pairs of a number n
    Returns a list of tuples containing factor pairs
    """
    factors = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            factors.append((i, n // i))
    return factors

def calculate_metrics(M, N, K, num_cores, R, C, dataflow, partition, P):
    # Initialize variables to track best results
    min_compute = float('inf')
    min_memory = float('inf')
    min_compute_config_memory = float('inf')  # Memory for min compute configuration
    min_memory_config_compute = float('inf')  # Compute for min memory configuration
    
    # Set dataflow-specific parameters
    if dataflow == 'os':
        Sr, Sc, T = M, N, K
    elif dataflow == 'ws':
        Sr, Sc, T = K, M, N
    elif dataflow == 'is':
        Sr, Sc, T = K, N, M
    else:
        return float('inf'), float('inf'), float('inf'), float('inf')
    
    if partition == 'spatial':
        for pair in P:
            # First orientation
            compute = (2*R+C+T-2)*math.ceil(Sr/pair[0]/R)*math.ceil(Sc/pair[1]/C)
            memory = num_cores*(Sr*T/pair[0]+Sc*T/pair[1]+Sc*Sr/(pair[0]*pair[1]))
            
            if compute < min_compute:
                min_compute = compute
                min_compute_config_memory = memory
            if memory < min_memory:
                min_memory = memory
                min_memory_config_compute = compute
            
            # Second orientation
            compute = (2*R+C+T-2)*math.ceil(Sr/pair[1]/R)*math.ceil(Sc/pair[0]/C)
            memory = num_cores*(Sr*T/pair[1]+Sc*T/pair[0]+Sc*Sr/(pair[0]*pair[1]))
            
            if compute < min_compute:
                min_compute = compute
                min_compute_config_memory = memory
            if memory < min_memory:
                min_memory = memory
                min_memory_config_compute = compute
            
    elif partition == 'spatio-temporal1':
        for pair in P:
            # First orientation
            compute = (2*R+C+math.ceil(T/pair[0])-2)*math.ceil(Sr/R)*math.ceil(Sc/pair[1]/C)
            memory = num_cores*(Sr*T/pair[0]+Sc*T/(pair[0]*pair[1])+Sc*Sr/pair[1])
            
            if compute < min_compute:
                min_compute = compute
                min_compute_config_memory = memory
            if memory < min_memory:
                min_memory = memory
                min_memory_config_compute = compute
            
            # Second orientation
            compute = (2*R+C+math.ceil(T/pair[1])-2)*math.ceil(Sr/R)*math.ceil(Sc/pair[0]/C)
            memory = num_cores*(Sr*T/pair[1]+Sc*T/(pair[1]*pair[0])+Sc*Sr/pair[0])
            
            if compute < min_compute:
                min_compute = compute
                min_compute_config_memory = memory
            if memory < min_memory:
                min_memory = memory
                min_memory_config_compute = compute
            
    elif partition == 'spatio-temporal2':
        for pair in P:
            # First orientation
            compute = (2*R+C+math.ceil(T/pair[0])-2)*math.ceil(Sr/pair[1]/R)*math.ceil(Sc/C)
            memory = num_cores*(Sr*T/(pair[0]*pair[1])+Sc*T/(pair[0])+Sc*Sr/pair[1])
            
            if compute < min_compute:
                min_compute = compute
                min_compute_config_memory = memory
            if memory < min_memory:
                min_memory = memory
                min_memory_config_compute = compute
            
            # Second orientation
            compute = (2*R+C+math.ceil(T/pair[1])-2)*math.ceil(Sr/pair[0]/R)*math.ceil(Sc/C)
            memory = num_cores*(Sr*T/(pair[0]*pair[1])+Sc*T/(pair[1])+Sc*Sr/pair[0])
            
            if compute < min_compute:
                min_compute = compute
                min_compute_config_memory = memory
            if memory < min_memory:
                min_memory = memory
                min_memory_config_compute = compute
    
    return min_memory, min_compute, min_memory_config_compute, min_compute_config_memory

# Parameters ranges
M_values = [1000, 5000, 10000]
N_values = [1000, 5000, 10000]
K_values = [1000, 5000, 10000]
R_values = [8,16,32]
C_values = [8,16,32]
num_cores_values = [16,32,64]

# Lists to store results
results = []

# Iterate through all combinations
count = 0
total_count = 0

for M in M_values:
    for N in N_values:
        for K in K_values:
            for R in R_values:
                for C in C_values:
                    for num_cores in num_cores_values:
                        # Get factor pairs
                        P = find_factors(num_cores)
                        
                        # Initialize lists to store results for each partition
                        dataflows = ['os', 'ws', 'is']
                        partitions = ['spatial', 'spatio-temporal1', 'spatio-temporal2']

                        # Calculate best metrics for each partition
                        best_metrics = {}
                        for partition in partitions:
                            best_memory = float('inf')
                            best_compute = float('inf')
                            best_memory_compute = float('inf')
                            best_compute_memory = float('inf')
                            bes_memory_times_compute = float('inf')
                            
                            for dataflow in dataflows:
                                memory, compute, memory_compute, compute_memory = calculate_metrics(
                                    M, N, K, num_cores, R, C, dataflow, partition, P)
                              
                                if memory < best_memory:
                                    best_memory = memory
                                    best_memory_compute = memory_compute
                                if compute < best_compute:
                                    best_compute = compute
                                    best_compute_memory = compute_memory
                                if memory*compute < bes_memory_times_compute:
                                    bes_memory_times_compute = memory*compute
                            best_metrics[f'best_memory_{partition}'] = best_memory
                            best_metrics[f'best_compute_{partition}'] = best_compute
                            best_metrics[f'best_memory_compute_{partition}'] = best_memory_compute
                            best_metrics[f'best_compute_memory_{partition}'] = best_compute_memory
                            best_metrics[f'best_memory_times_compute_{partition}'] = bes_memory_times_compute
                        
                        # Create row dictionary
                        row = {
                            'M': M,
                            'N': N,
                            'K': K,
                            'num_cores': num_cores,
                            'R': R,
                            'C': C,
                            'best_memory_spatial': best_metrics['best_memory_spatial'],
                            'best_memory_compute_spatial': best_metrics['best_memory_compute_spatial'],
                            'best_compute_spatial': best_metrics['best_compute_spatial'],
                            'best_compute_memory_spatial': best_metrics['best_compute_memory_spatial'],
                            'best_memory_times_compute_spatial': best_metrics['best_memory_times_compute_spatial'],
                            'best_memory_spatio-temporal1': best_metrics['best_memory_spatio-temporal1'],
                            'best_memory_compute_spatio-temporal1': best_metrics['best_memory_compute_spatio-temporal1'],
                            'best_compute_spatio-temporal1': best_metrics['best_compute_spatio-temporal1'],
                            'best_compute_memory_spatio-temporal1': best_metrics['best_compute_memory_spatio-temporal1'],
                            'best_memory_times_compute_spatio-temporal1': best_metrics['best_memory_times_compute_spatio-temporal1'],
                            'best_memory_spatio-temporal2': best_metrics['best_memory_spatio-temporal2'],
                            'best_memory_compute_spatio-temporal2': best_metrics['best_memory_compute_spatio-temporal2'],
                            'best_compute_spatio-temporal2': best_metrics['best_compute_spatio-temporal2'],
                            'best_compute_memory_spatio-temporal2': best_metrics['best_compute_memory_spatio-temporal2'],
                            'best_memory_times_compute_spatio-temporal2': best_metrics['best_memory_times_compute_spatio-temporal2'],
                        }
                        p1 = row['best_memory_spatial']*row['best_memory_compute_spatial']
                        p2 = row['best_compute_spatial']*row['best_compute_memory_spatial']
                        
                        p3 = row['best_memory_spatio-temporal1']*row['best_memory_compute_spatio-temporal1']
                        p4 = row['best_compute_spatio-temporal1']*row['best_compute_memory_spatio-temporal1']
                        
                        p5 = row['best_memory_spatio-temporal2']*row['best_memory_compute_spatio-temporal2']
                        p6 = row['best_compute_spatio-temporal2']*row['best_compute_memory_spatio-temporal2']

                        spatial_p = min(p1,p2) 
                        spatiotemporal_p = min(p3, p4, p5, p6) 
                        if spatiotemporal_p < spatial_p:
                            count = count + 1
                        total_count += 1
                        results.append(row)

# Create DataFrame from all results
df = pd.DataFrame(results)

# Print first few rows

# Save to CSV
df.to_csv('results.csv', index=False)

# Print total number of configurations
print(f"\nTotal configurations analyzed: {len(df)}")

