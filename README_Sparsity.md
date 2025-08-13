# Sparsity in SCALE-Sim v3

## Sparsity features in SCALE-Sim

Sparsity is a key property in many deep learning models, where a large portion of tensor elements—such as weights or activations—are either zero or very small. It is typically measured as the ratio of nonzero elements (N) to the total elements (M) in a given block, expressed as the sparsity ratio N:M.

Expanding upon its predecessor, SCALE-Sim v3 introduces advanced support for layer-wise and row-wise sparsity, optimizing systolic array architectures for more efficient neural network execution.

- Layer-Wise Sparsity: Recognizing that different layers in a neural network exhibit varying degrees of sparsity, SCALE-Sim v3 allows for customizable sparsity configurations across layers. Early layers in CNNs often feature denser connections, while deeper layers naturally accommodate higher sparsity. This flexibility enables more accurate modeling of real-world neural network behavior.

- Row-Wise Sparsity: By enforcing a fixed number of nonzero elements per row, this approach provides granular control over sparsity at the workload level. SCALE-Sim v3 implements row-wise sparsity for different N:M ratios, ensuring that each group of M elements in a row contains precisely N nonzero values. To maintain computational efficiency, the framework constrains sparsity ratios to N ≤ M/2, as exceeding this threshold leads to denser configurations, diminishing the performance gains typically associated with sparsity.

## Steps to run sparsity simulations in SCALE-Sim v3

Various files need to be modified in order for the user to interact with the tool's sparsity features. This section explains the different steps involved in utilizing the advanced features of sparsity in SCALE-Sim v3.

1. Configuration file: <br>
Check the ```sparsity.cfg``` config file in ```scale-sim-v2\configs``` folder for more details on how to specify sparsity information. Here is additional information to be specified in the config file:
    ```
    [sparsity]
    SparsitySupport : true
    SparseRep : ellpack_block
    OptimizedMapping : true
    BlockSize : 4
    ```

    1. SparsitySupport: 
        - true -> enables sparsity feature in SCALE-Sim <br>
        - false -> disables sparsity feature in SCALE-Sim
    2. SparseRep: 
        - csr -> Compressed Sparse Row 
        - csc -> Compressed Sparse Column
        - ellpack_block -> Blocked ELLPACK
    3. OptimizedMapping: 
        - true -> enables row-sparsity
        - false -> disables row-sparsity (layer-wise sparsity is the default configuration).
    4. BlockSize: M in N:M ratio for row-wise sparsity
    5. RandomNumberGeneratorSeed: A seed value to be used for random number generator used in row-wise sparsity logic. The default value is set to 40. The user can omit any changes to this setting if no control over seed is desired.

2. Topology file: <br>
An extra column named "Sparsity" has been added. The entries in this column has be in the "N:M" format. See ```scale-sim-v2\topologies\sparsity\gemm.csv``` for more details.
    ```
    Layer Name, M, N, K, Sparsity,
    GEMM_1, 3, 5, 16, 3:4,
    GEMM_2, 1, 5, 16, 1:4,
    ```

3. Command to run sparsity simulation:

    The command is no different from the default command explained in ```README.md```. Care needs to be taken to input the right files. Example commands for CONV and GEMM operations are as follows:
    ```
    python scalesim/scale.py -c configs/sparsity.cfg -t topologies/sparsity/conv.csv -p sparsity_results
    ```
    ```
    python scalesim/scale.py -c configs/sparsity.cfg -t topologies/sparsity/gemm.csv -p sparsity_results -i gemm
    ```

## Results

Along with the usual outputs/reports generated, the simulator generates a ```SPARSE_REPORT.csv``` file, providing key insights and metrics related to sparsity. This report includes details such as the sparsity representation, original filter storage, and new filter storage, which encompasses both the compressed filter matrix and its associated metadata.

We introduce a new metric, referred to as ```Filter Metadata SRAM Bandwidth```, to quantify the amount of metadata being accessed. This metric is calculated for each layer of the CNN model and represents the total number of metadata words read from the filter SRAM to the Processing Elements (PEs) during computation cycles, measured in ```words/cycle```. If the ```SparsitySupport``` option is set to false, the filter metadata SRAM bandwidth is considered zero.

## Developers

Main developer for sparsity:
* Nikhil Chandra (@NikhilChandraNcbs)

Contributers:
* Ritik Raj (@ritikraj7)

Maintainers and Advisors
* Tushar Krishna