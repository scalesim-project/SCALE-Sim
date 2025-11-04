"""
This module implements the 'systolic_compute_is' class, which simulates a systolic array with Weight
Stationary (WS) dataflow. It handles operand prefetching, demand matrix creation, and performance
metrics such as mapping efficiency and compute utilization. It also tracks read and write requests
for IFMAP, Filter, and OFMAP operations.
"""

import math
import numpy as np
from tqdm import tqdm
from scalesim.scale_config import scale_config as cfg
from scalesim.compute.compression import compression as cp

class systolic_compute_ws:
    """
    Class that computes the output using Weight Stationary dataflow.
    """
    #
    def __init__(self):
        """
        __init__ method.
        """
        # Params set by user
        self.config = cfg()
        self.sparsity_ratio_N = 1
        self.sparsity_ratio_M = 1

        self.ifmap_op_mat = np.zeros((1, 1))
        self.ofmap_op_mat = np.zeros((1, 1))
        self.filter_op_mat = np.zeros((1, 1))

        # Derived parameters
        self.Sr = 0
        self.Sc = 0
        self.T = 0

        self.arr_row = 0
        self.arr_col = 0

        self.row_fold = 1
        self.col_fold = 1

        # Generated matrices
        self.ifmap_op_mat_trans = np.zeros((1,1))
        self.ifmap_prefetch_matrix = np.zeros((1,1))
        self.filter_prefetch_matrix = np.zeros((1,1))

        self.ifmap_demand_matrix = np.zeros((1,1))
        self.ofmap_demand_matrix = np.zeros((1,1))
        self.filter_demand_matrix = np.zeros((1,1))

        self.ifmap_op_mat_original = np.zeros((1,1))
        self.sparsity_filter_array = np.zeros((1,1))

        # Generated metrics
        self.ifmap_reads = 0
        self.filter_reads = 0
        self.ofmap_writes = 0

        self.mapping_efficiency_per_fold = []
        self.compute_utility_per_fold = []

        # Flags
        self.params_set_flag = False
        self.prefetch_mat_ready_flag = False
        self.demand_mat_ready_flag = False

        # Compression
        self.compression = cp()

    #
    def set_params(self,
                   config_obj=cfg(),
                   ifmap_op_mat = np.zeros((1,1)),
                   ofmap_op_mat = np.zeros((1,1)),
                   filter_op_mat = np.zeros((1,1)),
                   sparsity_ratio_N = 1,
                   sparsity_ratio_M = 1,
                   ifmap_op_mat_original = np.zeros((1,1)),
                   sparsity_filter_array = np.zeros((1,1))
                ):
        """
        Method to set the weight stationary run parameters for housekeeping.
        """

        self.config = config_obj
        self.ifmap_op_mat = ifmap_op_mat
        self.filter_op_mat = filter_op_mat
        self.ofmap_op_mat = ofmap_op_mat
        self.sparsity_ratio_N = sparsity_ratio_N
        self.sparsity_ratio_M = sparsity_ratio_M

        self.ifmap_op_mat_original = ifmap_op_mat_original
        self.sparsity_filter_array = sparsity_filter_array

        ifmap_col = self.ifmap_op_mat.shape[1]
        filter_row = self.filter_op_mat.shape[0]

        if not self.config.sparsity_support and not self.config.sparsity_optimized_mapping:
            assert ifmap_col == filter_row, "Dimension mismatch between operands"

        self.Sr = self.ifmap_op_mat.shape[1]
        self.Sc = self.filter_op_mat.shape[1]
        self.T = self.ifmap_op_mat.shape[0]

        self.arr_row, self.arr_col = self.config.get_array_dims()

        self.row_fold = math.ceil(self.Sr / self.arr_row) # to be used for prefetch matrices
        self.row_fold_demand_matrices = math.ceil(self.filter_op_mat.shape[0] / self.arr_row)
        self.col_fold = math.ceil(self.Sc / self.arr_col)

        self.params_set_flag = True

    #
    def create_prefetch_matrices(self):
        """
        Method to create ifmap and filter prefetch matrices. These matrices are prefetched in the
        SRAM before running memory simulation.
        """
        assert self.params_set_flag, 'Parameters are not set'

        self.create_ifmap_prefetch_mat()
        self.create_filter_prefetch_mat()

        self.prefetch_mat_ready_flag = True

    #
    def create_ifmap_prefetch_mat(self):
        """
        Method to create ifmap prefetch matrix.
        """
        assert self.params_set_flag, 'Parameters are not set'

        for fr in range(self.row_fold):
            start_col_idx = fr * self.arr_row
            end_col_idx = min(start_col_idx + self.arr_row, self.Sr)

            delta = self.arr_row - (end_col_idx - start_col_idx)

            this_fold_prefetch = self.ifmap_op_mat[:,start_col_idx: end_col_idx]

            #If there is under utilization, fill them with null requests
            if delta > 0:
                null_req_mat = np.ones((self.T, delta)) * -1
                this_fold_prefetch = np.concatenate((this_fold_prefetch, null_req_mat), axis=1)

            if fr == 0:
                self.ifmap_prefetch_matrix = this_fold_prefetch
            else:
                self.ifmap_prefetch_matrix = \
                    np.concatenate((self.ifmap_prefetch_matrix, this_fold_prefetch), axis=0)

        # Fixing ISSUE #15, #16
        # Roll out the matrices along the diagonal to account for temporal locality when there is a
        # skew in demand

        M, N = self.ifmap_prefetch_matrix.shape
        num_elems = M * N
        num_diags = M + N
        prefetches = np.zeros((1, num_elems))
        idx = 0

        pbar = tqdm(total=M * N, disable=True)
        # print('DEBUG: Total = ' + str(num_elems) + ' Diags = ' + str(num_diags))

        for diag_id in range(num_diags):
            max_row_id = min(diag_id, M - 1)
            min_row_id = max(0, diag_id - N + 1)
            valid_rows = max_row_id - min_row_id + 1

            for offset in range(valid_rows):
                row_id = max_row_id - offset
                col_id = diag_id - row_id

                elem = self.ifmap_prefetch_matrix[row_id][col_id]
                prefetches[0, idx] = elem
                idx += 1
                pbar.update(1)

        pbar.close()
        self.ifmap_prefetch_matrix = prefetches

    #
    def create_filter_prefetch_mat(self):
        """
        Method to create filter prefetch matrix.
        """
        assert self.params_set_flag, 'Parameters are not set'

        for fc in range(self.col_fold):
            col_start_id = fc * self.arr_col
            col_end_id = min(col_start_id + self.arr_col, self.Sc)

            delta = self.arr_col - (col_end_id - col_start_id)

            this_fold_prefetch = self.filter_op_mat[:,col_start_id:col_end_id]

            if delta > 0:
                null_req_mat = np.ones((self.filter_op_mat.shape[0], delta)) * -1 # self.Sr
                this_fold_prefetch = np.concatenate((this_fold_prefetch, null_req_mat), axis=1)

            if fc == 0:
                self.filter_prefetch_matrix = this_fold_prefetch
            else:
                self.filter_prefetch_matrix = \
                    np.concatenate((self.filter_prefetch_matrix, this_fold_prefetch), axis=0)

        # Note: ISSUE #15: no skewing happens in the Filter for WS so this issue does not apply.

    #
    def create_demand_matrices(self):
        """
        Method to create ifmap, filter and ofmap demand matrices from the operand matrices. They
        contain several folds of ifmap, filter and ofmap demands. The folding happens because
        operand matrices are generally larger than systolic array dimensions.
        """
        # NCBS: check this once, create new assert for row_stationary, if...else
        assert self.params_set_flag, 'Parameters are not set'

        self.create_ifmap_demand_mat()
        self.create_filter_demand_mat()
        self.create_ofmap_demand_mat()   

        # assert self.ifmap_demand_matrix.shape[0] == self.filter_demand_matrix.shape[0], \
        #        'IFMAP and Filter demands out of sync'
        # assert self.ofmap_demand_matrix.shape[0] == self.filter_demand_matrix.shape[0], \
        #        'OFMAP and Filter demands out of sync'
        if not self.config.sparsity_optimized_mapping:
            assert self.ifmap_demand_matrix.shape[1] == self.arr_row, 'IFMAP demands exceed the rows'
        assert self.filter_demand_matrix.shape[1] == self.arr_col,'Filter demands exceed the cols'
        assert self.ofmap_demand_matrix.shape[1] == self.arr_col, 'OFMAP demands exceed the cols'

        self.demand_mat_ready_flag = True

    #
    def create_ifmap_demand_mat(self):
        """
        Method to create ifmap demand matrix.
        """
        assert self.params_set_flag, 'Parameters are not set'

        inter_fold_gap_prefix = self.arr_row
        inter_fold_gap_prefix_mat = np.ones((inter_fold_gap_prefix, self.arr_row)) * -1

        inter_fold_gap_suffix = self.arr_col - 1

        inter_fold_gap_suffix_mat = np.ones((inter_fold_gap_suffix, self.arr_row)) * -1

        metadata_conversion_mat = [ [ ] ]
        if False:
            if self.config.sparsity_support is True:
                if self.config.sparsity_representation == 'csr':
                    metadata_conversion_mat = np.ones((1, self.arr_col)) * -1
                elif self.config.sparsity_representation == 'csc':
                    metadata_conversion_mat = np.ones((1, self.arr_col)) * -1
                elif self.config.sparsity_representation == 'ellpack_block':
                    metadata_conversion_mat = np.ones((0, self.arr_col)) * -1

        ifmap_demand_matrix_list = []

        for fc in range(self.col_fold):
            # for fr in range(self.row_fold):
            for fr in range(self.row_fold_demand_matrices):
                if self.config.sparsity_support and self.config.sparsity_optimized_mapping:
                    col_start_id = fr * (self.arr_row * 2) # Since we need 2 tiles
                    col_end_idx = min(col_start_id + (self.arr_row * 2), self.Sr)
                    delta = (self.arr_row * 2) - (col_end_idx - col_start_id)
                    this_fold_demand = self.ifmap_op_mat_original[:,col_start_id: col_end_idx]
                else:
                    col_start_id = fr * self.arr_row
                    col_end_idx = min(col_start_id + self.arr_row, self.Sr)
                    delta = self.arr_row - (col_end_idx - col_start_id)

                    # Indexing the cols with row start and row end idx are correct
                    # See the comment on ifmap_prefetch generation
                    this_fold_demand = self.ifmap_op_mat[:,col_start_id: col_end_idx]

                # Need to add custom skew for row-wise sparsity
                if self.config.sparsity_support and self.config.sparsity_optimized_mapping:
                    this_fold_demand = skew_matrix_row_sparsity(this_fold_demand, self.arr_row, \
                                                                self.config.sparsity_block_size)
                
                if self.config.sparsity_support:
                    if self.config.sparsity_optimized_mapping:
                        self.ifmap_reads += this_fold_demand.shape[0] * this_fold_demand.shape[1]
                    else:
                        # A single block of input is shared among M/N rows, hence a row needs to be
                        # read M/N times (assume absence of any broadcast)
                        self.ifmap_reads += this_fold_demand.shape[0] * this_fold_demand.shape[1] * \
                                            (self.sparsity_ratio_M / self.sparsity_ratio_N)
                else:
                    self.ifmap_reads += this_fold_demand.shape[0] * this_fold_demand.shape[1]

                # Take into account under utilization
                if not self.config.sparsity_optimized_mapping:
                    if delta > 0:
                        null_req_mat = np.ones((self.T, delta)) * -1
                        this_fold_demand = np.concatenate((this_fold_demand, null_req_mat), axis=1)
                
                if self.config.sparsity_support and self.config.sparsity_optimized_mapping:
                    if inter_fold_gap_prefix_mat.shape[1] < this_fold_demand.shape[1]:
                        inter_fold_gap_prefix_mat = np.pad(
                            inter_fold_gap_prefix_mat,
                            ((0, 0), (0, this_fold_demand.shape[1] - inter_fold_gap_prefix_mat.shape[1])),  # Pad only columns, not rows
                            constant_values=-1
                        )

                # Account for the cycles for weights to load
                this_fold_demand = np.concatenate((inter_fold_gap_prefix_mat, this_fold_demand),
                                                  axis=0)

                # Account for the cycles for final output to drain out
                if self.config.sparsity_support and self.config.sparsity_optimized_mapping:
                    if inter_fold_gap_suffix_mat.shape[1] < this_fold_demand.shape[1]:
                        inter_fold_gap_suffix_mat = np.pad(
                            inter_fold_gap_suffix_mat,
                            ((0, 0), (0, this_fold_demand.shape[1] - inter_fold_gap_suffix_mat.shape[1])),  # Pad only columns, not rows
                            constant_values=-1
                        )
                this_fold_demand = np.concatenate((this_fold_demand, inter_fold_gap_suffix_mat),
                                                  axis=0)                

                # Add skew to the IFMAP demand matrix to reflect systolic pipeline fill
                if not self.config.sparsity_optimized_mapping:
                    this_fold_demand = skew_matrix(this_fold_demand)

                ifmap_demand_matrix_list.append(this_fold_demand)

        self.ifmap_demand_matrix = np.concatenate(ifmap_demand_matrix_list)

        if False:
            if self.config.sparsity_support is True:
                self.ifmap_demand_matrix = \
                    np.concatenate((metadata_conversion_mat, self.ifmap_demand_matrix), axis=0)

    # END of IFMAP demand generation

    #
    def create_filter_demand_mat(self):
        """
        Method to create filter demand matrix.
        """
        assert self.params_set_flag, 'Parameters are not set'

        inter_fold_gap_suffix = self.arr_row + self.arr_col + self.T - 2
        inter_fold_gap_suffix_mat = np.ones((inter_fold_gap_suffix, self.arr_col)) * -1

        metadata_conversion_mat = [ [ ] ]
        if False:
            if self.config.sparsity_support is True:
                if self.config.sparsity_representation == 'csr':
                    metadata_conversion_mat = np.ones((1, self.arr_col)) * -1
                elif self.config.sparsity_representation == 'csc':
                    metadata_conversion_mat = np.ones((1, self.arr_col)) * -1
                elif self.config.sparsity_representation == 'ellpack_block':
                    metadata_conversion_mat = np.ones((0, self.arr_col)) * -1

        filter_demand_matrix_list = []
        for fc in range(self.col_fold):
            # for fr in range(self.row_fold):
            for fr in range(self.row_fold_demand_matrices):
                row_start_id = fr * self.arr_row
                # row_end_idx = min(row_start_id + self.arr_row, self.Sr)
                row_end_idx = min(row_start_id + self.arr_row, self.filter_op_mat.shape[0])
                row_delta = self.arr_row - (row_end_idx - row_start_id)

                col_start_id = fc * self.arr_col
                col_end_idx = min(col_start_id + self.arr_col, self.Sc)
                col_delta = self.arr_col - (col_end_idx - col_start_id)

                this_fold_demand = \
                    self.filter_op_mat[row_start_id:row_end_idx, col_start_id: col_end_idx]
                self.filter_reads += this_fold_demand.shape[0] * this_fold_demand.shape[1]

                # Take into account under utilization
                if col_delta > 0:
                    null_req_mat = np.ones((this_fold_demand.shape[0], col_delta)) * -1
                    this_fold_demand = np.concatenate((this_fold_demand, null_req_mat), axis=1)

                if row_delta > 0:
                    null_req_mat = np.ones((row_delta, self.arr_col)) * -1
                    this_fold_demand = np.concatenate((this_fold_demand, null_req_mat), axis=0)

                # The filters are needed to be filled in reverse order to ensure that
                # top element is pushed in last to maintain alignment with the input elements
                this_fold_demand = np.flip(this_fold_demand, 0)

                sum_sparse = sum(list(row).count(-1) for row in this_fold_demand)

                # Time for inputs to stream and the partial sums to drain out
                this_fold_demand = np.concatenate((this_fold_demand, inter_fold_gap_suffix_mat),
                                                  axis=0)

                # Calculate the mapping efficiency
                row_used = min(self.arr_row, row_end_idx - row_start_id)
                col_used = min(self.arr_col, col_end_idx - col_start_id)
                mac_used = row_used * col_used

                # mapping_eff_this_fold = mac_used / (self.arr_row * self.arr_col)
                mapping_eff_this_fold = \
                    ((self.arr_row * self.arr_col) - sum_sparse) / (self.arr_row * self.arr_col)

                cycles_this_fold = this_fold_demand.shape[0] + this_fold_demand.shape[1] - 1
                compute_cycles_this_fold = mac_used * self.T
                compute_util_this_fold = \
                    compute_cycles_this_fold / (self.arr_row * self.arr_col * cycles_this_fold)

                self.mapping_efficiency_per_fold.append(mapping_eff_this_fold)
                self.compute_utility_per_fold.append(compute_util_this_fold)

                filter_demand_matrix_list.append(this_fold_demand)
                #if fr == 0 and fc == 0:
                #    self.filter_demand_matrix = this_fold_demand
                #else:
                #    self.filter_demand_matrix = \
                #       np.concatenate((self.filter_demand_matrix, this_fold_demand), axis=0)
        
        self.filter_demand_matrix = np.concatenate(filter_demand_matrix_list)

        if False:
            if self.config.sparsity_support is True:
                self.filter_demand_matrix = \
                    np.concatenate((metadata_conversion_mat, self.filter_demand_matrix), axis=0)

        # No skew needed in filters for weight stationary

    #
    def create_ofmap_demand_mat(self):
        """
        Method to create ofmap demand matrix.
        """
        assert self.params_set_flag, 'Parameters are not set'

        inter_fold_gap_prefix = 2 * self.arr_row - 1
        inter_fold_gap_prefix_mat = np.ones((inter_fold_gap_prefix, self.arr_col)) * -1

        metadata_conversion_mat = [ [ ] ]
        if False:
            if self.config.sparsity_support is True:
                if self.config.sparsity_representation == 'csr':
                    metadata_conversion_mat = np.ones((1, self.arr_col)) * -1
                elif self.config.sparsity_representation == 'csc':
                    metadata_conversion_mat = np.ones((1, self.arr_col)) * -1
                elif self.config.sparsity_representation == 'ellpack_block':
                    metadata_conversion_mat = np.ones((0, self.arr_col)) * -1

        ofmap_demand_matrix_list = []

        for fc in range(self.col_fold):
            # for fr in range(self.row_fold):
            for fr in range(self.row_fold_demand_matrices):
                col_start_id = fc * self.arr_col
                col_end_idx = min(col_start_id + self.arr_col, self.Sc) # self.Sc
                col_delta = self.arr_col - (col_end_idx - col_start_id)

                this_fold_demand = self.ofmap_op_mat[:, col_start_id: col_end_idx]
                self.ofmap_writes += this_fold_demand.shape[0] * this_fold_demand.shape[1]

                # Adding null requests when there is under utilization ie. no mapping along a few
                # rows or cols
                if col_delta > 0:
                    null_req_mat = np.ones((this_fold_demand.shape[0], col_delta)) * -1
                    this_fold_demand = np.concatenate((this_fold_demand, null_req_mat), axis=1)

                # Now add the prefix matrix
                # These are the null demands to account for when the operands are streamed in
                # and the OFMAPS are not ready
                this_fold_demand = np.concatenate((inter_fold_gap_prefix_mat, this_fold_demand),
                                                  axis=0)

                # Add skew to the OFMAP demand matrix to reflect systolic pipeline fill
                this_fold_demand = skew_matrix(this_fold_demand)

                ofmap_demand_matrix_list.append(this_fold_demand)
                #if fr == 0 and fc == 0:
                #    self.ofmap_demand_matrix = this_fold_demand
                #else:
                #    self.ofmap_demand_matrix = \
                #       np.concatenate((self.ofmap_demand_matrix, this_fold_demand), axis=0)

        self.ofmap_demand_matrix = np.concatenate(ofmap_demand_matrix_list)

        if False:
            if self.config.sparsity_support is True:
                self.ofmap_demand_matrix = \
                    np.concatenate((metadata_conversion_mat, self.ofmap_demand_matrix), axis=0)

    # END of OFMAP demand generation

    #
    def get_ifmap_prefetch_mat(self):
        """
        Method to get ifmap prefetch matrix.
        """
        if not self.prefetch_mat_ready_flag:
            self.create_prefetch_matrices()

        return self.ifmap_prefetch_matrix

    #
    def get_filter_prefetch_mat(self):
        """
        Method to get filter prefetch matrix.
        """
        if not self.prefetch_mat_ready_flag:
            self.create_prefetch_matrices()

        return self.filter_prefetch_matrix

    #
    def get_prefetch_matrices(self):
        """
        Method to get ifmap and filter prefetch matrices.
        """
        if not self.prefetch_mat_ready_flag:
            self.create_prefetch_matrices()

        return self.ifmap_prefetch_matrix, self.filter_prefetch_matrix

    #
    def get_ifmap_demand_mat(self):
        """
        Method to get ifmap demand matrix.
        """
        if not self.demand_mat_ready_flag:
            self.create_demand_matrices()

        return self.ifmap_demand_matrix

    #
    def get_filter_demand_mat(self):
        """
        Method to get filter demand matrix.
        """
        if not self.demand_mat_ready_flag:
            self.create_demand_matrices()

        return self.filter_demand_matrix

    #
    def get_ofmap_demand_mat(self):
        """
        Method to get ofmap demand matrix.
        """
        if not self.demand_mat_ready_flag:
            self.create_demand_matrices()

        return self.ofmap_demand_matrix

    #
    def get_demand_matrices(self):
        """
        Method to get ifmap, filter and ofmap demand matrices.
        """
        if not self.demand_mat_ready_flag:
            self.create_demand_matrices()

        return self.ifmap_demand_matrix, self.filter_demand_matrix, self.ofmap_demand_matrix

    #
    def get_avg_mapping_efficiency(self):
        """
        Method to get average mapping efficincy on the systolic array.
        """
        assert self.demand_mat_ready_flag, 'Computes not ready yet'

        agg = sum(self.mapping_efficiency_per_fold)
        num = len(self.mapping_efficiency_per_fold)

        avg_mapping_eff = agg / num

        return avg_mapping_eff

    #
    def get_avg_compute_utilization(self):
        """
        Method to get average compute utilization on the systolic array.
        """
        assert self.demand_mat_ready_flag, 'Computes not ready yet'

        agg = sum(self.compute_utility_per_fold)
        num = len(self.compute_utility_per_fold)

        avg_compute_util = agg / num

        return avg_compute_util

    #
    def get_ifmap_requests(self):
        """
        Method to get the number of ifmap read requests.
        """
        assert self.demand_mat_ready_flag, 'Computes not ready yet'
        return self.ifmap_reads

    #
    def get_filter_requests(self):
        """
        Method to get the number of filter read requests.
        """
        assert self.demand_mat_ready_flag, 'Computes not ready yet'
        return self.filter_reads

    #
    def get_ofmap_requests(self):
        """
        Method to get the number of ofmap read requests.
        """
        assert self.demand_mat_ready_flag, 'Computes not ready yet'
        return self.ofmap_writes


#
def skew_matrix(input_matrix_np):
    """
    Method to add skew to the input matix to maintain systolic array flow.
    Example:
        Input matrix:
        1 1 1 1 1 1 1 1 1

        Output matrix:
            1 1 1
          1 1 1
        1 1 1
    """
    rows, cols = input_matrix_np.shape

    out_matrix_np = np.full((rows + cols - 1, cols), -1, dtype=input_matrix_np.dtype)

    for c in range(cols):
        out_matrix_np[c:c + rows, c] = input_matrix_np[:, c]

    return out_matrix_np

#
def skew_matrix_row_sparsity(input_matrix, arr_row, block_size):
    # Step 1: Ensure the number of columns is arr_row * 2 as we are combining 2 tiles
    num_tiles = 2
    num_cols = input_matrix.shape[1]
    if num_cols < arr_row * num_tiles:
        padding = arr_row * num_tiles - num_cols
        input_matrix = np.pad(input_matrix, ((0, 0), (0, padding)), constant_values=-1)
    
    # Step 2: Split into blocks
    blocks = []
    for row in input_matrix:
        row_blocks = [row[i:i+block_size] for i in range(0, len(row), block_size)]
        blocks.append(row_blocks)

    # Step 3: Add extra copies to match block_size / 2 copies per column
    repeated_blocks = []
    for block_row in blocks:
        new_row = []
        for block in block_row:
            new_row.extend([block] * (block_size // num_tiles))
        repeated_blocks.append(new_row)

    # Step 4: Apply skew to the repeated blocks
    output_matrix = []
    num_block_rows = len(repeated_blocks)
    num_block_cols = (len(repeated_blocks[0][0]) * block_size) // num_tiles

    for i in range(num_block_rows + arr_row - 1):
        row = []
        for j in range(len(repeated_blocks[0])):
            block_row_idx = i - j
            if 0 <= block_row_idx < num_block_rows:
                row.append(repeated_blocks[block_row_idx][j])
            else:
                row.append([-1] * block_size)
        row = np.concatenate(row)
        output_matrix.append(row)
    
    return np.array(output_matrix, dtype=input_matrix.dtype)
