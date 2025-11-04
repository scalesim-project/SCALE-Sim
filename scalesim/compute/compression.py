"""
This file contains the 'compression' class that defines the various compression formats such as
Compressed Sparse Row, Compressed Sparse Column and Blocked ELLPACK. The class provides methods to
obtain the Filter and Metadata storages along with original storage for all the compression formats
mentioned.
"""

import math
import numpy as np

class compression:
    """
    This class deals with calculating the filter storage for with/without sparsity cases.
    """
    #
    def compress_to_csr(self, matrix):
        """
        Method to compress a given matrix using the Compressed Sparse Row (CSR) format.
        """
        rows, cols = matrix.shape
        data = []
        col_id = []
        row_ptr = [0]

        for i in range(rows):
            # Find non-zero elements in current row
            non_zero_indices = np.nonzero(matrix[i])[0]
            data.extend(matrix[i, non_zero_indices])
            col_id.extend(non_zero_indices)
            row_ptr.append(len(data))  # Update starting index for next row

        original_storage = rows * cols
        new_storage = (2 * len(col_id)) + len(row_ptr)
        metadata_storage = len(col_id) + len(row_ptr)

        return np.array(data), np.array(col_id), np.array(row_ptr), original_storage, new_storage, \
               metadata_storage

    #
    def compress_to_csc(self, matrix):
        """
        Method to compress a given matrix using the Compressed Sparse Column (CSC) format.
        """
        rows, cols = matrix.shape
        data = []
        row_id = []
        col_ptr = [0] * (cols + 1)

        for j in range(cols):
            # Find non-zero elements in current column
            non_zero_indices = np.nonzero(matrix[:, j])[0]
            data.extend(matrix[non_zero_indices, j])
            row_id.extend(non_zero_indices)
            col_ptr[j + 1] = len(data)  # Update starting index for next column

        original_storage = rows * cols
        new_storage = (2 * len(row_id)) + len(col_ptr)
        metadata_storage = len(row_id) + len(col_ptr)

        return np.array(data), np.array(row_id), np.array(col_ptr), original_storage, new_storage, \
               metadata_storage

    #
    def compress_to_ellpack_block(self, matrix, filter_op_mat, sparsity_ratio_M):
        """
        Method to compress a given matrix using the Blocked ELLPACK format.
        """
        # Each entry of the original matrix requires 1 word size.
        # Each entry in filter_op_mat will require log2(M) bits of metadata.
        original_rows, original_cols = matrix.shape
        new_rows, new_cols = filter_op_mat.shape

        # Units: Words (considering 1 word = 4 bytes = 32 bits)
        original_storage = original_rows * original_cols
        metadata_storage = ((new_rows * new_cols) * math.ceil(math.log2(sparsity_ratio_M))) / 32
        new_storage = (new_rows * new_cols) + metadata_storage

        return original_storage, new_storage, metadata_storage

    #
    def get_csr_storage(self, matrix):
        """
        Method to get the sizes of original filter, compressed dense filter matrix and its metadata
        when CSR is used as the comprtession format.
        """
        data, col_id, row_ptr, original_storage, new_storage, metadata_storage = \
            self.compress_to_csr(matrix)
        return original_storage, new_storage, metadata_storage

    #
    def get_csc_storage(self, matrix):
        """
        Method to get the sizes of original filter, compressed dense filter matrix and its metadata
        when CSC is used as the comprtession format.
        """
        data, row_id, col_ptr, original_storage, new_storage, metadata_storage = \
            self.compress_to_csc(matrix)
        return original_storage, new_storage, metadata_storage

    #
    def get_ellpack_block_storage(self, matrix, filter_op_mat, sparsity_ratio_M):
        """
        Method to get the sizes of original filter, compressed dense filter matrix and its metadata
        when Blocked ELLPACK is used as the comprtession format.
        """
        original_storage, new_storage, metadata_storage = \
            self.compress_to_ellpack_block(matrix, filter_op_mat, sparsity_ratio_M)
        return original_storage, new_storage, metadata_storage
