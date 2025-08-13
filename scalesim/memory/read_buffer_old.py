"""
Double buffer read memory implementation
"""
# TODO: Verification Pending
import math
import numpy as np
from tqdm import tqdm

from scalesim.memory.read_port import read_port


class read_buffer:
    """
    Class which runs the memory simulation of double buffered ifmap/filter SRAM. The double
    buffering helps to hide the DRAM latency when the SRAM is servicing requests from the systolic
    array using one of the buffers while the other buffer prefetches from the DRAM.
    """
    #
    def __init__(self):
        """
        __init__ method.
        """
        # Buffer properties: User specified
        self.total_size_bytes = 128
        self.word_size = 1         # Bytes
        self.active_buf_frac = 0.9
        self.hit_latency = 1       # Cycles after which a request is served if already in the buffer

        # Buffer properties: Calculated
        self.total_size_elems = math.floor(self.total_size_bytes / self.word_size)
        self.active_buf_size = int(math.ceil(self.total_size_elems * 0.9))
        self.prefetch_buf_size = self.total_size_elems - self.active_buf_size

        # Backing interface properties
        self.backing_buffer = read_port()
        self.req_gen_bandwidth = 100            # words per cycle

        # Status of the buffer
        self.hashed_buffer = {}
        self.active_buffer_set_limits = []
        self.prefetch_buffer_set_limits = []
        self.active_buffer_contents = set()

        # Variables to enable prefetching
        self.fetch_matrix = np.ones((1, 1))
        self.last_prefect_cycle = -1
        self.next_line_prefetch_idx = 0
        self.next_col_prefetch_idx = 0
        self.prefetch_buffer_contents = []

        # Access counts
        self.num_access = 0

        # Trace matrix
        self.trace_matrix = np.ones((1, 1))

        # Flags
        self.active_buf_full_flag = False
        self.hashed_buffer_valid = False
        self.trace_valid = False

    #
    def set_params(self, backing_buf_obj,
                   total_size_bytes=1, word_size=1, active_buf_frac=0.9,
                   hit_latency=1, backing_buf_bw=1
                   ):
        """
        Method to set the ifmap/filter double buffered memory simulation parameters for
        housekeeping.
        """

        self.total_size_bytes = total_size_bytes
        self.word_size = word_size

        assert 0.5 <= active_buf_frac < 1, "Valid active buf frac [0.5,1)"
        self.active_buf_frac = round(active_buf_frac, 2)
        self.hit_latency = hit_latency

        self.backing_buffer = backing_buf_obj
        self.req_gen_bandwidth = backing_buf_bw

        # Calculate these based on the values provided
        self.total_size_elems = math.floor(self.total_size_bytes / self.word_size)
        self.active_buf_size = int(math.ceil(self.total_size_elems * self.active_buf_frac))
        self.prefetch_buf_size = self.total_size_elems - self.active_buf_size

    #
    def reset(self): # TODO: check if all resets are working propoerly
        """
        Method to reset the read buffer parameters.
        """
        # Buffer properties: User specified
        self.total_size_bytes = 128
        self.word_size = 1  # Bytes
        self.active_buf_frac = 0.9
        self.hit_latency = 1  # Cycles after which a request is served if already in the buffer

        # Buffer properties: Calculated
        self.total_size_elems = math.floor(self.total_size_bytes / self.word_size)
        self.active_buf_size = int(math.ceil(self.total_size_elems * 0.9))
        self.prefetch_buf_size = self.total_size_elems - self.active_buf_size

        # Backing interface properties
        self.backing_buffer = read_port()
        self.req_gen_bandwidth = 100  # words per cycle

        # Status of the buffer
        self.hashed_buffer = {}
        self.active_buffer_set_limits = []
        self.prefetch_buffer_set_limits = []
        self.active_buffer_contents = set()

        # Variables to enable prefetching
        self.fetch_matrix = np.ones((1, 1))
        self.last_prefect_cycle = -1
        self.next_line_prefetch_idx = 0
        self.next_col_prefetch_idx = 0
        self.prefetch_buffer_contents = []

        # Access counts
        self.num_access = 0

        # Trace matrix
        self.trace_matrix = np.ones((1, 1))

        # Flags
        self.active_buf_full_flag = False
        self.hashed_buffer_valid = False
        self.trace_valid = False

    #
    def set_fetch_matrix(self, fetch_matrix_np):
        """
        Method to set the fetch matrix responsible for prefetching from the DRAM.
        """
        # The operand matrix determines what to pre-fetch into both active and prefetch buffers
        # req_gen_bandwidth is set to 100 by default and will be used in 'calc' mode
        # In 'user' mode, this will be set in the set_params

        num_elems = fetch_matrix_np.shape[0] * fetch_matrix_np.shape[1]
        num_lines = int(math.ceil(num_elems / self.req_gen_bandwidth))
        self.fetch_matrix = np.ones((num_lines, self.req_gen_bandwidth)) * -1

        # Put stuff into the fetch matrix
        # This is done to ensure that there is no shape mismatch
        # Not sure if this is the optimal way to do it or not
        for i in range(num_elems):
            src_row = math.floor(i / fetch_matrix_np.shape[1])
            src_col = math.floor(i % fetch_matrix_np.shape[1])

            dest_row = math.floor(i / self.req_gen_bandwidth)
            dest_col = math.floor(i % self.req_gen_bandwidth)

            self.fetch_matrix[dest_row][dest_col] = fetch_matrix_np[src_row][src_col]

        # Once the fetch matrices are set, populate the data structure for faster lookups and
        # servicing
        self.prepare_hashed_buffer()

    #
    def prepare_hashed_buffer(self):
        """
        Method to convert the fetch matrix into a hashed buffer for fast lookups.
        """
        elems_per_set = math.ceil(self.total_size_elems / 100)

        prefetch_rows = self.fetch_matrix.shape[0]
        prefetch_cols = self.fetch_matrix.shape[1]

        line_id = 0
        elem_ctr = 0
        current_line = set()

        for r in range(prefetch_rows):
            for c in range(prefetch_cols):
                elem = self.fetch_matrix[r][c]

                if not elem == -1:
                    current_line.add(elem)
                    elem_ctr += 1

                if not elem_ctr < elems_per_set:    # ie > or =
                    self.hashed_buffer[line_id] = current_line
                    line_id += 1
                    elem_ctr = 0
                    current_line = set()        # new set

        self.hashed_buffer[line_id] = current_line
        self.hashed_buffer_valid = True

    #
    def service_reads(self,
                      incoming_requests_arr_np, # 2D array with the requests
                      incoming_cycles_arr):     # 1D vector with the cycles at which req arrived
        """
        Method to service read requests coming from systolic array.
        """
        # Service the incoming read requests
        # returns a cycles array corresponding to the requests buffer
        # Logic: Always check if an addr is in active buffer.
        #        If hit, return with hit latency
        #        Else, make the contents of prefetch buffer as active and then check
        #              finish till an ongoing prefetch is done before reassiging prefetch buffer

        if not self.active_buf_full_flag:
            start_cycle = incoming_cycles_arr[0][0]
            # Needs to use the entire operand matrix
            # keeping in mind the tile order and everything
            self.prefetch_active_buffer(start_cycle=start_cycle)

        out_cycles_arr = []
        offset = self.hit_latency
        #for cycle, request_line in tqdm(zip(incoming_cycles_arr, incoming_requests_arr_np)):
        for i in tqdm(range(incoming_requests_arr_np.shape[0])):
            cycle = incoming_cycles_arr[i]
            request_line = set(incoming_requests_arr_np[i]) #shaves off a few seconds

            for addr in request_line:
                if addr == -1:
                    continue

                if addr not in self.active_buffer_contents: #this is super slow!!!
                    self.new_prefetch()
                    potential_stall_cycles = self.last_prefect_cycle - (cycle + offset)
                    # Offset increments if there were potential stalls
                    offset += potential_stall_cycles

            out_cycles = cycle + offset
            out_cycles_arr.append(out_cycles)

        out_cycles_arr_np = np.asarray(out_cycles_arr).reshape((len(out_cycles_arr), 1))

        return out_cycles_arr_np

    #
    def prefetch_active_buffer(self, start_cycle):
        """
        Method to prefetch the active read buffer before servicing individual memory requests.
        """
        # Depending on size of the active buffer, calculate the number of lines from op mat to fetch
        # Also, calculate the cycles arr for requests

        # 1. Preparing the requests:
        num_lines = math.ceil(self.active_buf_size / self.req_gen_bandwidth)
        if not num_lines < self.fetch_matrix.shape[0]:
            num_lines = self.fetch_matrix.shape[0]

        requested_data_size = num_lines * self.req_gen_bandwidth
        self.num_access += requested_data_size

        start_idx = 0
        end_idx = num_lines

        prefetch_requests = self.fetch_matrix[start_idx:end_idx, :]
        # 1.1 See if extra requests are made, if so nullify them
        self.next_col_prefetch_idx = 0
        if requested_data_size > self.active_buf_size:
            valid_cols = int(self.active_buf_size % self.req_gen_bandwidth)
            row = end_idx - 1
            self.next_col_prefetch_idx = valid_cols
            for col in range(valid_cols, self.req_gen_bandwidth):
                prefetch_requests[row][col] = -1

        # 2. Preparing the cycles array
        #    The start_cycle variable ensures that all the requests have been made before any
        #    incoming reads came
        cycles_arr = np.zeros((num_lines, 1))
        for i in range(cycles_arr.shape[0]):
            cycles_arr[i][0] = \
                -1 * (num_lines - start_cycle - (i - self.backing_buffer.get_latency()))

        # 3. Send the request and get the response cycles count
        response_cycles_arr = \
            self.backing_buffer.service_reads(incoming_cycles_arr=cycles_arr,
                                              incoming_requests_arr_np=prefetch_requests)

        # 4. Update the variables
        self.last_prefect_cycle = int(response_cycles_arr[-1][0])

        # Update the trace matrix
        self.trace_matrix = np.concatenate((response_cycles_arr, prefetch_requests), axis=1)
        self.trace_valid = True

        # Set active buffer contents
        num_elems_to_fetch = min(self.active_buf_size, requested_data_size)
        req_col_size = prefetch_requests.shape[1]
        for i in range(num_elems_to_fetch):
            r = int(i / req_col_size)
            c = int(i % req_col_size)

            addr = prefetch_requests[r][c]
            if not addr == -1:
                self.active_buffer_contents.add(addr)
        self.active_buf_full_flag = True

        # Set the line to be prefetched next
        if requested_data_size > self.active_buf_size:
            self.next_line_prefetch_idx = num_lines % self.fetch_matrix.shape[0]
        else:
            self.next_line_prefetch_idx = (num_lines + 1) % self.fetch_matrix.shape[0]

    #
    def new_prefetch(self):
        """
        Method to do a new prefetch. In a new prefetch, some portion of the original data needs to
        be deleted to accomodate the prefetched data In this case we overwrite some data in the
        active buffer with the prefetched data and then create a new prefetch request.
        """
        # In a new prefetch, some portion of the original data needs to be deleted to accomodate the
        # prefetched data.
        # In this case we overwrite some data in the active buffer with the prefetched data
        # And then create a new prefetch request
        # Also return when the prefetched data was made available

        # 1. Rewrite the active buffer
        assert self.active_buf_full_flag, 'Active buffer is empty'
        for i in range(len(self.prefetch_buffer_contents)):
            data = self.prefetch_buffer_contents[i]
            self.active_buffer_contents.add(data)

        self.prefetch_buffer_contents = []

        # 2. Create the request
        start_idx = self.next_line_prefetch_idx
        num_lines = math.ceil(self.prefetch_buf_size / self.req_gen_bandwidth)
        end_idx = start_idx + num_lines
        requested_data_size = num_lines * self.req_gen_bandwidth
        self.num_access += requested_data_size

        # In care we need to circle back
        if end_idx > self.fetch_matrix.shape[0]:
            last_idx = self.fetch_matrix.shape[0]
            prefetch_requests = self.fetch_matrix[start_idx:,:]

            new_end_idx = min(end_idx - last_idx, start_idx) # In case the entire array is engulfed
            prefetch_requests = \
                np.concatenate((prefetch_requests, self.fetch_matrix[:new_end_idx,:]))
        else:
            prefetch_requests = self.fetch_matrix[start_idx:end_idx, :]

        # Modify the prefetch request to drop unwanted addresses
        # a. Chomp the elements in the first line included in previous fetches
        for i in range(0, self.next_col_prefetch_idx):
            prefetch_requests[0][i] = -1

        # b. Chomp the excess elements in the last line
        if requested_data_size > self.active_buf_size:
            valid_cols = int(self.active_buf_size % self.req_gen_bandwidth)
            row = end_idx - 1
            for col in range(valid_cols, self.req_gen_bandwidth):
                prefetch_requests[row][col] = -1

        # 3. Create the request cycles
        cycles_arr = np.zeros((num_lines, 1))
        for i in range(cycles_arr.shape[0]):
            cycles_arr[i][0] = self.last_prefect_cycle + i

        # 4. Send the request
        response_cycles_arr = \
            self.backing_buffer.service_reads(incoming_cycles_arr=cycles_arr,
                                              incoming_requests_arr_np=prefetch_requests)

        # 5. Update the variables
        self.last_prefect_cycle = response_cycles_arr[-1][0]

        this_prefetch_trace = np.concatenate((response_cycles_arr, prefetch_requests), axis=1)
        self.trace_matrix = np.concatenate((self.trace_matrix, this_prefetch_trace), axis=0)

        # Set the contents of the prefetch buffer
        num_prefetch_reqs = prefetch_requests.shape[0] * prefetch_requests.shape[1]
        elems_to_prefectch = min(self.prefetch_buf_size, num_prefetch_reqs)
        req_col_size = prefetch_requests.shape[1]
        for i in range(elems_to_prefectch):
            r = int(i / req_col_size)
            c = int(i % req_col_size)

            addr = prefetch_requests[r][c]
            self.prefetch_buffer_contents.append(addr)

        # Set the line to be prefetched next
        if requested_data_size > self.active_buf_size:
            self.next_line_prefetch_idx = num_lines % self.fetch_matrix.shape[0]
        else:
            self.next_line_prefetch_idx = (num_lines + 1) % self.fetch_matrix.shape[1]

        # This does not need to return anything
        #return response_cycles_arr

    #
    def get_trace_matrix(self):
        """
        Method to get the read buffer trace matrix. It contains addresses requsted by the systolic
        array and the cycles (first column) at which the requests are made.
        """
        if not self.trace_valid:
            print('No trace has been generated yet')
            return

        return self.trace_matrix

    #
    def get_hit_latency(self):
        """
        Method to get hit latency of the read buffer.
        """
        return self.hit_latency

    #
    def get_num_accesses(self):
        """
        Method to get number of accesses of the read buffer if trace_valid flag is set.
        """
        assert self.trace_valid, 'Traces not ready yet'
        return self.num_access

    #
    def get_external_access_start_stop_cycles(self):
        """
        Method to get start and stop cycles of the read buffer if trace_valid flag is set.
        """
        assert self.trace_valid, 'Traces not ready yet'
        start_cycle = self.trace_matrix[0][0]
        end_cycle = self.trace_matrix[-1][0]

        return start_cycle, end_cycle

    #
    def print_trace(self, filename):
        """
        Method to write the read buffer trace matrix to a file.
        """
        if not self.trace_valid:
            print('No trace has been generated yet')
            return

        np.savetxt(filename, self.trace_matrix, fmt='%s', delimiter=",")
