"""
Buffer to stage the data to be written
"""
# TODO: Verification Pending
import math
import numpy as np
from tqdm import tqdm
from scalesim.memory.write_port import write_port


class write_buffer:
    """
    Class which runs the memory simulation of the OFMAP SRAM.
    """
    def __init__(self):
        """
        __init__ method.
        """
        # Buffer properties: User specified
        self.total_size_bytes = 128
        self.word_size = 1
        self.active_buf_frac = 0.9

        # Buffer properties: Calculated
        self.total_size_elems = math.floor(self.total_size_bytes / self.word_size)
        self.active_buf_size = int(math.ceil(self.total_size_elems * self.active_buf_frac))
        self.drain_buf_size = self.total_size_elems - self.active_buf_size

        # Backing interface properties
        self.backing_buffer = write_port()
        self.req_gen_bandwidth = 100

        # Status of the buffer
        self.free_space = self.total_size_elems
        self.active_buf_contents = []
        self.drain_buf_contents = []
        self.drain_end_cycle = 0
        #self.active_buf_start_line_id = -1
        #self.active_buf_end_line_id = -1
        #self.drain_buf_start_line_id = -1
        #self.drain_buf_end_line_id = -1

        # Access counts
        self.num_access = 0

        # Trace matrix
        self.trace_matrix = np.zeros((1, 1))

        # Flags
        # This variable determines where the new requests should be buffered
        # 0: Directly in the drain buffer
        # 1: In the active buffer, while the drain buffer is flushed
        self.state = 0

        self.trace_valid = False

    #
    def set_params(self, backing_buf_obj,
                   total_size_bytes=128, word_size=1, active_buf_frac=0.9,
                   backing_buf_bw=100
                   ):
        """
        Method to set the ofmap memory simulation parameters for housekeeping.
        """
        self.total_size_bytes = total_size_bytes
        self.word_size = word_size

        assert 0.5 <= active_buf_frac < 1, "Valid active buf frac [0.5,1)"
        self.active_buf_frac = active_buf_frac

        self.backing_buffer = backing_buf_obj
        self.req_gen_bandwidth = backing_buf_bw

        self.total_size_elems = math.floor(self.total_size_bytes / self.word_size)
        self.active_buf_size = int(math.ceil(self.total_size_elems * self.active_buf_frac))
        self.drain_buf_size = self.total_size_elems - self.active_buf_size
        self.free_space = self.total_size_elems

    #
    def reset(self):
        """
        Method to reset the write buffer parameters.
        """
        self.total_size_bytes = 128
        self.word_size = 1
        self.active_buf_frac = 0.9

        self.backing_buffer = write_buffer()
        self.req_gen_bandwidth = 100

        self.free_space = self.total_size_elems
        self.active_buf_contents = []
        self.drain_buf_contents = []
        self.drain_end_cycle = 0

        self.trace_matrix = np.zeros((1, 1))

        self.num_access = 0
        self.state = 0

        self.trace_valid = False

    #
    def service_writes(self, incoming_requests_arr_np, incoming_cycles_arr_np):
        """
        Method to service write requests coming from systolic array.
        """
        assert incoming_cycles_arr_np.shape[0] == incoming_requests_arr_np.shape[0], \
               'Cycles and requests do not match'
        out_cycles_arr = []

        offset = 0
        #for row, cycle in zip(incoming_requests_arr_np, incoming_cycles_arr_np):
        for i in tqdm(range(incoming_requests_arr_np.shape[0])):
            row = incoming_requests_arr_np[i]
            cycle = incoming_cycles_arr_np[i]
            current_cycle = cycle[0] + offset

            for elem in row:
                # Pay no attention to empty requests
                if elem == -1:
                    continue

                # Case 1: Drain buffer is empty
                #         Put the contents in drain buffer till its full
                if self.state == 0:
                    self.drain_buf_contents.append(elem)
                    self.free_space -= 1

                    if len(self.drain_buf_contents) >= self.drain_buf_size:
                        self.state = 1
                        self.drain_end_cycle = self.empty_drain_buf(empty_start_cycle=current_cycle)

                # Case 2: If drain buffer is not empty but active buffer is empty
                #         Put the contents in active buffer till drain buffer is free
                else:
                    if current_cycle < self.drain_end_cycle:
                        if len(self.active_buf_contents) < self.active_buf_size:
                            self.active_buf_contents.append(elem)
                            self.free_space -= 1
                        else:
                            offset += max(self.drain_end_cycle - current_cycle, 0)
                            current_cycle = self.drain_end_cycle
                    else:
                        for i in range(self.drain_buf_size):
                            elem = self.active_buf_contents[i]
                            self.drain_buf_contents.append(elem)
                            self.active_buf_contents.remove(elem)

                        self.drain_end_cycle = self.empty_drain_buf(empty_start_cycle=current_cycle)

            out_cycles_arr.append(current_cycle)

        num_lines = incoming_requests_arr_np.shape[0]
        out_cycles_arr_np = np.asarray(out_cycles_arr).reshape((num_lines, 1))
        return out_cycles_arr_np

    #
    def empty_drain_buf(self, empty_start_cycle=0):
        """
        Method to drain the drain buffer once the active buffer is full.
        """

        data_sz_to_drain = min(len(self.drain_buf_contents), self.drain_buf_size)

        num_lines = int(math.ceil(data_sz_to_drain / self.req_gen_bandwidth))
        delta_req = int(num_lines * self.req_gen_bandwidth) - data_sz_to_drain
        self.num_access += data_sz_to_drain

        for _ in range(delta_req):
            self.drain_buf_contents.append(-1)

        self.drain_buf_contents.sort()
        requests_arr_np = np.asarray(self.drain_buf_contents)
        requests_arr_np = requests_arr_np.reshape((num_lines, self.req_gen_bandwidth))
        self.drain_buf_contents = []

        cycles_arr = [x+empty_start_cycle for x in range(num_lines)]
        cycles_arr_np = np.asarray(cycles_arr).reshape((num_lines, 1))
        serviced_cycles_arr = self.backing_buffer.service_writes(requests_arr_np, cycles_arr_np)

        # Generate trace here
        if not self.trace_valid:
            self.trace_matrix = np.concatenate((serviced_cycles_arr, requests_arr_np), axis=1)
            self.trace_valid = True
        else:
            local_trace_matrix = np.concatenate((serviced_cycles_arr, requests_arr_np), axis=1)
            self.trace_matrix = np.concatenate((self.trace_matrix, local_trace_matrix), axis=0)

        service_end_cycle = serviced_cycles_arr[-1][0]
        self.free_space += data_sz_to_drain
        return service_end_cycle

    #
    def drain_active_buf(self):
        """
        Method to transfer all elements from the active buffer to the drain buffer.
        """
        while len(self.active_buf_contents) > 0:
            for i in range(self.drain_buf_size):
                elem = self.active_buf_contents[i]
                self.drain_buf_contents.append(elem)
                self.active_buf_contents.remove(elem)

            self.drain_end_cycle = self.empty_drain_buf(self.drain_end_cycle)

    #
    def empty_all_buffers(self, cycle):
        """
        Method to drain all of the active buffer.
        """
        if self.state == 0:
            self.drain_end_cycle = self.empty_drain_buf(empty_start_cycle=cycle)
            self.state = 1
        else:
            self.drain_active_buf()

    #
    def get_trace_matrix(self):
        """
        Method to get the write buffer trace matrix.
        """
        if not self.trace_valid:
            print('No trace has been generated yet')
            return

        return self.trace_matrix

    #
    def get_free_space(self):
        """
        Method to get free space of the write buffer.
        """
        return self.free_space

    #
    def get_num_accesses(self):
        """
        Method to get number of accesses of the write buffer if trace_valid flag is set.
        """
        assert self.trace_valid, 'Traces not ready yet'
        return self.num_access

    #
    def get_external_access_start_stop_cycles(self):
        """
        Method to get start and stop cycles of the write buffer if trace_valid flag is set.
        """
        assert self.trace_valid, 'Traces not ready yet'
        start_cycle = self.trace_matrix[0][0]
        end_cycle = self.trace_matrix[-1][0]

        return start_cycle, end_cycle

    #
    def print_trace(self, filename):
        """
        Method to write the write buffer trace matrix to a file.
        """
        if not self.trace_valid:
            print('No trace has been generated yet')
            return

        np.savetxt(filename, self.trace_matrix, fmt='%s', delimiter=",")
