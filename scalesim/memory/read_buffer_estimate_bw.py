"""
The 'ReadBufferEstimateBw' class simulates a double-buffered memory read buffer in estimate
bandwidth mode designed for systolic array computations.
"""

import math
import numpy as np

from scalesim.memory.read_port import read_port


class ReadBufferEstimateBw:
    """
    Class which service the estimate bandwidth mode in the read buffer.
    """
    #
    def __init__(self):
        """
        __init__ method.
        """
        # Buffer parameters
        self.word_size = 1
        self.active_buf_frac = 0.5
        self.total_size_bytes = 1
        self.total_size_elems = 1
        self.active_buf_size = 1
        self.prefetch_buf_size = 1

        self.hit_latency = 1

        # Backing buffer parameters
        self.backing_buffer = read_port()
        self.default_bandwidth = 1
        self.prefetch_bandwidth = 1

        # Access counts
        self.num_access = 0

        # Trace matrix
        self.trace_matrix = np.ones((1, 1))

        # Tracking variables
        self.num_items_per_set = -1
        self.elems_current_set = 0
        self.current_set_id = 0
        self.read_buffer_set_start_id = -1
        self.read_buffer_set_end_id = -1
        self.prefetch_buffer_set_start_id = -1
        self.prefetch_buffer_set_end_id = -1
        self.last_prefetch_start_cycle = -2
        self.last_prefetch_end_cycle = -1
        self.first_request_rcvd_cycle = 0

        # Internal data structures
        self.current_set = set()
        self.list_of_sets = []
        self.num_sets_active_buffer = 1
        self.num_sets_prefetch_buffer = 1

        # Flags
        self.first_request_seen = False
        self.params_set_flag = False
        self.active_buffer_prefetch_done = False
        self.trace_valid = False
        self.use_ramulator_trace = False

    #
    def set_params(self, backing_buf_obj,
                   total_size_bytes=1, word_size=1, active_buf_frac=0.9,
                   hit_latency=1, backing_buf_default_bw=1,
                   use_ramulator_trace = False):
        """
        Method to set the ifmap/filter double buffered memory simulation parameters for estimate
        bandwidth mode.
        """

        self.total_size_bytes = total_size_bytes
        self.word_size = word_size

        assert 0.5 <= active_buf_frac < 1, "Valid active buf frac [0.5,1)"
        self.active_buf_frac = round(active_buf_frac, 2)
        self.hit_latency = hit_latency

        self.backing_buffer = backing_buf_obj
        self.default_bandwidth = backing_buf_default_bw
        self.prefetch_bandwidth = self.default_bandwidth

        # Calculate these based on the values provided
        self.total_size_elems = math.floor(self.total_size_bytes / self.word_size)
        self.active_buf_size = int(math.ceil(self.total_size_elems * self.active_buf_frac))
        self.prefetch_buf_size = self.total_size_elems - self.active_buf_size

        #
        self.num_items_per_set = math.floor(self.total_size_elems / 100)
        self.num_sets_active_buffer = int(self.active_buf_frac * 100)
        self.num_sets_prefetch_buffer = 100 - self.num_sets_active_buffer

        self.current_set = set()
        self.current_set_id = 0
        self.list_of_sets = []
        self.read_buffer_set_start_id = 0
        self.read_buffer_set_end_id = self.num_sets_active_buffer - 1
        self.last_prefetch_start_cycle = -2
        self.last_prefetch_end_cycle = -1  # TODO: Check what the correct value is
        self.use_ramulator_trace = use_ramulator_trace

        #
        self.params_set_flag = True

    #
    def service_reads(self, incoming_requests_arr_np, incoming_cycles_arr):
        """
        Method to service read requests coming from systolic array in estimate bandwidth mode.
        """
        assert self.params_set_flag, 'Parameters are not set yet'
        assert incoming_cycles_arr.shape[0] == incoming_requests_arr_np.shape[0],\
               'Incoming cycles and requests dont match'

        outcycles = incoming_cycles_arr + self.hit_latency
        # In estimate mode, operation is stall free.
        # Therefore its always a hit

        # The following to track requests and maintain proper state of the buffer
        for i in range(incoming_requests_arr_np.shape[0]):
            cycle = int(incoming_cycles_arr[i][0])

            requests_this_cycle = incoming_requests_arr_np[i]
            if not self.first_request_seen:
                if max(requests_this_cycle) > -1:
                    self.first_request_rcvd_cycle = cycle
                    self.first_request_seen = True

            for addr in requests_this_cycle:
                if not addr == -1:
                    self.manage_prefetches(cycle, addr)

        return outcycles

    #
    def manage_prefetches(self, cycle, addr):
        """
        Method to manage prefetches in estimate bandwidth mode.
        """

        # If this is a new address, otherwise its a hit
        if self.check_hit(addr):
            return

        if addr not in self.current_set:
            self.current_set.add(addr)
            self.elems_current_set += 1

            if self.elems_current_set == self.num_items_per_set:
                self.list_of_sets += [self.current_set]
                self.current_set = set()
                self.elems_current_set = 0
                self.current_set_id += 1

                # This should be prefetched
                if self.current_set_id == self.read_buffer_set_end_id + 1:
                    if not self.active_buffer_prefetch_done:
                        self.prefetch_bandwidth = self.default_bandwidth
                        self.last_prefetch_end_cycle = \
                            self.first_request_rcvd_cycle - 1 - self.backing_buffer.get_latency()

                        cycles_needed = (self.num_sets_prefetch_buffer * self.num_items_per_set) \
                                        / self.prefetch_bandwidth
                        cycles_needed = math.ceil(cycles_needed)

                        self.last_prefetch_start_cycle = \
                            self.last_prefetch_end_cycle - cycles_needed + 1

                        self.prefetch()
                        self.prefetch_buffer_set_start_id =self.read_buffer_set_end_id + 1
                        self.prefetch_buffer_set_end_id = self.prefetch_buffer_set_start_id + \
                                                          self.num_sets_prefetch_buffer - 1
                        self.active_buffer_prefetch_done = True

                    else:
                        elems_to_prefetch = self.num_sets_prefetch_buffer * self.num_items_per_set
                        cycles_needed = \
                            self.last_prefetch_end_cycle - self.last_prefetch_start_cycle + 1
                        self.prefetch_bandwidth = math.ceil(elems_to_prefetch / cycles_needed)
                        self.prefetch()
                        self.prefetch_buffer_set_start_id += self.num_sets_prefetch_buffer
                        self.prefetch_buffer_set_end_id += self.num_sets_prefetch_buffer

                    # Solving memory leak by discarding sets that are no longer in use
                    i = self.read_buffer_set_start_id
                    for j in range(self.num_sets_prefetch_buffer):
                        self.list_of_sets[i+j] = None


                    self.read_buffer_set_start_id += self.num_sets_prefetch_buffer
                    self.read_buffer_set_end_id += self.num_sets_prefetch_buffer
                    self.last_prefetch_start_cycle = self.last_prefetch_end_cycle +1
                    self.last_prefetch_end_cycle = cycle

    #
    def check_hit(self, addr):
        """
        Method to check if the address is hit or miss in the active read buffer.
        """
        assert self.params_set_flag, 'Parameters are not set yet'

        start_set_idx = self.read_buffer_set_start_id
        end_set_idx = min(self.current_set_id, self.read_buffer_set_end_id + 1)

        if start_set_idx == end_set_idx:
            return False

        for idx in range(start_set_idx, end_set_idx):
            if addr in self.list_of_sets[idx]:
                return True

        return False

    #
    def complete_all_prefetches(self):
        """
        Method to complete all the prefetches in estimate bandwidth mode. Prefetch first the active
        buffer if not done before and then keep prefetching prefetch buffers.
        """
        assert self.params_set_flag, 'Parameters are not set yet'

        current_set_elems = list(self.current_set)
        if len(current_set_elems) > 0:
            self.list_of_sets += [self.current_set]
        else:
            self.current_set_id -= 1    # If there are no elems in this set, dont consider it

        if not self.active_buffer_prefetch_done:
            self.prefetch_bandwidth = self.default_bandwidth
            self.last_prefetch_end_cycle = -1 - self.backing_buffer.get_latency()

            num_sets_to_prefetch = self.current_set_id + 1
            self.num_sets_active_buffer = num_sets_to_prefetch

            cycles_needed = (num_sets_to_prefetch * self.num_items_per_set) \
                            / self.prefetch_bandwidth
            cycles_needed = math.ceil(cycles_needed)

            self.last_prefetch_start_cycle = self.last_prefetch_end_cycle - cycles_needed + 1

            self.prefetch()
            self.active_buffer_prefetch_done = True
        else:
            num_sets_to_prefetch = self.current_set_id - self.prefetch_buffer_set_start_id + 1
            self.prefetch_buffer_set_end_id = self.current_set_id
            elems_to_prefetch = num_sets_to_prefetch * self.num_items_per_set
            cycles_needed = self.last_prefetch_end_cycle - self.last_prefetch_start_cycle + 1
            self.prefetch_bandwidth = math.ceil(elems_to_prefetch / cycles_needed)
            self.prefetch()

    #
    def prefetch(self):
        """
        Method to do a new prefetch. In a new prefetch, some portion of the original data needs to
        be deleted to accomodate the prefetched data In this case we overwrite some data in the
        active buffer with the prefetched data and then create a new prefetch request.
        """
        assert self.params_set_flag, 'Parameters are not set yet'

        if not self.active_buffer_prefetch_done:
            start_set_idx = 0
            end_set_idx = self.num_sets_active_buffer - 1
        else:
            start_set_idx = self.prefetch_buffer_set_start_id
            end_set_idx = self.prefetch_buffer_set_end_id

        all_addresses = []
        for idx in range(start_set_idx, end_set_idx + 1):
            this_set = self.list_of_sets[idx]
            all_addresses += list(this_set)

        self.num_access += len(all_addresses)

        cycles_needed = self.last_prefetch_end_cycle - self.last_prefetch_start_cycle + 1
        max_prefetch_capacity = cycles_needed * self.prefetch_bandwidth

        delta = max_prefetch_capacity - len(all_addresses)

        if delta > 0:
            for _ in range(delta):
                all_addresses += [-1]

        prefetch_requests = np.asarray(all_addresses).reshape((cycles_needed,
                                                               self.prefetch_bandwidth))

        cycles_arr = np.zeros((cycles_needed,1))
        for i in range(cycles_arr.shape[0]):
            cycles_arr[i][0] = self.last_prefetch_start_cycle + i

        response_cycles_arr = \
            self.backing_buffer.service_reads(incoming_cycles_arr=cycles_arr,
                                              incoming_requests_arr_np=prefetch_requests)

        # Create / add elements to the trace matrix
        this_prefetch_traces = np.concatenate((response_cycles_arr, prefetch_requests), axis=1)

        if not self.trace_valid:
            self.trace_matrix = this_prefetch_traces
            self.trace_valid = True

        else:
            del_cols = self.trace_matrix.shape[1] - this_prefetch_traces.shape[1]
            if del_cols > 0:
                empty_cols = np.ones((this_prefetch_traces.shape[0], del_cols))
                this_prefetch_traces = np.concatenate((this_prefetch_traces, empty_cols), axis=1)

            elif del_cols < 0:
                del_cols = int(-1 * del_cols)
                empty_cols = np.ones((self.trace_matrix.shape[0], del_cols))
                self.trace_matrix = np.concatenate((self.trace_matrix, empty_cols), axis=1)

            self.trace_matrix = np.concatenate((self.trace_matrix, this_prefetch_traces), axis=0)

    #
    def get_latency(self):
        """
        Method to get hit latency of the read estimate buffer.
        """
        assert self.params_set_flag, 'Parameters are not valid'
        return self.hit_latency

    #
    def get_trace_matrix(self):
        """
        Method to get the read estimate buffer trace matrix. It contains addresses requsted by the
        systolic array and the cycles (first column) at which the requests are made.
        """
        if not self.trace_valid:
            print('No trace has been generated yet')
            return

        return self.trace_matrix

    #
    def get_hit_latency(self):
        """
        Method to get hit latency of the read estimate buffer.
        """
        return self.hit_latency

    #
    def get_num_accesses(self):
        """
        Method to get number of accesses of the read estimate buffer if trace_valid flag is set.
        """
        assert self.trace_valid, 'Traces not ready yet'
        return self.num_access

    #
    def get_external_access_start_stop_cycles(self):
        """
        Method to get start and stop cycles of the read estimate buffer if trace_valid flag is set.
        """
        assert self.trace_valid, 'Traces not ready yet'
        start_cycle = self.trace_matrix[0][0]
        end_cycle = self.trace_matrix[-1][0]

        return start_cycle, end_cycle

    #
    def print_trace(self, filename):
        """
        Method to write the read estimate buffer trace matrix to a file.
        """
        if not self.trace_valid:
            print('No trace has been generated yet')
            return

        np.savetxt(filename, self.trace_matrix, fmt='%s', delimiter=",")
