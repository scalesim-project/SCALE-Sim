"""
This file contains the 'double_buffered_scratchpad' class that handles memory module simulations of
double buffered SRAMs.
"""

import time
import os
import math
import numpy as np
from tqdm import tqdm

from scalesim.scale_config import scale_config as cfg
from scalesim.topology_utils import topologies as topo
from scalesim.memory.read_buffer import read_buffer as rdbuf
from scalesim.memory.read_buffer_estimate_bw import ReadBufferEstimateBw as rdbuf_est
from scalesim.memory.read_port import read_port as rdport
from scalesim.memory.write_buffer import write_buffer as wrbuf
from scalesim.memory.write_port import write_port as wrport

class double_buffered_scratchpad:
    """
    Class which runs the memory simulation of double buffered scratchpad memories (SRAMs). The
    double buffering helps to hide the DRAM latency when the SRAM is servicing requests from the
    systolic array using one of the buffers while the other buffer prefetches from the DRAM.
    """
    #
    def __init__(self):
        """
        __init__ method.
        """
        self.layer_id = 0
        self.ifmap_buf = rdbuf()
        self.filter_buf = rdbuf()
        self.ofmap_buf =wrbuf()

        self.ifmap_port = rdport()
        self.filter_port = rdport()
        self.ofmap_port = wrport()
        self.config = cfg()
        self.topo = topo()

        self.verbose = True

        self.ifmap_trace_matrix = np.zeros((1,1), dtype=int)
        self.filter_trace_matrix = np.zeros((1,1), dtype=int)
        self.ofmap_trace_matrix = np.zeros((1,1), dtype=int)

        # Metrics to gather for generating run reports
        self.total_cycles = 0
        self.compute_cycles = 0
        self.stall_cycles = 0

        self.avg_ifmap_dram_bw = 0
        self.avg_filter_dram_bw = 0
        self.avg_ofmap_dram_bw = 0

        self.ifmap_sram_start_cycle = 0
        self.ifmap_sram_stop_cycle = 0
        self.filter_sram_start_cycle = 0
        self.filter_sram_stop_cycle = 0
        self.ofmap_sram_start_cycle = 0
        self.ofmap_sram_stop_cycle = 0

        self.ifmap_dram_start_cycle = 0
        self.ifmap_dram_stop_cycle = 0
        self.ifmap_dram_reads = 0
        self.filter_dram_start_cycle = 0
        self.filter_dram_stop_cycle = 0
        self.filter_dram_reads = 0
        self.ofmap_dram_start_cycle = 0
        self.ofmap_dram_stop_cycle = 0
        self.ofmap_dram_writes = 0

        self.estimate_bandwidth_mode = False
        self.traces_valid = False
        self.params_valid_flag = True
        self.use_ramulator_trace = self.config.get_ramulator_trace()

        self.using_ifmap_custom_layout = False
        self.using_filter_custom_layout = False

        # Dynamic IFMAP/FILTER bank allocation state
        self.enable_dynamic_bank_allocation = False
        self.static_ifmap_sram_bank_num = 1
        self.static_filter_sram_bank_num = 1
        self.ifmap_sram_bank_port = 2
        self.filter_sram_bank_port = 2
        self.dynamic_ifmap_banks = set()
        self.dynamic_filter_banks = set()
        self.dynamic_unassigned_banks = []
        self.dynamic_target_ifmap_banks = 1
        self.dynamic_target_filter_banks = 1

    #
    def set_params(self,
                   layer_id=0,
                   verbose=True,
                   estimate_bandwidth_mode=False,
                   word_size=1,
                   ifmap_buf_size_bytes=2, filter_buf_size_bytes=2, ofmap_buf_size_bytes=2,
                   rd_buf_active_frac=0.5, wr_buf_active_frac=0.5,
                   ifmap_backing_buf_bw=1, filter_backing_buf_bw=1, ofmap_backing_buf_bw=1,
                   ifmap_sram_bank_num=1, ifmap_sram_bank_port=2, filter_sram_bank_num=1, filter_sram_bank_port=2,
                   using_ifmap_custom_layout=False, using_filter_custom_layout=False,
                   enable_dynamic_bank_allocation=False,
                   config=cfg(), topo=topo()
                   ):

        """
        Method to set the double buffered memory simulation parameters for housekeeping.
        """
        self.layer_id = layer_id
        self.topo = topo
        self.config = config
        self.use_ramulator_trace = config.get_ramulator_trace()
        self.static_ifmap_sram_bank_num = max(1, int(ifmap_sram_bank_num))
        self.static_filter_sram_bank_num = max(1, int(filter_sram_bank_num))
        self.ifmap_sram_bank_port = max(1, int(ifmap_sram_bank_port))
        self.filter_sram_bank_port = max(1, int(filter_sram_bank_port))

        self.estimate_bandwidth_mode = estimate_bandwidth_mode

        if self.estimate_bandwidth_mode:
            self.ifmap_buf = rdbuf_est()
            self.filter_buf = rdbuf_est()

            self.ifmap_buf.set_params(backing_buf_obj=self.ifmap_port,
                                      total_size_bytes=ifmap_buf_size_bytes,
                                      word_size=word_size,
                                      active_buf_frac=rd_buf_active_frac,
                                      backing_buf_default_bw=ifmap_backing_buf_bw,
                                      use_ramulator_trace=self.use_ramulator_trace
                                      )

            self.filter_buf.set_params(backing_buf_obj=self.filter_port,
                                       total_size_bytes=filter_buf_size_bytes,
                                       word_size=word_size,
                                       active_buf_frac=rd_buf_active_frac,
                                       backing_buf_default_bw=filter_backing_buf_bw,
                                       use_ramulator_trace=self.use_ramulator_trace
                                       )
        else:
            self.ifmap_buf = rdbuf()
            self.filter_buf = rdbuf()
            
            if self.use_ramulator_trace == True:
                root_path = os.getcwd()
                #topology_file = self.topo.split('.')[0]
                topology_file =''
                ifmap_dram_trace = (root_path+"/results/"+topology_file+"_ifmapFile"+str(layer_id)+".npy")
                filter_dram_trace = (root_path+"/results/"+topology_file+"_filterFile"+str(layer_id)+".npy")
                ofmap_dram_trace = (root_path+"/results/"+topology_file+"_ofmapFile"+str(layer_id)+".npy")
                self.ifmap_port.def_params(config = self.config, latency_file=ifmap_dram_trace)
                self.filter_port.def_params(config = self.config, latency_file=filter_dram_trace)
                self.ofmap_port.def_params(config=self.config, latency_file=ofmap_dram_trace)

            self.ifmap_buf.set_params(backing_buf_obj=self.ifmap_port,
                                      total_size_bytes=ifmap_buf_size_bytes,
                                      word_size=word_size,
                                      active_buf_frac=rd_buf_active_frac,
                                      backing_buf_bw=ifmap_backing_buf_bw,
                                      num_bank=ifmap_sram_bank_num,
                                      num_port=ifmap_sram_bank_port,
                                      enable_layout_evaluation=using_ifmap_custom_layout,
                                      use_ramulator_trace=self.use_ramulator_trace
                                      )

            self.filter_buf.set_params(backing_buf_obj=self.filter_port,
                                       total_size_bytes=filter_buf_size_bytes,
                                       word_size=word_size,
                                       active_buf_frac=rd_buf_active_frac,
                                       backing_buf_bw=filter_backing_buf_bw,
                                       num_bank=filter_sram_bank_num,
                                       num_port=filter_sram_bank_port,
                                       enable_layout_evaluation=using_filter_custom_layout,
                                       use_ramulator_trace=self.use_ramulator_trace
                                       )

        self.ofmap_buf.set_params(backing_buf_obj=self.ofmap_port,
                                  total_size_bytes=ofmap_buf_size_bytes,
                                  word_size=word_size,
                                  active_buf_frac=wr_buf_active_frac,
                                  backing_buf_bw=ofmap_backing_buf_bw)

        self.verbose = verbose

        self.using_ifmap_custom_layout = using_ifmap_custom_layout  
        self.using_filter_custom_layout = using_filter_custom_layout
        self.enable_dynamic_bank_allocation = bool(enable_dynamic_bank_allocation)
        if self.estimate_bandwidth_mode:
            self.enable_dynamic_bank_allocation = False
        if not (self.using_ifmap_custom_layout and self.using_filter_custom_layout):
            self.enable_dynamic_bank_allocation = False

        self.dynamic_ifmap_banks = set()
        self.dynamic_filter_banks = set()
        self.dynamic_unassigned_banks = []
        self.dynamic_target_ifmap_banks = 1
        self.dynamic_target_filter_banks = 1

        self.params_valid_flag = True


    #
    def set_read_buf_prefetch_matrices(self,
                                       ifmap_prefetch_mat=np.zeros((1,1)),
                                       filter_prefetch_mat=np.zeros((1,1))
                                       ):
        """
        Method to read ifmap and filter prefetch matrices generated in the compute simulation.
        """

        self.ifmap_buf.set_fetch_matrix(ifmap_prefetch_mat)
        self.filter_buf.set_fetch_matrix(filter_prefetch_mat)

    #
    def reset_buffer_states(self):
        """
        Method to reset ifmap, filter and ofmap SRAMs.
        """

        self.ifmap_buf.reset()
        self.filter_buf.reset()
        self.ofmap_buf.reset()

    # The following are just shell methods for users to control each mem individually
    def service_ifmap_reads(self,
                            incoming_requests_arr_np,   # 2D array with the requests
                            incoming_cycles_arr):
        """
        Method to service ifmap read requests coming from systolic array.
        """
        out_cycles_arr_np = self.ifmap_buf.service_reads(incoming_requests_arr_np,
                                                         incoming_cycles_arr)

        return out_cycles_arr_np

    #
    def service_filter_reads(self,
                            incoming_requests_arr_np,   # 2D array with the requests
                            incoming_cycles_arr):
        """
        Method to service filter read requests coming from systolic array.
        """
        out_cycles_arr_np = self.filter_buf.service_reads(incoming_requests_arr_np,
                                                          incoming_cycles_arr)

        return out_cycles_arr_np

    #
    def service_ofmap_writes(self,
                             incoming_requests_arr_np,  # 2D array with the requests
                             incoming_cycles_arr):
        """
        Method to service ofmap write requests coming from systolic array.
        """

        out_cycles_arr_np = self.ofmap_buf.service_writes(incoming_requests_arr_np,
                                                          incoming_cycles_arr, 1)

        return out_cycles_arr_np

    def _apply_dynamic_bank_topology(self):
        """
        Apply current dynamic bank assignment to IFMAP/FILTER read buffers.
        """
        ifmap_banks = max(1, len(self.dynamic_ifmap_banks))
        filter_banks = max(1, len(self.dynamic_filter_banks))
        self.ifmap_buf.update_bank_topology(num_bank=ifmap_banks,
                                            num_port=self.ifmap_sram_bank_port)
        self.filter_buf.update_bank_topology(num_bank=filter_banks,
                                             num_port=self.filter_sram_bank_port)

    def _assign_one_dynamic_bank(self, assign_to_ifmap):
        """
        Permanently assign one unassigned bank to IFMAP or FILTER.
        """
        if len(self.dynamic_unassigned_banks) == 0:
            return False

        bank_id = self.dynamic_unassigned_banks.pop(0)
        if assign_to_ifmap:
            self.dynamic_ifmap_banks.add(bank_id)
        else:
            self.dynamic_filter_banks.add(bank_id)

        self._apply_dynamic_bank_topology()
        return True

    def _estimate_required_banks(self, demand_line, num_port, total_banks):
        """
        Estimate required banks from instantaneous request pressure.
        """
        valid_reqs = int(np.count_nonzero(demand_line != -1))
        if valid_reqs == 0:
            return 1
        est_banks = int(math.ceil(valid_reqs / max(1, num_port)))
        est_banks = min(total_banks - 1, max(1, est_banks))
        return est_banks

    def _estimate_unique_demand_bytes(self, demand_mat, word_size=1):
        """
        Estimate unique demanded payload size in bytes for one operand.
        """
        flat_payload = demand_mat.reshape(-1)
        valid_payload = flat_payload[flat_payload != -1]
        if valid_payload.size == 0:
            return 0.0
        unique_words = np.unique(valid_payload).size
        return float(unique_words * max(1, int(word_size)))

    def _allocate_towards_target_distribution(self):
        """
        Allocate all free banks towards target IFMAP/FILTER distribution.
        """
        while len(self.dynamic_unassigned_banks) > 0:
            deficit_ifmap = max(0, self.dynamic_target_ifmap_banks - len(self.dynamic_ifmap_banks))
            deficit_filter = max(0, self.dynamic_target_filter_banks - len(self.dynamic_filter_banks))

            if deficit_ifmap == 0 and deficit_filter == 0:
                break

            if deficit_ifmap > deficit_filter:
                self._assign_one_dynamic_bank(assign_to_ifmap=True)
            elif deficit_filter > deficit_ifmap:
                self._assign_one_dynamic_bank(assign_to_ifmap=False)
            else:
                self._assign_one_dynamic_bank(assign_to_ifmap=(len(self.dynamic_ifmap_banks) <= len(self.dynamic_filter_banks)))

    def _initialize_dynamic_bank_allocator(self, ifmap_demand_mat, filter_demand_mat):
        """
        Initialize bank pools and assign banks to balance capacity utilization.
        """
        total_banks = self.static_ifmap_sram_bank_num + self.static_filter_sram_bank_num
        if total_banks < 2:
            self.enable_dynamic_bank_allocation = False
            return

        # Start with one dedicated bank each, and keep the rest in a free pool.
        self.dynamic_ifmap_banks = {0}
        self.dynamic_filter_banks = {1}
        self.dynamic_unassigned_banks = list(range(2, total_banks))
        self._apply_dynamic_bank_topology()

        if len(self.dynamic_unassigned_banks) == 0:
            return

        ifmap_need_bytes = self._estimate_unique_demand_bytes(ifmap_demand_mat,
                                                               word_size=getattr(self.ifmap_buf, 'word_size', 1))
        filter_need_bytes = self._estimate_unique_demand_bytes(filter_demand_mat,
                                                                word_size=getattr(self.filter_buf, 'word_size', 1))

        ifmap_per_bank_capacity = max(1.0, self.ifmap_buf.total_size_bytes / max(1, self.static_ifmap_sram_bank_num))
        filter_per_bank_capacity = max(1.0, self.filter_buf.total_size_bytes / max(1, self.static_filter_sram_bank_num))

        ifmap_weight = ifmap_need_bytes / ifmap_per_bank_capacity
        filter_weight = filter_need_bytes / filter_per_bank_capacity

        if ifmap_weight <= 0 and filter_weight <= 0:
            ifmap_weight = 1.0
            filter_weight = 1.0

        target_ifmap = int(round(total_banks * (ifmap_weight / (ifmap_weight + filter_weight))))
        target_ifmap = min(total_banks - 1, max(1, target_ifmap))
        target_filter = total_banks - target_ifmap

        self.dynamic_target_ifmap_banks = target_ifmap
        self.dynamic_target_filter_banks = target_filter

        self._allocate_towards_target_distribution()

        while len(self.dynamic_unassigned_banks) > 0:
            if ifmap_weight >= filter_weight:
                self._assign_one_dynamic_bank(assign_to_ifmap=True)
            else:
                self._assign_one_dynamic_bank(assign_to_ifmap=False)

    def _dynamic_allocate_from_demand(self, ifmap_demand_line, filter_demand_line):
        """
        Allocate remaining banks towards precomputed target distribution.
        """
        if len(self.dynamic_unassigned_banks) == 0:
            return
        self._allocate_towards_target_distribution()

    def _dynamic_allocate_from_stall_feedback(self, ifmap_stall, filter_stall):
        """
        Allocate one extra bank only when target distribution is not yet reached.
        """
        if len(self.dynamic_unassigned_banks) == 0:
            return

        if ifmap_stall <= 0 and filter_stall <= 0:
            return

        deficit_ifmap = max(0, self.dynamic_target_ifmap_banks - len(self.dynamic_ifmap_banks))
        deficit_filter = max(0, self.dynamic_target_filter_banks - len(self.dynamic_filter_banks))
        if deficit_ifmap == 0 and deficit_filter == 0:
            return

        if ifmap_stall > filter_stall:
            self._assign_one_dynamic_bank(assign_to_ifmap=True)
        elif filter_stall > ifmap_stall:
            self._assign_one_dynamic_bank(assign_to_ifmap=False)

    #
    def service_memory_requests(self, ifmap_demand_mat, filter_demand_mat, ofmap_demand_mat):
        """
        Method to run the memory simulation of ifmap, filter and ofmap SRAMs together and generate
        the traces.
        """
        assert self.params_valid_flag, 'Memories not initialized yet'

        ofmap_lines = ofmap_demand_mat.shape[0]

        self.total_cycles = 0
        self.stall_cycles = 0

        ifmap_hit_latency = self.ifmap_buf.get_hit_latency()
        filter_hit_latency = self.filter_buf.get_hit_latency()

        ifmap_serviced_cycles = []
        filter_serviced_cycles = []
        ofmap_serviced_cycles = []

        if self.enable_dynamic_bank_allocation:
            self._initialize_dynamic_bank_allocator(ifmap_demand_mat, filter_demand_mat)

        pbar_disable = not self.verbose
        for i in tqdm(range(ofmap_lines), disable=pbar_disable):

            cycle_arr = np.zeros((1,1)) + i + self.stall_cycles

            ifmap_demand_line = ifmap_demand_mat[i, :].reshape((1,ifmap_demand_mat.shape[1]))
            filter_demand_line = filter_demand_mat[i, :].reshape((1, filter_demand_mat.shape[1]))

            if self.enable_dynamic_bank_allocation:
                # Permanent one-way assignment from free pool to IFMAP/FILTER.
                self._dynamic_allocate_from_demand(ifmap_demand_line, filter_demand_line)

            ifmap_cycle_out = \
                self.ifmap_buf.service_reads(incoming_requests_arr_np=ifmap_demand_line,
                                             incoming_cycles_arr=cycle_arr)
            ifmap_serviced_cycles += [ifmap_cycle_out[0]]
            ifmap_stalls = ifmap_cycle_out[0] - cycle_arr[0] - ifmap_hit_latency

            filter_cycle_out = \
                self.filter_buf.service_reads(incoming_requests_arr_np=filter_demand_line,
                                              incoming_cycles_arr=cycle_arr)
            filter_serviced_cycles += [filter_cycle_out[0]]
            filter_stalls = filter_cycle_out[0] - cycle_arr[0] - filter_hit_latency

            if self.enable_dynamic_bank_allocation:
                self._dynamic_allocate_from_stall_feedback(ifmap_stall=float(ifmap_stalls[0]),
                                                           filter_stall=float(filter_stalls[0]))

            ofmap_demand_line = ofmap_demand_mat[i, :].reshape((1, ofmap_demand_mat.shape[1]))
            ofmap_cycle_out = \
                self.ofmap_buf.service_writes(incoming_requests_arr_np=ofmap_demand_line,
                                              incoming_cycles_arr_np=cycle_arr)
            ofmap_serviced_cycles += [ofmap_cycle_out[0]]
            ofmap_stalls = ofmap_cycle_out[0] - cycle_arr[0]

            self.stall_cycles += int(max(ifmap_stalls[0], filter_stalls[0], ofmap_stalls[0]))
            #self.stall_cycles += ifmap_stalls[0] + filter_stalls[0] + ofmap_stalls[0]

        if self.estimate_bandwidth_mode:
            # IDE shows warning as complete_all_prefetches is not implemented in read_buffer class
            # It's harmless since read_buffer_estimate_bw is instantiated in estimate bandwidth mode
            self.ifmap_buf.complete_all_prefetches()
            self.filter_buf.complete_all_prefetches()

        self.ofmap_buf.empty_all_buffers(ofmap_serviced_cycles[-1])

        # Prepare the traces
        ifmap_services_cycles_np = \
            np.asarray(ifmap_serviced_cycles).reshape((len(ifmap_serviced_cycles), 1))
        self.ifmap_trace_matrix = np.concatenate((ifmap_services_cycles_np, ifmap_demand_mat),
                                                 axis=1)

        filter_services_cycles_np = \
            np.asarray(filter_serviced_cycles).reshape((len(filter_serviced_cycles), 1))
        self.filter_trace_matrix = np.concatenate((filter_services_cycles_np, filter_demand_mat),
                                                  axis=1)

        ofmap_services_cycles_np = \
            np.asarray(ofmap_serviced_cycles).reshape((len(ofmap_serviced_cycles), 1))
        self.ofmap_trace_matrix = np.concatenate((ofmap_services_cycles_np, ofmap_demand_mat),
                                                 axis=1)
        #self.total_cycles = int(ofmap_serviced_cycles[-1][0])
        ## Probable fault in sanity check
        self.total_cycles = int(max(ofmap_serviced_cycles))

        # END of serving demands from memory
        self.traces_valid = True

    # This is the trace computation logic of this memory system
    # Anand: This is too complex, perform the serve cycle by cycle for the requests
    def service_memory_requests_old(self, ifmap_demand_mat, filter_demand_mat, ofmap_demand_mat):
        """
        This is the trace computation logic of this memory system.
        """
        # TODO: assert sanity check
        assert self.params_valid_flag, 'Memories not initialized yet'

        # Logic:
        # Stalls can occur in both read and write portions and interfere with each other
        # We mitigate interference by picking a window in which there are no write stall,
        # ie, there is sufficient free space in the write buffer

        # The three demand mats have the same shape though
        ofmap_lines_remaining = ofmap_demand_mat.shape[0]
        start_line_idx = 0
        end_line_idx = 0

        first = True
        cycle_offset = 0
        self.total_cycles = 0
        self.stall_cycles = 0

        # Status bar
        pbar_disable = not self.verbose #or True
        pbar = tqdm(total=ofmap_lines_remaining, disable=pbar_disable)

        avg_read_time_series = []

        while ofmap_lines_remaining > 0:
            loop_start_time = time.time()
            ofmap_free_space = self.ofmap_buf.get_free_space()

            # Find the number of lines till the ofmap_free_space is filled up
            count = 0
            while not count > ofmap_free_space:
                this_line = ofmap_demand_mat[end_line_idx]
                for elem in this_line:
                    if not elem == -1:
                        count += 1

                if not count > ofmap_free_space:
                    end_line_idx += 1
                    # Limit check
                    if not end_line_idx < ofmap_demand_mat.shape[0]:
                        end_line_idx = ofmap_demand_mat.shape[0] - 1
                        count = ofmap_free_space + 1
                else:   # Send request with minimal data ie one line of the requests
                    end_line_idx += 1
            # END of line counting

            num_lines = end_line_idx - start_line_idx + 1
            this_req_cycles_arr = [int(x + cycle_offset) for x in range(num_lines)]
            this_req_cycles_arr_np = np.asarray(this_req_cycles_arr).reshape((num_lines,1))

            this_req_ifmap_demands = ifmap_demand_mat[start_line_idx:(end_line_idx + 1), :]
            this_req_filter_demands = filter_demand_mat[start_line_idx:(end_line_idx + 1), :]
            this_req_ofmap_demands = ofmap_demand_mat[start_line_idx:(end_line_idx + 1), :]

            no_stall_cycles = num_lines     # Since the cycles are consecutive at this point

            time_start = time.time()
            ifmap_cycles_out = \
                self.ifmap_buf.service_reads(incoming_requests_arr_np=this_req_ifmap_demands,
                                             incoming_cycles_arr=this_req_cycles_arr_np)
            time_end = time.time()
            delta = time_end - time_start
            avg_read_time_series.append(delta)

            # Take care of the incurred stalls when launching demands for filter_reads
            # Note: Stalls incurred on reading line i in ifmap reflect the request cycles for line
            #       i+1 in filter
            ifmap_hit_latency = self.ifmap_buf.get_hit_latency()
            # Vec - vec - scalar
            ifmap_stalls = ifmap_cycles_out - this_req_cycles_arr_np - ifmap_hit_latency
            # Shift by one row
            ifmap_stalls = np.concatenate((np.zeros((1,1)), ifmap_stalls[0:-1]), axis=0)
            this_req_cycles_arr_np = this_req_cycles_arr_np + ifmap_stalls

            time_start = time.time()
            filter_cycles_out = \
                self.filter_buf.service_reads(incoming_requests_arr_np=this_req_filter_demands,
                                              incoming_cycles_arr=this_req_cycles_arr_np)
            time_end = time.time()
            delta = time_end - time_start
            avg_read_time_series.append(delta)

            # Take care of stalls again --> The entire array stops when there is a stall
            filter_hit_latency = self.filter_buf.get_hit_latency()
            # Vec - vec - scalar
            filter_stalls = filter_cycles_out - this_req_cycles_arr_np - filter_hit_latency
            # Shift by one row
            filter_stalls = np.concatenate((np.zeros((1, 1)), filter_stalls[0:-1]), axis=0)
            this_req_cycles_arr_np = this_req_cycles_arr_np + filter_stalls

            ofmap_cycles_out = \
                self.ofmap_buf.service_writes(incoming_requests_arr_np=this_req_ofmap_demands,
                                              incoming_cycles_arr_np=this_req_cycles_arr_np)

            # Make the trace matrices
            this_req_ifmap_trace_matrix = \
                np.concatenate((ifmap_cycles_out, this_req_ifmap_demands), axis=1)
            this_req_filter_trace_matrix = \
                np.concatenate((filter_cycles_out, this_req_filter_demands), axis=1)
            this_req_ofmap_trace_matrix = \
                np.concatenate((ofmap_cycles_out, this_req_ofmap_demands), axis=1)

            actual_cycles = ofmap_cycles_out[-1][0] - this_req_cycles_arr_np[0][0] + 1
            num_stalls = actual_cycles - no_stall_cycles

            self.stall_cycles += num_stalls
            self.total_cycles = ofmap_cycles_out[-1][0] + 1         # OFMAP is served the last

            if first:
                first = False
                self.ifmap_trace_matrix = this_req_ifmap_trace_matrix
                self.filter_trace_matrix = this_req_filter_trace_matrix
                self.ofmap_trace_matrix = this_req_ofmap_trace_matrix
            else:
                self.ifmap_trace_matrix = \
                    np.concatenate((self.ifmap_trace_matrix, this_req_ifmap_trace_matrix), axis=0)
                self.filter_trace_matrix = \
                    np.concatenate((self.filter_trace_matrix, this_req_filter_trace_matrix), axis=0)
                self.ofmap_trace_matrix = \
                    np.concatenate((self.ofmap_trace_matrix, this_req_ofmap_trace_matrix), axis=0)

            # Update the local variable for another iteration of the while loop
            cycle_offset = ofmap_cycles_out[-1][0] + 1
            start_line_idx = end_line_idx + 1

            pbar.update(num_lines)
            # Cutoff at 0
            ofmap_lines_remaining = max(ofmap_demand_mat.shape[0] - (end_line_idx + 1), 0)
            # print("DEBUG: " + str(end_line_idx))

            if end_line_idx > ofmap_demand_mat.shape[0]:
                print('Trap')

            # if int(ofmap_lines_remaining % 1000) == 0:
            #     print("DEBUG: " + str(ofmap_lines_remaining))

            # loop_end_time = time.time()
            # loop_time = loop_end_time - loop_start_time
            # print('DEBUG: Time taken in one iteration: ' + str(loop_time))

        # At this stage there might still be some data in the active buffer of the OFMAP scratchpad
        # The following drains it and generates the OFMAP
        drain_start_cycle = self.ofmap_trace_matrix[-1][0] + 1
        self.ofmap_buf.empty_all_buffers(drain_start_cycle)

        #avg_read_time = sum(avg_read_time_series) / len(avg_read_time_series)
        #print('DEBUG: Avg time to service reads= ' + str(avg_read_time))

        pbar.close()
        # END of serving demands from memory
        self.traces_valid = True

    #
    def get_total_compute_cycles(self):
        """
        Method to get the total number of compute cycles if trace_valid flag is set.
        """
        assert self.traces_valid, 'Traces not generated yet'
        return self.total_cycles

    #
    def get_stall_cycles(self):
        """
        Method to get the number of stall cycles if trace_valid flag is set.
        """
        assert self.traces_valid, 'Traces not generated yet'
        return int(self.stall_cycles)

    def get_final_ifmap_filter_bank_allocation(self):
        """
        Method to get final IFMAP/FILTER bank ownership after simulation.
        """
        assert self.params_valid_flag, 'Memories not initialized yet'

        if self.enable_dynamic_bank_allocation and len(self.dynamic_ifmap_banks) > 0 and len(self.dynamic_filter_banks) > 0:
            return len(self.dynamic_ifmap_banks), len(self.dynamic_filter_banks)

        return int(self.static_ifmap_sram_bank_num), int(self.static_filter_sram_bank_num)

    def _get_unique_payload_words(self, trace_matrix):
        """
        Count unique valid addresses from a trace matrix payload region.
        """
        if trace_matrix is None or trace_matrix.size == 0:
            return 0

        payload = trace_matrix[:, 1:]
        if payload.size == 0:
            return 0

        flat_payload = payload.reshape(-1)
        valid = flat_payload[flat_payload != -1]
        if valid.size == 0:
            return 0

        return int(np.unique(valid).size)

    def get_ifmap_filter_bank_capacity_utilization(self):
        """
        Return IFMAP/FILTER bank capacity utilization.

        Utilization definition:
        used_capacity / (bank_count * per_bank_capacity)
        """
        assert self.traces_valid, 'Traces not generated yet'

        final_ifmap_banks, final_filter_banks = self.get_final_ifmap_filter_bank_allocation()

        # Per-bank capacity is derived from configured static banks.
        ifmap_per_bank_capacity = max(1.0, self.ifmap_buf.total_size_bytes / max(1, self.static_ifmap_sram_bank_num))
        filter_per_bank_capacity = max(1.0, self.filter_buf.total_size_bytes / max(1, self.static_filter_sram_bank_num))

        ifmap_total_capacity = max(1.0, final_ifmap_banks * ifmap_per_bank_capacity)
        filter_total_capacity = max(1.0, final_filter_banks * filter_per_bank_capacity)

        ifmap_used_words = self._get_unique_payload_words(self.ifmap_trace_matrix)
        filter_used_words = self._get_unique_payload_words(self.filter_trace_matrix)

        ifmap_word_size = max(1, int(getattr(self.ifmap_buf, 'word_size', 1)))
        filter_word_size = max(1, int(getattr(self.filter_buf, 'word_size', 1)))

        ifmap_used_capacity = ifmap_used_words * ifmap_word_size
        filter_used_capacity = filter_used_words * filter_word_size

        # Cap to 100% as a capacity-utilization metric.
        ifmap_used_capacity = min(ifmap_used_capacity, ifmap_total_capacity)
        filter_used_capacity = min(filter_used_capacity, filter_total_capacity)

        ifmap_util = ifmap_used_capacity / ifmap_total_capacity
        filter_util = filter_used_capacity / filter_total_capacity

        return float(ifmap_util), float(filter_util)

    #
    def get_ifmap_sram_start_stop_cycles(self):
        """
        Method to get the start and stop cycles of ifmap SRAM requests by the systolic array if
        trace_valid flag is set.
        """
        assert self.traces_valid, 'Traces not generated yet'

        done = False
        for ridx in range(self.ifmap_trace_matrix.shape[0]):
            if done:
                break
            row = self.ifmap_trace_matrix[ridx,1:]
            for addr in row:
                if not addr == -1:
                    self.ifmap_sram_start_cycle = self.ifmap_trace_matrix[ridx][0]
                    done = True
                    break

        done = False
        for ridx in range(self.ifmap_trace_matrix.shape[0]):
            if done:
                break
            ridx = -1 * (ridx + 1)
            row = self.ifmap_trace_matrix[ridx,1:]
            for addr in row:
                if not addr == -1:
                    self.ifmap_sram_stop_cycle  = self.ifmap_trace_matrix[ridx][0]
                    done = True
                    break

        return self.ifmap_sram_start_cycle, self.ifmap_sram_stop_cycle

    #
    def get_filter_sram_start_stop_cycles(self):
        """
        Method to get the start and stop cycles of filter SRAM requests by the systolic array if
        trace_valid flag is set.
        """
        assert self.traces_valid, 'Traces not generated yet'

        done = False
        for ridx in range(self.filter_trace_matrix.shape[0]):

            if done:
                break
            row = self.filter_trace_matrix[ridx, 1:]
            for addr in row:
                if not addr == -1:
                    self.filter_sram_start_cycle = self.filter_trace_matrix[ridx][0]
                    done = True
                    break

        done = False
        for ridx in range(self.filter_trace_matrix.shape[0]):
            if done:
                break
            ridx = -1 * (ridx + 1)

            row = self.filter_trace_matrix[ridx, 1:]
            for addr in row:
                if not addr == -1:
                    self.filter_sram_stop_cycle = self.filter_trace_matrix[ridx][0]
                    done = True
                    break

        return self.filter_sram_start_cycle, self.filter_sram_stop_cycle

    #
    def get_ofmap_sram_start_stop_cycles(self):
        """
        Method to get the start and stop cycles of ofmap SRAM requests by the systolic array if
        trace_valid flag is set.
        """
        assert self.traces_valid, 'Traces not generated yet'

        done = False
        for ridx in range(self.ofmap_trace_matrix.shape[0]):
            if done:
                break
            row = self.ofmap_trace_matrix[ridx, 1:]
            for addr in row:
                if not addr == -1:
                    self.ofmap_sram_start_cycle = self.ofmap_trace_matrix[ridx][0]
                    done = True
                    break

        done = False
        for ridx in range(self.ofmap_trace_matrix.shape[0]):
            if done:
                break
            ridx = -1 * (ridx + 1)
            row = self.ofmap_trace_matrix[ridx, 1:]
            for addr in row:
                if not addr == -1:
                    self.ofmap_sram_stop_cycle = self.ofmap_trace_matrix[ridx][0]
                    done = True
                    break

        return self.ofmap_sram_start_cycle, self.ofmap_sram_stop_cycle

    #
    def get_ifmap_dram_details(self):
        """
        Method to get the start cycle, stop cycle and number of reads of DRAM requests made by the
        ifmap SRAM if trace_valid flag is set.
        """
        assert self.traces_valid, 'Traces not generated yet'

        self.ifmap_dram_reads = self.ifmap_buf.get_num_accesses()
        self.ifmap_dram_start_cycle, self.ifmap_dram_stop_cycle \
            = self.ifmap_buf.get_external_access_start_stop_cycles()

        return self.ifmap_dram_start_cycle, self.ifmap_dram_stop_cycle, self.ifmap_dram_reads

    #
    def get_filter_dram_details(self):
        """
        Method to get the start cycle, stop cycle and number of reads of DRAM requests made by the
        filter SRAM if trace_valid flag is set.
        """
        assert self.traces_valid, 'Traces not generated yet'

        self.filter_dram_reads = self.filter_buf.get_num_accesses()
        self.filter_dram_start_cycle, self.filter_dram_stop_cycle \
            = self.filter_buf.get_external_access_start_stop_cycles()

        return self.filter_dram_start_cycle, self.filter_dram_stop_cycle, self.filter_dram_reads

    #
    def get_ofmap_dram_details(self):
        """
        Method to get the start cycle, stop cycle and number of writes of DRAM requests made by the
        ofmap SRAM if trace_valid flag is set.
        """
        assert self.traces_valid, 'Traces not generated yet'

        self.ofmap_dram_writes = self.ofmap_buf.get_num_accesses()
        self.ofmap_dram_start_cycle, self.ofmap_dram_stop_cycle \
            = self.ofmap_buf.get_external_access_start_stop_cycles()

        return self.ofmap_dram_start_cycle, self.ofmap_dram_stop_cycle, self.ofmap_dram_writes

    #
    def get_ifmap_sram_trace_matrix(self):
        """
        Method to get the ifmap SRAM trace matrix. It contains addresses requsted by the systolic
        array and the cycles (first column) at which the requests are made.
        """
        assert self.traces_valid, 'Traces not generated yet'
        return self.ifmap_trace_matrix

    #
    def get_filter_sram_trace_matrix(self):
        """
        Method to get the filter SRAM trace matrix. It contains addresses requsted by the systolic
        array and the cycles (first column) at which the requests are made.
        """
        assert self.traces_valid, 'Traces not generated yet'
        return self.filter_trace_matrix

    #
    def get_ofmap_sram_trace_matrix(self):
        """
        Method to get the ofmap SRAM trace matrix. It contains addresses requsted by the systolic
        array and the cycles (first column) at which the requests are made.
        """
        assert self.traces_valid, 'Traces not generated yet'
        return self.ofmap_trace_matrix

    #
    def get_sram_trace_matrices(self):
        """
        Method to get the ifmap, filter and ofmap SRAM trace matrices.
        """
        assert self.traces_valid, 'Traces not generated yet'
        return self.ifmap_trace_matrix, self.filter_trace_matrix, self.ofmap_trace_matrix

    #
    def get_ifmap_dram_trace_matrix(self):
        """
        Method to get the ifmap DRAM trace matrix. It contains addresses requsted by the ifmap SRAM
        and the cycles (first column) at which the requests are made.
        """
        return self.ifmap_buf.get_trace_matrix()

    #
    def get_filter_dram_trace_matrix(self):
        """
        Method to get the filter DRAM trace matrix. It contains addresses requsted by the filter
        SRAM and the cycles (first column) at which the requests are made.
        """
        return self.filter_buf.get_trace_matrix()

    #
    def get_ofmap_dram_trace_matrix(self):
        """
        Method to get the ofmap DRAM trace matrix. It contains addresses requsted by the ofmap SRAM
        and the cycles (first column) at which the requests are made.
        """
        return self.ofmap_buf.get_trace_matrix()

    #
    def get_dram_trace_matrices(self):
        """
        Method to get the ifmap, filter and ofmap DRAM trace matrices
        """
        dram_ifmap_trace = self.ifmap_buf.get_trace_matrix()
        dram_filter_trace = self.filter_buf.get_trace_matrix()
        dram_ofmap_trace = self.ofmap_buf.get_trace_matrix()

        return dram_ifmap_trace, dram_filter_trace, dram_ofmap_trace

    #
    def print_ifmap_sram_trace(self, filename):
        """
        Method to write the ifmap SRAM trace matrix to a file if trace_valid flag is set.
        """
        assert self.traces_valid, 'Traces not generated yet'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.savetxt(filename, self.ifmap_trace_matrix, fmt='%i', delimiter=",")

    #
    def print_filter_sram_trace(self, filename):
        """
        Method to write the filter SRAM trace matrix to a file if trace_valid flag is set.
        """
        assert self.traces_valid, 'Traces not generated yet'
        np.savetxt(filename, self.filter_trace_matrix, fmt='%i', delimiter=",")

    #
    def print_ofmap_sram_trace(self, filename):
        """
        Method to write the Ofmap SRAM trace matrix to a file if trace_valid flag is set.
        """
        assert self.traces_valid, 'Traces not generated yet'
        np.savetxt(filename, self.ofmap_trace_matrix, fmt='%i', delimiter=",")

    #
    def print_ifmap_dram_trace(self, filename):
        """
        Method to write the ifmap DRAM trace matrix to a file.
        """
        self.ifmap_buf.print_trace(filename)

    #
    def print_filter_dram_trace(self, filename):
        """
        Method to write the filter DRAM trace matrix to a file.
        """
        self.filter_buf.print_trace(filename)

    #
    def print_ofmap_dram_trace(self, filename):
        """
        Method to write the iomap DRAM trace matrix to a file.
        """
        self.ofmap_buf.print_trace(filename)
