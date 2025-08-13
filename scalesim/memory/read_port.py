"""
External DRAM read requests serviced by Ramulator
"""
import numpy as np
from scalesim.scale_config import scale_config as config
from bisect import bisect_left

class read_port:
    """
    Class to define external DRAM memory requests with Ramulator
    """
    #
    def __init__(self):
        """
        __init__ method.
        """
        self.latency = 1
        self.ramulator_trace = False
        self.latency_matrix = []
        self.bw = 10
        self.request_queue_size = 100
        self.request_queue_status = 0
        self.stall_cycles = 0
        self.request_array = []
        self.count = 0
        self.config = config()
    #
    def def_params( self,
                    config = config(),
                    latency_file = ''
                ):
        """
        Method to define the paths of ramulator trace numpy files 
        and read request queue sizes.
        """
        self.config = config
        self.ramulator_trace = self.config.get_ramulator_trace()
        self.request_queue_size = self.config.get_req_buf_sz_rd()
        self.bw = self.config.get_bandwidths_as_list()[0]
        if self.ramulator_trace == True:
            self.latency_matrix = np.load(latency_file)
            #print(f"Latency file is {latency_file}")
        self.stall_cycles=0
        self.latency = 1
        
    def set_params(self, latency):
        """
        Method to set the backing buffer hit latency for housekeeping.
        """
        self.latency = latency

    #
    def get_latency(self):
        """
        Method to get the backing buffer hit latency for housekeeping.
        """
        return self.latency
    
    def find_latency(self):
        """
        Method to map DRAM return path latency for each transactions.
        """
        if(self.count < len(self.latency_matrix)):
            latency_out = self.latency_matrix[self.count]
            self.count+=1
        else:
            latency_out = self.latency
        if(latency_out > 10000):
            latency_out = 1
        return latency_out

    # The incoming read requests will be needed when the capability of port is expanded
    # At the moment its kept for compatibility
    def service_reads(self, incoming_requests_arr_np, incoming_cycles_arr):
        """
        Method to service read request by the read buffer.
        Check for hit in the request queue or add the DRAM
        roundtrip latency for each transaction reported by 
        Ramulator.
        """
        if self.ramulator_trace is False:
            out_cycles_arr = incoming_cycles_arr + self.latency
            return out_cycles_arr

        updated_req_timestamp = incoming_cycles_arr[0]
        out_cycles_arr = np.zeros(incoming_requests_arr_np.shape[0])
        for i in range(len(incoming_cycles_arr)):
            out_cycles_arr[i] = incoming_cycles_arr[i] + self.stall_cycles + self.find_latency()
            #print(str(incoming_cycles_arr[i]) + ' ' + str(out_cycles_arr[i]) + ' ' +str(self.stall_cycles))
            self.request_array.append(out_cycles_arr[i])
            if len(self.request_array) == self.request_queue_size:
                updated_req_timestamp = incoming_cycles_arr[i] + self.stall_cycles
                self.request_array.sort()
                if self.request_array[0] >= updated_req_timestamp:
                    self.stall_cycles += self.request_array[0] - updated_req_timestamp
                    #print(f"stall cycle: {self.stall_cycles}. request array: {self.request_array[0]} updated_req_timestamp: {updated_req_timestamp}")
                    updated_req_timestamp = self.request_array[0]
                    self.request_array.pop(0)
                else:
                    index = bisect_left(self.request_array,updated_req_timestamp)
                    if index == len(self.request_array):
                        self.request_array = []
                    else:
                        self.request_array = self.request_array[index:]
            elif len(self.request_array) > self.request_queue_size:
                self.request_array = self.request_array[-self.request_queue_size:]
                #print(f"stall cycle: {self.stall_cycles}. request array: {self.request_array[0]} updated_req_timestamp: {updated_req_timestamp}")
        
        self.stall_cycles=0
        return out_cycles_arr