"""
External DRAM write requests serviced by Ramulator
"""
import numpy as np
from scalesim.scale_config import scale_config as config
from bisect import bisect_left

# This is shell module to ensure continuity

class write_port:
    """
    Class to define external DRAM memory requests with Ramulator
    """
    #
    def __init__(self):
        """
        __init__ module.
        """
        self.latency = 0
        self.ramulator_trace = False
        self.latency_matrix = []
        self.bw = 10
        self.request_queue_size = 100
        self.request_queue_status = 0
        self.stall_cycles = 0
        self.request_array = []
        self.count = 0
        self.config = config()
    
    def def_params( self,
                    config = config(),
                    latency_file =''
                ):
        """
        Method to define the paths of ramulator trace numpy files 
        and write request queue sizes.
        """
        self.config = config
        self.ramulator_trace = self.config.get_ramulator_trace()
        self.request_queue_size = self.config.get_req_buf_sz_wr()
        self.bw = self.config.get_bandwidths_as_list()[0]
        if self.ramulator_trace == True:
            self.latency_matrix = np.load(latency_file)
        self.latency=0
    #

    def find_latency(self):
        """
        Method to map DRAM return path latency for each transactions.
        """
        if(self.count < len(self.latency_matrix)):
            latency_out = self.latency_matrix[self.count]
            #print(str(self.count)+ ' ' + str(latency_out))
            self.count+=1
        else:
            latency_out = self.latency
        if(latency_out > 10000):
            latency_out = 0

        return latency_out

    def service_writes(self, incoming_requests_arr_np, incoming_cycles_arr_np):
        """
        Method to service read request by the read buffer.
        Check for hit in the request queue or add the DRAM
        roundtrip latency for each transaction reported by 
        Ramulator.
        """
        if self.ramulator_trace == False:
            out_cycles_arr_np = incoming_cycles_arr_np + self.latency
            out_cycles_arr_np = out_cycles_arr_np.reshape((out_cycles_arr_np.shape[0], 1))
            return out_cycles_arr_np

        updated_req_timestamp = incoming_cycles_arr_np[0][0]
        print(updated_req_timestamp)
        out_cycles_arr = np.zeros(incoming_cycles_arr_np.shape[0])
        for i in range(len(incoming_cycles_arr_np)):
            out_cycles_arr[i] = incoming_cycles_arr_np[i][0] + self.stall_cycles + self.find_latency()
            self.request_array.append(out_cycles_arr[i])
            if len(self.request_array) == self.request_queue_size:
                updated_req_timestamp = incoming_cycles_arr_np[i][0] + self.stall_cycles
                self.request_array.sort()
                if self.request_array[0] >= updated_req_timestamp:
                    self.stall_cycles += self.request_array[0] - updated_req_timestamp
                    updated_req_timestamp = self.request_array[0]
                    self.request_array.pop(0)
                else:
                    index = bisect_left(self.request_array,updated_req_timestamp)
                    if index == len(self.request_array):
                        self.request_array = []
                        print("Empty array")
                    else:
                        self.request_array = self.request_array[index:]
            elif len(self.request_array) > self.request_queue_size:
                self.request_array = self.request_array[-self.request_queue_size:]
        # --- Placeholder for zeroization logic
        #if flush == 0:
        #    self.stall_cycles = 0

        self.stall_cycles =0
        return out_cycles_arr
