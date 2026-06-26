import  numpy as np
from enum import unique
from inspect import trace
###############################################
# USER SET PARAMETERS FOR ENERGY CALCULATIONS #

Avg_saved_cycles = 2

SRAM_bank_size = Avg_saved_cycles
SRAM_row_size = 5
DRAM_bank_size = Avg_saved_cycles
DRAM_row_size = 5

###############################################


class energymodel:

    def __init__(self):

        self.SRAM_bank_size = 0
        self.SRAM_row_size = 0
        self.DRAM_bank_size = 0
        self.DRAMrow_size = 0


    def convert_to_accelergy_params(self, layer_id, memory_system, top_path):

        self.SRAM_bank_size  = SRAM_bank_size
        self.SRAM_row_size = SRAM_row_size
        self.DRAM_bank_size  = DRAM_bank_size
        self.DRAM_row_size = DRAM_row_size

        self.ifmap_sram_repeat = self.sram_repeat_check(bank_size=self.SRAM_bank_size, memory = memory_system.ifmap_trace_matrix, row_size=self.SRAM_row_size)
        self.ifmap_dram_repeat = self.dram_repeat_check(bank_size=self.DRAM_bank_size, memory = memory_system.ifmap_buf, row_size=self.DRAM_row_size)
        self.filter_sram_repeat = self.sram_repeat_check(bank_size=self.SRAM_bank_size, memory = memory_system.filter_trace_matrix, row_size=self.SRAM_row_size)
        self.filter_dram_repeat = self.dram_repeat_check(bank_size=self.DRAM_bank_size, memory = memory_system.filter_buf, row_size=self.DRAM_row_size)
        self.ofmap_sram_repeat = self.sram_repeat_check(bank_size=self.SRAM_bank_size, memory = memory_system.ifmap_trace_matrix, row_size=self.SRAM_row_size)
        self.ofmap_dram_repeat = self.dram_repeat_check(bank_size=self.DRAM_bank_size, memory = memory_system.ofmap_buf, row_size=self.DRAM_row_size)
        
        self.generate_report(top_path, layer_id)


    def generate_report(self, top_path, layer_id):
        repeat_report_name = top_path + '/REPEAT_CYCLE.csv'

        if layer_id == 0:
            repeat_report = open(repeat_report_name, 'w')
            header = 'layer_num,'
            header += 'ifmap_sram_repeat,filter_sram_repeat,ofmap_sram_repeat,'
            header += 'ifmap_dram_repeat,filter_dram_repeat,ofmap_dram_repeatn,\n'
            repeat_report.write(header)
        
        else:
            repeat_report = open(repeat_report_name, 'a')
        
        repeat_report.write(str(layer_id)+',')
        repeat_report.write(str(self.ifmap_sram_repeat)+',')
        repeat_report.write(str(self.filter_sram_repeat)+',')
        repeat_report.write(str(self.ofmap_sram_repeat)+',')
        repeat_report.write(str(self.ifmap_dram_repeat)+',')
        repeat_report.write(str(self.filter_dram_repeat)+',')
        repeat_report.write(str(self.ofmap_dram_repeat)+',\n')
        repeat_report.close()



    # Function to reduce redundancy
    def repeat_check_each_trace(self, previous_cycle, current_cycle, row_size):

        unique_elements_1 = np.unique(previous_cycle // row_size)
        unique_elements_2, unique_count_same_cycle = np.unique(current_cycle // row_size, return_counts = True) # output: 1D

        count_same_cycle = np.sum((unique_count_same_cycle - 1) * np.all([unique_count_same_cycle >=2, unique_elements_2 >= 0], axis=0))

        unique_element, unique_count = np.unique(np.concatenate((unique_elements_1, unique_elements_2), axis=0), return_counts=True)

        total_repeat = count_same_cycle + np.sum(np.all([unique_element >= 0, unique_count >= 2], axis=0))

        return total_repeat 


    # Function for DRAM
    def dram_repeat_check(self, bank_size, memory, row_size = 1):
        if not memory.trace_valid:
            print('No trace has been generated yet')
            return

        trace_repeat_count = 0

        for index in range(1, memory.trace_matrix.shape[0]):

            previous_cycle = memory.trace_matrix[max(index-bank_size, 0):index, 1:] - (np.abs(memory.trace_matrix[max(index-bank_size, 0):index, 1:]) == 1) * 2 
            current_cycle = memory.trace_matrix[index, 1:] - (np.abs(memory.trace_matrix[index, 1:]) == 1) * 2

            trace_repeat_count += self.repeat_check_each_trace(previous_cycle, current_cycle, row_size)

        return trace_repeat_count

    def sram_repeat_check(self,  bank_size, memory, row_size = 1):
        #if not memory.trace_valid:
        #    print('No trace has been generated yet')
        #    return

        trace_repeat_count = 0

        for index in range(1, memory.shape[0]):

            previous_cycle = memory[max(index-bank_size, 0):index, 1:] - (np.abs(memory[max(index-bank_size, 0):index, 1:]) == 1) * 2 
            current_cycle = memory[index, 1:] - (np.abs(memory[index, 1:]) == 1) * 2

            trace_repeat_count += self.repeat_check_each_trace(previous_cycle, current_cycle, row_size)

        return trace_repeat_count
