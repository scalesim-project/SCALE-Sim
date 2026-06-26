import pandas as pd
import numpy as np
import os, sys
import yaml
from yaml import dump
from collections import OrderedDict
import argparse
import configparser as cp
import copy

# Load data from SCALE-Sim summary files
def load_detail_report_data(data_dir='.', run_name=''):
    csv_filename = data_dir + '/' + run_name + '/DETAILED_ACCESS_REPORT.csv'
    bandwidths_df = pd.read_csv(csv_filename, sep=r'\s*,\s*', engine='python')
    return bandwidths_df

def load_repeat_report_data(data_dir='.', run_name=''):
    csv_filename = data_dir + '/' + run_name + '/REPEAT_CYCLE.csv'
    repeat_df = pd.read_csv(csv_filename, sep=r'\s*,\s*', engine='python')
    return repeat_df

def load_compute_report_data(data_dir='.', run_name=''):
    csv_filename = data_dir + '/' + run_name + '/COMPUTE_REPORT.csv'
    compute_df = pd.read_csv(csv_filename, sep=r'\s*,\s*', engine='python')
    return compute_df

# Write final yaml file
def write_yaml_file(filepath, content):
    """
    if file exists at filepath, overwite the file, if not, create a new file
    :param filepath: string that specifies the destination file path
    :param content: yaml string that needs to be written to the destination file
    :return: None
    """
    if os.path.exists(filepath):
        os.remove(filepath)
    create_folder(os.path.dirname(filepath))
    out_file = open(filepath, 'a')
    out_file.write(dump( content, default_flow_style= False))

def create_folder(directory):
    """
    Checks the existence of a directory, if does not exist, create a new one
    :param directory: path to directory under concern
    :return: None
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('ERROR: Creating directory. ' +  directory)
        sys.exit()    

# Generate yaml block for SRAM and PE
def yaml_name_generator(name, sram, address_delta, data_delta, counts):

    if name == 'idle':
        action_count = {
                'name': name,
                'counts': counts
            }    
        
    else:
        action_count = {
                'name': name,
                'arguments':
                {
                    'address_delta' : address_delta,
                    'data_delta' : data_delta
                },
                'counts': counts
            }    
        
    return action_count

# Generate yaml block for DRAM
def yaml_name_generator_dram(name, counts):
    action_count = {
            'name': name,
            'counts': counts
    }    
    return action_count    

def yaml_name_generator_mac(name, counts):
    action_count = {
        'name': name,
        'counts': counts
    }
    return action_count

def yaml_generator(name, contents):
    action_count = {
        'name' : name,
        'action_counts' : contents
        }
    
    
    return action_count

#PARAMETER SETTINGS
parser = argparse.ArgumentParser()
parser.add_argument("--saved_folder", help='location of the run file')
parser.add_argument("--run_name", help='name of the scalesim run(final directory)')
parser.add_argument("--arch_name", help='name of the architecture which will be used in yaml file')
parser.add_argument("--SRAM_row_size", help='SRAM_row_size')
parser.add_argument("--DRAM_row_size", help='DRAM_row_size')
parser.add_argument("--config", help='config file under configs/')
parser.add_argument("--SRAM_repeat_check", help='if it is true, it is filtering out the repeat case', default='True')

args = parser.parse_args()

################### 
saved_folder  = args.saved_folder
run_name = args.run_name
arch_name = args.arch_name
SRAM_row_size = int(args.SRAM_row_size)
DRAM_row_size = int(args.DRAM_row_size)
config = args.config
SRAM_repeat_check = args.SRAM_repeat_check == 'True'
###################


#################################
# Create action counts for SRAM #
#################################
detail_access = load_detail_report_data(os.path.join(os.path.join(os.getcwd(), os.pardir), saved_folder), run_name)
repeat_access = load_repeat_report_data(os.path.join(os.path.join(os.getcwd(), os.pardir), saved_folder), run_name)

#Collapse by row
detail_access = np.sum(detail_access.to_numpy()[:,0:-1], axis=0)
repeat_access = np.sum(repeat_access.to_numpy()[:,0:-1], axis=0)

#Column index: Detailed access
layerID = 0
SRAM_ifmap_start_cycle = 1
SRAM_ifmap_stop_cycle = 2
SRAM_ifmap_reads = 3
SRAM_filter_start_cycle = 4
SRAM_filter_stop_cycle = 5
SRAM_filter_reads = 6
SRAM_ofmap_start_cycle = 7
SRAM_ofmap_stop_cycle = 8
SRAM_ofmap_writes = 9

DRAM_ifmap_start_cycle = 10
DRAM_ifmap_stop_cycle = 11
DRAM_ifmap_reads = 12
DRAM_filter_start_cycle = 13
DRAM_filter_stop_cycle = 14
DRAM_filter_reads = 15
DRAM_ofmap_start_cycle = 16
DRAM_ofmap_stop_cycle = 17
DRAM_ofmap_writes = 18

IFMAP_Write_Count = 19
IFMAP_Read_Count = 20
Filter_Write_Count = 21
Filter_Read_Count = 22
OFMAP_Write_Count = 23
OFMAP_Read_Count = 24

#Column index: Repeated access
layerID = 0
ifmap_sram_repeat = 1
filter_sram_repeat = 2
ofmap_sram_repeat = 3
ifmap_dram_repeat = 4
filter_dram_repeat = 5
ofmap_dram_repeat = 6

# get PE array size from configs/xx.cfg
config_filename = os.path.join(os.path.join(os.getcwd(), os.pardir), config)
config = cp.ConfigParser()
config.read(config_filename)
arrayheight = int(config.get('architecture_presets', 'ArrayHeight'))
arraywidth = int(config.get('architecture_presets', 'ArrayWidth'))
dataflow = config.get('architecture_presets', 'Dataflow')
PE_array_size = arrayheight * arraywidth

if(SRAM_repeat_check):
    SRAM_ifmap_idle = int((detail_access[SRAM_ifmap_stop_cycle]-detail_access[SRAM_ifmap_start_cycle]+1) * arrayheight - detail_access[SRAM_ifmap_reads])
    SRAM_ifmap_random = int(detail_access[SRAM_ifmap_reads] - repeat_access[ifmap_sram_repeat])
    SRAM_ifmap_repeat = int(repeat_access[ifmap_sram_repeat])

    SRAM_filter_idle = int((detail_access[SRAM_filter_stop_cycle]-detail_access[SRAM_filter_start_cycle]+1) * arraywidth - detail_access[SRAM_filter_reads])
    SRAM_filter_random = int(detail_access[SRAM_filter_reads] - repeat_access[filter_sram_repeat])
    SRAM_filter_repeat = int(repeat_access[filter_sram_repeat])

    SRAM_ofmap_idle = int((detail_access[SRAM_ofmap_stop_cycle]-detail_access[SRAM_ofmap_start_cycle]+1) * arraywidth - detail_access[SRAM_ofmap_writes])
    SRAM_ofmap_random = int(detail_access[SRAM_ofmap_writes] - repeat_access[ofmap_sram_repeat])
    SRAM_ofmap_repeat = int(repeat_access[ofmap_sram_repeat])
else:
    SRAM_ifmap_idle = int((detail_access[SRAM_ifmap_stop_cycle]-detail_access[SRAM_ifmap_start_cycle]+1) * arrayheight - detail_access[SRAM_ifmap_reads])
    SRAM_ifmap_random = int(detail_access[SRAM_ifmap_reads])
    SRAM_ifmap_repeat = 0

    SRAM_filter_idle = int((detail_access[SRAM_filter_stop_cycle]-detail_access[SRAM_filter_start_cycle]+1) * arraywidth - detail_access[SRAM_filter_reads])
    SRAM_filter_random = int(detail_access[SRAM_filter_reads])
    SRAM_filter_repeat = 0

    SRAM_ofmap_idle = int((detail_access[SRAM_ofmap_stop_cycle]-detail_access[SRAM_ofmap_start_cycle]+1) * arraywidth - detail_access[SRAM_ofmap_writes])
    SRAM_ofmap_random = int(detail_access[SRAM_ofmap_writes])
    SRAM_ofmap_repeat = 0

#Calculate the results (DRAM)
DRAM_ifmap_idle = int((detail_access[DRAM_ifmap_stop_cycle]-detail_access[DRAM_ifmap_start_cycle]+1) * DRAM_row_size - detail_access[DRAM_ifmap_reads])
DRAM_ifmap_random = int(detail_access[DRAM_ifmap_reads])

DRAM_filter_idle = int((detail_access[DRAM_filter_stop_cycle]-detail_access[DRAM_filter_start_cycle]+1) * DRAM_row_size - detail_access[DRAM_filter_reads])
DRAM_filter_random = int(detail_access[DRAM_filter_reads])

DRAM_ofmap_idle = int((detail_access[DRAM_ofmap_stop_cycle]-detail_access[DRAM_ofmap_start_cycle]+1) * DRAM_row_size - detail_access[DRAM_ofmap_writes])
DRAM_ofmap_random = int(detail_access[DRAM_ofmap_writes])


###############################
# Create action counts for PE #
###############################

# get MAC action counts
compute_access = load_compute_report_data(os.path.join(os.path.join(os.getcwd(), os.pardir), saved_folder), run_name)
layer_name = compute_access['LayerID'].values.tolist()
layer_utilization = (compute_access['Compute Util %']/100).values.tolist()
layer_unutilization = (1-compute_access['Compute Util %']/100).values.tolist()
layer_cycle = compute_access['Total Cycles'].values.tolist()

PE_MAC_random = sum(layer_cycle)

PE_weights_spad_write = int(detail_access[Filter_Write_Count])/PE_array_size
PE_weights_spad_read = int(detail_access[Filter_Read_Count])/PE_array_size
PE_ifmap_spad_write = int(detail_access[IFMAP_Write_Count])/PE_array_size
PE_ifmap_spad_read = int(detail_access[IFMAP_Read_Count])/PE_array_size
PE_psum_spad_write = int(detail_access[OFMAP_Write_Count])/PE_array_size
PE_psum_spad_read = int(detail_access[OFMAP_Read_Count])/PE_array_size

# Note: if use power/clock gating for MAC
# PE_MAC_random = int(sum([x * y for x,y in zip(layer_cycle,layer_utilization)]))
# PE_MAC_gated = int(sum([x * y for x,y in zip(layer_cycle,layer_unutilization)]))


###################################
# Create action counts yaml block #
###################################

action_counts = []

DRAM_ifmap = []
DRAM_ifmap.append(yaml_name_generator_dram('read', DRAM_ifmap_random))
DRAM_ifmap.append(yaml_name_generator_dram('idle', DRAM_ifmap_idle))
action_counts.append(yaml_generator(arch_name+'.ifmap_dram', DRAM_ifmap))

DRAM_filter = []
DRAM_filter.append(yaml_name_generator_dram('read', DRAM_filter_random))
DRAM_filter.append(yaml_name_generator_dram('idle', DRAM_filter_idle))
action_counts.append(yaml_generator(arch_name+'.weights_dram', DRAM_filter))

DRAM_ofmap = []
DRAM_ofmap.append(yaml_name_generator_dram('write', DRAM_ofmap_random))
DRAM_ofmap.append(yaml_name_generator_dram('idle', DRAM_ofmap_idle))
action_counts.append(yaml_generator(arch_name+'.psum_dram', DRAM_ofmap))


SRAM_ifmap = []
SRAM_ifmap.append(yaml_name_generator('read', 1, 1, 1, SRAM_ifmap_random))
SRAM_ifmap.append(yaml_name_generator('read', 1, 0, 0, SRAM_ifmap_repeat))
SRAM_ifmap.append(yaml_name_generator('idle', 1, 0, 0, SRAM_ifmap_idle))
action_counts.append(yaml_generator(arch_name+'.ifmap_glb', SRAM_ifmap))

SRAM_filter = []
SRAM_filter.append(yaml_name_generator('read', 1, 1, 1, SRAM_filter_random))
SRAM_filter.append(yaml_name_generator('read', 1, 0, 0, SRAM_filter_repeat))
SRAM_filter.append(yaml_name_generator('idle', 1, 0, 0, SRAM_filter_idle))
action_counts.append(yaml_generator(arch_name+'.weights_glb', SRAM_filter))

SRAM_ofmap = []
SRAM_ofmap.append(yaml_name_generator('update', 1, 1, 1, SRAM_ofmap_random))
SRAM_ofmap.append(yaml_name_generator('update', 1, 0, 0, SRAM_ofmap_repeat))
SRAM_ofmap.append(yaml_name_generator('idle', 1, 0, 0, SRAM_ofmap_idle))
action_counts.append(yaml_generator(arch_name+'.psum_glb', SRAM_ofmap))

PE_weight = []
PE_weight.append(yaml_name_generator('write', 1, 1, 1, PE_weights_spad_write))
PE_weight.append(yaml_name_generator('read', 1, 0, 0, PE_weights_spad_read))
for n in range(PE_array_size):
    action_counts.append(yaml_generator(arch_name+'.PE['+str(n)+'].weights_spad', copy.deepcopy(PE_weight)))

PE_ifmap = []
PE_ifmap.append(yaml_name_generator('write', 1, 1, 1, PE_ifmap_spad_write))
PE_ifmap.append(yaml_name_generator('read', 1, 0, 0, PE_ifmap_spad_read))
for n in range(PE_array_size):
    action_counts.append(yaml_generator(arch_name+'.PE['+str(n)+'].ifmap_spad', copy.deepcopy(PE_ifmap)))

if dataflow == 'os':
    PE_psum = []
    PE_psum.append(yaml_name_generator('write', 1, 0, 0, PE_psum_spad_write))
    PE_psum.append(yaml_name_generator('read', 1, 0, 0, PE_psum_spad_read))
    for n in range(PE_array_size):
        action_counts.append(yaml_generator(arch_name+'.PE['+str(n)+'].psum_spad', copy.deepcopy(PE_psum)))
else:
    PE_psum = []
    PE_psum.append(yaml_name_generator('write', 1, 1, 1, PE_psum_spad_write))
    PE_psum.append(yaml_name_generator('read', 1, 0, 0, PE_psum_spad_read))
    for n in range(PE_array_size):
        action_counts.append(yaml_generator(arch_name+'.PE['+str(n)+'].psum_spad', copy.deepcopy(PE_psum)))

PE_MAC = []
# PE_MAC.append(yaml_name_generator_mac('mac_gated', PE_MAC_gated))
PE_MAC.append(yaml_name_generator_mac('mac_random', PE_MAC_random))
for n in range(PE_array_size):
    action_counts.append(yaml_generator(arch_name+'.PE['+str(n)+'].mac', copy.deepcopy(PE_MAC)))

action_counts_header = {'action_counts': {'version': 0.3,
                                                                              'local': action_counts}}

write_yaml_file(os.path.join(os.path.join(os.path.join(os.getcwd(), os.pardir), saved_folder, run_name),'action_count.yaml'), action_counts_header)                                                                        
