import argparse
import os
import yaml
from collections import OrderedDict

def build_PE_Array(ArrayHeight, ArrayWidth):
    acclg_pe = {}
    num_pes = int(ArrayHeight) * int(ArrayWidth)
    acclg_pe['name'] = 'PE[0..'+str(num_pes-1)+']'
    acclg_pe['attributes'] = {'memory_width': 16}
    
    alu = {'name':'mac', 'class':'custom_intmac','attributes':{'datawidth':8}}
    ifm_rf_attr = {'depth':1,'n_rdwr_ports':1,'width':1}
    ifm_spad  = {'name':'ifmap_spad', 'class':'custom_regfile','attributes':ifm_rf_attr}
    wgt_rf_attr = {'depth':1,'n_rdwr_ports':1,'width':1}
    wgt_spad = {'name':'weights_spad', 'class':'custom_regfile','attributes':wgt_rf_attr}
    psum_rf_attr = {'depth':1,'n_rdwr_ports':1,'width':1}
    psum_spad = {'name':'psum_spad', 'class':'custom_regfile','attributes':psum_rf_attr}

    acclg_pe['local'] = [alu, ifm_spad, wgt_spad, psum_spad]
    
    return acclg_pe


def build_GLB(NameGLB, SramSz, MemoryBanks):
    MemoryBanks = int(MemoryBanks)
    SramSz = int(SramSz) * 1024
    acclg_glb = {}
    acclg_glb['name'] = NameGLB
    acclg_glb['class'] = 'smartbuffer_SRAM'
    acclg_glb['attributes'] = {'memory_width':32, 'n_banks':MemoryBanks, 'bank_depth':SramSz ,'n_buffets':1,'memory_depth':'bank_depth * n_banks'}

    return acclg_glb


def build_DRAM(NameDRAM):
    acclg_dram = {}
    acclg_dram['name'] = NameDRAM
    acclg_dram['class'] = 'custom_DRAM'
    acclg_dram['attributes'] = {'width':32}

    return acclg_dram


def build_module(name, attributes=None, local=None, subtree=None):
    acclg_module = {}
    acclg_module['name'] = name

    if attributes is not None:
        acclg_module['attributes'] = attributes
    acclg_module['local'] = local

    if attributes is not None:
        acclg_module['subtree'] = subtree


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', metavar='SCALE-sim Config file', type=str,
                        help="Path to the systolic array config file")
    parser.add_argument('-t', metavar='SCALE-Sim Topology file', type=str,
                        help="Path to the topology file")
    parser.add_argument('-p', metavar='SCALE-Sim log dir', type=str,
                        help="Path to SCALE-Sim log dir")
    parser.add_argument('-o', metavar='final output dir', type=str,
                        default="./results",
                        help="Path to output dir")

    args = parser.parse_args()
    config = args.c 
    topology = args.t
    scsim_log = args.p
    outdir = args.o

    # extract from scsim.cfg
    scsim = {}
    with open(config, 'r') as fr:
        for line in fr:
            if 'run_name' in line:
                key = line.split("=")[0].strip()
                val = line.split("=")[1].strip()
                scsim[key] = val
            elif ':' in line:
                key = line.split(':')[0]
                val = line.split(':')[1].strip()
                scsim[key] = val

    # build acclg
    acclg = {}
    acclg['version'] = 0.3

    # build PEs
    PEs = build_PE_Array(scsim['ArrayHeight'], scsim['ArrayWidth'])

    # build GLBs
    GLBs = [build_GLB('weights_glb', scsim['FilterSramSzkB'], scsim['MemoryBanks']), 
            build_GLB('ifmap_glb', scsim['IfmapSramSzkB'], scsim['MemoryBanks']), 
            build_GLB('psum_glb', scsim['OfmapSramSzkB'], scsim['MemoryBanks'])]

    # build DRAMs
    DRAMs = [build_DRAM('weights_dram'), build_DRAM('ifmap_dram'), build_DRAM('psum_dram')]

    subtree_0 = {}
    subtree_0['name'] = 'systolic_array'
    subtree_0['attributes'] = {'technology':'40nm'}
    subtree_0['local'] = GLBs + DRAMs
    subtree_0['subtree'] = [PEs]

    acclg['subtree'] = [subtree_0]

    # wrap-up 
    acclg = {'architecture':acclg}

    # dump to acclg.yaml
    with open('accelergy_input/architecture.yaml', 'w') as fw:
        data = yaml.safe_dump(acclg, fw, sort_keys=False)

    # prepare command for action count extraction
    scsim_rundir = '../scalesim'
    with open('create_action_count.sh', 'w') as fw:
        saved_folder = scsim_log
        run_name = scsim['run_name']
        arch_name = 'systolic_array'
        SRAM_row_size = scsim['SRAM_row_size']
        DRAM_row_size = scsim['DRAM_row_size']

        fw.writelines('#!/bin/bash'+'\n\n')
        command = "python3 create_action_count.py --saved_folder " + saved_folder + " --run_name " + run_name + " --arch_name " + arch_name \
                + " --SRAM_row_size " + SRAM_row_size + " --DRAM_row_size " + DRAM_row_size + " --config " + config
        fw.writelines(command+'\n\n')
        
        command = "cp " + os.path.join('..',saved_folder,run_name,'action_count.yaml') + " ./accelergy_input/action_count.yaml"
        fw.writelines(command+'\n\n')
 
        scale_sim_output_dest =  os.path.join(outdir,"scale_sim_output_"+run_name)

        if os.path.isdir(scale_sim_output_dest):
            command = "rm -rf " + scale_sim_output_dest
            fw.writelines(command+'\n\n')

        command = "mv " + os.path.join('..',saved_folder,run_name) + "  " + scale_sim_output_dest
        fw.writelines(command+'\n\n')

    os.system("chmod 777 create_action_count.sh")

    # prepare command for running accelergy
    with open('./run_accelergy.sh', 'w') as fw:
        command = "accelergy accelergy_input/*.yaml accelergy_input/components/*.yaml -o accelergy_output/"+scsim['run_name']+" -v 1"
        fw.writelines(command+'\n\n')
        
        accelergy_output_dest = os.path.join(outdir, "accelergy_output_"+run_name)
        
        if os.path.isdir(accelergy_output_dest):
            command = "rm -rf " + accelergy_output_dest
            fw.writelines(command+'\n\n')
        
        command = "mv ./accelergy_output/"+run_name + "  " + os.path.join(outdir, "accelergy_output_"+run_name)
        fw.writelines(command+'\n\n')

        # command = "rm -rf ./accelergy_output" 
        # fw.writelines(command+'\n\n')

    os.system("chmod 777 ./run_accelergy.sh")
