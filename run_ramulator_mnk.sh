#!/bin/bash
topology="topologies/ispass25_models/${1}.csv"
python3 scalesim/scale.py -c ./configs/google.cfg -t ${topology} -i gemm > ${1}_${2}_orig_out
python3 scripts/dram_sim.py -run_name GoogleTPU_v1_os
python3 scripts/dram_latency.py
python3 scalesim/scale.py -c ./configs/google_ramulator.cfg -t ${topology} -i gemm > ${1}_${2}_stall_out
cp ${1}_${2}_orig_out ./results/dram_results/stall_cycles/${1}_${2}_orig_out
cp ${1}_${2}_stall_out ./results/dram_results/stall_cycles/${1}_${2}_stall_out
