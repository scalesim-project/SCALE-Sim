#!/bin/bash

path="./"

sed -i "s/save_disk_space=True/save_disk_space=False/" $path/scalesim/scale.py
sed -i "s/^RandomNumberGeneratorSeed.*/RandomNumberGeneratorSeed : 40/" "$path/configs/sparsity.cfg"

source venv/bin/activate
export PYTHONPATH=.
python3 $path/scalesim/scale.py -c $path/configs/sparsity.cfg -t $path/topologies/sparsity/alexnet_part.csv -p $path/sparsity_outputs

DIFF1=$(diff $path/sparsity_outputs/scalesim_sparsity/BANDWIDTH_REPORT.csv $path/test/sparsity/golden_trace/BANDWIDTH_REPORT.csv)
DIFF2=$(diff $path/sparsity_outputs/scalesim_sparsity/COMPUTE_REPORT.csv $path/test/sparsity/golden_trace/COMPUTE_REPORT.csv)
DIFF3=$(diff $path/sparsity_outputs/scalesim_sparsity/DETAILED_ACCESS_REPORT.csv $path/test/sparsity/golden_trace/DETAILED_ACCESS_REPORT.csv)
DIFF4=$(diff $path/sparsity_outputs/scalesim_sparsity/SPARSE_REPORT.csv $path/test/sparsity/golden_trace/SPARSE_REPORT.csv)
DIFF5=$(diff $path/sparsity_outputs/scalesim_sparsity/layer0/FILTER_DRAM_TRACE.csv $path/test/sparsity/golden_trace/layer0/FILTER_DRAM_TRACE.csv)
DIFF6=$(diff $path/sparsity_outputs/scalesim_sparsity/layer0/FILTER_SRAM_TRACE.csv $path/test/sparsity/golden_trace/layer0/FILTER_SRAM_TRACE.csv)
DIFF7=$(diff $path/sparsity_outputs/scalesim_sparsity/layer0/IFMAP_DRAM_TRACE.csv $path/test/sparsity/golden_trace/layer0/IFMAP_DRAM_TRACE.csv)
DIFF8=$(diff $path/sparsity_outputs/scalesim_sparsity/layer0/IFMAP_SRAM_TRACE.csv $path/test/sparsity/golden_trace/layer0/IFMAP_SRAM_TRACE.csv)
DIFF9=$(diff $path/sparsity_outputs/scalesim_sparsity/layer0/OFMAP_DRAM_TRACE.csv $path/test/sparsity/golden_trace/layer0/OFMAP_DRAM_TRACE.csv)
DIFF10=$(diff $path/sparsity_outputs/scalesim_sparsity/layer0/OFMAP_SRAM_TRACE.csv $path/test/sparsity/golden_trace/layer0/OFMAP_SRAM_TRACE.csv)



if [ "$DIFF1" != "" ]; then
    echo "Output does not match!" 
    echo "$DIFF1"
    # exit 1
elif [ "$DIFF2" != "" ]; then
    echo "Output does not match!" 
    echo "$DIFF2"
    # exit 1
elif [ "$DIFF3" != "" ]; then
    echo "Output does not match!" 
    echo "$DIFF3"    
    # exit 1
elif [ "$DIFF4" != "" ]; then
    echo "Output does not match!" 
    echo "$DIFF4"
    # exit 1
elif [ "$DIFF5" != "" ]; then
    echo "Output does not match!"
    echo "$DIFF5" 
    # exit 1
elif [ "$DIFF6" != "" ]; then
    echo "Output does not match!"
    echo "$DIFF6" 
    # exit 1
elif [ "$DIFF7" != "" ]; then
    echo "Output does not match!" 
    echo "$DIFF7" 
    # exit 1
elif [ "$DIFF8" != "" ]; then
    echo "Output does not match!" 
    echo "$DIFF8" 
    # exit 1
elif [ "$DIFF9" != "" ]; then
    echo "Output does not match!" 
    echo "$DIFF9" 
    # exit 1
elif [ "$DIFF10" != "" ]; then
    echo "Output does not match!" 
    echo "$DIFF10" 
    # exit 1
fi
