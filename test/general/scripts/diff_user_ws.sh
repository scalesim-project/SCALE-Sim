#!/bin/bash

path="./"

sed -i "2s/.*/run_name = scale_example_run_32x32_ws/" $path/configs/scale.cfg
sed -i "s/Dataflow : os/Dataflow : ws/g" $path/configs/scale.cfg
sed -i "s/Dataflow : is/Dataflow : ws/g" $path/configs/scale.cfg
sed -i 's/InterfaceBandwidth: CALC/InterfaceBandwidth: USER/g' $path/configs/scale.cfg
sed -i "s/save_disk_space=True/save_disk_space=False/" $path/scalesim/scale.py

source venv/bin/activate
export PYTHONPATH=.
python3 $path/scalesim/scale.py -c $path/configs/scale.cfg -t $path/topologies/GEMM_mnk/test_mnk_input.csv -i gemm -p $path/test_runs

DIFF1=$(diff $path/test_runs/scale_example_run_32x32_ws/BANDWIDTH_REPORT.csv $path/test/general/golden_trace_user_ws/BANDWIDTH_REPORT.csv)
DIFF2=$(diff $path/test_runs/scale_example_run_32x32_ws/COMPUTE_REPORT.csv $path/test/general/golden_trace_user_ws/COMPUTE_REPORT.csv)
DIFF3=$(diff $path/test_runs/scale_example_run_32x32_ws/DETAILED_ACCESS_REPORT.csv $path/test/general/golden_trace_user_ws/DETAILED_ACCESS_REPORT.csv)
DIFF4=$(diff $path/test_runs/scale_example_run_32x32_ws/layer0/FILTER_DRAM_TRACE.csv $path/test/general/golden_trace_user_ws/layer0/FILTER_DRAM_TRACE.csv)
DIFF5=$(diff $path/test_runs/scale_example_run_32x32_ws/layer0/FILTER_SRAM_TRACE.csv $path/test/general/golden_trace_user_ws/layer0/FILTER_SRAM_TRACE.csv)
DIFF6=$(diff $path/test_runs/scale_example_run_32x32_ws/layer0/IFMAP_DRAM_TRACE.csv $path/test/general/golden_trace_user_ws/layer0/IFMAP_DRAM_TRACE.csv)
DIFF7=$(diff $path/test_runs/scale_example_run_32x32_ws/layer0/IFMAP_SRAM_TRACE.csv $path/test/general/golden_trace_user_ws/layer0/IFMAP_SRAM_TRACE.csv)
DIFF8=$(diff $path/test_runs/scale_example_run_32x32_ws/layer0/OFMAP_DRAM_TRACE.csv $path/test/general/golden_trace_user_ws/layer0/OFMAP_DRAM_TRACE.csv)
DIFF9=$(diff $path/test_runs/scale_example_run_32x32_ws/layer0/OFMAP_SRAM_TRACE.csv $path/test/general/golden_trace_user_ws/layer0/OFMAP_SRAM_TRACE.csv)


if [ "$DIFF1" != "" ]; then
    echo "Output does not match!" 
    echo "$DIFF1"
    exit 1
elif [ "$DIFF2" != "" ]; then
    echo "Output does not match!" 
    echo "$DIFF2"
    exit 1
elif [ "$DIFF3" != "" ]; then
    echo "Output does not match!" 
    echo "$DIFF3"    
    exit 1
elif [ "$DIFF4" != "" ]; then
    echo "Output does not match!" 
    echo "$DIFF4"
    exit 1
elif [ "$DIFF5" != "" ]; then
    echo "Output does not match!"
    echo "$DIFF5" 
    exit 1
elif [ "$DIFF6" != "" ]; then
    echo "Output does not match!"
    echo "$DIFF6" 
    exit 1
elif [ "$DIFF7" != "" ]; then
    echo "Output does not match!" 
    echo "$DIFF7" 
    exit 1
elif [ "$DIFF8" != "" ]; then
    echo "Output does not match!" 
    echo "$DIFF8" 
    exit 1
elif [ "$DIFF9" != "" ]; then
    echo "Output does not match!" 
    echo "$DIFF9" 
    exit 1
fi