#!/bin/bash

########### User Input ##########################

while getopts c:t:p:o:i: flag
do
    case "${flag}" in
        c) scsimCfg=${OPTARG};;
        t) scsimTplg=${OPTARG};;
        p) scsimOutput=${OPTARG};;        
        o) allOutput=${OPTARG};;
        i) topoOption=${OPTARG};;
    esac
done 

if [[ $scsimCfg == ""  ||  $scsimTplg == ""  || $allOutput == "" ]]; then
 echo "Not enough input files privoded"
 echo "./run_all.sh -c <path_to_config_file> -t <path_to_topology_file> -p <path_to_scale-sim_log_dir> -o <path_to_final_output_dir>"
 exit 0
fi

echo "config file: $scsimCfg";
echo "topology file: $scsimTplg";
echo "scsim log dir: $scsimOutput";
echo "output dir: $allOutput";
echo "topology option: $topoOption";

################################################

# Ensure log and output directories exist before resolving absolute paths
if [[ -n "$scsimOutput" ]]; then
    mkdir -p "$scsimOutput" || { echo "Failed to create scsim log dir at $scsimOutput"; exit 1; }
fi

if [[ -n "$allOutput" ]]; then
    mkdir -p "$allOutput" || { echo "Failed to create output dir at $allOutput"; exit 1; }
fi

scsimCfg=$(realpath "$scsimCfg")
scsimTplg=$(realpath "$scsimTplg")
scsimOutput=$(realpath "$scsimOutput")
allOutput=$(realpath "$allOutput")

rm -f accelergy_input/*.yaml

# Generate Accelergy::architecture.yaml from ScaleSim::scale.cfg
python3 preprocess.py -c $scsimCfg -t $scsimTplg -p $scsimOutput -o $allOutput

# Run Scale-sim
cd ..
python3 scale.py -c $scsimCfg -t $scsimTplg -p $scsimOutput -i $topoOption

# Extract Accelergy::action_count.yaml from ScaleSim::reuslts
cd rundir-accelergy
./create_action_count.sh


# Run Accelergy
./run_accelergy.sh


# Post-Process
# gen_plot.ipynb
