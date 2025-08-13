#!/bin/bash

git checkout configs/google.cfg
git checkout configs/google_ramulator.cfg
mkdir results
mkdir results/dram_results
mkdir results/dram_results/mem_bw
ramulator_path="./submodules/ramulator/configs"
outpath="./results/dram_results/mem_bw"
sed -i "s/channels = [0-9]\+/channels = 1/g" $ramulator_path/DDR4-config.cfg
echo "Running scalesimV3 with r1c1 configuration"
./run_ramulator.sh resnet18 512
mkdir $outpath/Resnet18_r1c1
mv results/DDR4* $outpath/Resnet18_r1c1 
sed -i "s/channels = 1/channels = 2/g" $ramulator_path/DDR4-config.cfg
echo "Running scalesimV3 with r1c2 configuration"
./run_ramulator.sh resnet18 512
mkdir $outpath/Resnet18_r1c2
mv results/DDR4* $outpath/Resnet18_r1c2
sed -i "s/channels = 2/channels = 4/g" $ramulator_path/DDR4-config.cfg
echo "Running scalesimV3 with r1c4 configuration"
./run_ramulator.sh resnet18 512
mkdir $outpath/Resnet18_r1c4
mv results/DDR4* $outpath/Resnet18_r1c4
sed -i "s/channels = 4/channels = 8/g" $ramulator_path/DDR4-config.cfg
echo "Running scalesimV3 with r1c8 configuration"
./run_ramulator.sh resnet18 512
mkdir $outpath/Resnet18_r1c8
mv results/DDR4* $outpath/Resnet18_r1c8

echo "Plotting for Figure 9 stall results"
mkdir Exp1
mv ./results/dram_results/mem_bw/* Exp1/
python scripts/plots/mem_bw_plot.py
echo "Figure 9 plot generation successful!!!"
