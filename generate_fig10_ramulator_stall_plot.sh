#!/bin/bash

path="./"
mkdir results
mkdir results/dram_results
mkdir results/dram_results/stall_cycles
echo "Running scalesimV3 for Alexnet with buffer size 512"
./run_ramulator.sh alexnet 512
echo "Running scalesimV3 for Resnet18 with buffer size 512"
./run_ramulator.sh resnet18 512
echo "Running scalesimV3 for vit_s with buffer size 512"
./run_ramulator_mnk.sh vit_s 512
echo "Running scalesimV3 for vit_b with buffer size 512"
./run_ramulator_mnk.sh vit_b 512
echo "Running scalesimV3 for vit_bg with buffer size 512"
./run_ramulator_mnk.sh vit_bg 512
echo "Running scalesimV3 for vit_h with buffer size 512"
./run_ramulator_mnk.sh vit_h 512
echo "Running scalesimV3 for vit_l with buffer size 512"
./run_ramulator_mnk.sh vit_l 512
echo "Complete run with the 512 buffer size"

sed -i "s/ReadRequestBuffer: 512/ReadRequestBuffer: 128/g" $path/configs/google.cfg
sed -i "s/WriteRequestBuffer: 512/WriteRequestBuffer: 128/g" $path/configs/google.cfg
sed -i "s/ReadRequestBuffer: 512/ReadRequestBuffer: 128/g" $path/configs/google_ramulator.cfg
sed -i "s/WriteRequestBuffer: 512/WriteRequestBuffer: 128/g" $path/configs/google_ramulator.cfg
echo "Running scalesimV3 for Alexnet with buffer size 128"
./run_ramulator.sh alexnet 128
echo "Running scalesimV3 for Resnet18 with buffer size 128"
./run_ramulator.sh resnet18 128
echo "Running scalesimV3 for vit_s with buffer size 128"
./run_ramulator_mnk.sh vit_s 128
echo "Running scalesimV3 for vit_b with buffer size 128"
./run_ramulator_mnk.sh vit_b 128
echo "Running scalesimV3 for vit_bg with buffer size 128"
./run_ramulator_mnk.sh vit_bg 128
echo "Running scalesimV3 for vit_h with buffer size 128"
./run_ramulator_mnk.sh vit_h 128
echo "Running scalesimV3 for vit_l with buffer size 128"
./run_ramulator_mnk.sh vit_l 128
echo "Complete run with the 128 buffer size"

sed -i "s/ReadRequestBuffer: 128/ReadRequestBuffer: 32/g" $path/configs/google.cfg
sed -i "s/WriteRequestBuffer: 128/WriteRequestBuffer: 32/g" $path/configs/google.cfg
sed -i "s/ReadRequestBuffer: 128/ReadRequestBuffer: 32/g" $path/configs/google_ramulator.cfg
sed -i "s/WriteRequestBuffer: 128/WriteRequestBuffer: 32/g" $path/configs/google_ramulator.cfg
echo "Running scalesimV3 for Alexnet with buffer size 32"
./run_ramulator.sh alexnet 32
echo "Running scalesimV3 for Resnet18 with buffer size 32"
./run_ramulator.sh resnet18 32
echo "Running scalesimV3 for vit_s with buffer size 32"
./run_ramulator_mnk.sh vit_s 32
echo "Running scalesimV3 for vit_b with buffer size 32"
./run_ramulator_mnk.sh vit_b 32
echo "Running scalesimV3 for vit_bg with buffer size 32"
./run_ramulator_mnk.sh vit_bg 32
echo "Running scalesimV3 for vit_h with buffer size 32"
./run_ramulator_mnk.sh vit_h 32
echo "Running scalesimV3 for vit_l with buffer size 32"
./run_ramulator_mnk.sh vit_l 32
echo "Complete run with the 128 buffer size"
git checkout configs/google.cfg
git checkout configs/google_ramulator.cfg

echo "Plotting for Figure 10 stall results"
mkdir Exp2
rm -rf Exp2/*
mv ./results/dram_results/stall_cycles Exp2/
python scripts/plots/stall_plot.py
echo "Figure 10 plot generation successful!!!"

