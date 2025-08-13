#!/bin/bash
# Baseline
python3 ./scalesim/scale.py -c configs/scale.cfg -t topologies/conv_nets/Resnet18.csv -p ./ >> log_bankconflict_resnet18
# python3 ./scalesim/scale.py -c configs/scale.cfg -t topologies/conv_nets/alexnet_full.csv -p ./ >> log_bankconflict_alexnet
python3 ./scalesim/scale.py -i gemm -c configs/scale.cfg -t topologies/GEMM_mnk/vit_s.csv -p ./ >> log_bankconflict_vit_s
