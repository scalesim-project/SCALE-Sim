#!/bin/bash

python3 create_action_count.py --saved_folder ../test_runs --run_name scale_example_run_128x128_ws --arch_name systolic_array --SRAM_row_size 2 --DRAM_row_size 2 --config ../configs/ispass_ae_run_128x128_ws.cfg

cp ../test_runs/scale_example_run_128x128_ws/action_count.yaml ./accelergy_input/action_count.yaml

mv ../test_runs/scale_example_run_128x128_ws  ./output/scale_sim_output_scale_example_run_128x128_ws

