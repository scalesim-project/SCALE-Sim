#!/bin/bash

# Run the commands sequentially
echo "Running run_all.sh for OS..."
./run_all.sh -c ../configs/ispass_ae_run_128x128_os.cfg -t ../topologies/vit_s.csv -p ../test_runs/ -o ./output

echo "Running run_all.sh for WS..."
./run_all.sh -c ../configs/ispass_ae_run_128x128_ws.cfg -t ../topologies/vit_s.csv -p ../test_runs/ -o ./output

echo "Running run_all.sh for IS..."
./run_all.sh -c ../configs/ispass_ae_run_128x128_is.cfg -t ../topologies/vit_s.csv -p ../test_runs/ -o ./output

# Run the Python script for result extraction
echo "Extracting results..."
python output/result_extraction.py

echo "All tasks completed!"
