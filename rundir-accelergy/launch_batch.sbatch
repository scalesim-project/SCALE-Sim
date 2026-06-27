#!/bin/bash
#SBATCH --job-name=scalesim-run         # Job name
#SBATCH --output=job_%j.out      # Standard output (%j = job ID)
#SBATCH --error=job_%j.err       # Standard error
#SBATCH --time=7-00:00:00         # Walltime (hh:mm:ss)
#SBATCH --partition=batch_long.q     # Partition/queue
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks=1               # Number of tasks
#SBATCH --cpus-per-task=8        # CPUs per task
#SBATCH --mem=64G                # Total memory

## Load Python environment (choose one method)
module load lang/Python/3.10.8-GCCcore-12.2.0

## (2) If using virtualenv
source ../.venv/bin/activate

echo "Running inside Python environment: $(which python)"

# Run your bash script
./run_all.sh -c ../configs/sweep_study/64_os.cfg -t ../topologies/conv_nets/Resnet50_parsed.csv -p ../test_runs/ -o ./output
