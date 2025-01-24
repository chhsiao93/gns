#!/bin/bash

#SBATCH -J s2d_fix         # Job name
#SBATCH -o s2d_fix.o%j     # Name of stdout output file
#SBATCH -e s2d_fix.e%j     # Name of stderr error file
#SBATCH -p gpu-a100-small              # Queue (partition) name
#SBATCH -N 1                     # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                 # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 12:00:00          # Run time (hh:mm:ss)
#SBATCH --mail-type=all      # Send email at begin and end of job
#SBATCH --mail-user=chhsiao@utexas.edu
#SBATCH -A BCS20003          # Project/Allocation name (req'd if you have more than 1)

# fail on error
set -e

# start in slurm_scripts
cd ..
source start_venv.sh

# assume data is already downloaded and hardcode WaterDropSample
data="SandSinglePhi"
python3 -m gns.train --data_path="${SCRATCH}/gns/${data}/dataset/" \
--model_path="${SCRATCH}/gns/${data}/fix_gns/" \
--output_path="${SCRATCH}/gns/${data}/rollouts/" \
--nsave_steps=10000 \
--cuda_device_number=0 \
--ntraining_steps=1000000 \
--model_file="latest" \
--train_state_file="latest"
