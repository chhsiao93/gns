#!/bin/bash

#SBATCH -J train         # Job name
#SBATCH -o train.o%j     # Name of stdout output file
#SBATCH -e train.e%j     # Name of stderr error file
#SBATCH -p rtx-dev              # Queue (partition) name
#SBATCH -N 1                     # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                 # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 0:05:00          # Run time (hh:mm:ss)
#SBATCH --mail-type=all      # Send email at begin and end of job


# Train for a few steps.
NODEFILE=/tmp/hostfile
scontrol show hostnames  > $NODEFILE
NNODES=$(< $NODEFILE wc -l)

CONTAINER=$1
n_gpu_per_node=$2
mpiexec.hydra -np $NNODES -ppn 1 ../slurm_scripts/launch_helper.sh $CONTAINER $n_gpu_per_node