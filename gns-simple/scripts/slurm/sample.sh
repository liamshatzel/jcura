#!/bin/bash

#SBATCH -J sample-waterdrop  # Job name
#SBATCH -o sample-waterdrop.o%j     # Name of stdout output file
#SBATCH -e sample-waterdrop.e%j     # Name of stderr error file
#SBATCH --gres=gpu:1
#SBATCH -N 1                     # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                 # Total # of mpi tasks (should be 1 for serial)
#SBATCH --mem=8G
#SBATCH -t 01:00:00          # Run time (hh:mm:ss)
#SBATCH --mail-user=<stevenbobyn@gmail.com>
#SBATCH --mail-type=ALL

# fail on error
set -e

# start in slurm_scripts
cd ../..

dataset="WaterDropSample"

module purge
module load python
source venv/bin/activate
python train.py \
    --data-path ${SCRATCH}/224w-gns/datasets/${dataset} \
    --output ${SCRATCH}/224w-gns/datasets/${dataset} \
    --epoch 10 \
    --eval-interval 1000 \
    --vis-interval 1000 \
    --save-interval 1000
