#!/bin/bash

#SBATCH -J sandramps# Job name
#SBATCH -o sandramps.o%j     # Name of stdout output file
#SBATCH -e sandramps.e%j     # Name of stderr error file
#SBATCH --gres=gpu:1
#SBATCH -N 1                     # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                 # Total # of mpi tasks (should be 1 for serial)
#SBATCH --mem=16G
#SBATCH -t 12:00:00          # Run time (hh:mm:ss)
#SBATCH --mail-user=<stevenbobyn@gmail.com>
#SBATCH --mail-type=ALL

# fail on error
set -e

# start in slurm_scripts
cd ../..

dataset="SandRamps"

module purge
module load python
source venv/bin/activate
python train.py \
    --data-path ${SCRATCH}/224w-gns/datasets/${dataset} \
    --output ${SCRATCH}/224w-gns/datasets/${dataset} \
    --epoch 10 \
    --eval-interval 100000 \
    --vis-interval 100000 \
    --save-interval 100000
