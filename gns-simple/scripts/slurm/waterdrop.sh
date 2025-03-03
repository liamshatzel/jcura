#!/bin/bash
#SBATCH --account=def-brandonh
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=15:0:0
#SBATCH --mail-user=lshatzel@uvic.ca
#SBATCH --mail-type=ALL

# fail on error
set -e

# start in slurm_scripts
cd ../..

dataset="WaterDrop"

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
