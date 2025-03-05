#!/bin/bash
#SBATCH --account=def-brandonh
#SBATCH --mem=32G
#SBATCH --gres=gpu:p100:1
#SBATCH --ntasks-per-node=8
#SBATCH --time=10:0:0
#SBATCH --mail-user=lshatzel@uvic.ca
#SBATCH --mail-type=ALL

# fail on error
set -e

# start in slurm_scripts
# cd ../..

dataset="Sand"

module purge
module load python
source venv/bin/activate
# python train.py \
#     --data-path datasets/WaterDropSample \
#     --output datasets/WaterDropSample \ 
#     --epoch 1000 \
#     --eval-interval 1000 \
#     --vis-interval 1000 \
#     --save-interval 1000

python train.py --data-path datasets/${dataset} --output datasets/${dataset} --epoch 1000 --eval-interval 1000 --vis-interval 1000 --save-interval 1000
    # ${SCRATCH}/224w-gns/datasets/${dataset} \
    # ${SCRATCH}/224w-gns/datasets/${dataset} \
