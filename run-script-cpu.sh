#!/bin/bash
#SBATCH --account=def-brandonh
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=5:0:0
#SBATCH --mail-user=lshatzel@uvic.ca
#SBATCH --mail-type=ALL
module purge
module load python/3.10 scipy-stack
source venv/bin/activate
python3 -m gns.train --data_path="datasets/WaterDropSample/" --ntraining_steps=1000000 --wandb_sweep=False --wandb_enable=True --force_cpu=True
# python3 -m gns.train --data_path="datasets/WaterDropSample/" --ntraining_steps=10000 --wandb_sweep=False --wandb_enable=False --batch_size=32 --lr_init=0.00001
