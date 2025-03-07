# Credit to @sbobyn
import argparse
import random
import os

from mpm88 import MPM88

import taichi as ti
import numpy as np
from tqdm import tqdm

ti.init(arch=ti.cpu)

parser = argparse.ArgumentParser()
parser.add_argument("--n_train", type=int, default=1000)
parser.add_argument("--n_valid", type=int, default=30)
parser.add_argument("--n_test", type=int, default=30)
parser.add_argument("--n_steps", type=int, default=1001)
parser.add_argument("--n_substeps", type=int, default=50)
parser.add_argument("--datapath", type=str, default="datasets/mpm88")

args = parser.parse_args()

# generate dataset

all_n_particles = []
all_simulations = []

for i in range(args.n_train + args.n_valid + args.n_test):
    n_particles = random.randint(200, 1200)
    all_n_particles.append(n_particles)
    x_history = np.zeros((args.n_steps, n_particles, 2), dtype=np.float32)
    all_simulations.append(x_history)

for i, n_particles in enumerate(all_n_particles):
    mpm = MPM88(n_particles)
    print(
        f"Generating simulation {i}/{len(all_n_particles)} with {n_particles} particles"
    )
    for s in tqdm(range(args.n_steps)):
        for substep in range(args.n_substeps):
            mpm.substep()
        # TODO: Fix because only works for 1001 case and water
        all_simulations[i][s] = mpm.x.to_numpy()

os.makedirs(args.datapath, exist_ok=True)

for split, n in [
    ("train", args.n_train),
    ("valid", args.n_valid),
    ("test", args.n_test),
]:
    np.savez(
        f"{args.datapath}/positions_{split}.npz",
        *all_simulations[:n],
    )

print("Done!")


