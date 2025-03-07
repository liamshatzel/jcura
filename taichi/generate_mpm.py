import argparse
import random
import os

from mpm88 import MPM88
import taichi as ti
import numpy as np
from tqdm import tqdm

ti.init(arch=ti.gpu)

parser = argparse.ArgumentParser()
parser.add_argument("--n_train", type=int, default=1000)
parser.add_argument("--n_valid", type=int, default=30)
parser.add_argument("--n_test", type=int, default=30)
parser.add_argument("--n_steps", type=int, default=1001)
parser.add_argument("--n_substeps", type=int, default=50)
parser.add_argument("--datapath", type=str, default="datasets/mpm88")

args = parser.parse_args()

os.makedirs(args.datapath, exist_ok=True)

for split, n in [
    ("train", args.n_train),
    ("valid", args.n_valid),
    ("test", args.n_test),
]:
    all_simulations = []  # Store tuples directly

    for i in range(n):
        n_particles = random.randint(200, 1200)
        x_history = np.zeros((args.n_steps, n_particles, 2), dtype=np.float32)
        constant_array = np.full(n_particles, 5, dtype=np.float32)  # Equivalent to 5 * np.ones(n_particles)

        mpm = MPM88(n_particles)
        print(f"Generating simulation {i}/{n} with {n_particles} particles")

        for s in tqdm(range(args.n_steps)):
            for substep in range(args.n_substeps):
                mpm.substep()
            x_history[s] = mpm.x.to_numpy()

        # Store the tuple in the list
        all_simulations.append((x_history, constant_array))

    # Save the list of tuples as an object array in .npz
    np.savez(f"{args.datapath}/{split}.npz", gns_data=np.array(all_simulations, dtype=object))

print("Done!")
