import argparse
import os
import json

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default="datasets/mpm88")
    parser.add_argument("--default_connectivity_radius", type=float, default=0.015)

    args = parser.parse_args()

    ds = np.load(os.path.join(args.datapath, f"positions_train.npz"))

    positions = []
    for f in ds:
        positions.append(ds[f])

    # compute bounds
    bounds = np.zeros((2, 2), dtype=np.float64)
    for pos in positions:
        # horizontal bounds
        bounds[0][0] = min(bounds[0][0], pos[:, 0].min())
        bounds[0][1] = max(bounds[0][1], pos[:, 0].max())
        # vertical bounds
        bounds[1][0] = min(bounds[1][0], pos[:, 1].min())
        bounds[1][1] = max(bounds[1][1], pos[:, 1].max())

    # compute velocities as finite difference
    velocities = []
    for i in range(1, len(positions)):
        velocities.append(positions[i][1:] - positions[i][:-1])

    # compute mean and std of velocities
    vel_mean = np.zeros((2), dtype=np.float64)
    vel_std = np.zeros((2), dtype=np.float64)
    for vel in velocities:
        vel_mean += vel.mean(axis=(0, 1))
        vel_std += vel.std(axis=(0, 1))
    vel_mean /= len(velocities)
    vel_std /= len(velocities)
    vel_mean, vel_std

    # compute accelerations as finite difference
    accelerations = []
    for i in range(1, len(velocities)):
        accelerations.append(velocities[i][1:] - velocities[i][:-1])

    # compute mean and std of accelerations
    acc_mean = np.zeros((2), dtype=np.float64)
    acc_std = np.zeros((2), dtype=np.float64)
    for acc in accelerations:
        acc_mean += acc.mean(axis=(0, 1))
        acc_std += acc.std(axis=(0, 1))
    acc_mean /= len(accelerations)
    acc_std /= len(accelerations)
    acc_mean, acc_std

    sequence_length = positions[0].shape[0]
    dim = positions[0].shape[2]

    metadata = {
        "bounds": bounds.tolist(),
        "vel_mean": vel_mean.tolist(),
        "vel_std": vel_std.tolist(),
        "acc_mean": acc_mean.tolist(),
        "acc_std": acc_std.tolist(),
        "dim": dim,
        "sequence_length": sequence_length,
        "default_connectivity_radius": args.default_connectivity_radius,
    }

    with open(os.path.join(args.datapath, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()