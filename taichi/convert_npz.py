import os
import json
import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default="datasets/mpm88")
    args = parser.parse_args()

    for split in ["train", "valid", "test"]:
        ds = np.load(os.path.join(args.datapath, f"positions_{split}.npz"))

        data_shape = {}
        file_offset = {}
        i = 0
        for f in ds:
            sim = ds[f]
            data = {
                "particle_type": 5 * np.ones((sim.shape[1]), dtype=np.int64),
                "position": sim,
            }
            shape = {}
            for key, value in data.items():
                filename = os.path.join(args.datapath, split + "_" + key + ".dat")
                offset = file_offset.get(key, 0)
                if key == "particle_type":
                    assert value.dtype == np.int64, value.dtype
                else:
                    assert value.dtype == np.float32, value.dtype
                mode = "r+" if os.path.exists(filename) else "w+"
                array = np.memmap(
                    filename,
                    dtype=value.dtype,
                    mode=mode,
                    offset=offset * value.dtype.itemsize,
                    shape=value.shape,
                )
                array[:] = value
                shape[key] = {"offset": offset, "shape": value.shape}
                file_offset[key] = offset + value.size
            data_shape[i] = shape
            i += 1
        with open(os.path.join(args.datapath, split + "_offset.json"), "w") as f:
            json.dump(data_shape, f, indent=2)


if __name__ == "__main__":
    main()