import os
import h5py
import numpy as np
from itertools import accumulate
from torch.utils.data import Dataset

_dir = "/data/1e404a5c-140b-4e30-af3a-ee453536e9d8/lc0"
boards_filename = _dir + "/boards.h5"
filename = _dir + "/evalHDF0"


class EVAL_HDF_Dataset(Dataset):
    def __init__(self, source_dir):
        super().__init__()
        self.source_dir = source_dir

        # Read inventory file
        with open(os.path.join(self.source_dir, "inventory.txt"), "r") as file:
            self.inventory = file.readlines()

        # Parse inventory
        sizes, self.filenames = zip(
            *[line.strip().split() for line in self.inventory[1:]]
        )
        self.sizes = [int(size) for size in sizes]
        self.len = sum(self.sizes)
        self.breaks = np.array(list(accumulate(self.sizes)))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # Find the corresponding HDF5 file
        hdf_idx = (self.breaks > idx).argmax().item()
        board_idx = idx - sum(self.sizes[:hdf_idx])
        hdf_path = os.path.join(self.source_dir, self.filenames[hdf_idx])

        # Read from the HDF5 file
        with h5py.File(hdf_path, "r") as hf:
            if "boards" not in hf or "scores" not in hf:
                raise ValueError(f"Missing datasets in {hdf_path}")
            board = hf["boards"][board_idx]
            score = hf["scores"][board_idx]

        return board, score


dataset = EVAL_HDF_Dataset(_dir)
for idx in range(10):
    board, score = dataset[idx]
    print(f"Sample {idx}:")
    print(board)
    print(score)