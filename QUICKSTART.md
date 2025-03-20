## Setup

`python3 -m venv venv`
`pip install -r requirements.txt`
`cp models/chessGPT/torch/model.py model.py`
`cp models/chessGPT/train_chessGPT.py train.py`
`cp models/chessGPT/torch/model_config.yaml model_config.yaml`

## Running

`python run.py`

## Training

`python train.py`

## Notes

to train:
isc train model.isc

validate:
cp /root/<output>/<path>/latest_pt/checkpoint.pt /root/chess-hackathon/checkpoint.pt

show experiments:
isc experiments

nlayers: 14
embed_dim: 768
nhead: 12
head_dim: 64
ff_dim: 3072
dropout: 0.1
rope: True
causal: True
norm_first: True
ghost: False
device: "cuda"

--bs 24 # Batch size per GPU (max your VRAM allows)
--lr 0.0002 # Learning rate
--ws 800 # Warmup steps
--grad-accum 4 # Gradient accumulation steps
--save-steps 100 # Checkpoint frequency

torchrun --nproc_per_node=8 \
 --rdzv_id=chess_train \
 --rdzv_backend=c10d \
 train.py \
 --bs 24 \
 --lr 0.0002 \
 --ws 800 \
 --grad-accum 4 \
 --model-config model_config.yaml \
 --dataset-id your_dataset_id

75M model:

nlayers: 24
embed_dim: 512
nhead: 8
head_dim: 64
ff_dim: 2048
dropout: 0.05
rope: True
causal: True
norm_first: False
ghost: True

input_artifact_id_list = ["261fc020-3b24-498e-a10d-da1f9ee11db8"]

### Example PGNs (with metadata) for chessGPT from GM dataset

```
PGN 1:
1.e4 c5 2.c3 Nf6 3.e5 Nd5 4.d4 Nc6 5.Nf3 cxd4 6.cxd4 e6 7.a3 d6 8.Bd3 Qa5+ 9.Bd2 Qb6 10.Nc3 Nxc3 11.Bxc3 dxe5 12.dxe5 Be7 13.O-O Bd7 14.Nd2 Qc7 15.Qg4 O-O-O 16.Rfc1 Kb8 17.Qc4 Rc8 18.b4 f6 19.Nf3 Qb6 20.Qe4 f5 21.Qe1 a6 22.Rab1 g5 23.Nd2 Nd4 24.Qe3 Rxc3 25.Rxc3 f4 26.Qe1 g4 27.Ne4 Bc6 28.Nc5 Ka7 29.a4 Bf3 30.a5 Qd8 31.Bc4 Bxc5 32.bxc5 Qh4 33.gxf3 gxf3 34.Kh1 Rg8 35.Qe4 Rg7 36.Qxd4 Qg5 37.c6+ Kb8 38.c7+ Rxc7 39.Rg1 Qh5 40.Rg8+ Rc8 41.Qd6+ Ka7  1-0
Metadata: [Event "Wch U16"]
[Site "Wattignies"]
[Date "1976.08.27"]
[Round "?"]
[White "Chandler, Murray G"]
[Black "Kasparov, Gary"]
[Result "1-0"]
[WhiteElo ""]
[BlackElo ""]
[ECO "B22"]

PGN 2:
1.e4 e6 2.d4 d5 3.Nd2 Nf6 4.e5 Ne4 5.Nxe4 dxe4 6.Be3 b6 7.Ne2 Bb7 8.Ng3 c5 9.dxc5 Qxd1+ 10.Rxd1 Bxc5 11.Bxc5 bxc5 12.Bb5+ Ke7 13.O-O Bc6 14.Bxc6 Nxc6 15.Nxe4 Nxe5 16.Nxc5 Rac8 17.b4 Rhd8 18.f3 a5 19.c3 axb4 20.cxb4 Nc6 21.Rxd8 Rxd8 22.b5 Nb4 23.Rb1 Nd5 24.b6 Rb8 25.b7 Nc7 26.Rb6 Kd8 27.Kf2 Nd5 28.Rd6+ Kc7 29.Rxd5 exd5 30.Na6+ Kxb7 31.Nxb8 Kxb8 32.Ke3 Kc7 33.Kd4 Kc6 34.a4 Kb6 35.Kxd5 Ka5 36.Kd6 Kxa4 37.Ke7 f5 38.Kf7  1-0
Metadata: [Event "Wch U16"]
[Site "Wattignies"]
[Date "1976.??.??"]
[Round "?"]
[White "Kasparov, Gary"]
[Black "Galle, Andre"]
[Result "1-0"]
[WhiteElo ""]
[BlackElo ""]
[ECO "C11"]
```

### Example PGNs (with metadata) for chessVision from GM dataset

```
(venv) root@d4e1f354774b:~# python3 /tmp/read_h5.py
Sample 1:
[[10  8  9 11 12  9  8 10]
 [ 7  7  7  7  7  7  7  7]
 [ 6  6  6  6  6  6  6  6]
 [ 6  6  6  6  6  6  6  6]
 [ 6  6  6  6  6  6  6  6]
 [ 6  6  6  6  6  6  6  4]
 [ 5  5  5  5  5  5  5  5]
 [ 2  4  3  1  0  3  6  2]]
-64

Sample 2:
[[10  8  9 11 12  9  8 10]
 [ 7  7  7  7  7  7  7  7]
 [ 6  6  6  6  6  6  6  6]
 [ 6  6  6  6  6  6  6  6]
 [ 6  6  6  6  6  6  6  6]
 [ 6  6  6  6  6  4  6  6]
 [ 5  5  5  5  5  5  5  5]
 [ 2  4  3  1  0  3  6  2]]
24
```

### Code to read the hdf5 files

Code to read hdf5 file (read first 10 pgns):

```
import h5py

flocation = "/data/83f72080-4aee-445b-874f-7190d6880e13/gm"
flocation = flocation + "/pgnHDF0.h5"
# Open the HDF5 file
with h5py.File('pgnHDF0.h5', 'r') as file:
    # Load PGN and metadata datasets
    pgn_dataset = file['pgn']
    meta_dataset = file['meta']

    # Read the first 10 PGNs and metadata
    first_10_pgns = [pgn.decode('utf-8') for pgn in pgn_dataset[:10]]
    first_10_meta = [meta.decode('utf-8') for meta in meta_dataset[:10]]

    # Print PGNs and their corresponding metadata
    for i, (pgn, meta) in enumerate(zip(first_10_pgns, first_10_meta), 1):
        print(f"PGN {i}:\n{pgn}\nMetadata: {meta}\n{'='*50}")
```

Code to read eval hdf5 file (read first 10 pgns):

```
import os
import h5py
import numpy as np
from itertools import accumulate
from torch.utils.data import Dataset

_dir = "/data/1e404a5c-140b-4e30-af3a-ee453536e9d8/gm"
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
```
