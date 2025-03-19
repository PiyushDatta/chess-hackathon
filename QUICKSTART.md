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

--bs 24              # Batch size per GPU (max your VRAM allows)
--lr 0.0002          # Learning rate
--ws 800             # Warmup steps
--grad-accum 4       # Gradient accumulation steps
--save-steps 100     # Checkpoint frequency


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