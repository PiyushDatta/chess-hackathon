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

## ISC Training

`isc train model.isc`

## To see tensorboard

`tensorboard --logdir=training_output`

## To use bottleneck

`python3 -m torch.utils.bottleneck train.py --stop-after-test-max-limit-train-batches`

## Training notes

#### Training session 1

    -   Chess vision model (last winner)
    -   Base training script with profiler
    -   chessVision_profiler.png
    -   Batch throughput: 60-64 (examples seen)
    -   Training loss at 28 at 100 steps
    -   Train loss/rank corr at -2.88 at 100 steps

#### Training session 2

    -   Chess vision model (last winner)
    -   Base training script with profiler (can enable/disable, default is false)
    -   Dataloaders during training (num_workers=8, pin_memory=True, prefetch_factor=4,  # Add prefetching, persistent_workers=True  # Keep workers alive)
        - Data pinning
        - worker threads of 8
        - prefetch_factor = 4
        - Keep workers alive (persistent_workers)
    -   Batch size changed to 32
    -   AMP autocast disabled (forward pass with FP32), with grad scalar enabled (backward pass mixed precision)
    -   Batch throughput:  84-88 (examples seen)
    -   Training loss at 27-28 at 100 steps
    -   Train loss/rank corr at -2.93 at 100 steps

```
Epoch [0] Step [22 / 147,631] Batch [137 / 885,786] Lr: [2.2e-05], Avg Loss [30.162], Rank Corr.: [0.387%], Batch Time: [2.199 s], Total train time: [317.582 s], Batch count: [138], Avg batch time [2.301 s], Throughput: [87.3 ex/s]  13,451.922 ms,       320.93 s total
```

```
data_pi@DESKTOP-OBJ3JKF:/mnt/c/Users/data_pi/Documents/programming/chess-hackathon$ ./check_gpu_stats.sh
timestamp, name, utilization.gpu [%], utilization.memory [%], memory.total [MiB], memory.free [MiB], memory.used [MiB], temperature.gpu, power.draw [W]
025/03/21 02:06:50.642, NVIDIA GeForce RTX 3090, 3 %, 2 %, 24576 MiB, 22382 MiB, 1945 MiB, 36, 38.63 W
2025/03/21 02:06:51.651, NVIDIA GeForce RTX 3090, 19 %, 3 %, 24576 MiB, 22379 MiB, 1948 MiB, 36, 36.19 W
2025/03/21 02:06:52.661, NVIDIA GeForce RTX 3090, 2 %, 2 %, 24576 MiB, 22377 MiB, 1950 MiB, 36, 38.41 W
2025/03/21 02:06:53.669, NVIDIA GeForce RTX 3090, 36 %, 7 %, 24576 MiB, 22377 MiB, 1950 MiB, 36, 37.08 W
2025/03/21 02:06:54.679, NVIDIA GeForce RTX 3090, 17 %, 2 %, 24576 MiB, 22377 MiB, 1950 MiB, 36, 38.45 W
2025/03/21 02:06:55.692, NVIDIA GeForce RTX 3090, 1 %, 2 %, 24576 MiB, 22388 MiB, 1939 MiB, 36, 38.28 W
2025/03/21 02:06:56.702, NVIDIA GeForce RTX 3090, 16 %, 2 %, 24576 MiB, 22388 MiB, 1939 MiB, 36, 37.59 W
2025/03/21 02:06:57.710, NVIDIA GeForce RTX 3090, 7 %, 3 %, 24576 MiB, 22379 MiB, 1948 MiB, 36, 36.33 W
2025/03/21 02:06:58.734, NVIDIA GeForce RTX 3090, 29 %, 6 %, 24576 MiB, 22382 MiB, 1945 MiB, 36, 35.29 W
2025/03/21 02:06:59.746, NVIDIA GeForce RTX 3090, 7 %, 4 %, 24576 MiB, 22388 MiB, 1939 MiB, 36, 38.64 W
2025/03/21 02:07:00.755, NVIDIA GeForce RTX 3090, 25 %, 7 %, 24576 MiB, 22388 MiB, 1939 MiB, 36, 37.08 W
```

```
--------------------------------------------------------------------------------
  Environment Summary
--------------------------------------------------------------------------------
PyTorch 2.3.1+cu121 DEBUG compiled w/ CUDA 12.1
Running with Python 3.10 and CUDA 11.8.89

`pip3 list` truncated output:
numpy==1.26.4
torch==2.3.1
torch-geometric==2.6.0
torchvision==0.18.1
triton==2.3.1
--------------------------------------------------------------------------------
  cProfile output
--------------------------------------------------------------------------------
         8255353 function calls (8211910 primitive calls) in 151.110 seconds

   Ordered by: internal time
   List reduced from 7151 to 15 due to restriction <15>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    90985   75.502    0.001   75.502    0.001 {built-in method posix.stat}
     3275   42.669    0.013   48.247    0.015 {built-in method builtins.sorted}
     3202    5.644    0.002    5.644    0.002 {method 'index' of 'list' objects}
       50    4.726    0.095    4.726    0.095 {method 'run_backward' of 'torch._C._EngineBase' objects}
    11962    3.493    0.000    3.493    0.000 {built-in method posix.lstat}
       12    2.132    0.178    2.132    0.178 {method 'tolist' of 'torch._C.TensorBase' objects}
      350    2.122    0.006    4.857    0.014 {built-in method torch.conv2d}
     1327    1.850    0.001    1.850    0.001 {built-in method io.open_code}
        3    1.260    0.420    1.260    0.420 {built-in method torch.randperm}
        1    0.697    0.697    0.697    0.697 {built-in method torch._C._distributed_c10d._verify_params_across_processes}
     1328    0.669    0.001    0.669    0.001 {method 'read' of '_io.BufferedReader' objects}
     1375    0.648    0.000    0.648    0.000 {method '__exit__' of '_io._IOBase' objects}
    33/31    0.444    0.013    0.453    0.015 {built-in method _imp.create_dynamic}
    10621    0.444    0.000   70.767    0.007 /usr/lib/python3.10/traceback.py:338(extract)
        1    0.424    0.424  151.112  151.112 train.py:1(<module>)


--------------------------------------------------------------------------------
  autograd profiler output (CPU mode)
--------------------------------------------------------------------------------
        top 15 events sorted by cpu_time_total

-----------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                               Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
-----------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
    DistributedDataParallel.forward        24.53%   -5088.000us     -6614.21%        1.372s        1.372s             1
    DistributedDataParallel.forward        -9.51%       1.973ms     -6575.83%        1.364s        1.364s             1
    DistributedDataParallel.forward       -10.13%       2.101ms     -6559.27%        1.361s        1.361s             1
    DistributedDataParallel.forward       -10.16%       2.108ms     -6493.39%        1.347s        1.347s             1
    DistributedDataParallel.forward        -9.69%       2.011ms     -6485.87%        1.346s        1.346s             1
    DistributedDataParallel.forward       -10.64%       2.207ms     -6481.87%        1.345s        1.345s             1
    DistributedDataParallel.forward        74.66%  -15489.000us     -6479.59%        1.344s        1.344s             1
    DistributedDataParallel.forward        23.96%   -4970.000us     -6473.66%        1.343s        1.343s             1
    DistributedDataParallel.forward       -10.64%       2.208ms     -6472.72%        1.343s        1.343s             1
    DistributedDataParallel.forward       -16.35%       3.393ms     -6470.04%        1.342s        1.342s             1
    DistributedDataParallel.forward       -11.94%       2.478ms     -6458.99%        1.340s        1.340s             1
    DistributedDataParallel.forward        -9.18%       1.904ms     -6428.63%        1.334s        1.334s             1
    DistributedDataParallel.forward        27.52%   -5709.000us     -6427.19%        1.333s        1.333s             1
    DistributedDataParallel.forward        27.57%   -5719.000us     -6421.21%        1.332s        1.332s             1
    DistributedDataParallel.forward        20.02%   -4154.000us     -6415.42%        1.331s        1.331s             1
-----------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: -20746.000us

--------------------------------------------------------------------------------
  autograd profiler output (CUDA mode)
--------------------------------------------------------------------------------
        top 15 events sorted by cpu_time_total

        Because the autograd profiler uses the CUDA event API,
        the CUDA time column reports approximately max(cuda_time, cpu_time).
        Please ignore this output if your code does not use CUDA.

-----------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                               Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-----------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
    DistributedDataParallel.forward         6.19%       2.961ms      3326.87%        1.592s        1.592s       1.279ms         7.56%        1.592s        1.592s             1
    DistributedDataParallel.forward         8.02%       3.841ms      2891.41%        1.384s        1.384s       1.363ms         8.06%        1.384s        1.384s             1
    DistributedDataParallel.forward         7.40%       3.543ms      2879.84%        1.378s        1.378s     989.000us         5.85%        1.378s        1.378s             1
    DistributedDataParallel.forward         7.13%       3.414ms      2819.34%        1.350s        1.350s       1.543ms         9.12%        1.349s        1.349s             1
    DistributedDataParallel.forward         6.41%       3.068ms      2796.58%        1.339s        1.339s       1.258ms         7.44%        1.339s        1.339s             1
    DistributedDataParallel.forward         6.37%       3.051ms      2791.80%        1.336s        1.336s     944.000us         5.58%        1.336s        1.336s             1
    DistributedDataParallel.forward         7.44%       3.560ms      2767.96%        1.325s        1.325s     953.000us         5.64%        1.325s        1.325s             1
    DistributedDataParallel.forward         6.33%       3.030ms      2764.64%        1.323s        1.323s     839.000us         4.96%        1.323s        1.323s             1
    DistributedDataParallel.forward         6.28%       3.008ms      2752.58%        1.318s        1.318s       1.093ms         6.46%        1.317s        1.317s             1
    DistributedDataParallel.forward         6.69%       3.201ms      2739.66%        1.311s        1.311s       1.415ms         8.37%        1.309s        1.309s             1
    DistributedDataParallel.forward         6.38%       3.054ms      2737.54%        1.310s        1.310s       1.208ms         7.14%        1.310s        1.310s             1
    DistributedDataParallel.forward         7.03%       3.363ms      2735.03%        1.309s        1.309s     936.000us         5.53%        1.310s        1.310s             1
    DistributedDataParallel.forward         6.24%       2.987ms      2734.34%        1.309s        1.309s       1.129ms         6.68%        1.309s        1.309s             1
    DistributedDataParallel.forward         5.95%       2.850ms      2731.28%        1.307s        1.307s     729.000us         4.31%        1.307s        1.307s             1
    DistributedDataParallel.forward         6.13%       2.935ms      2730.57%        1.307s        1.307s       1.234ms         7.30%        1.307s        1.307s             1
-----------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 47.866ms
Self CUDA time total: 16.912ms
```

#### Training session 3

    -   Chess vision model (last winner)
    -   Base training script with profiler (can enable/disable, default is false)
    -   Dataloaders during training (num_workers=8, pin_memory=True, prefetch_factor=4,  # Add prefetching, persistent_workers=True  # Keep workers alive)
        - Data pinning
        - worker threads of 8
        - prefetch_factor = 4
        - Keep workers alive (persistent_workers)
    -   Batch size changed to 32
    -   AMP autocast disabled (forward pass with FP32), with grad scalar enabled (backward pass mixed precision)
    -   Model configs changed:
        - Make larger, from 1M to 12.8M params (12,798,081)
            nlayers: 4
            embed_dim: 256
            inner_dim: 352
            attention_dim: 224
    -   Use flash attention
    -   Batch throughput:  55-65 (examples seen)
    -   Training loss at 27-28 at 100 steps
    -   Train loss/rank corr at -2.93 at 100 steps

```
Epoch [0] Step [8 / 147,631] Batch [53 / 885,786] Lr: [8e-06], Avg Loss [27.029], Rank Corr.: [-5.021%], Batch Time: [3.367 s], Total train time: [188.297 s], Batch count: [54], Avg batch time [3.487 s], Throughput: [57.0 ex/s]  20,438.752 ms,       192.50 s total
```

```
data_pi@DESKTOP-OBJ3JKF:/mnt/c/Users/data_pi/Documents/programming/chess-hackathon$ ./check_gpu_stats.sh
timestamp, name, utilization.gpu [%], utilization.memory [%], memory.total [MiB], memory.free [MiB], memory.used [MiB], temperature.gpu, power.draw [W]
2025/03/21 04:44:07.665, NVIDIA GeForce RTX 3090, 19 %, 5 %, 24576 MiB, 22154 MiB, 2173 MiB, 35, 40.72 W
2025/03/21 04:44:08.672, NVIDIA GeForce RTX 3090, 5 %, 4 %, 24576 MiB, 22154 MiB, 2173 MiB, 35, 43.28 W
2025/03/21 04:44:09.686, NVIDIA GeForce RTX 3090, 14 %, 2 %, 24576 MiB, 22157 MiB, 2170 MiB, 35, 41.70 W
2025/03/21 04:44:10.694, NVIDIA GeForce RTX 3090, 14 %, 1 %, 24576 MiB, 22157 MiB, 2170 MiB, 35, 41.97 W
2025/03/21 04:44:11.703, NVIDIA GeForce RTX 3090, 31 %, 14 %, 24576 MiB, 22159 MiB, 2168 MiB, 35, 44.06 W
2025/03/21 04:44:12.710, NVIDIA GeForce RTX 3090, 13 %, 7 %, 24576 MiB, 22159 MiB, 2168 MiB, 35, 40.71 W
2025/03/21 04:44:13.724, NVIDIA GeForce RTX 3090, 11 %, 2 %, 24576 MiB, 22159 MiB, 2168 MiB, 35, 39.95 W
2025/03/21 04:44:14.732, NVIDIA GeForce RTX 3090, 58 %, 32 %, 24576 MiB, 22159 MiB, 2168 MiB, 35, 40.47 W
2025/03/21 04:44:15.744, NVIDIA GeForce RTX 3090, 11 %, 2 %, 24576 MiB, 22159 MiB, 2168 MiB, 35, 43.37 W
2025/03/21 04:44:16.751, NVIDIA GeForce RTX 3090, 13 %, 2 %, 24576 MiB, 22159 MiB, 2168 MiB, 35, 42.99 W
2025/03/21 04:44:17.762, NVIDIA GeForce RTX 3090, 44 %, 21 %, 24576 MiB, 22159 MiB, 2168 MiB, 35, 40.41 W
2025/03/21 04:44:18.770, NVIDIA GeForce RTX 3090, 10 %, 8 %, 24576 MiB, 22159 MiB, 2168 MiB, 35, 39.34 W
2025/03/21 04:44:19.784, NVIDIA GeForce RTX 3090, 18 %, 2 %, 24576 MiB, 22157 MiB, 2170 MiB, 35, 43.51 W
2025/03/21 04:44:20.792, NVIDIA GeForce RTX 3090, 8 %, 0 %, 24576 MiB, 22109 MiB, 2218 MiB, 37, 100.66 W
```

#### Training session 3

    -   Chess vision model (last winner)
    -   Base training script with profiler (can enable/disable, default is false)
    -   Dataloaders during training (num_workers=8, pin_memory=True, prefetch_factor=4,  # Add prefetching, persistent_workers=True  # Keep workers alive)
        - Data pinning
        - worker threads of 8
        - prefetch_factor = 4
        - Keep workers alive (persistent_workers)
    -   Batch size changed to 32
    -   AMP with grad scalar enabled (backward pass mixed precision), but with autocast disabled (forward pass with FP32)
    -   Using FSDP instead of DDP
    -   Model configs changed:
        - Make larger, from 1M to 12.8M params (12,798,081)
            nlayers: 4            # increased from 2
            embed_dim: 128        # increased from 64
            inner_dim: 640        # increased from 320
            attention_dim: 128    # increased from 64
            use_1x1conv: True
            dropout: 0.5    -   Use flash attention
    -   Use dataloader_num_workers = 12, from 8
    -   Use torch.backends.cudnn.benchmark = True, helps PyTorch pick the best convolution algorithms for input sizes, which speeds up training when the input dimensions don’t change much
    -   Use torch.set_float32_matmul_precision("high") can improve numerical precision during float32 matrix multiplications, which might benefit the stability of the training
    -   Actually use flash attention this time
    -   Change attention module to be more efficient
    -   Change residual module to be more efficient
    -   Change warm up steps to 25
    -   Batch throughput:  84-88 (examples seen)
    -   Training loss at 27-28 at 100 steps
    -   Train loss/rank corr at -2.93 at 100 steps

```
Epoch [0] Step [21 / 147,631] Batch [131 / 885,786] Lr: [2.1e-05], Avg Loss [27.309], Rank Corr.: [0.509%], Batch Time: [3.963 s], Total train time: [548.317 s], Batch count: [132], Avg batch time [4.154 s], Throughput: [48.5 ex/s]  23,947.377 ms,       552.39 s total
```

```
2025/03/21 17:25:32.797, NVIDIA GeForce RTX 3090, 24 %, 5 %, 24576 MiB, 22303 MiB, 2024 MiB, 34, 36.13 W
2025/03/21 17:25:33.812, NVIDIA GeForce RTX 3090, 5 %, 3 %, 24576 MiB, 22302 MiB, 2025 MiB, 34, 36.76 W
2025/03/21 17:25:34.820, NVIDIA GeForce RTX 3090, 8 %, 4 %, 24576 MiB, 22304 MiB, 2023 MiB, 34, 32.89 W
2025/03/21 17:25:35.835, NVIDIA GeForce RTX 3090, 8 %, 4 %, 24576 MiB, 22304 MiB, 2023 MiB, 34, 33.40 W
2025/03/21 17:25:36.844, NVIDIA GeForce RTX 3090, 61 %, 28 %, 24576 MiB, 22299 MiB, 2028 MiB, 34, 35.66 W
2025/03/21 17:25:37.853, NVIDIA GeForce RTX 3090, 36 %, 26 %, 24576 MiB, 22294 MiB, 2033 MiB, 34, 38.44 W
2025/03/21 17:25:38.861, NVIDIA GeForce RTX 3090, 5 %, 3 %, 24576 MiB, 22295 MiB, 2032 MiB, 34, 32.76 W
2025/03/21 17:25:39.870, NVIDIA GeForce RTX 3090, 6 %, 3 %, 24576 MiB, 22294 MiB, 2033 MiB, 34, 32.29 W
2025/03/21 17:25:40.879, NVIDIA GeForce RTX 3090, 53 %, 22 %, 24576 MiB, 22298 MiB, 2029 MiB, 34, 34.98 W
2025/03/21 17:25:41.890, NVIDIA GeForce RTX 3090, 17 %, 2 %, 24576 MiB, 22303 MiB, 2024 MiB, 34, 39.00 W
2025/03/21 17:25:42.898, NVIDIA GeForce RTX 3090, 6 %, 4 %, 24576 MiB, 22308 MiB, 2019 MiB, 34, 33.97
```

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

### To submit:

```
cp checkpoint.pt /shared/rat_exterminator/.
cp chess_gameplay.py /shared/rat_exterminator/.
cp model.py /shared/rat_exterminator/.
cp model_config.yaml /shared/rat_exterminator/.
cp pre_submission_val.py /shared/rat_exterminator/.
```
