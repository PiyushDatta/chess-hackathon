### Setup

```
python3 -m venv venv
pip install -r requirements.txt
```

### Running

```
python run.py
```

### ISC training

```
Go to train.py and change local_gpu to false
```

```
source /root/.chess/bin/activate
isc train model.isc
```
Check experiments with
```
isc experiments
```

### Local training
Copy train_chessGPT.py to train.py
```
cp /root/chess-hackathon/models/chessGPT/train_chessGPT.py train.py
```

Make the following changes to train.py
```
 def main(args, timer):
+    local_gpu = True
     dist.init_process_group("nccl")  # Expects RANK set in environment variable
     rank = int(os.environ["RANK"])  # Rank of this GPU in cluster
     world_size = int(os.environ["WORLD_SIZE"]) # Total number of GPUs in the cluster
@@ -57,10 +58,16 @@ def main(args, timer):
         print(f"TrainConfig: {args}")
     timer.report("Setup for distributed training")
 
+    if local_gpu:
+        args.save_dir = "/root/chess-hackathon/checkpoint.pt"
     saver = AtomicDirectory(args.save_dir)
     timer.report("Validated checkpoint path")
 
-    data_path = "/data"
+    if local_gpu:
+        data_path = "/data/gm"
+    else:
+        data_path = "/data"
```

Run from directory /root/chess-hackathon/train.py
```
torchrun --nnodes=1 --nproc-per-node=1  train.py
```