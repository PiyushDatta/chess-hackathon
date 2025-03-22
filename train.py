from cycling_utils import TimestampedTimer

timer = TimestampedTimer("Imported TimestampedTimer")

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import torch.autograd.profiler as profiler
from pathlib import Path
import argparse
import os
import socket
import yaml
import time
import math

from cycling_utils import (
    InterruptableDistributedSampler,
    MetricsTracker,
    AtomicDirectory,
    atomic_torch_save,
)

from utils.optimizers import Lamb
from utils.datasets import EVAL_HDF_Dataset
from torch.utils.data import Sampler
from model import Model

timer.report("Completed imports")

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-config",
        help="model config path",
        type=Path,
        default="/root/chess-hackathon/model_config.yaml",
    )
    parser.add_argument(
        "--load-path",
        help="path to checkpoint.pt file to resume from",
        type=Path,
        default="/root/chess-hackathon/recover/checkpoint.pt",
    )
    parser.add_argument("--bs", help="batch size", type=int, default=64)
    parser.add_argument("--lr", help="learning rate", type=float, default=0.001)
    parser.add_argument("--wd", help="weight decay", type=float, default=0.01)
    parser.add_argument(
        "--ws", help="learning rate warm up steps", type=int, default=25
    )
    parser.add_argument(
        "--grad-accum", help="gradient accumulation steps", type=int, default=5
    )
    parser.add_argument(
        "--save-steps", help="saving interval steps", type=int, default=25
    )
    parser.add_argument(
        "--dataset-id", help="Dataset ID for the dataset", type=str, default=''
    )
    parser.add_argument(
        "--stop-after-test-max-limit-train-batches", action="store_true", help="Stop training after our max limit batches", default=False
    )
    parser.add_argument(
        "--amp-enabled", help="Enable/Disable Automatic Mixed Precision (AMP)", type=bool, default=True
    )
    parser.add_argument(
        "--fsdp-enabled", help="Enable/Disable Fully Sharded Data Parallel (FSDP) training", type=bool, default=False
    )
    return parser


def logish_transform(data):
    """Zero-symmetric log-transformation."""
    return torch.sign(data) * torch.log1p(torch.abs(data))


def spearmans_rho(a, b):
    """Spearman's rank correlation coefficient"""
    assert len(a) == len(b), "ERROR: Vectors must be of equal length"
    n = len(a)
    a_ranks = [sorted(a).index(i) for i in a]
    b_ranks = [sorted(b).index(j) for j in b]
    a_ranks_mean = sum(a_ranks) / n
    b_ranks_mean = sum(b_ranks) / n
    rank_covariance = (
        sum(
            [
                (a_rank - a_ranks_mean) * (b_rank - b_ranks_mean)
                for a_rank, b_rank in zip(a_ranks, b_ranks)
            ]
        )
        / n
    )
    a_ranks_sd = (sum([(a_rank - a_ranks_mean) ** 2 for a_rank in a_ranks]) / n) ** 0.5
    b_ranks_sd = (sum([(b_rank - b_ranks_mean) ** 2 for b_rank in b_ranks]) / n) ** 0.5
    return rank_covariance / (a_ranks_sd * b_ranks_sd + 1e-8)


def main(args, timer):
    TESTING_LOCAL = False
    do_barrier = False
    global_batch_count = 0
    global_batch_max_limit = 50
    global_batch_start_time = time.time()  
    if TESTING_LOCAL:
        rank = int(os.environ.get("RANK", 0))  # Default to 0 if not set
        args.world_size = int(
            os.environ.get("WORLD_SIZE", 1)
        )  # Default to 1 if not set
        args.device_id = int(os.environ.get("LOCAL_RANK", 0))  # Default to 0 if not set
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(args.world_size)
        os.environ["LOCAL_RANK"] = str(args.device_id)
        os.environ["MASTER_ADDR"] = "0000"
        os.environ["MASTER_PORT"] = "3555"
        current_path = os.getcwd()
        training_output_path = os.path.join(current_path, "training_output")
        args.load_path = current_path
        args.model_config = os.path.join(current_path, "model_config.yaml")
        os.environ["LOSSY_ARTIFACT_PATH"] = training_output_path
        os.environ["CHECKPOINT_ARTIFACT_PATH"] = training_output_path
    else:
        rank = int(os.environ["RANK"])  # Rank of this GPU in cluster
        args.world_size = int(
            os.environ["WORLD_SIZE"]
        )  # Total number of GPUs in the cluster
        args.device_id = int(os.environ["LOCAL_RANK"])  # Rank on local node
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("Number of GPUs:", torch.cuda.device_count())
            print("Current device:", torch.cuda.current_device())
            print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    if not dist.is_initialized() and torch.cuda.is_available():
        dist.init_process_group("nccl")  # Expects RANK set in environment variable
    args.is_master = rank == 0  # Master node for saving / reporting
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device_id)  # Enables calling 'cuda'
    torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    if args.device_id == 0:
        hostname = socket.gethostname()
        print("Hostname:", hostname)
        print(f"TrainConfig: {args}")
    timer.report("Setup for distributed training")

    if TESTING_LOCAL:
        # Create the /data/ directory if it doesn't exist and define data_path
        data_path = f"/data/1e404a5c-140b-4e30-af3a-ee453536e9d8/lc0"
        # data_path = f"/data/lc0"
        # data_path = f"/data/gm"
        ######################################################################
        # TO DOWNLOAD DATASET
        ######################################################################
        # from huggingface_hub import snapshot_download, login
        # os.makedirs(data_path, exist_ok=True)
        # Specify the repository id from Hugging Face Hub
        # repo_id = "<REPO_ID>"
        # Download the snapshot to data_path
        # login(token="<TOKEN_ID>")
        # snapshot_download(repo_id=repo_id, local_dir=data_path)
        ######################################################################
    else:
        data_path = f"/data/{args.dataset_id}/gm"
        # data_path = f"/data/{args.dataset_id}/lc0"
    dataset = EVAL_HDF_Dataset(data_path)
    random_generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(
        dataset, [0.8, 0.2], generator=random_generator
    )
    timer.report(
        f"Intitialized datasets with {len(train_dataset):,} training and {len(test_dataset):,} test board evaluations."
    )

    model_config = yaml.safe_load(open(args.model_config))
    if args.device_id == 0:
        print(f"ModelConfig: {model_config}")
    model_config["device"] = "cuda"
    model = Model(**model_config)
    if torch.cuda.is_available():
        model = model.to(args.device_id)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
    timer.report(f"Initialized model with {params:,} params, moved to device")

    train_sampler = InterruptableDistributedSampler(train_dataset)
    test_sampler = InterruptableDistributedSampler(test_dataset)

    dataloader_num_workers = 32
    dataloader_prefetch_factor = 4
    dataloader_persistent_workers = True
    if args.stop_after_test_max_limit_train_batches:
        # TODO(piydatta): Remove once bug in torch.utils.bottleneck is gone.
        # Multiple workers for dataloader is not supported.
        # [rank0]:[W profiler_kineto.cpp:472] 
        # Failed to record CUDA event. ../torch/csrc/profiler/stubs/cuda.cpp:48: CUDA initialization error. This can occur if one runs the profiler in CUDA mode on code that creates a DataLoader with num_workers > 0. This operation is currently unsupported; potential workarounds are: (1) don't use the profiler in CUDA mode or (2) use num_workers=0 in the DataLoader or (3) Don't profile the data loading portion of your code. https://github.com/pytorch/pytorch/issues/6313 tracks profiler support for multi-worker DataLoader.
        dataloader_num_workers = 0
        dataloader_prefetch_factor = None
        dataloader_persistent_workers = False
    print(f"Loading dataloaders with {dataloader_num_workers} workers")
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.bs, 
        sampler=train_sampler,
        num_workers=dataloader_num_workers,  # Add workers for data loading
        pin_memory=True,  # Speed up GPU transfers
        prefetch_factor=dataloader_prefetch_factor,  # Add prefetching
        persistent_workers=dataloader_persistent_workers  # Keep workers alive
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.bs, 
        sampler=test_sampler,
        num_workers=dataloader_num_workers,
        pin_memory=True,  # Speed up GPU transfers
        prefetch_factor=dataloader_prefetch_factor,  # Add prefetching
        persistent_workers=dataloader_persistent_workers  # Keep workers alive
    )
    timer.report("Prepared dataloaders")
    # compile model before ddp/fsdp.
    model = torch.jit.script(model)
    # Use FSDP if enabled; else fall back to DDP.
    if args.fsdp_enabled:
        model = FSDP(model)
        timer.report("Using FSDP for distributed training")
    else:
        model = DDP(model, device_ids=[args.device_id])
        timer.report("Using DDP for distributed training")

    loss_fn = nn.MSELoss(reduction="sum")
    # optimizer = Lamb(model.parameters(), lr=args.lr, weight_decay=args.wd)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    metrics = {
        "train": MetricsTracker(),
        "test": MetricsTracker(),
        "best_rank_corr": float("-inf"),
    }

    if args.is_master:
        writer = SummaryWriter(log_dir=os.environ["LOSSY_ARTIFACT_PATH"])

    output_directory = os.environ["CHECKPOINT_ARTIFACT_PATH"]
    saver = AtomicDirectory(output_directory=output_directory, is_master=args.is_master)

    # set the checkpoint_path if there is one to resume from
    checkpoint_path = None
    latest_symlink_file_path = os.path.join(output_directory, saver.symlink_name)
    if os.path.islink(latest_symlink_file_path):
        latest_checkpoint_path = os.readlink(latest_symlink_file_path)
        checkpoint_path = os.path.join(latest_checkpoint_path, "checkpoint.pt")
    elif args.load_path:
        # assume user has provided a full path to a checkpoint to resume
        if os.path.isfile(args.load_path):
            checkpoint_path = args.load_path

    scaler = GradScaler(enabled=args.amp_enabled)
    if checkpoint_path:
        # load checkpoint
        timer.report(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{args.device_id}")
        # For DDP, the underlying module is stored under model.module
        if args.fsdp_enabled:
            model.load_state_dict(checkpoint["model"])
        else:
            model.module.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        train_dataloader.sampler.load_state_dict(checkpoint["train_sampler"])
        test_dataloader.sampler.load_state_dict(checkpoint["test_sampler"])
        metrics = checkpoint["metrics"]
        timer = checkpoint["timer"]
        timer.start_time = time.time()
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        timer.report("Retrieved saved checkpoint")
        if do_barrier:
            dist.barrier()
    else:
        print(f"No checkpoint path at {checkpoint_path}, not loading checkpoint.")

    ## TRAINING

    for epoch in range(train_dataloader.sampler.epoch, 10_000):

        train_dataloader.sampler.set_epoch(epoch)
        test_dataloader.sampler.set_epoch(epoch)

        ## TRAIN

        timer.report(f"Training epoch {epoch}")
        train_batches_per_epoch = len(train_dataloader)
        train_steps_per_epoch = math.ceil(train_batches_per_epoch / args.grad_accum)
        optimizer.zero_grad()
        model.train()

        # Disable/Enable profiling on the first few batches for detailed timing.
        profiling_enabled = False 
        # profiling_enabled = (epoch in (0,1,2,3,4,5))

        for boards, scores in train_dataloader:
            global_batch_count += 1
            if args.stop_after_test_max_limit_train_batches and global_batch_count > global_batch_max_limit:
                timer.report(f"Stopping training after {global_batch_max_limit} batches as requested.")
                break
            # Start timing the batch
            batch_start_time = time.time()  
            # Determine the current step
            batch = train_dataloader.sampler.progress // train_dataloader.batch_size
            is_accum_batch = (batch + 1) % args.grad_accum == 0
            is_last_batch = (batch + 1) == train_batches_per_epoch
            is_save_batch = ((batch + 1) % args.save_steps == 0) or is_last_batch

            # If profiling is enabled, use the context manager for a specific training step.
            if profiling_enabled and is_save_batch:
                with profiler.profile(record_shapes=True, use_cuda=True) as prof:
                    boards, scores = boards.to(args.device_id,non_blocking=True), scores.to(args.device_id,non_blocking=True)
                    scores = logish_transform(scores)
                    logits = model(boards)
                    loss = loss_fn(logits, scores) / args.grad_accum
                    loss.backward()
                prof.export_chrome_trace(chrome_trace_file)
                # print(prof.key_averages().table(row_limit=10))
                print("Outputting key averages")
                print(prof.key_averages().table(row_limit=10, top_level_events_only=True))
            else:
                boards, scores = boards.to(args.device_id,non_blocking=True), scores.to(args.device_id,non_blocking=True)
                scores = logish_transform(scores)  # suspect this might help
                # TODO(Piyush): Enable autocast once the bug is resolved
                # BUG: https://github.com/pytorch/pytorch/issues/40497
                # This always leads to NaN issues
                # with autocast(enabled=args.amp_enabled):
                with autocast(enabled=False):
                    logits = model(boards)
                    loss = loss_fn(logits, scores) / args.grad_accum

            if args.amp_enabled:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            train_dataloader.sampler.advance(len(scores))

            # How accurately do our model scores rank the batch of moves?
            rank_corr = spearmans_rho(logits, scores)
            # Compute batch time
            batch_time = time.time() - batch_start_time
            total_train_time = time.time() - global_batch_start_time
            avg_batch_time = total_train_time /  global_batch_count
            # Update running metrics for average batch time
            metrics["train"].update(
                {
                    "examples_seen": len(scores),
                    "accum_loss": loss.item() * args.grad_accum,  # undo loss scale
                    "rank_corr": rank_corr,
                    "total_train_time": total_train_time,
                    "batch_count": global_batch_count,
                    "avg_batch_time": avg_batch_time
                }
            )

            if is_accum_batch or is_last_batch:
                if args.amp_enabled:
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                step = batch // args.grad_accum

                # learning rate warmup
                lr_factor = min((epoch * train_steps_per_epoch + step) / args.ws, 1)
                next_lr = lr_factor * args.lr
                for g in optimizer.param_groups:
                    g["lr"] = next_lr

                metrics["train"].reduce()
                rpt = metrics["train"].local
                avg_loss = rpt["accum_loss"] / rpt["examples_seen"]
                rpt_rank_corr = (
                    100
                    * rpt["rank_corr"]
                    / ((batch % args.grad_accum + 1) * args.world_size)
                )
                throughput = rpt["examples_seen"] / batch_time
                # report
                report = f"""\ 
Epoch [{epoch:,}] Step [{step:,} / {train_steps_per_epoch:,}] Batch [{batch:,} / {train_batches_per_epoch:,}] Lr: [{lr_factor * args.lr:,.3}], \
Avg Loss [{avg_loss:,.3f}], Rank Corr.: [{rpt_rank_corr:,.3f}%], Batch Time: [{batch_time:.3f} s], Total train time: [{total_train_time:.3f} s], Batch count: [{global_batch_count:,}], Avg batch time [{avg_batch_time:.3f} s], Throughput: [{throughput:.1f} ex/s]"""
                timer.report(report)
                if args.is_master:
                    total_progress = batch + epoch * train_batches_per_epoch
                    writer.add_scalar("train/learn_rate", next_lr, total_progress)
                    writer.add_scalar("train/loss", avg_loss, total_progress)
                    writer.add_scalar("train/batch_rank_corr", rpt_rank_corr, total_progress)
                    writer.add_scalar("train/batch_time", batch_time, total_progress)
                    writer.add_scalar("train/throughput", throughput, total_progress)
                    writer.add_scalar("train/avg_batch_time", avg_batch_time, total_progress)
                    chrome_trace_file = os.path.join(os.environ["CHECKPOINT_ARTIFACT_PATH"], f"chrome_trace_epoch_{epoch}_step_{step}.json")
                metrics["train"].reset_local()

            # Saving
            if is_save_batch:
                checkpoint_directory = saver.prepare_checkpoint_directory()

                if args.is_master:
                    # Save checkpoint
                    if do_barrier:
                        dist.barrier()
                    atomic_torch_save(
                        {
                            "model": model.module.state_dict() if args.fsdp_enabled is False else model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "train_sampler": train_dataloader.sampler.state_dict(),
                            "test_sampler": test_dataloader.sampler.state_dict(),
                            "metrics": metrics,
                            "timer": timer,
                            "scaler": scaler.state_dict()
                        },
                        os.path.join(checkpoint_directory, "checkpoint.pt"),
                    )
                print(f"Saving checkpoint to {checkpoint_directory}")
                saver.symlink_latest(checkpoint_directory)

        # Check if we should exit training after the current epoch
        if args.stop_after_test_max_limit_train_batches and global_batch_count > global_batch_max_limit:
            timer.report(f"Stopping training after {global_batch_max_limit} batches as requested.")
            break  

        ## TESTING ##

        timer.report(f"Testing epoch {epoch}")
        test_batches_per_epoch = len(test_dataloader)
        model.eval()

        with torch.no_grad():
            for boards, scores in test_dataloader:

                # Determine the current step
                batch = test_dataloader.sampler.progress // test_dataloader.batch_size
                is_last_batch = (batch + 1) == test_batches_per_epoch
                is_save_batch = ((batch + 1) % args.save_steps == 0) or is_last_batch

                scores = logish_transform(scores)  # suspect this might help
                boards, scores = boards.to(args.device_id), scores.to(args.device_id)

                logits = model(boards)
                loss = loss_fn(logits, scores)
                test_dataloader.sampler.advance(len(scores))

                # How accurately do our model scores rank the batch of moves?
                rank_corr = spearmans_rho(logits, scores)

                metrics["test"].update(
                    {
                        "examples_seen": len(scores),
                        "accum_loss": loss.item(),
                        "rank_corr": rank_corr,
                    }
                )

                # Reporting
                if is_last_batch:
                    metrics["test"].reduce()
                    rpt = metrics["test"].local
                    avg_loss = rpt["accum_loss"] / rpt["examples_seen"]
                    rpt_rank_corr = (
                        100
                        * rpt["rank_corr"]
                        / (test_batches_per_epoch * args.world_size)
                    )
                    report = f"Epoch [{epoch}] Evaluation, Avg Loss [{avg_loss:,.3f}], Rank Corr. [{rpt_rank_corr:,.3f}%]"
                    timer.report(report)
                    metrics["test"].reset_local()

                    if args.is_master:
                        writer.add_scalar("test/loss", avg_loss, epoch)
                        writer.add_scalar("test/batch_rank_corr", rpt_rank_corr, epoch)

                # Saving
                if is_save_batch:
                    # force save checkpoint if test performance improves
                    if is_last_batch and (rpt_rank_corr > metrics["best_rank_corr"]):
                        force_save = True
                        metrics["best_rank_corr"] = rpt_rank_corr
                    else:
                        force_save = False

                    checkpoint_directory = saver.prepare_checkpoint_directory(
                        force_save=force_save
                    )

                    if args.is_master:
                        # Save checkpoint
                        if do_barrier:
                            dist.barrier()
                        atomic_torch_save(
                            {
                                "model": model.module.state_dict() if args.fsdp_enabled is False else model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "train_sampler": train_dataloader.sampler.state_dict(),
                                "test_sampler": test_dataloader.sampler.state_dict(),
                                "metrics": metrics,
                                "timer": timer,
                            },
                            os.path.join(checkpoint_directory, "checkpoint.pt"),
                        )

                    saver.symlink_latest(checkpoint_directory) 

        train_dataloader.sampler.reset_progress()
        test_dataloader.sampler.reset_progress()


timer.report("Defined functions")
if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args, timer)
