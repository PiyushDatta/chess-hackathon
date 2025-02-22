from cycling_utils import TimestampedTimer

timer = TimestampedTimer("Imported TimestampedTimer")

import torch
import torch.nn as nn
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    StateDictType,
    FullStateDictConfig,
    FullOptimStateDictConfig
)

# from torch.distributed.fsdp import StateDictType
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
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

from utils.train_utils import topk_accuracy, softmax
from utils.optimizers import Lamb
from utils.datasets import PGN_HDF_Dataset
from model import Model


timer.report("Completed imports")

default_bs = 32
default_lr = 0.001
default_work_step = 500


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-config",
        help="model config path",
        type=Path,
        default="/root/chess-hackathon/model_config.yaml",
    )
    parser.add_argument(
        "--save-dir",
        help="save checkpoint path",
        type=Path,
        default=os.getenv("OUTPUT_PATH"),
    )
    # parser.add_argument("--save-dir", help="save checkpoint path", type=Path, default="/root/chess-hackathon/checkpoint.pt")
    parser.add_argument(
        "--load-path",
        help="path to checkpoint.pt file to resume from",
        type=Path,
        default="/root/chess-hackathon/recover/checkpoint.pt",
    )
    parser.add_argument("--bs", help="batch size", type=int, default=default_bs)
    # parser.add_argument("--lr", help="learning rate", type=float, default=0.001)
    parser.add_argument("--lr", help="learning rate", type=float, default=default_lr)
    parser.add_argument("--wd", help="weight decay", type=float, default=0.01)
    parser.add_argument(
        "--ws", help="learning rate warm up steps", type=int, default=default_work_step
    )
    parser.add_argument(
        "--grad-accum", help="gradient accumulation steps", type=int, default=5
    )
    parser.add_argument(
        "--save-steps", help="saving interval steps", type=int, default=20
    )
    parser.add_argument(
        "--dataset-id", help="Dataset ID for the dataset", type=str, required=True
    )

    return parser


def save_checkpoint(
    model,
    optimizer,
    train_dataloader,
    test_dataloader,
    metrics,
    timer,
    checkpoint_path,
    args,
    fsdp=False,
):
    if not args.is_master and fsdp:
        timer.report(f" FSDP NOT saving checkpoint from {checkpoint_path} since not is_master (rank == 0)")
        return
    if fsdp:
        timer.report(f"Saving fsdp")
        # Save non-FSDP state
        checkpoint = {
            'train_dataloader': train_dataloader.state_dict() if hasattr(train_dataloader, 'state_dict') else None,
            'test_dataloader': test_dataloader.state_dict() if hasattr(test_dataloader, 'state_dict') else None,
            'metrics': metrics,
            'timer': timer,
            'args': args
        }
        # Save model state - using FullStateDictConfig for model
        model_state_dict_config = FullStateDictConfig(offload_to_cpu=False, rank0_only=True)
        with FullyShardedDataParallel.state_dict_type(model, StateDictType.FULL_STATE_DICT, model_state_dict_config):
            checkpoint['model'] = model.state_dict()
        # Save optimizer state - using FullOptimStateDictConfig for optimizer
        optim_state_dict_config = FullStateDictConfig(offload_to_cpu=False, rank0_only=True)
        with FullyShardedDataParallel.state_dict_type(model, StateDictType.FULL_STATE_DICT, optim_state_dict_config):
            checkpoint['optimizer'] = FullyShardedDataParallel.optim_state_dict(model, optimizer)
        # Save to disk
        atomic_torch_save(
            checkpoint,
            checkpoint_path,
        )
        timer.report(f"Saved fsdp")
        # timer.report(f"Barrier for fsdp")
        # torch.distributed.barrier()
    else:
        state_dict = model.state_dict()
        atomic_torch_save(
            {
                "model": state_dict,
                "optimizer": optimizer.state_dict(),
                "train_sampler": train_dataloader.sampler.state_dict(),
                "test_sampler": test_dataloader.sampler.state_dict(),
                "metrics": metrics,
                "timer": timer,
            },
            checkpoint_path,
        )
    timer.report(f"Saving checkpoint into {checkpoint_path}")


def load_checkpoint(
    model,
    optimizer,
    train_dataloader,
    test_dataloader,
    metrics,
    timer,
    checkpoint_path,
    args,
    fsdp=False,
):
    timer.report(f"Loading checkpoint from {checkpoint_path}")
    if fsdp:
        timer.report(f"Loading fsdp")
        if not args.is_master:
            timer.report(f"NOT loading checkpoint from {checkpoint_path} since not rank 0")
            checkpoint = None
        else:
            timer.report(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
        # Load model state dict
        full_state_dict_config = FullStateDictConfig(offload_to_cpu=False, rank0_only=True)
        with FullyShardedDataParallel.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
            model_state_dict = checkpoint['model'] if args.is_master else None
            torch.distributed.broadcast_object_list([model_state_dict], src=0)
            model.load_state_dict(model_state_dict[0])
        # Load optimizer state dict
        optim_state_dict_config = FullStateDictConfig(offload_to_cpu=False, rank0_only=True)
        with FullyShardedDataParallel.state_dict_type(model, StateDictType.FULL_STATE_DICT, optim_state_dict_config):
            optim_state_dict = checkpoint['optimizer'] if args.is_master else None
            torch.distributed.broadcast_object_list([optim_state_dict], src=0)
            optimizer.load_state_dict(FullyShardedDataParallel.optim_state_dict_to_load(optim_state_dict[0], model, optimizer))
        # Load other components on rank 0 only
        if args.is_master:
            if checkpoint.get('train_dataloader') and hasattr(train_dataloader, 'load_state_dict'):
                train_dataloader.load_state_dict(checkpoint['train_dataloader'])
            if checkpoint.get('test_dataloader') and hasattr(test_dataloader, 'load_state_dict'):
                test_dataloader.load_state_dict(checkpoint['test_dataloader'])
            metrics.update(checkpoint['metrics'])
            timer.update(checkpoint['timer'])
        # Synchronize all processes
        # torch.distributed.barrier()
        timer.report(f"Loaded fsdp")
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer"])
    train_dataloader.sampler.load_state_dict(checkpoint["train_sampler"])
    test_dataloader.sampler.load_state_dict(checkpoint["test_sampler"])
    metrics = checkpoint["metrics"]
    timer = checkpoint["timer"]
    timer.start_time = time.time()
    if args.is_master:
        timer.report(f"Retrieved saved checkpoint from {checkpoint_path}")


def main(args, timer):
    # fsdp = True
    fsdp = False
    rank = int(os.environ["RANK"])  # Rank of this GPU in cluster
    world_size = int(os.environ["WORLD_SIZE"])  # Total number of GPUs in the cluster
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size
    )  # Expects RANK set in environment variable
    args.device_id = int(os.environ["LOCAL_RANK"])  # Rank on local node
    args.is_master = rank == 0  # Master node for saving / reporting
    torch.cuda.set_device(args.device_id)  # Enables calling 'cuda'
    torch.autograd.set_detect_anomaly(True)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    if args.device_id == 0:
        hostname = socket.gethostname()
        print("Hostname:", hostname)
        print(f"TrainConfig: {args}")
    timer.report("Setup for distributed training")

    saver = AtomicDirectory(output_directory=args.save_dir, is_master=args.is_master)
    timer.report("Validated checkpoint path")

    data_path = f"/data/{args.dataset_id}/lc0"
    dataset = PGN_HDF_Dataset(data_path)
    random_generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(
        dataset, [0.8, 0.2], generator=random_generator
    )
    timer.report(
        f"Intitialized datasets with {len(train_dataset):,} training and {len(test_dataset):,} test PGNs."
    )

    train_sampler = InterruptableDistributedSampler(train_dataset)
    test_sampler = InterruptableDistributedSampler(test_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        sampler=train_sampler,
        num_workers=4,  # Increase number of workers
        pin_memory=True,  # Enable pinned memory
        prefetch_factor=2,  # Prefetch batches
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.bs,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )
    timer.report("Prepared dataloaders")

    model_config = yaml.safe_load(open(args.model_config))
    if args.device_id == 0:
        print(f"ModelConfig: {model_config}")
    model_config["device"] = "cuda"
    model = Model(**model_config)
    model = model.to(args.device_id)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
    print(f"Initialized model with {params:,} params, moved to device")
    if fsdp:
        model = FullyShardedDataParallel(
            model,
            # use_orig_params=True,
            # mixed_precision=torch.distributed.fsdp.MixedPrecision(
            #     param_dtype=torch.float16,
            #     reduce_dtype=torch.float16,
            # ),
            device_id=args.device_id,
        )
    else:
        model = DDP(model, device_ids=[args.device_id], find_unused_parameters=True)
    # model = torch.compile(model)
    timer.report("Prepared model for distributed training")

    loss_fn = nn.CrossEntropyLoss(reduction="none")
    optimizer = Lamb(model.parameters(), lr=args.lr, weight_decay=args.wd)
    metrics = {"train": MetricsTracker(), "test": MetricsTracker()}

    if args.is_master:
        writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "tb"))

    checkpoint_path = None
    local_resume_path = os.path.join(args.save_dir, saver.symlink_name)
    if os.path.islink(local_resume_path):
        checkpoint = os.path.join(os.readlink(local_resume_path), "checkpoint.pt")
        if os.path.isfile(checkpoint):
            checkpoint_path = checkpoint
    elif args.load_path:
        if os.path.isfile(args.load_path):
            checkpoint_path = args.load_path
    timer.report(f"Checkpoint path: {checkpoint_path}")
    if checkpoint_path:
        load_checkpoint(
            model,
            optimizer,
            train_dataloader,
            test_dataloader,
            metrics,
            timer,
            checkpoint_path,
            args,
            fsdp=fsdp,
        )
        # if args.is_master:
        #     timer.report(f"Loading checkpoint from {checkpoint_path}")
        # # 1. Set proper state dict type for loading
        # with FullyShardedDataParallel.state_dict_type(
        #     model,
        #     StateDictType.FULL_STATE_DICT,
        #     state_dict_config=FullStateDictConfig(
        #         rank0_only=True
        #     ),
        # ):
        #     # 2. Load checkpoint on CPU first
        #     checkpoint = torch.load(checkpoint_path, map_location="cpu")
        #     # 3. Filter FSDP-specific prefixes if needed
        #     fixed_state_dict = {
        #         k.replace("_fsdp_wrapped_module.", ""): v
        #         for k, v in checkpoint["model"].items()
        #     }
        #     # 4. Load with strict=False to handle architecture changes
        #     model.load_state_dict(fixed_state_dict, strict=False)
        # # checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{args.device_id}")
        # # with FullyShardedDataParallel.summon_full_params(model):
        # #     model.load_state_dict(checkpoint["model"])
        # optimizer.load_state_dict(checkpoint["optimizer"])
        # train_dataloader.sampler.load_state_dict(checkpoint["train_sampler"])
        # test_dataloader.sampler.load_state_dict(checkpoint["test_sampler"])
        # metrics = checkpoint["metrics"]
        # timer = checkpoint["timer"]
        # timer.start_time = time.time()
        # timer.report("Retrieved saved checkpoint")

    # train loop
    for epoch in range(train_dataloader.sampler.epoch, 10_000):
        with train_dataloader.sampler.in_epoch(epoch):
            timer.report(f"Training epoch {epoch}")
            train_batches_per_epoch = len(train_dataloader)
            train_steps_per_epoch = math.ceil(train_batches_per_epoch / args.grad_accum)
            optimizer.zero_grad()
            model.train()
            ddp_loss = torch.zeros(2).to(args.device_id)

            for pgn_batch in train_dataloader:
                # Determine the current step
                batch = train_dataloader.sampler.progress // train_dataloader.batch_size
                is_save_batch = (batch + 1) % args.save_steps == 0
                is_accum_batch = (batch + 1) % args.grad_accum == 0
                is_last_batch = (batch + 1) == train_batches_per_epoch

                # Prepare checkpoint directory
                if (is_save_batch or is_last_batch) and args.is_master:
                    checkpoint_directory = saver.prepare_checkpoint_directory()

                logits, targets, target_pad_mask = model(pgn_batch)
                flat_logits = logits.flatten(end_dim=1)
                flat_targets = targets.flatten()
                flat_mask = torch.logical_not(target_pad_mask.flatten())
                loss = loss_fn(flat_logits, flat_targets) * flat_mask
                loss = loss.sum() / args.grad_accum
                loss.backward()

                train_dataloader.sampler.advance(len(pgn_batch))

                count_real, [top1_correct, top5_correct] = topk_accuracy(
                    flat_logits, flat_targets, ks=[1, 5], mask=flat_mask
                )

                char_probs = softmax(flat_logits)
                entropies = -char_probs * torch.log2(char_probs + 1e-8)
                total_prediction_entropy = entropies[flat_mask].sum()

                metrics["train"].update(
                    {
                        "gen_tokens": count_real,
                        "accum_loss": loss.item() * args.grad_accum,
                        "top1_correct": top1_correct.item(),
                        "top5_correct": top5_correct.item(),
                        "uncertainty": total_prediction_entropy.item(),
                    }
                )

                if is_accum_batch or is_last_batch:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    step = batch // args.grad_accum
                    ddp_loss[0] += loss.item()
                    ddp_loss[1] += len(pgn_batch)

                    # learning rate warmup
                    lr_factor = min((epoch * train_steps_per_epoch + step) / args.ws, 1)
                    next_lr = lr_factor * args.lr
                    for g in optimizer.param_groups:
                        g["lr"] = next_lr

                    metrics["train"].reduce()
                    rpt = metrics["train"].local
                    avg_loss = rpt["accum_loss"] / rpt["gen_tokens"]
                    rpt_top1 = 100 * rpt["top1_correct"] / rpt["gen_tokens"]
                    rpt_top5 = 100 * rpt["top5_correct"] / rpt["gen_tokens"]
                    rpt_uncertainty = rpt["uncertainty"] / rpt["gen_tokens"]
                    report = f"""\
Epoch [{epoch:,}] Step [{step:,} / {train_steps_per_epoch:,}] Batch [{batch:,} / {train_batches_per_epoch:,}] Lr: [{lr_factor * args.lr:,.3}], \
Avg Loss [{avg_loss:,.3f}], Top1: [{rpt_top1:,.3f}%], Top5: [{rpt_top5:,.3f}%], \
Uncertainty: [{rpt_uncertainty:,.3f}], Tokens: {rpt['gen_tokens']:,.0f}"""
                    timer.report(report)
                    metrics["train"].reset_local()

                    if args.is_master:
                        total_progress = batch + epoch * train_batches_per_epoch
                        writer.add_scalar("train/learn_rate", next_lr, total_progress)
                        writer.add_scalar("train/loss", avg_loss, total_progress)
                        writer.add_scalar(
                            "train/uncertainty", rpt_uncertainty, total_progress
                        )

                # Saving
                if (is_save_batch or is_last_batch) and args.is_master:
                    checkpoint_path = os.path.join(
                        checkpoint_directory, "checkpoint.pt"
                    )
                    save_checkpoint(
                        model,
                        optimizer,
                        train_dataloader,
                        test_dataloader,
                        metrics,
                        timer,
                        checkpoint_path,
                        args,
                        fsdp=fsdp,
                    )
                    saver.atomic_symlink(checkpoint_directory)

            if fsdp:
                dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
                if args.is_master:
                    timer.report(
                        "FSDP Train Epoch: {} \tLoss: {:.6f}".format(
                            epoch, ddp_loss[0] / ddp_loss[1]
                        )
                    )
            with test_dataloader.sampler.in_epoch(epoch):
                timer.report(f"Testing epoch {epoch}")
                test_batches_per_epoch = len(test_dataloader)
                model.eval()

                with torch.no_grad():
                    for pgn_batch in test_dataloader:
                        # Determine the current step
                        batch = (
                            test_dataloader.sampler.progress
                            // test_dataloader.batch_size
                        )
                        is_save_batch = (batch + 1) % args.save_steps == 0
                        is_last_batch = (batch + 1) == test_batches_per_epoch

                        # Prepare checkpoint directory
                        if (is_save_batch or is_last_batch) and args.is_master:
                            checkpoint_directory = saver.prepare_checkpoint_directory()

                        logits, targets, target_pad_mask = model(pgn_batch)

                        flat_logits = logits.flatten(end_dim=1)
                        flat_targets = targets.flatten()
                        flat_mask = torch.logical_not(target_pad_mask.flatten())
                        loss = (loss_fn(flat_logits, flat_targets) * flat_mask).sum()
                        test_dataloader.sampler.advance(len(pgn_batch))

                        count_real, [top1_correct, top5_correct] = topk_accuracy(
                            flat_logits, flat_targets, ks=[1, 5], mask=flat_mask
                        )

                        char_probs = softmax(flat_logits)
                        entropies = -char_probs * torch.log2(char_probs + 1e-8)
                        total_prediction_entropy = entropies[flat_mask].sum()

                        metrics["test"].update(
                            {
                                "gen_tokens": count_real,
                                "accum_loss": loss.item(),
                                "top1_correct": top1_correct.item(),
                                "top5_correct": top5_correct.item(),
                                "uncertainty": total_prediction_entropy.item(),
                            }
                        )

                        # Reporting
                        if is_last_batch:
                            metrics["test"].reduce()
                            rpt = metrics["test"].local
                            avg_loss = rpt["accum_loss"] / rpt["gen_tokens"]
                            rpt_top1 = 100 * rpt["top1_correct"] / rpt["gen_tokens"]
                            rpt_top5 = 100 * rpt["top5_correct"] / rpt["gen_tokens"]
                            rpt_uncertainty = rpt["uncertainty"] / rpt["gen_tokens"]
                            report = f"""\
Epoch [{epoch}] Evaluation, Avg Loss [{avg_loss:,.3f}], \
Top1 [{rpt_top1:,.3f}%], Top5 [{rpt_top5:,.3f}%], \
Uncertainty: [{rpt_uncertainty:,.3f}]"""
                            timer.report(report)
                            metrics["test"].reset_local()

                            if args.is_master:
                                writer.add_scalar("test/loss", avg_loss, epoch)
                                writer.add_scalar(
                                    "test/uncertainty", rpt_uncertainty, epoch
                                )

                        # Saving
                        if (is_save_batch or is_last_batch) and args.is_master:
                            timer.report(
                                f"Saving after test batch [{batch} / {test_batches_per_epoch}]"
                            )
                            checkpoint_path = os.path.join(
                                checkpoint_directory, "checkpoint.pt"
                            )
                            save_checkpoint(
                                model,
                                optimizer,
                                train_dataloader,
                                test_dataloader,
                                metrics,
                                timer,
                                checkpoint_path,
                                args,
                                fsdp=fsdp,
                            )
                            saver.atomic_symlink(checkpoint_directory)

                            # # Save checkpoint
                            # atomic_torch_save(
                            #     {
                            #         "model": model.module.state_dict(),
                            #         "optimizer": optimizer.state_dict(),
                            #         "train_sampler": train_dataloader.sampler.state_dict(),
                            #         "test_sampler": test_dataloader.sampler.state_dict(),
                            #         "metrics": metrics,
                            #         "timer": timer,
                            #     },
                            #     os.path.join(checkpoint_directory, "checkpoint.pt"),
                            # )
                            # saver.atomic_symlink(checkpoint_directory)


timer.report("Defined functions")
if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args, timer)
