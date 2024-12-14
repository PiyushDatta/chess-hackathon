from cycling_utils import TimestampedTimer

timer = TimestampedTimer("Imported TimestampedTimer")

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
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
from mcts import MCTS

timer.report("Completed imports")

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", help="model config path", type=Path, default="/root/chess-hackathon/model_config.yaml")
    parser.add_argument("--save-dir", help="save checkpoint path", type=Path, default=os.getenv("OUTPUT_PATH"))
    parser.add_argument("--load-path", help="path to checkpoint.pt file to resume from", type=Path, default="/root/chess-hackathon/recover/checkpoint.pt")
    parser.add_argument("--bs", help="batch size", type=int, default=1)
    parser.add_argument("--lr", help="learning rate", type=float, default=0.01)
    parser.add_argument("--wd", help="weight decay", type=float, default=0.01)
    parser.add_argument("--ws", help="learning rate warm up steps", type=int, default=1000)
    parser.add_argument("--grad-accum", help="gradient accumulation steps", type=int, default=10)
    parser.add_argument("--save-steps", help="saving interval steps", type=int, default=20)
    return parser

def augment_pgn_with_mcts(pgn_batch, mcts_moves):
    augmented_pgn_batch = []
    for pgn, move in zip(pgn_batch, mcts_moves):
        # Split the PGN to get the moves and the result
        moves_and_result = pgn.split('{')[0]
        moves_and_result = moves_and_result.strip()
        moves_and_result = moves_and_result.split(' ')
        result = moves_and_result[-1]
        # All moves except the result
        moves = ' '.join(moves_and_result[:-1])  
        moves = moves.strip()
        # To determine the current move number
        # Each move is split by space, so count half for the move number
        move_number = len(moves.split()) // 2
        # Insert the MCTS move with the correct move number
        mcts_move_with_number = f"{move_number + 1}.{move}"
        # Construct the augmented PGN: add the MCTS move and then append the result
        augmented_pgn = f"{moves} {mcts_move_with_number} {result}"
        augmented_pgn_batch.append(augmented_pgn)
    return augmented_pgn_batch

def main(args, timer):
    dist.init_process_group("nccl")  # Expects RANK set in environment variable
    rank = int(os.environ["RANK"])  # Rank of this GPU in cluster
    world_size = int(os.environ["WORLD_SIZE"]) # Total number of GPUs in the cluster
    args.device_id = int(os.environ["LOCAL_RANK"])  # Rank on local node
    args.is_master = rank == 0  # Master node for saving / reporting
    torch.cuda.set_device(args.device_id)  # Enables calling 'cuda'
    torch.autograd.set_detect_anomaly(True) 

    if args.device_id == 0:
        hostname = socket.gethostname()
        print("Hostname:", hostname)
        print(f"TrainConfig: {args}")
    timer.report("Setup for distributed training")

    local_gpu = True
    if local_gpu:
        args.save_dir = "/root/chess-hackathon/checkpoints"
        args.load_path = "/root/chess-hackathon/checkpoints/latest_pt"
        data_path = "/data/lc0"
    else:
        data_path = "/data/lc0"

    saver = AtomicDirectory(args.save_dir)
    timer.report("Validated checkpoint path")

    dataset = PGN_HDF_Dataset(data_path)
    timer.report(f"Intitialized dataset with {len(dataset):,} PGNs.")

    random_generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2], generator=random_generator)

    train_sampler = InterruptableDistributedSampler(train_dataset)
    test_sampler = InterruptableDistributedSampler(test_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=args.bs, sampler=test_sampler)
    timer.report("Prepared dataloaders")

    model_config = yaml.safe_load(open(args.model_config))
    if args.device_id == 0:
        print(f"ModelConfig: {model_config}")
    model_config["device"] = 'cuda'
    model = Model(**model_config)
    model = model.to(args.device_id)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
    timer.report(f"Initialized model with {params:,} params, moved to device")

    model = DDP(model, device_ids=[args.device_id])
    timer.report("Prepared model for distributed training")

    loss_fn = nn.CrossEntropyLoss(reduction='none')
    optimizer = Lamb(model.parameters(), lr=args.lr, weight_decay=args.wd)
    metrics = {"train": MetricsTracker(), "test": MetricsTracker()}

    checkpoint_path = None
    local_resume_path = os.path.join(args.save_dir, saver.symlink_name)
    if os.path.islink(local_resume_path):
        checkpoint = os.path.join(os.readlink(local_resume_path), "checkpoint.pt")
        if os.path.isfile(checkpoint):
            checkpoint_path = checkpoint  
    elif args.load_path:
        if os.path.isfile(args.load_path):
            checkpoint_path = args.load_path
    if checkpoint_path:
        if args.is_master:
            timer.report(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{args.device_id}")
        model.module.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        train_dataloader.sampler.load_state_dict(checkpoint["train_sampler"])
        test_dataloader.sampler.load_state_dict(checkpoint["test_sampler"])
        metrics = checkpoint["metrics"]
        timer = checkpoint["timer"]
        timer.start_time = time.time()
        timer.report("Retrieved saved checkpoint")

    for epoch in range(train_dataloader.sampler.epoch, 10_000):
        with train_dataloader.sampler.in_epoch(epoch):
            timer.report(f"Training epoch {epoch}")
            train_batches_per_epoch = len(train_dataloader)
            train_steps_per_epoch = math.ceil(train_batches_per_epoch / args.grad_accum)
            optimizer.zero_grad()
            model.train()
            model.compile()
            mcts = MCTS(model, timer=timer, device=f"cuda:{args.device_id}", num_simulations=10)

            for pgn_batch in train_dataloader:

                # Determine the current step
                batch = train_dataloader.sampler.progress // train_dataloader.batch_size
                is_save_batch = (batch + 1) % args.save_steps == 0
                is_accum_batch = (batch + 1) % args.grad_accum == 0
                is_last_batch = (batch + 1) == train_batches_per_epoch

                # Prepare checkpoint directory
                if (is_save_batch or is_last_batch) and args.is_master:
                    checkpoint_directory = saver.prepare_checkpoint_directory()

                # timer.report("MCTS making moves")
                mcts_moves = mcts.select_best_moves(pgn_batch)
                # Augment the PGN batch with MCTS moves
                # timer.report(f"Augmenting moves {mcts_moves} to pgn batch")
                augmented_pgn_batch = augment_pgn_with_mcts(pgn_batch, mcts_moves)
                # timer.report(f"Augmented board: {augmented_pgn_batch}")
                # print(f"BATCH\npgn_batch:{pgn_batch}\nmcts_moves:{mcts_moves}\naugmented_pgn_batch:{augmented_pgn_batch}\n")
                # timer.report("Having model predict moves based on augmented pgn batch")
                logits, targets, target_pad_mask = model(augmented_pgn_batch)
                
                flat_logits = logits.flatten(end_dim=1)
                flat_targets = targets.flatten()
                flat_mask = torch.logical_not(target_pad_mask.flatten())
                loss = loss_fn(flat_logits, flat_targets) * flat_mask
                loss = loss.sum() / args.grad_accum

                loss.backward()
                train_dataloader.sampler.advance(len(pgn_batch))

                count_real, [top1_correct, top5_correct] = topk_accuracy(flat_logits, flat_targets, ks=[1, 5], mask=flat_mask)

                char_probs = softmax(flat_logits)
                entropies = -char_probs * torch.log2(char_probs + 1e-8)
                total_prediction_entropy = entropies[flat_mask].sum()

                metrics["train"].update({
                    "gen_tokens": count_real,
                    "accum_loss": loss.item() * args.grad_accum, 
                    "top1_correct": top1_correct.item(), 
                    "top5_correct": top5_correct.item(),
                    "uncertainty": total_prediction_entropy.item()
                })

                if is_accum_batch or is_last_batch:
                    optimizer.step()
                    optimizer.zero_grad()
                    step = batch // args.grad_accum
                    
                    # learning rate warmup
                    lr_factor = min((epoch * train_steps_per_epoch + step) / args.ws, 1)
                    for g in optimizer.param_groups:
                        g['lr'] = lr_factor * args.lr
                    
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

                # Saving
                if (is_save_batch or is_last_batch) and args.is_master:
                    # Save checkpoint
                    atomic_torch_save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "train_sampler": train_dataloader.sampler.state_dict(),
                            "test_sampler": test_dataloader.sampler.state_dict(),
                            "metrics": metrics,
                            "timer": timer
                        },
                        os.path.join(checkpoint_directory, "checkpoint.pt"),
                    )
                    saver.atomic_symlink(checkpoint_directory)

            with test_dataloader.sampler.in_epoch(epoch):
                timer.report(f"Testing epoch {epoch}")
                test_batches_per_epoch = len(test_dataloader)
                model.eval()

                with torch.no_grad():
                    for pgn_batch in test_dataloader:

                        # Determine the current step
                        batch = test_dataloader.sampler.progress // test_dataloader.batch_size
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

                        count_real, [top1_correct, top5_correct] = topk_accuracy(flat_logits, flat_targets, ks=[1, 5], mask=flat_mask)

                        char_probs = softmax(flat_logits)
                        entropies = -char_probs * torch.log2(char_probs + 1e-8)
                        total_prediction_entropy = entropies[flat_mask].sum()

                        metrics["test"].update({
                            "gen_tokens": count_real,
                            "accum_loss": loss.item(), 
                            "top1_correct": top1_correct.item(), 
                            "top5_correct": top5_correct.item(),
                            "uncertainty": total_prediction_entropy.item()
                        })
                        
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
                        
                        # Saving
                        if (is_save_batch or is_last_batch) and args.is_master:
                            # Save checkpoint
                            atomic_torch_save(
                                {
                                    "model": model.module.state_dict(),
                                    "optimizer": optimizer.state_dict(),
                                    "train_sampler": train_dataloader.sampler.state_dict(),
                                    "test_sampler": test_dataloader.sampler.state_dict(),
                                    "metrics": metrics,
                                    "timer": timer
                                },
                                os.path.join(checkpoint_directory, "checkpoint.pt"),
                            )
                            saver.atomic_symlink(checkpoint_directory)


timer.report("Defined functions")
if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args, timer)