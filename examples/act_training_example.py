"""Example of training ACT policy with DDP support.

This example demonstrates:
1. Creating ACT policy with custom configuration
2. Setting up distributed training with DDP
3. Training loop with gradient accumulation
4. Checkpoint saving and loading
5. Evaluation

Usage:
    # Single GPU
    python act_training_example.py

    # Multi-GPU (4 GPUs)
    torchrun --nproc_per_node=4 act_training_example.py --distributed
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from waleo.policy import (
    ACTConfig,
    ACTPolicy,
    DDPCheckpointManager,
    cleanup_ddp,
    get_rank,
    get_world_size,
    is_main_process,
    print_rank_0,
    reduce_dict,
    setup_ddp,
    wrap_policy_ddp,
)


# Mock dataset for demonstration
class MockRobotDataset(Dataset):
    """Mock dataset for demonstration."""

    def __init__(self, num_samples: int = 1000, chunk_size: int = 10):
        self.num_samples = num_samples
        self.chunk_size = chunk_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "observation.images.top": torch.randn(3, 128, 128),
            "observation.state": torch.randn(14),
            "actions": torch.randn(self.chunk_size, 7),
        }


def create_dataloader(args, is_train=True):
    """Create dataloader with optional distributed sampler."""
    dataset = MockRobotDataset(
        num_samples=1000 if is_train else 200, chunk_size=args.chunk_size
    )

    sampler = None
    shuffle = is_train

    if args.distributed:
        sampler = DistributedSampler(
            dataset, num_replicas=get_world_size(), rank=get_rank(), shuffle=shuffle
        )
        shuffle = False  # Sampler handles shuffling

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return dataloader


def train_epoch(policy, dataloader, optimizer, device, args):
    """Train for one epoch."""
    policy.train()

    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        loss, logs = policy.forward(batch)

        # Backward pass
        loss = loss / args.gradient_accumulation_steps
        loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)

            optimizer.step()
            optimizer.zero_grad()

        # Accumulate metrics
        total_loss += logs["loss"]
        total_mae += logs["mae"]
        num_batches += 1

        # Log progress
        if is_main_process() and batch_idx % args.log_interval == 0:
            print(
                f"  Batch [{batch_idx}/{len(dataloader)}] "
                f"Loss: {logs['loss']:.4f} MAE: {logs['mae']:.4f}"
            )

    # Average metrics
    metrics = {
        "loss": torch.tensor(total_loss / num_batches).to(device),
        "mae": torch.tensor(total_mae / num_batches).to(device),
    }

    # Reduce metrics across processes if distributed
    if args.distributed:
        metrics = reduce_dict(metrics, average=True)

    return {k: v.item() for k, v in metrics.items()}


def evaluate(policy, dataloader, device, args):
    """Evaluate policy."""
    policy.eval()

    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            loss, logs = policy.forward(batch)

            # Accumulate metrics
            total_loss += logs["loss"]
            total_mae += logs["mae"]
            num_batches += 1

    # Average metrics
    metrics = {
        "loss": torch.tensor(total_loss / num_batches).to(device),
        "mae": torch.tensor(total_mae / num_batches).to(device),
    }

    # Reduce metrics across processes if distributed
    if args.distributed:
        metrics = reduce_dict(metrics, average=True)

    return {k: v.item() for k, v in metrics.items()}


def main(args):
    """Main training function."""

    # Setup distributed training
    if args.distributed:
        rank = int(torch.distributed.get_rank())
        world_size = int(torch.distributed.get_world_size())
        setup_ddp(rank, world_size, backend="nccl")
        device_id = rank
    else:
        rank = 0
        world_size = 1
        device_id = 0

    device = torch.device(f"cuda:{device_id}")

    print_rank_0("=" * 80)
    print_rank_0("ACT Policy Training Example")
    print_rank_0("=" * 80)
    print_rank_0(f"World size: {world_size}")
    print_rank_0(f"Device: {device}")

    # Create ACT configuration
    config = ACTConfig(
        name="act",
        input_shapes={
            "observation.images.top": (3, 128, 128),
            "observation.state": (14,),
        },
        output_shapes={"action": (7,)},
        device=str(device),
        # ACT-specific
        backbone="resnet18",
        backbone_pretrained=True,
        hidden_dim=512,
        nheads=8,
        enc_layers=4,
        dec_layers=1,
        dim_feedforward=3200,
        dropout=0.1,
        chunk_size=args.chunk_size,
        camera_names=["top"],
        state_dim=14,
        num_queries=args.chunk_size,
    )

    print_rank_0(f"\nACT Configuration:")
    print_rank_0(f"  Backbone: {config.backbone}")
    print_rank_0(f"  Hidden dim: {config.hidden_dim}")
    print_rank_0(f"  Chunk size: {config.chunk_size}")

    # Create policy
    policy = ACTPolicy(config)
    policy = policy.to(device)

    # Wrap with DDP if distributed
    if args.distributed:
        policy = wrap_policy_ddp(policy, device_id)
        print_rank_0("Policy wrapped with DDP")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        policy.module.get_optim_params() if args.distributed else policy.get_optim_params(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print_rank_0(f"\nOptimizer:")
    print_rank_0(f"  Learning rate: {args.learning_rate}")
    print_rank_0(f"  Weight decay: {args.weight_decay}")

    # Create dataloaders
    train_loader = create_dataloader(args, is_train=True)
    val_loader = create_dataloader(args, is_train=False)

    print_rank_0(f"\nDataloaders:")
    print_rank_0(f"  Train batches: {len(train_loader)}")
    print_rank_0(f"  Val batches: {len(val_loader)}")

    # Checkpoint manager
    checkpoint_manager = DDPCheckpointManager(args.save_dir)

    # Training loop
    print_rank_0("\n" + "=" * 80)
    print_rank_0("Starting Training")
    print_rank_0("=" * 80)

    best_val_loss = float("inf")

    for epoch in range(args.num_epochs):
        print_rank_0(f"\nEpoch {epoch + 1}/{args.num_epochs}")

        # Set epoch for distributed sampler
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        # Train
        train_metrics = train_epoch(policy, train_loader, optimizer, device, args)
        print_rank_0(
            f"  Train - Loss: {train_metrics['loss']:.4f} MAE: {train_metrics['mae']:.4f}"
        )

        # Evaluate
        val_metrics = evaluate(policy, val_loader, device, args)
        print_rank_0(
            f"  Val   - Loss: {val_metrics['loss']:.4f} MAE: {val_metrics['mae']:.4f}"
        )

        # Save checkpoint
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            checkpoint_manager.save_checkpoint(
                policy=policy,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"train": train_metrics, "val": val_metrics},
                filename="best_checkpoint.pth",
            )
            print_rank_0(f"  Saved best checkpoint (val_loss: {best_val_loss:.4f})")

        # Save periodic checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_manager.save_checkpoint(
                policy=policy,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"train": train_metrics, "val": val_metrics},
                filename=f"checkpoint_epoch_{epoch + 1}.pth",
            )

    print_rank_0("\n" + "=" * 80)
    print_rank_0("Training Complete!")
    print_rank_0(f"Best validation loss: {best_val_loss:.4f}")
    print_rank_0("=" * 80)

    # Cleanup
    if args.distributed:
        cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ACT Policy Training Example")

    # Training args
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps"
    )
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm")

    # ACT args
    parser.add_argument("--chunk-size", type=int, default=10, help="Action chunk size")

    # Data args
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")

    # Logging and saving
    parser.add_argument("--log-interval", type=int, default=10, help="Log interval (batches)")
    parser.add_argument("--save-interval", type=int, default=5, help="Save interval (epochs)")
    parser.add_argument(
        "--save-dir", type=str, default="./checkpoints/act", help="Checkpoint directory"
    )

    # Distributed args
    parser.add_argument(
        "--distributed", action="store_true", help="Use distributed training (DDP)"
    )

    args = parser.parse_args()

    main(args)
