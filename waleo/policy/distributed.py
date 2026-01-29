"""Distributed training utilities for policies."""

import os
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_ddp(rank: int, world_size: int, backend: str = "nccl"):
    """Initialize distributed training.

    Args:
        rank: Rank of current process
        world_size: Total number of processes
        backend: Backend to use ("nccl", "gloo", "mpi")
    """
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")

    # Initialize process group
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    # Set device
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    """Get current process rank.

    Returns:
        Rank (0 if not distributed)
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes.

    Returns:
        World size (1 if not distributed)
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if current process is main process (rank 0).

    Returns:
        True if main process
    """
    return get_rank() == 0


@contextmanager
def main_process_first():
    """Context manager to run code on main process first, then others.

    Useful for downloading datasets, initializing resources, etc.

    Example:
        with main_process_first():
            # This runs on rank 0 first, then others
            download_dataset()
    """
    if not is_main_process():
        # Wait for main process
        if dist.is_initialized():
            dist.barrier()

    yield

    if is_main_process():
        # Signal other processes to continue
        if dist.is_initialized():
            dist.barrier()


def wrap_policy_ddp(policy: torch.nn.Module, device_id: int) -> DDP:
    """Wrap policy with DistributedDataParallel.

    Args:
        policy: Policy to wrap
        device_id: GPU device ID

    Returns:
        DDP-wrapped policy
    """
    policy = policy.to(device_id)
    return DDP(policy, device_ids=[device_id], output_device=device_id)


def reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    """Reduce tensor across all processes.

    Args:
        tensor: Tensor to reduce
        average: Whether to average (vs sum)

    Returns:
        Reduced tensor
    """
    if not dist.is_initialized():
        return tensor

    # Clone to avoid modifying original
    tensor = tensor.clone()

    # All-reduce
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    if average:
        tensor /= get_world_size()

    return tensor


def reduce_dict(input_dict: dict[str, torch.Tensor], average: bool = True) -> dict[str, torch.Tensor]:
    """Reduce dictionary of tensors across all processes.

    Args:
        input_dict: Dictionary of tensors
        average: Whether to average (vs sum)

    Returns:
        Dictionary with reduced tensors
    """
    if not dist.is_initialized():
        return input_dict

    output_dict = {}
    for key, tensor in input_dict.items():
        output_dict[key] = reduce_tensor(tensor, average=average)

    return output_dict


def gather_tensors(tensor: torch.Tensor) -> list[torch.Tensor]:
    """Gather tensors from all processes to main process.

    Args:
        tensor: Tensor to gather

    Returns:
        List of tensors from all processes (only on rank 0, empty list on others)
    """
    if not dist.is_initialized():
        return [tensor]

    world_size = get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]

    dist.all_gather(gathered, tensor)

    return gathered


class DDPCheckpointManager:
    """Checkpoint manager for DDP training.

    Ensures that only the main process saves checkpoints, and all processes
    can load them correctly.
    """

    def __init__(self, save_dir: str):
        """Initialize checkpoint manager.

        Args:
            save_dir: Directory to save checkpoints
        """
        self.save_dir = save_dir

        if is_main_process():
            os.makedirs(save_dir, exist_ok=True)

    def save_checkpoint(
        self,
        policy: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict,
        filename: str = "checkpoint.pth",
    ):
        """Save checkpoint (only on main process).

        Args:
            policy: Policy (can be DDP-wrapped)
            optimizer: Optimizer
            epoch: Current epoch
            metrics: Training metrics
            filename: Checkpoint filename
        """
        if not is_main_process():
            return

        # Get underlying module if DDP-wrapped
        model = policy.module if isinstance(policy, DDP) else policy

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }

        save_path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, save_path)

    def load_checkpoint(
        self,
        policy: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        filename: str = "checkpoint.pth",
    ) -> dict:
        """Load checkpoint (on all processes).

        Args:
            policy: Policy (can be DDP-wrapped)
            optimizer: Optimizer (optional)
            filename: Checkpoint filename

        Returns:
            Checkpoint dictionary
        """
        load_path = os.path.join(self.save_dir, filename)

        # Load checkpoint
        checkpoint = torch.load(load_path, map_location=f"cuda:{get_rank()}")

        # Get underlying module if DDP-wrapped
        model = policy.module if isinstance(policy, DDP) else policy

        # Load state
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint


def print_rank_0(message: str):
    """Print message only on rank 0.

    Args:
        message: Message to print
    """
    if is_main_process():
        print(message)


def synchronize():
    """Synchronize all processes.

    Waits for all processes to reach this point before continuing.
    """
    if dist.is_initialized():
        dist.barrier()
