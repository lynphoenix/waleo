"""
梯度同步工具

提供跨进程的梯度归约、广播和收集操作
"""

import torch
import torch.distributed as dist
from typing import List, Optional


def all_reduce(
    tensor: torch.Tensor,
    op: str = "avg"
) -> torch.Tensor:
    """跨进程归约张量

    Args:
        tensor: 要归约的张量
        op: 操作类型 ("avg", "sum", "max", "min")

    Returns:
        torch.Tensor: 归约后的张量

    Raises:
        RuntimeError: 如果分布式未初始化
        ValueError: 如果操作类型无效
    """
    if not dist.is_initialized():
        raise RuntimeError("Distributed is not initialized")

    if op == "avg":
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(dist.get_world_size())
    elif op == "sum":
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    elif op == "max":
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    elif op == "min":
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    else:
        raise ValueError(f"Unknown operation: {op}")

    return tensor


def broadcast(
    tensor: torch.Tensor,
    src: int = 0
) -> torch.Tensor:
    """从主进程广播张量

    Args:
        tensor: 要广播的张量
        src: 源进程 rank

    Returns:
        torch.Tensor: 广播后的张量

    Raises:
        RuntimeError: 如果分布式未初始化
    """
    if not dist.is_initialized():
        raise RuntimeError("Distributed is not initialized")

    dist.broadcast(tensor, src=src)
    return tensor


def all_gather(
    tensors: List[torch.Tensor]
) -> List[torch.Tensor]:
    """收集所有进程的张量

    Args:
        tensors: 每个进程的张量列表

    Returns:
        List[torch.Tensor]: 所有进程的张量列表

    Raises:
        RuntimeError: 如果分布式未初始化
    """
    if not dist.is_initialized():
        raise RuntimeError("Distributed is not initialized")

    if not tensors:
        return []

    world_size = dist.get_world_size()

    # 收集所有张量
    gathered = []
    for tensor in tensors:
        # 创建收集缓冲区
        shape = list(tensor.shape)
        shape[0] = shape[0] * world_size
        gathered_tensor = torch.zeros(shape, dtype=tensor.device, device=tensor.device)

        # 执行收集
        dist.all_gather_into_tensor(gathered_tensor, tensor)
        gathered.append(gathered_tensor)

    return gathered


def reduce_scalar(
    value: float,
    op: str = "avg"
) -> float:
    """归约标量值

    Args:
        value: 要归约的标量值
        op: 操作类型 ("avg", "sum")

    Returns:
        float: 归约后的值

    Raises:
        RuntimeError: 如果分布式未初始化
        ValueError: 如果操作类型无效
    """
    if not dist.is_initialized():
        return value

    tensor = torch.tensor([value], dtype=torch.float32)
    tensor = all_reduce(tensor, op=op)
    return tensor.item()
