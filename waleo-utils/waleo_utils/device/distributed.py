"""
分布式设备支持

提供单机多卡和多机多卡的分布式设备管理
"""

import os
import torch
import torch.distributed as dist
from typing import Optional

# 全局分布式状态
_DISTRIBUTED_INITIALIZED = False
_WORLD_SIZE = 1
_RANK = 0
_LOCAL_RANK = 0


def setup_distributed(backend: str = "nccl") -> None:
    """初始化分布式环境（单机多卡）

    自动检测环境变量并初始化进程组

    Args:
        backend: 通信后端 ("nccl", "gloo")

    Raises:
        RuntimeError: 如果初始化失败
    """
    global _DISTRIBUTED_INITIALIZED, _WORLD_SIZE, _RANK, _LOCAL_RANK

    if _DISTRIBUTED_INITIALIZED:
        return

    # 检查环境变量（由 torchrun 或启动脚本设置）
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        # 未设置环境变量，不使用分布式
        return

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # 初始化进程组
    if backend == "nccl" and not torch.cuda.is_available():
        raise RuntimeError("NCCL backend requires CUDA")

    try:
        dist.init_process_group(
            backend=backend,
            world_size=world_size,
            rank=rank,
        )
        _DISTRIBUTED_INITIALIZED = True
        _WORLD_SIZE = world_size
        _RANK = rank
        _LOCAL_RANK = local_rank

        # 设置当前设备
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    except Exception as e:
        raise RuntimeError(f"Failed to initialize distributed: {e}")


def setup_distributed_multinode(
    init_method: str,
    world_size: int,
    rank: int,
    backend: str = "nccl",
    local_rank: Optional[int] = None
) -> None:
    """初始化多机分布式环境

    Args:
        init_method: 初始化方法，例如 "tcp://192.168.1.1:12345"
        world_size: 总进程数
        rank: 当前进程的 rank
        backend: 通信后端
        local_rank: 本地 rank（可选，默认为 rank % GPU 数）

    Raises:
        RuntimeError: 如果初始化失败
    """
    global _DISTRIBUTED_INITIALIZED, _WORLD_SIZE, _RANK, _LOCAL_RANK

    if _DISTRIBUTED_INITIALIZED:
        return

    if local_rank is None:
        # 自动推断 local_rank
        if torch.cuda.is_available():
            local_rank = rank % torch.cuda.device_count()
        else:
            local_rank = 0

    try:
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )
        _DISTRIBUTED_INITIALIZED = True
        _WORLD_SIZE = world_size
        _RANK = rank
        _LOCAL_RANK = local_rank

        # 设置当前设备
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    except Exception as e:
        raise RuntimeError(f"Failed to initialize multinode distributed: {e}")


def get_world_size() -> int:
    """获取进程总数

    Returns:
        int: 总进程数（未初始化分布式时返回 1）
    """
    if _DISTRIBUTED_INITIALIZED:
        return _WORLD_SIZE
    return 1


def get_rank() -> int:
    """获取当前进程 rank

    Returns:
        int: 当前进程 rank（未初始化分布式时返回 0）
    """
    if _DISTRIBUTED_INITIALIZED:
        return _RANK
    return 0


def get_local_rank() -> int:
    """获取本地进程 rank

    Returns:
        int: 本地进程 rank（未初始化分布式时返回 0）
    """
    if _DISTRIBUTED_INITIALIZED:
        return _LOCAL_RANK
    return 0


def is_main_process() -> bool:
    """是否为主进程

    Returns:
        bool: 是否为主进程（rank 0）
    """
    return get_rank() == 0


def barrier() -> None:
    """同步所有进程

    阻塞直到所有进程都调用此函数
    """
    if _DISTRIBUTED_INITIALIZED:
        dist.barrier()


def destroy_distributed() -> None:
    """清理分布式环境

    停止所有分布式操作
    """
    global _DISTRIBUTED_INITIALIZED, _WORLD_SIZE, _RANK, _LOCAL_RANK

    if _DISTRIBUTED_INITIALIZED:
        dist.destroy_process_group()
        _DISTRIBUTED_INITIALIZED = False
        _WORLD_SIZE = 1
        _RANK = 0
        _LOCAL_RANK = 0
