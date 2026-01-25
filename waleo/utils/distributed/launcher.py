"""
多进程启动器

提供多进程和多机训练的启动功能
"""

import os
import subprocess
import torch.multiprocessing as mp
from typing import Callable, Tuple, List, Dict, Optional


def launch_multi_process(
    num_gpus: int,
    func: Callable,
    args: Tuple = (),
) -> None:
    """启动多进程训练（单机多卡）

    Args:
        num_gpus: GPU 数量
        func: 每个进程的训练函数，签名为 func(rank, world_size, *args)
        args: 传递给 func 的额外参数
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for multi-GPU training")

    if num_gpus > torch.cuda.device_count():
        raise ValueError(f"Requested {num_gpus} GPUs but only {torch.cuda.device_count()} available")

    # 设置环境变量
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # 启动进程
    mp.spawn(
        _worker_fn,
        args=(num_gpus, func, args),
        nprocs=num_gpus,
        join=True,
    )


def _worker_fn(
    rank: int,
    num_gpus: int,
    func: Callable,
    args: Tuple,
) -> None:
    """Worker 进程入口函数

    Args:
        rank: 当前进程 rank
        num_gpus: 总 GPU 数量
        func: 训练函数
        args: 额外参数
    """
    # 设置环境变量
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(num_gpus)
    os.environ["LOCAL_RANK"] = str(rank)

    # 调用训练函数
    func(rank, num_gpus, *args)


def launch_multinode(
    nodes_config: List[Dict[str, any]],
    func: Callable,
    args: Tuple = (),
) -> None:
    """启动多机多卡训练

    Args:
        nodes_config: 节点配置列表，每个节点包含：
            - host: 主机地址
            - num_gpus: GPU 数量
        func: 每个进程的训练函数
        args: 传递给 func 的额外参数
    """
    if not nodes_config:
        raise ValueError("nodes_config cannot be empty")

    # 主节点信息
    master_node = nodes_config[0]
    master_addr = master_node["host"]
    master_port = "12355"

    # 计算总进程数
    world_size = sum(node["num_gpus"] for node in nodes_config)

    # 在当前节点启动进程
    current_rank = 0
    for node in nodes_config:
        if node["host"] == _get_hostname():
            # 这是当前节点，启动进程
            num_gpus = node["num_gpus"]
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = master_port

            mp.spawn(
                _multinode_worker_fn,
                args=(num_gpus, current_rank, world_size, func, args),
                nprocs=num_gpus,
                join=True,
            )
            break

        current_rank += node["num_gpus"]


def _multinode_worker_fn(
    local_rank: int,
    num_gpus: int,
    rank_offset: int,
    world_size: int,
    func: Callable,
    args: Tuple,
) -> None:
    """多机 Worker 进程入口函数

    Args:
        local_rank: 本地 rank
        num_gpus: 本地 GPU 数量
        rank_offset: 全局 rank 偏移量
        world_size: 总进程数
        func: 训练函数
        args: 额外参数
    """
    # 计算全局 rank
    global_rank = rank_offset + local_rank

    # 设置环境变量
    os.environ["RANK"] = str(global_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)

    # 调用训练函数
    func(global_rank, world_size, *args)


def run_main(
    rank: int,
    world_size: int,
    func: Callable,
    args: Tuple = (),
) -> None:
    """单进程主函数

    用于分布式训练的单进程入口

    Args:
        rank: 当前进程 rank
        world_size: 总进程数
        func: 训练函数
        args: 额外参数
    """
    func(rank, world_size, *args)


def _get_hostname() -> str:
    """获取当前主机名

    Returns:
        str: 主机名
    """
    import socket
    return socket.gethostname()
