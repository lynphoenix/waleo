"""
分布式状态管理

提供分布式训练状态的封装和管理
"""

import torch
import torch.distributed as dist
from typing import Optional


class DistributedState:
    """分布式状态管理类

    封装分布式训练的状态信息，提供便捷的访问接口
    """

    _instance: Optional["DistributedState"] = None

    def __new__(cls) -> "DistributedState":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化分布式状态"""
        if dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
            self._local_rank = int(
                float(torch.cuda.current_device())
                if torch.cuda.is_available() else 0
            )
        else:
            self._rank = 0
            self._world_size = 1
            self._local_rank = 0

    @property
    def rank(self) -> int:
        """当前进程 rank"""
        if dist.is_initialized():
            self._rank = dist.get_rank()
        return self._rank

    @property
    def world_size(self) -> int:
        """总进程数"""
        if dist.is_initialized():
            self._world_size = dist.get_world_size()
        return self._world_size

    @property
    def local_rank(self) -> int:
        """本地进程 rank"""
        if dist.is_initialized() and torch.cuda.is_available():
            self._local_rank = torch.cuda.current_device()
        return self._local_rank

    @property
    def is_distributed(self) -> bool:
        """是否使用分布式"""
        return dist.is_initialized() and self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        """是否为主进程"""
        return self.rank == 0

    def get_device(self) -> torch.device:
        """获取当前进程的设备"""
        if torch.cuda.is_available():
            return torch.device(f"cuda:{self.local_rank}")
        else:
            return torch.device("cpu")

    def barrier(self) -> None:
        """同步所有进程"""
        if dist.is_initialized():
            dist.barrier()

    @classmethod
    def get_instance(cls) -> "DistributedState":
        """获取单例实例

        Returns:
            DistributedState: 状态实例
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """重置单例实例"""
        cls._instance = None


# 便捷函数
def get_state() -> DistributedState:
    """获取分布式状态实例

    Returns:
        DistributedState: 状态实例
    """
    return DistributedState.get_instance()
