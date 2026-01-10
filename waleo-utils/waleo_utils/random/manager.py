"""
随机数管理工具

提供可复现的随机数生成和状态管理
"""

import random
import numpy as np
import torch
from typing import Optional, Dict, Any
from pathlib import Path


class RNGManager:
    """随机数生成器管理器

    管理所有随机数生成器的状态，确保可复现性
    """

    def __init__(self, seed: Optional[int] = None):
        """初始化 RNG 管理器

        Args:
            seed: 随机种子
        """
        self._seed = seed
        if seed is not None:
            self.seed(seed)

    def seed(self, seed: int) -> None:
        """设置所有随机数生成器的种子

        Args:
            seed: 随机种子
        """
        self._seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # 如果 CUDA 可用，设置 CUDA 随机种子
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # 确保卷积操作的可复现性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @property
    def seed_value(self) -> Optional[int]:
        """获取当前种子值"""
        return self._seed

    def get_state(self) -> Dict[str, Any]:
        """获取所有 RNG 的当前状态

        Returns:
            Dict: 状态字典
        """
        state = {
            "seed": self._seed,
            "random": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }

        # 如果 CUDA 可用，保存 CUDA RNG 状态
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()

        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        """设置所有 RNG 的状态

        Args:
            state: 状态字典
        """
        self._seed = state.get("seed")
        random.setstate(state["random"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])

        # 如果 CUDA 状态存在，恢复 CUDA RNG 状态
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])

    def save_state(self, path: Path) -> None:
        """保存 RNG 状态到文件

        Args:
            path: 保存路径
        """
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self.get_state(), f)

    def load_state(self, path: Path) -> None:
        """从文件加载 RNG 状态

        Args:
            path: 文件路径
        """
        import pickle
        path = Path(path)

        with open(path, "rb") as f:
            state = pickle.load(f)

        self.set_state(state)

    def fork_rng(self, seed: Optional[int] = None) -> "ForkedRNG":
        """创建分支 RNG

        Args:
            seed: 新 RNG 的种子，如果为 None 则使用当前种子

        Returns:
            ForkedRNG: 分支的 RNG 上下文管理器
        """
        if seed is None:
            seed = self._seed
        return ForkedRNG(seed)


class ForkedRNG:
    """分支 RNG 上下文管理器

    在临时上下文中使用不同的随机种子
    """

    def __init__(self, seed: int):
        """初始化分支 RNG

        Args:
            seed: 随机种子
        """
        self.seed = seed
        self._state = None

    def __enter__(self) -> "ForkedRNG":
        """进入上下文，保存当前状态并设置新种子"""
        self._state = {
            "random": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }

        if torch.cuda.is_available():
            self._state["torch_cuda"] = torch.cuda.get_rng_state_all()

        # 设置新种子
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文，恢复原状态"""
        random.setstate(self._state["random"])
        np.random.set_state(self._state["numpy"])
        torch.set_rng_state(self._state["torch"])

        if "torch_cuda" in self._state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self._state["torch_cuda"])


def set_seed(seed: int) -> None:
    """设置全局随机种子

    Args:
        seed: 随机种子
    """
    manager = RNGManager(seed)
    return manager


def get_seed() -> Optional[int]:
    """获取当前随机种子

    Returns:
        Optional[int]: 当前种子值
    """
    # 尝试从 torch 获取初始种子
    try:
        return torch.initial_seed()
    except Exception:
        return None


def seed_worker(worker_id: int) -> None:
    """为 DataLoader worker 设置种子

    Args:
        worker_id: worker ID
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
