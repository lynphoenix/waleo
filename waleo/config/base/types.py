"""
基础类型定义

提供配置管理系统中使用的枚举类型和数据类
"""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Tuple, Any, runtime_checkable


class FeatureType(Enum):
    """特征类型枚举"""
    STATE = "STATE"
    VISUAL = "VISUAL"
    ENV = "ENV"
    ACTION = "ACTION"
    REWARD = "REWARD"


class NormalizationMode(Enum):
    """归一化模式枚举"""
    MIN_MAX = "MIN_MAX"
    MEAN_STD = "MEAN_STD"
    IDENTITY = "IDENTITY"


@dataclass
class PolicyFeature:
    """策略特征定义

    Args:
        type: 特征类型
        shape: 特征形状
    """
    type: FeatureType
    shape: Tuple[int, ...]

    def __post_init__(self):
        """验证特征定义"""
        if not isinstance(self.type, FeatureType):
            raise ValueError(f"type must be FeatureType, got {type(self.type)}")

        if not isinstance(self.shape, tuple):
            raise ValueError(f"shape must be tuple, got {type(self.shape)}")

        if len(self.shape) == 0:
            raise ValueError("shape cannot be empty")

        if any(dim <= 0 for dim in self.shape):
            raise ValueError(f"shape dimensions must be positive, got {self.shape}")


@runtime_checkable
class DictLike(Protocol):
    """类字典接口协议

    任何实现了 __getitem__ 和 __contains__ 的对象都符合此协议
    """
    def __getitem__(self, key: str) -> Any: ...
    def __contains__(self, key: str) -> bool: ...
