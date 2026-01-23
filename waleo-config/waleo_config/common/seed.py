"""
随机种子配置
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class SeedConfig:
    """"随机种子配置

    Args:
        seed: 随机种子（None 表示不固定）
        cuda_deterministic: 是否使用 CUDA 确定性模式
        benchmark: 是否启用 cudnn.benchmark
    """
    seed: int = 1337
    cuda_deterministic: bool = True
    benchmark: bool = False

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "SeedConfig":
        """从字典创建配置"""
        return cls(**config_dict)

    def validate(self) -> bool:
        """验证配置"""
        if self.seed is not None and self.seed < 0:
            return False
        return True
