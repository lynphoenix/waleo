"""
路径配置
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PathConfig:
    """"路径配置

    Args:
        root_dir: 项目根目录
        data_dir: 数据目录
        cache_dir: 缓存目录
        output_dir: 输出目录
        checkpoint_dir: 检查点目录
    """
    root_dir: str = "."
    data_dir: str = "data"
    cache_dir: str = ".cache"
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"

    @property
    def full_data_dir(self) -> Path:
        """获取完整数据目录路径"""
        return Path(self.root_dir) / self.data_dir

    @property
    def full_cache_dir(self) -> Path:
        """获取完整缓存目录路径"""
        return Path(self.root_dir) / self.cache_dir

    @property
    def full_output_dir(self) -> Path:
        """获取完整输出目录路径"""
        return Path(self.root_dir) / self.output_dir

    @property
    def full_checkpoint_dir(self) -> Path:
        """获取完整检查点目录路径"""
        return Path(self.root_dir) / self.checkpoint_dir

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "PathConfig":
        """从字典创建配置"""
        return cls(**config_dict)

    def validate(self) -> bool:
        """验证配置"""
        return True
