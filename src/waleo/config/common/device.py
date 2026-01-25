"""
设备配置

所有模块都使用的设备相关配置
"""

from dataclasses import dataclass
from typing import Any
from pathlib import Path


@dataclass
class DeviceConfig:
    """"设备配置

    Args:
        device_type: 设备类型 ("cuda", "mps", "cpu", "auto")
        cuda_device: CUDA 设备编号 (0, 1, ...)
        use_deterministic: 是否使用确定性算法（Torch）
        mixed_precision: 是否使用混合精度
    """
    device_type: str = "auto"  # auto, cuda, mps, cpu
    cuda_device: int = 0
    use_deterministic: bool = False
    mixed_precision: bool = False

    def __post_init__(self):
        """配置验证"""
        valid_types = ["auto", "cuda", "mps", "cpu"]
        if self.device_type not in valid_types:
            raise ValueError(
                f"Invalid device_type: {self.device_type}. " +
                f"Must be one of {valid_types}"
            )

    @property
    def device(self) -> str:
        """获取实际的设备字符串"""
        if self.device_type == "auto":
            import torch
            if torch.cuda.is_available():
                return f"cuda:{self.cuda_device}"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.device_type

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "DeviceConfig":
        """从字典创建配置"""
        return cls(**config_dict)

    def validate(self) -> bool:
        """验证配置"""
        try:
            self.__post_init__()
            return True
        except (ValueError, TypeError):
            return False
