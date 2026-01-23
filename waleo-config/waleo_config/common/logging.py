"""
日志配置
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class LogConfig:
    """"日志配置

    Args:
        level: 日志级别
        log_dir: 日志目录
        use_tensorboard: 是否使用 TensorBoard
        use_wandb: 是否使用 Weights & Biases
        console_output: 是否输出到控制台
    """
    level: str = "INFO"
    log_dir: Optional[str] = None
    use_tensorboard: bool = True
    use_wandb: bool = False
    console_output: bool = True

    def __post_init__(self):
        """配置验证"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {self.level}")

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "LogConfig":
        """从字典创建配置"""
        return cls(**config_dict)

    def validate(self) -> bool:
        """验证配置"""
        try:
            self.__post_init__()
            return True
        except (ValueError, TypeError):
            return False
