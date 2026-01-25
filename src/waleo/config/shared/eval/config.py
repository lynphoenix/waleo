"""
评估配置模块

提供策略评估相关的配置类
"""

import warnings
from dataclasses import dataclass
from typing import Any


@dataclass
class EvalConfig:
    """评估配置

    Args:
        n_episodes: 评估回合数
        batch_size: 并行环境数量
        use_async_envs: 使用异步环境（多进程）
        save_video: 保存评估视频
        video_fps: 视频帧率
    """
    n_episodes: int = 50
    batch_size: int = 50
    use_async_envs: bool = False
    save_video: bool = True
    video_fps: int = 30

    def __post_init__(self):
        """配置验证"""
        if self.n_episodes <= 0:
            raise ValueError(f"n_episodes must be positive, got {self.n_episodes}")

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.video_fps <= 0:
            raise ValueError(f"video_fps must be positive, got {self.video_fps}")

        if self.batch_size > self.n_episodes:
            warnings.warn(
                f"Eval batch size ({self.batch_size}) > n_episodes ({self.n_episodes}). "
                "Some environments will be unused."
            )

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "EvalConfig":
        """从字典创建配置"""
        return cls(**config_dict)

    def validate(self) -> bool:
        """验证配置"""
        try:
            self.__post_init__()
            return True
        except (ValueError, TypeError):
            return False
