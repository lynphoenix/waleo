"""Waleo 配置管理模块

提供统一的配置管理系统，支持数据类定义和类型验证。
"""

__version__ = "0.1.0"

# 环境配置
from waleo.config.env_config import EnvConfig, CameraConfig

__all__ = [
    "__version__",
    "EnvConfig",
    "CameraConfig",
]
