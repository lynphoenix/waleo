"""
通用配置模块

包含所有模块都使用的配置类
"""

from waleo.config.common.device import DeviceConfig
from waleo.config.common.logging import LogConfig
from waleo.config.common.path import PathConfig
from waleo.config.common.seed import SeedConfig

__all__ = [
    "DeviceConfig",
    "LogConfig",
    "PathConfig",
    "SeedConfig",
]
