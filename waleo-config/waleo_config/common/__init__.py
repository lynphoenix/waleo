"""
通用配置模块

包含所有模块都使用的配置类
"""

from waleo_config.common.device import DeviceConfig
from waleo_config.common.logging import LogConfig
from waleo_config.common.path import PathConfig
from waleo_config.common.seed import SeedConfig

__all__ = [
    "DeviceConfig",
    "LogConfig",
    "PathConfig",
    "SeedConfig",
]
