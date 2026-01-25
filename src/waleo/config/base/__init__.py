"""
基础配置类模块

提供配置管理的基础类型和协议
"""

from waleo.config.base.types import (
    FeatureType,
    NormalizationMode,
    PolicyFeature,
    DictLike,
)

from waleo.config.base.protocol import (
    ConfigProtocol,
    FileConfigProtocol,
)

__all__ = [
    # 类型
    "FeatureType",
    "NormalizationMode",
    "PolicyFeature",
    "DictLike",
    # 协议
    "ConfigProtocol",
    "FileConfigProtocol",
]
