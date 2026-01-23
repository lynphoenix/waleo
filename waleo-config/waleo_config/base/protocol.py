"""
配置协议定义

定义配置类的接口和行为协议
"""

from typing import Protocol, Any, runtime_checkable
from pathlib import Path


@runtime_checkable
class ConfigProtocol(Protocol):
    """配置对象协议

    任何配置类都应该实现此协议
    """
    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        ...

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "ConfigProtocol":
        """从字典创建配置"""
        ...

    def validate(self) -> bool:
        """验证配置"""
        ...


@runtime_checkable
class FileConfigProtocol(Protocol):
    """文件配置协议

    支持从文件加载和保存配置
    """
    @classmethod
    def from_yaml(cls, path: Path | str) -> "FileConfigProtocol":
        """从 YAML 文件加载"""
        ...

    @classmethod
    def from_json(cls, path: Path | str) -> "FileConfigProtocol":
        """从 JSON 文件加载"""
        ...

    def to_yaml(self, path: Path | str) -> None:
        """保存为 YAML 文件"""
        ...

    def to_json(self, path: Path | str) -> None:
        """保存为 JSON 文件"""
        ...
