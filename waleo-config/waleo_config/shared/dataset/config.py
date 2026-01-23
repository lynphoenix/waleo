"""
数据集配置模块

提供数据集相关的配置类
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Any


@dataclass
class ImageTransformsConfig:
    """图像变换配置"""
    enable: bool = True
    random_resize: bool = False
    random_crop: bool = False
    horizontal_flip: bool = False
    vertical_flip: bool = False


@dataclass
class DatasetConfig:
    """数据集配置

    Args:
        repo_id: 数据集仓库 ID (如 "waleo/pusht")
        root: 数据集存储根目录
        episodes: 指定使用的 episode 列表
        image_transforms: 图像变换配置
        revision: Hub 版本
        use_imagenet_stats: 使用 ImageNet 统计信息
        video_backend: 视频后端 ("pyav" 或 "video_codec")
    """
    # 必需参数
    repo_id: str

    # 可选参数
    root: Optional[str] = None
    episodes: Optional[List[int]] = None
    image_transforms: ImageTransformsConfig = field(default_factory=ImageTransformsConfig)
    revision: Optional[str] = None
    use_imagenet_stats: bool = True
    video_backend: str = "pyav"

    def __post_init__(self):
        """配置验证"""
        if self.video_backend not in ["pyav", "video_codec"]:
            raise ValueError(
                f"Invalid video_backend: {self.video_backend}. " +
                "Must be 'pyav' or 'video_codec'"
            )

        if not self.repo_id:
            raise ValueError("repo_id cannot be empty")

        if self.root is not None:
            root_path = Path(self.root)
            if root_path.exists() and not root_path.is_dir():
                raise ValueError(f"root must be a directory, got {self.root}")

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        from dataclasses import asdict
        result = asdict(self)
        # 处理 ImageTransformsConfig 嵌套
        result["image_transforms"] = asdict(self.image_transforms)
        return result

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "DatasetConfig":
        """从字典创建配置"""
        # 处理 image_transforms 嵌套
        if "image_transforms" in config_dict and isinstance(config_dict["image_transforms"], dict):
            config_dict = config_dict.copy()
            config_dict["image_transforms"] = ImageTransformsConfig(**config_dict["image_transforms"])
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "DatasetConfig":
        """从 YAML 文件加载"""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required to load YAML files. Install with: pip install pyyaml")

        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_json(cls, path: Path | str) -> "DatasetConfig":
        """从 JSON 文件加载"""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_yaml(self, path: Path | str) -> None:
        """保存为 YAML 文件"""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required to save YAML files. Install with: pip install pyyaml")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    def to_json(self, path: Path | str) -> None:
        """保存为 JSON 文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def validate(self) -> bool:
        """验证配置"""
        try:
            self.__post_init__()
            return True
        except (ValueError, TypeError):
            return False
