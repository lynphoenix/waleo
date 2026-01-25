"""
训练配置模块

提供训练相关的配置类
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Tuple


@dataclass
class OptimizerConfig:
    """优化器配置"""
    type: str = "adam"
    lr: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0

    def __post_init__(self):
        """配置验证"""
        valid_types = ["adam", "adamw", "sgd"]
        if self.type not in valid_types:
            raise ValueError(
                f"Invalid optimizer type: {self.type}. " +
                f"Must be one of {valid_types}"
            )

        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")

        if len(self.betas) != 2:
            raise ValueError(f"betas must have 2 elements, got {len(self.betas)}")

        if not all(0 < b < 1 for b in self.betas):
            raise ValueError(f"betas must be in (0, 1), got {self.betas}")

        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}")

        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")


@dataclass
class SchedulerConfig:
    """学习率调度器配置"""
    type: str = "cosine"
    num_warmup_steps: int = 500
    min_lr_ratio: float = 0.0

    def __post_init__(self):
        """配置验证"""
        valid_types = ["cosine", "step", "constant"]
        if self.type not in valid_types:
            raise ValueError(
                f"Invalid scheduler type: {self.type}. " +
                f"Must be one of {valid_types}"
            )

        if self.num_warmup_steps < 0:
            raise ValueError(f"num_warmup_steps must be non-negative, got {self.num_warmup_steps}")

        if not 0 <= self.min_lr_ratio <= 1:
            raise ValueError(f"min_lr_ratio must be in [0, 1], got {self.min_lr_ratio}")


@dataclass
class CheckpointConfig:
    """检查点配置"""
    save_every: int = 5000
    save_total_limit: int = 3
    save_optimizer: bool = True

    def __post_init__(self):
        """配置验证"""
        if self.save_every <= 0:
            raise ValueError(f"save_every must be positive, got {self.save_every}")

        if self.save_total_limit <= 0:
            raise ValueError(f"save_total_limit must be positive, got {self.save_total_limit}")


@dataclass
class TrainingConfig:
    """训练配置

    Args:
        offline_steps: 离线训练步数
        online_steps: 在线训练步数
        rollout_n_episodes: 回放回合数
        rollout_batch_size: 回放批大小
        batch_size: 训练批大小
        num_workers: 数据加载工作进程数
        seed: 随机种子
        device: 训练设备
        log_freq: 日志记录频率
        eval_freq: 评估频率
        optimizer: 优化器配置
        scheduler: 学习率调度器配置
        checkpoint: 检查点配置
    """
    # 数据集相关
    offline_steps: int = 50000
    online_steps: int = 0
    rollout_n_episodes: int = 50
    rollout_batch_size: int = 8

    # 训练超参数
    batch_size: int = 64
    num_workers: int = 4
    seed: int = 1337

    # 设备和日志
    device: str = "cuda"
    log_freq: int = 100
    eval_freq: int = 5000

    # 子配置
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    def __post_init__(self):
        """配置验证"""
        if self.offline_steps <= 0:
            raise ValueError(f"offline_steps must be positive, got {self.offline_steps}")

        if self.online_steps < 0:
            raise ValueError(f"online_steps must be non-negative, got {self.online_steps}")

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {self.num_workers}")

        if self.device not in ["cuda", "mps", "cpu"]:
            raise ValueError(
                f"Invalid device: {self.device}. " +
                "Must be 'cuda', 'mps', or 'cpu'"
            )

        if self.log_freq <= 0:
            raise ValueError(f"log_freq must be positive, got {self.log_freq}")

        if self.eval_freq <= 0:
            raise ValueError(f"eval_freq must be positive, got {self.eval_freq}")

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        from dataclasses import asdict
        result = asdict(self)
        # 处理嵌套配置
        result["optimizer"] = asdict(self.optimizer)
        result["scheduler"] = asdict(self.scheduler)
        result["checkpoint"] = asdict(self.checkpoint)
        # 将 betas 元组转换为列表，以便 JSON/YAML 序列化
        if "betas" in result["optimizer"] and isinstance(result["optimizer"]["betas"], tuple):
            result["optimizer"]["betas"] = list(result["optimizer"]["betas"])
        return result

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TrainingConfig":
        """从字典创建配置"""
        # 复制以避免修改原始字典
        config_dict = config_dict.copy()
        
        # 处理嵌套配置
        if "optimizer" in config_dict and isinstance(config_dict["optimizer"], dict):
            opt_dict = config_dict["optimizer"]
            # 将 betas 列表转换为元组
            if "betas" in opt_dict and isinstance(opt_dict["betas"], list):
                opt_dict = opt_dict.copy()
                opt_dict["betas"] = tuple(opt_dict["betas"])
            config_dict["optimizer"] = OptimizerConfig(**opt_dict)

        if "scheduler" in config_dict and isinstance(config_dict["scheduler"], dict):
            config_dict["scheduler"] = SchedulerConfig(**config_dict["scheduler"])

        if "checkpoint" in config_dict and isinstance(config_dict["checkpoint"], dict):
            config_dict["checkpoint"] = CheckpointConfig(**config_dict["checkpoint"])

        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "TrainingConfig":
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
    def from_json(cls, path: Path | str) -> "TrainingConfig":
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
        # 创建一个临时配置来检查验证
        try:
            # 使用 from_dict 创建新实例，避免 __post_init__ 在构造时抛出异常
            config_dict = self.to_dict()
            TrainingConfig.from_dict(config_dict)
            return True
        except (ValueError, TypeError):
            return False
