"""
跨模块共享配置

包含多个模块共享但不是所有模块都使用的配置
"""

from waleo_config.shared.dataset.config import (
    DatasetConfig,
    ImageTransformsConfig,
)

from waleo_config.shared.training.config import (
    TrainingConfig,
    OptimizerConfig,
    SchedulerConfig,
    CheckpointConfig,
)

from waleo_config.shared.eval.config import EvalConfig

__all__ = [
    # 数据集配置 (M03/M13/M18 使用)
    "DatasetConfig",
    "ImageTransformsConfig",
    # 训练配置 (M13 使用)
    "TrainingConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "CheckpointConfig",
    # 评估配置 (M13/M18 使用)
    "EvalConfig",
]
