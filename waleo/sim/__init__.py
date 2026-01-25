"""Waleo 仿真环境模块

提供 Gym/Gymnasium 兼容的仿真环境基类和接口。
"""

__version__ = "0.1.0"

# 环境基类
from waleo.sim.base import BaseEnv, EnvWrapper, VectorEnv, RobotEnv

# 仿真后端
from waleo.sim.backends import (
    SimulationBackend,
    MuJoCoBackend,
    PyBulletBackend,
    ManiSkillBackend,
)

# 配置类
from waleo.sim.config import EnvConfig, CameraConfig

__all__ = [
    # 版本
    "__version__",
    # 环境基类
    "BaseEnv",
    "EnvWrapper",
    "VectorEnv",
    "RobotEnv",
    # 仿真后端
    "SimulationBackend",
    "MuJoCoBackend",
    "PyBulletBackend",
    "ManiSkillBackend",
    # 配置
    "EnvConfig",
    "CameraConfig",
]
