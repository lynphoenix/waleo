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

# 工厂函数（一行创建环境）
from waleo.sim.factory import (
    make_env,
    make,
    list_available_robots,
    list_available_tasks,
)

# 机器人注册中心
from waleo.sim.registry import (
    get_robot_registry,
    get_asset_resolver,
)

# 包装器
from waleo.sim.wrappers import (
    CustomRobotWrapper,
    TaskConfigWrapper,
    CameraConfigWrapper,
    create_wrapped_env,
)

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
    # 工厂函数（推荐使用）
    "make_env",
    "make",
    "list_available_robots",
    "list_available_tasks",
    # 注册中心
    "get_robot_registry",
    "get_asset_resolver",
    # 包装器（高级用法）
    "CustomRobotWrapper",
    "TaskConfigWrapper",
    "CameraConfigWrapper",
    "create_wrapped_env",
]
