"""Waleo 仿真环境模块

提供统一的仿真环境创建接口，支持多种后端。

推荐使用 make_env() 作为统一入口。
"""

__version__ = "0.2.0"

# 工厂函数（推荐使用）
from waleo.sim.factory import (
    make_env,
    make,
    list_available_robots,
    list_available_tasks,
    list_available_backends,
    get_backend,
    register_backend,
)

# 仿真后端
from waleo.sim.backends import (
    SimulationBackend,
    ManiSkillBackend,
    ManiSkill3Backend,  # 别名
    BackendError,
    BackendUnavailableError,
    BackendCreateError,
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

# 配置类
from waleo.sim.config import EnvConfig, CameraConfig

__all__ = [
    # 版本
    "__version__",
    # 工厂函数（推荐使用）
    "make_env",
    "make",
    "list_available_robots",
    "list_available_tasks",
    "list_available_backends",
    "get_backend",
    "register_backend",
    # 仿真后端
    "SimulationBackend",
    "ManiSkillBackend",
    "ManiSkill3Backend",
    "BackendError",
    "BackendUnavailableError",
    "BackendCreateError",
    # 注册中心
    "get_robot_registry",
    "get_asset_resolver",
    # 包装器（高级用法）
    "CustomRobotWrapper",
    "TaskConfigWrapper",
    "CameraConfigWrapper",
    "create_wrapped_env",
    # 配置
    "EnvConfig",
    "CameraConfig",
]
