"""仿真后端模块

提供多种仿真后端的统一抽象接口。

所有后端都继承自 SimulationBackend 基类，确保一致的调用方式。

示例:
    >>> from waleo.sim.backends import ManiSkillBackend, MuJoCoBackend
    >>> env = ManiSkillBackend.create("PickCube-v1", num_envs=8)
    >>> env = MuJoCoBackend.create("Ant-v4")

    >>> # 或通过 factory
    >>> from waleo.sim import make_env
    >>> env = make_env("PickCube-v1", backend="maniskill")
    >>> env = make_env("Ant-v4", backend="mujoco")
"""

from waleo.sim.backends.base import (
    SimulationBackend,
    BackendError,
    BackendUnavailableError,
    BackendCreateError,
)
from waleo.sim.backends.maniskill import ManiSkillBackend
from waleo.sim.backends.mujoco import MuJoCoBackend
from waleo.sim.backends.pybullet import PyBulletBackend
from waleo.sim.backends.isaacsim import IsaacSimBackend

__all__ = [
    # 抽象基类
    "SimulationBackend",
    # 异常
    "BackendError",
    "BackendUnavailableError",
    "BackendCreateError",
    # 后端实现
    "ManiSkillBackend",
    "MuJoCoBackend",
    "PyBulletBackend",
    "IsaacSimBackend",
    # 别名（向后兼容）
    "ManiSkill3Backend",
]

# 别名（向后兼容）
ManiSkill3Backend = ManiSkillBackend


def get_available_backends() -> list[str]:
    """获取所有可用的后端

    Returns:
        后端名称列表

    Example:
        >>> get_available_backends()
        ['maniskill', 'mujoco', 'pybullet', 'isaacsim']
    """
    backends = ["maniskill", "mujoco", "pybullet", "isaacsim"]
    # 过滤掉不可用的后端
    available = []
    backend_classes = {
        "maniskill": ManiSkillBackend,
        "mujoco": MuJoCoBackend,
        "pybullet": PyBulletBackend,
        "isaacsim": IsaacSimBackend,
    }
    for backend_name in backends:
        try:
            if backend_classes[backend_name].is_available():
                available.append(backend_name)
        except Exception:
            pass
    return available
