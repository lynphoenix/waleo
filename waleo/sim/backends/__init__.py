"""仿真后端模块

提供多种仿真后端的统一接口。
"""

from waleo.sim.backends.backend import SimulationBackend
from waleo.sim.backends.mujoco import MuJoCoBackend
from waleo.sim.backends.pybullet import PyBulletBackend
from waleo.sim.backends.maniskill import ManiSkillBackend

__all__ = [
    "SimulationBackend",
    "MuJoCoBackend",
    "PyBulletBackend",
    "ManiSkillBackend",
]


def get_backend(backend_name: str) -> type:
    """根据名称获取后端类

    Args:
        backend_name: 后端名称 ("mujoco", "pybullet", "maniskill")

    Returns:
        backend_class: 后端类

    Raises:
        ValueError: 如果后端名称不支持
    """
    backends = {
        "mujoco": MuJoCoBackend,
        "pybullet": PyBulletBackend,
        "maniskill": ManiSkillBackend,
    }

    backend_class = backends.get(backend_name.lower())
    if backend_class is None:
        raise ValueError(
            f"Unsupported backend: {backend_name}. "
            f"Supported backends: {list(backends.keys())}"
        )

    return backend_class
