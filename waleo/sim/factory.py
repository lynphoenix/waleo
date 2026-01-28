"""环境工厂函数

提供统一的环境创建接口，支持多种仿真后端。

所有后端通过相同的接口调用，内部路由到具体后端实现。
"""

from typing import Any, Optional, List, TYPE_CHECKING
import warnings

from waleo.sim.backends.base import SimulationBackend, BackendUnavailableError, BackendCreateError
from waleo.sim.backends.maniskill import ManiSkillBackend
from waleo.sim.backends.mujoco import MuJoCoBackend
from waleo.sim.backends.pybullet import PyBulletBackend
from waleo.sim.backends.isaacsim import IsaacSimBackend
from waleo.sim.registry.robot import get_robot_registry
from waleo.sim.wrappers.custom_robot import create_wrapped_env

if TYPE_CHECKING:
    import gymnasium as gym


# 后端注册表
_BACKENDS: dict[str, type[SimulationBackend]] = {
    "maniskill": ManiSkillBackend,
    "mujoco": MuJoCoBackend,
    "pybullet": PyBulletBackend,
    "isaacsim": IsaacSimBackend,
}


def register_backend(name: str, backend_cls: type[SimulationBackend]) -> None:
    """注册新的后端

    Args:
        name: 后端名称
        backend_cls: 后端类

    Example:
        >>> from waleo.sim.factory import register_backend
        >>> register_backend("my_backend", MyBackend)
    """
    _BACKENDS[name.lower()] = backend_cls


def get_backend(name: str) -> type[SimulationBackend]:
    """获取后端类

    Args:
        name: 后端名称

    Returns:
        后端类

    Raises:
        ValueError: 后端不存在

    Example:
        >>> backend_cls = get_backend("maniskill")
        >>> env = backend_cls.create("PickCube-v1")
    """
    name_lower = name.lower()
    if name_lower not in _BACKENDS:
        available = ", ".join(_BACKENDS.keys())
        raise ValueError(
            f"Unknown backend: {name}. Available backends: {available}"
        )
    return _BACKENDS[name_lower]


def make_env(
    task: str,
    robot: str = "panda",
    backend: str = "maniskill",
    num_envs: int = 1,
    robot_id: Optional[str] = None,
    **kwargs
) -> Any:
    """创建仿真环境（统一入口）

    设计原则：通用参数放在函数签名中，后端特定参数通过 **kwargs 传递。

    通用参数:
        task: 任务名称（各后端定义不同）
        robot: 内置机器人名称（如 "panda"）
        backend: 仿真后端 ("maniskill", "mujoco", "pybullet")
        num_envs: 并行环境数量
        robot_id: 自定义机器人 ID（如 "RJ2506"）

    后端特定参数 (通过 kwargs 传递):
        ManiSkill:
            obs_mode: "state", "state_dict", "rgbd", "pointcloud"
            control_mode: "pd_joint_pos", "pd_ee_delta_pose"
            render_mode: "human", "rgb_array", None
        MuJoCo:
            frame_skip: int
            render_mode: "human", "rgb_array", None
        PyBullet:
            physics: "gui", "direct"
            render_mode: "human", "rgb_array", None

    Args:
        task: 任务名称
        robot: 内置机器人名称
        backend: 仿真后端
        num_envs: 并行环境数量
        robot_id: 自定义机器人 ID，优先级高于 robot
        **kwargs: 后端特定参数

    Returns:
        gym.Env: 配置好的环境实例

    Raises:
        ValueError: 后端不支持
        BackendUnavailableError: 后端依赖未安装
        BackendCreateError: 环境创建失败

    Examples:
        >>> # ManiSkill (使用 obs_mode, control_mode)
        >>> env = make_env("PickCube-v1", obs_mode="state", control_mode="pd_joint_pos")

        >>> # MuJoCo (使用 frame_skip)
        >>> env = make_env("Ant-v4", backend="mujoco", frame_skip=5)

        >>> # 使用自定义机器人
        >>> env = make_env("PickCube-v1", robot_id="RJ2506", obs_mode="state")

        >>> # 多环境并行
        >>> env = make_env("PickCube-v1", num_envs=64)
    """
    # 处理机器人 ID（优先使用 robot_id）
    robot_id = robot_id or robot
    robot_id_upper = robot_id.upper()

    # 获取注册表
    registry = get_robot_registry()
    is_custom_robot = registry.is_registered(robot_id_upper)

    # 构建基础创建参数（通用参数）
    create_kwargs = {
        "num_envs": num_envs,
    }

    # 如果是自定义机器人，应用 robot spec 中的默认参数
    if is_custom_robot:
        spec = registry.get(robot_id_upper)
        # spec.default_kwargs 包含后端特定的默认参数
        # 例如：{"control_mode": "pd_joint_pos"} for ManiSkill
        #       {"frame_skip": 5} for MuJoCo
        if spec.default_kwargs:
            create_kwargs.update(spec.default_kwargs)

    # 合并用户传入的后端特定参数（用户参数优先级更高）
    create_kwargs.update(kwargs)

    # 添加后端特定的机器人参数
    # 注意：这里需要在后端 create 方法中处理不同后端的 robot 参数格式
    if backend == "maniskill":
        # ManiSkill 使用 robot_uids (小写)
        create_kwargs["robot_uids"] = robot_id.lower()
    elif backend in ("mujoco", "pybullet"):
        # MuJoCo/PyBullet 可能使用不同的参数名
        # 具体由各后端处理
        create_kwargs["robot_name"] = robot_id.lower()

    # 获取后端并创建环境
    backend_cls = get_backend(backend)

    try:
        env = backend_cls.create(task, **create_kwargs)
    except BackendUnavailableError as e:
        raise
    except BackendCreateError as e:
        raise
    except Exception as e:
        raise BackendCreateError(backend, task, str(e)) from e

    # 应用自定义机器人配置
    if is_custom_robot:
        spec = registry.get(robot_id_upper)
        task_config = spec.get_task_config(task)

        if task_config:
            env = create_wrapped_env(env, robot_id_upper, task, task_config)
        else:
            available = list(spec.task_configs.keys())
            warnings.warn(
                f"Robot '{robot_id_upper}' has no config for task '{task}'. "
                f"Available tasks: {available}. Using default config.",
                UserWarning
            )

    return env


def list_available_backends() -> List[str]:
    """列出所有可用的后端

    Returns:
        后端名称列表

    Example:
        >>> list_available_backends()
        ['maniskill', 'mujoco', 'pybullet']
    """
    return list(_BACKENDS.keys())


def list_available_robots() -> List[str]:
    """列出所有可用的机器人

    包括：
    - ManiSkill 内置机器人
    - Waleo 注册的自定义机器人

    Returns:
        机器人名称列表

    Example:
        >>> robots = list_available_robots()
        >>> print(robots)
        ['ALLEGRO_HAND', 'DCLAW', 'FETCH', 'PANDA', 'RJ2506', 'XARM7']
    """
    registry = get_robot_registry()
    custom_robots = registry.list_robots()

    # ManiSkill 内置机器人
    builtin_robots = [
        "PANDA",
        "FETCH",
        "XARM7",
        "ALLEGRO_HAND",
        "DCLAW",
    ]

    # 合并去重
    all_robots = sorted(set(custom_robots + builtin_robots))
    return all_robots


def list_available_tasks(backend: str = "maniskill") -> List[str]:
    """列出指定后端的可用任务

    Args:
        backend: 后端名称

    Returns:
        任务名称列表

    Example:
        >>> tasks = list_available_tasks("maniskill")
        >>> print(tasks[:5])
        ['ManiSkillPickCube-v1', 'ManiSkillPushCube-v1', ...]
    """
    backend_cls = get_backend(backend)

    # 如果后端有 list_available_tasks 方法，使用它
    if hasattr(backend_cls, "list_available_tasks"):
        return backend_cls.list_available_tasks()

    # 否则尝试从 gym 注册表获取
    try:
        import gymnasium as gym
        all_envs = gym.envs.registry.keys()
        tasks = [e for e in all_envs if not e.startswith("__")]
        return sorted(tasks)
    except Exception:
        return []


# 便捷别名
make = make_env
