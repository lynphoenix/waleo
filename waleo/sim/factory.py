"""环境工厂函数

提供便捷的环境创建接口，自动处理自定义机器人配置。
"""

import gymnasium as gym
from typing import Optional, Dict, Any, Union, List
import warnings

from waleo.sim.registry.robot import get_robot_registry
from waleo.sim.wrappers.custom_robot import create_wrapped_env


def make_env(
    task: str,
    robot: str = "panda",
    backend: str = "maniskill",
    num_envs: int = 1,
    render_mode: Optional[str] = None,
    obs_mode: Optional[str] = None,
    control_mode: Optional[str] = None,
    sim_freq: int = 500,
    control_freq: int = 20,
    **kwargs
) -> gym.Env:
    """创建仿真环境（一行代码）

    自动处理自定义机器人配置，包括：
    - 机器人 pose 调整
    - Keyframes 配置
    - 物体配置
    - 相机配置

    Args:
        task: 任务名称，例如：
            - "PickCube-v1"
            - "StackCube-v1"
            - "PegInsertionSide-v1"
        robot: 机器人名称，例如：
            - "panda" (Franka Emika Panda，ManiSkill 内置)
            - "fetch" (Fetch Robot，ManiSkill 内置)
            - "rj2506" (自定义机器人)
            - "RJ2506" (大小写不敏感)
        backend: 仿真后端，支持：
            - "maniskill" (ManiSkill2/3，GPU 加速)
            - "mujoco" (MuJoCo)
            - "pybullet" (PyBullet)
        num_envs: 并行环境数量（GPU 仿真支持批处理）
        render_mode: 渲染模式，例如 "human", "rgb_array", "cameras"
        obs_mode: 观测模式，例如 "state", "rgbd", "pointcloud"
        control_mode: 控制模式，例如 "pd_joint_delta_pos", "pd_ee_delta_pose"
        sim_freq: 仿真频率 (Hz)
        control_freq: 控制频率 (Hz)
        **kwargs: 传递给 gym.make() 的其他参数

    Returns:
        gym.Env: 配置好的环境实例

    Examples:
        >>> # 使用内置机器人
        >>> env = make_env("PickCube-v1", robot="panda", num_envs=1)

        >>> # 使用自定义机器人（自动应用配置）
        >>> env = make_env("PickCube-v1", robot="rj2506", num_envs=512)

        >>> # 指定观测和控制模式
        >>> env = make_env(
        ...     "PickCube-v1",
        ...     robot="panda",
        ...     obs_mode="rgbd",
        ...     control_mode="pd_ee_delta_pose",
        ...     num_envs=128
        ... )

        >>> # 启用可视化
        >>> env = make_env("PickCube-v1", robot="panda", render_mode="human")

    Raises:
        ValueError: 如果后端不支持或任务名称无效
        ImportError: 如果后端依赖未安装
    """
    # 标准化机器人名称（大小写不敏感）
    robot = robot.upper()

    # 获取机器人注册中心
    registry = get_robot_registry()

    # 检查是否是自定义机器人
    is_custom_robot = registry.is_registered(robot)

    # 构建 gym.make() 参数
    make_kwargs = {
        "num_envs": num_envs,
        "sim_freq": sim_freq,
        "control_freq": control_freq,
    }

    # 添加可选参数
    if render_mode is not None:
        make_kwargs["render_mode"] = render_mode
    if obs_mode is not None:
        make_kwargs["obs_mode"] = obs_mode
    if control_mode is not None:
        make_kwargs["control_mode"] = control_mode

    # 如果是自定义机器人，使用小写名称作为 robot_uids
    if is_custom_robot:
        spec = registry.get(robot)
        make_kwargs["robot_uids"] = robot.lower()

        # 如果 spec 指定了控制模式且用户未指定，使用 spec 的控制模式
        if control_mode is None and spec.control_mode:
            make_kwargs["control_mode"] = spec.control_mode
    else:
        # ManiSkill 内置机器人
        make_kwargs["robot_uids"] = robot.lower()

    # 合并用户提供的额外参数
    make_kwargs.update(kwargs)

    # 根据后端创建环境
    if backend == "maniskill":
        env = _make_maniskill_env(task, **make_kwargs)
    elif backend == "mujoco":
        env = _make_mujoco_env(task, **make_kwargs)
    elif backend == "pybullet":
        env = _make_pybullet_env(task, **make_kwargs)
    else:
        raise ValueError(
            f"Unsupported backend: {backend}. "
            f"Supported backends: maniskill, mujoco, pybullet"
        )

    # 如果是自定义机器人，应用包装器
    if is_custom_robot:
        spec = registry.get(robot)

        # 尝试获取任务特定配置
        task_config = spec.get_task_config(task)

        if task_config:
            # 应用所有包装器
            env = create_wrapped_env(env, robot, task_config)
        else:
            # 没有任务特定配置，给出警告
            warnings.warn(
                f"Robot '{robot}' does not have config for task '{task}'. "
                f"Available tasks: {list(spec.task_configs.keys())}. "
                f"Using default configuration.",
                UserWarning
            )

    return env


def _make_maniskill_env(task: str, **kwargs) -> gym.Env:
    """创建 ManiSkill 环境

    Args:
        task: 任务名称
        **kwargs: 传递给 gym.make() 的参数

    Returns:
        ManiSkill 环境实例

    Raises:
        ImportError: 如果 ManiSkill 未安装
    """
    try:
        import mani_skill.envs  # noqa
    except ImportError:
        raise ImportError(
            "ManiSkill backend requires mani_skill package. "
            "Install it with: pip install mani-skill"
        )

    # 创建 ManiSkill 环境
    env = gym.make(task, **kwargs)
    return env


def _make_mujoco_env(task: str, **kwargs) -> gym.Env:
    """创建 MuJoCo 环境

    Args:
        task: 任务名称
        **kwargs: 传递给 gym.make() 的参数

    Returns:
        MuJoCo 环境实例

    Raises:
        ImportError: 如果 MuJoCo 未安装
        NotImplementedError: 暂未实现
    """
    try:
        import mujoco  # noqa
    except ImportError:
        raise ImportError(
            "MuJoCo backend requires mujoco package. "
            "Install it with: pip install mujoco"
        )

    # TODO: 实现 MuJoCo 后端
    raise NotImplementedError("MuJoCo backend not yet implemented")


def _make_pybullet_env(task: str, **kwargs) -> gym.Env:
    """创建 PyBullet 环境

    Args:
        task: 任务名称
        **kwargs: 传递给 gym.make() 的参数

    Returns:
        PyBullet 环境实例

    Raises:
        ImportError: 如果 PyBullet 未安装
        NotImplementedError: 暂未实现
    """
    try:
        import pybullet  # noqa
    except ImportError:
        raise ImportError(
            "PyBullet backend requires pybullet package. "
            "Install it with: pip install pybullet"
        )

    # TODO: 实现 PyBullet 后端
    raise NotImplementedError("PyBullet backend not yet implemented")


def list_available_robots() -> List[str]:
    """列出所有可用的机器人

    包括：
    - ManiSkill 内置机器人
    - Waleo 注册的自定义机器人

    Returns:
        机器人名称列表

    Examples:
        >>> robots = list_available_robots()
        >>> print(robots)
        ['PANDA', 'FETCH', 'XARM7', 'RJ2506', ...]
    """
    registry = get_robot_registry()
    custom_robots = registry.list_robots()

    # ManiSkill 内置机器人列表（部分）
    # 完整列表见：https://maniskill.readthedocs.io/en/latest/user_guide/concepts/agents.html
    builtin_robots = [
        "PANDA",
        "FETCH",
        "XARM7",
        "ALLEGRO_HAND",
        "DCLAW",
    ]

    # 合并并去重
    all_robots = list(set(custom_robots + builtin_robots))
    all_robots.sort()

    return all_robots


def list_available_tasks(backend: str = "maniskill") -> List[str]:
    """列出指定后端的可用任务

    Args:
        backend: 仿真后端名称

    Returns:
        任务名称列表

    Examples:
        >>> tasks = list_available_tasks("maniskill")
        >>> print(tasks[:5])
        ['PickCube-v1', 'StackCube-v1', 'PegInsertionSide-v1', ...]
    """
    if backend == "maniskill":
        try:
            # 获取所有已注册的 ManiSkill 环境
            all_envs = gym.envs.registry.keys()
            maniskill_tasks = [
                env_id for env_id in all_envs
                if not env_id.startswith("__")  # 过滤内部环境
            ]
            return sorted(maniskill_tasks)
        except Exception as e:
            warnings.warn(f"Failed to list ManiSkill tasks: {e}", UserWarning)
            return []
    else:
        warnings.warn(
            f"Backend '{backend}' not yet supported for task listing",
            UserWarning
        )
        return []


# 便捷别名
make = make_env
