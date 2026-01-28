"""ManiSkill3 仿真后端

ManiSkill3 是一个基于 SAPIEN 的机器人操作仿真框架，提供：
- 内置任务库（PickCube, PushCube, StackCube 等）
- GPU 加速的物理仿真
- 高质量渲染
- Gymnasium 标准接口

文档: https://maniskill.readthedocs.io/
"""

from typing import Any, TYPE_CHECKING

from waleo.sim.backends.base import SimulationBackend, BackendUnavailableError, BackendCreateError

if TYPE_CHECKING:
    import gymnasium as gym


class ManiSkillBackend(SimulationBackend):
    """ManiSkill3 仿真后端

    为了让 ManiSkill3 环境通过 gym.make() 使用，我们需要：
    1. 导入 ManiSkill 环境类
    2. 注册到 gymnasium 环境注册表
    """

    import_name = "mani_skill"

    # ManiSkill 环境类映射
    _ENV_CLASSES = {
        "PickCube-v1": "mani_skill.envs.tasks.tabletop.pick_cube:PickCubeEnv",
        "PushCube-v1": "mani_skill.envs.tasks.tabletop.push_cube:PushCubeEnv",
        "StackCube-v1": "mani_skill.envs.tasks.tabletop.stack_cube:StackCubeEnv",
    }

    @classmethod
    def _register_envs(cls):
        """注册 ManiSkill 环境到 gymnasium

        ManiSkill3 环境需要手动注册到 gymnasium 才能通过 gym.make() 使用。
        """
        import gymnasium as gym
        from gymnasium.envs.registration import register

        # 注册 PickCube
        try:
            from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
            register(
                id="PickCube-v1",
                entry_point="mani_skill.envs.tasks.tabletop.pick_cube:PickCubeEnv",
            )
        except Exception as e:
            pass  # 可能已经注册或其他错误

        # 注册其他环境可以在这里添加...

    @classmethod
    def create(cls, task: str, **kwargs) -> Any:
        """创建 ManiSkill3 环境

        Args:
            task: 任务名称
                - "PickCube-v1", "PushCube-v1", "StackCube-v1" 等
            **kwargs: 环境参数
                - num_envs: 并行环境数量 (默认: 1)
                - obs_mode: 观察模式
                    - "state": 扁平状态向量（默认，推荐用于 RL）
                    - "state_dict": 状态字典
                    - "rgbd": RGB-D 图像
                    - "pointcloud": 点云
                - control_mode: 控制模式
                    - "pd_joint_pos": 关节位置控制（默认）
                    - "pd_ee_delta_pose": 末端执行器增量姿态
                - robot_uids: 机器人 ID
                    - "panda": Franka Panda（默认）
                    - "rj2506": 自定义机器人

        Returns:
            环境实例

        Raises:
            BackendUnavailableError: mani_skill 未安装
            BackendCreateError: 环境创建失败
        """
        # 检查依赖
        try:
            cls._check_dependencies()
        except ImportError as e:
            raise BackendUnavailableError(cls.name(), cls.import_name) from e

        # 尝试注册环境
        cls._register_envs()

        # 标准化任务名称
        task = cls._normalize_task_name(task)

        # 合并默认参数
        create_kwargs = cls.get_default_kwargs()
        create_kwargs.update(kwargs)

        # 延迟导入 gymnasium
        import gymnasium as gym

        # 尝试通过 gym.make 创建
        try:
            env = gym.make(task, **create_kwargs)
            return env
        except Exception:
            # 如果 gym.make 失败，直接导入并创建
            return cls._create_direct(task, **create_kwargs)

    @classmethod
    def _create_direct(cls, task: str, **kwargs) -> Any:
        """直接创建环境（备用方法）

        当 gym.make() 不可用时使用。
        """
        # 解析任务名到环境类
        env_class_path = cls._ENV_CLASSES.get(task)
        if not env_class_path:
            raise BackendCreateError(
                cls.name(),
                task,
                f"Unknown task: {task}. Supported: {list(cls._ENV_CLASSES.keys())}"
            )

        # 动态导入
        module_path, class_name = env_class_path.split(":")
        from importlib import import_module
        module = import_module(module_path)
        env_class = getattr(module, class_name)

        # 创建环境
        return env_class(**kwargs)

    @classmethod
    def _normalize_task_name(cls, task: str) -> str:
        """标准化任务名称

        Args:
            task: 用户输入的任务名

        Returns:
            str: 标准化的任务名

        Examples:
            >>> _normalize_task_name("PickCube-v1")
            'PickCube-v1'
            >>> _normalize_task_name("PickCube")
            'PickCube-v1'
        """
        # 如果已经有版本号，直接返回
        if "-v" in task:
            return task

        # 如果没有版本号，添加默认版本 -v1
        return f"{task}-v1"

    @classmethod
    def get_default_kwargs(cls) -> dict[str, Any]:
        """获取 ManiSkill3 默认参数

        使用 "state" 模式（扁平观察）作为默认，更适合大多数 RL 算法。
        """
        return {
            "num_envs": 1,
            "obs_mode": "state",  # 扁平观察，兼容大多数 RL 算法
            "control_mode": "pd_joint_pos",
            "robot_uids": "panda",
        }

    @classmethod
    def list_available_tasks(cls) -> list[str]:
        """列出所有可用的 ManiSkill3 任务

        Returns:
            list[str]: 任务名称列表
        """
        return list(cls._ENV_CLASSES.keys())


# 为了向后兼容，保留别名
ManiSkill3Backend = ManiSkillBackend
