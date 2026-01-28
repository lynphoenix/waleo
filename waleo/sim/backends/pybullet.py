"""PyBullet 仿真后端

PyBullet 是一个开源的物理引擎，提供：
- 开源免费，易于安装
- 支持多种机器人模型（URDF）
- 内置多个机器人示例（Kuka, Panda, Jaco 等）
- 跨平台支持

安装: pip install pybullet
文档: https://pybullet.org/
"""

from typing import Any

from waleo.sim.backends.base import SimulationBackend, BackendUnavailableError, BackendCreateError


class PyBulletBackend(SimulationBackend):
    """PyBullet 仿真后端

    使用 pybullet-gymnasium 接口创建 PyBullet 环境。
    支持标准机器人操作任务。
    """

    import_name = "pybullet"

    @classmethod
    def create(cls, task: str, **kwargs) -> Any:
        """创建 PyBullet 环境

        Args:
            task: 任务名称
                - 使用 gym.make() 兼容的环境ID
                - 或自定义 URDF 文件路径（需要额外配置）
            **kwargs: 环境参数
                - num_envs: 并行环境数量
                - render_mode: "human", "rgb_array", None
                - physics: "gui" 或 "direct"（PyBullet 特定）
                - 普通参数会传递给 gym.make()

        Returns:
            环境实例

        Raises:
            BackendUnavailableError: pybullet 未安装
            BackendCreateError: 环境创建失败

        Examples:
            >>> env = PyBulletBackend.create("KukaBulletEnv-v0")
            >>> env = PyBulletBackend.create("AntPyBulletEnv-v0")
        """
        # 检查依赖
        try:
            cls._check_dependencies()
        except ImportError as e:
            raise BackendUnavailableError(cls.name(), cls.import_name) from e

        # 合并默认参数
        create_kwargs = cls.get_default_kwargs()
        create_kwargs.update(kwargs)

        # 延迟导入 gymnasium
        import gymnasium as gym

        # 创建环境
        try:
            env = gym.make(task, **create_kwargs)
            return env
        except Exception as e:
            raise BackendCreateError(cls.name(), task, str(e)) from e

    @classmethod
    def get_default_kwargs(cls) -> dict[str, Any]:
        """获取 PyBullet 默认参数"""
        return {
            # PyBullet 特定的默认参数可以在这里添加
        }

    @classmethod
    def list_available_tasks(cls) -> list[str]:
        """列出所有可用的 PyBullet 任务

        Returns:
            list[str]: 常用 PyBullet 环境名称列表
        """
        return [
            # pybullet-gymnasium 环境
            "KukaBulletEnv-v0",
            "KukaDiverseObjectGrasping-v0",
            "InvertedPendulumBulletEnv-v0",
            "InvertedDoublePendulumBulletEnv-v0",
            "InvertedPendulumSwingupBulletEnv-v0",
            "CartPoleBulletEnv-v0",
            "MorphingAntBulletEnv-v0",
            "WalkerBaseBulletEnv-v0",
            "HalfCheetahBulletEnv-v0",
            "AntBulletEnv-v0",
            "HopperBulletEnv-v0",
            "HumanoidBulletEnv-v0",
            "HumanoidFlagrunBulletEnv-v0",
            "HumanoidFlagrunHarderBulletEnv-v0",
            "ReacherBulletEnv-v0",
            "StrikerBulletEnv-v0",
            "ThrowerBulletEnv-v0",
            "PusherBulletEnv-v0",
            "RacecarBulletEnv-v0",
        ]
