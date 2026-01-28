"""MuJoCo 仿真后端

MuJoCo 是一个高速物理仿真引擎，特别适合：
- 快速原型开发
- 大规模并行训练
- 经典控制任务（Ant, Hopper, Walker 等）

安装: pip install mujoco
文档: https://mujoco.readthedocs.io/
"""

from typing import Any

from waleo.sim.backends.base import SimulationBackend, BackendUnavailableError, BackendCreateError


class MuJoCoBackend(SimulationBackend):
    """MuJoCo 仿真后端

    使用 gymnasium 标准接口创建 MuJoCo 环境。
    支持 MuJoCo v3/v4/v5 系列环境。
    """

    import_name = "mujoco"

    @classmethod
    def create(cls, task: str, **kwargs) -> Any:
        """创建 MuJoCo 环境

        Args:
            task: 任务名称（MuJoCo 环境ID）
                - "Ant-v4", "Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"
                - "Humanoid-v4", "Swimmer-v4", "InvertedPendulum-v4"
                - 或自定义 XML 文件路径
            **kwargs: 环境参数
                - num_envs: 并行环境数量
                - frame_skip: 每个动作的物理步数（默认: 1）
                - render_mode: "human", "rgb_array", None
                - 普通参数会传递给 gym.make()

        Returns:
            环境实例

        Raises:
            BackendUnavailableError: mujoco 未安装
            BackendCreateError: 环境创建失败

        Examples:
            >>> env = MuJoCoBackend.create("Ant-v4")
            >>> env = MuJoCoBackend.create("Ant-v4", frame_skip=5)
            >>> env = MuJoCoBackend.create("custom.xml")
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
        """获取 MuJoCo 默认参数"""
        return {
            "frame_skip": 1,
        }

    @classmethod
    def list_available_tasks(cls) -> list[str]:
        """列出所有可用的 MuJoCo 任务

        Returns:
            list[str]: 常用 MuJoCo 环境名称列表
        """
        return [
            "Ant-v4",
            "Hopper-v4",
            "Walker2d-v4",
            "HalfCheetah-v4",
            "Humanoid-v4",
            "Swimmer-v4",
            "InvertedPendulum-v4",
            "InvertedDoublePendulum-v4",
            "Reacher-v4",
            "Pusher-v4",
            "Thrower-v4",
            "Striker-v4",
        ]
