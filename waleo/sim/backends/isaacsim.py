"""IsaacSim 仿真后端

IsaacSim 是 NVIDIA 开发的高性能机器人仿真器，提供：
- 基于 NVIDIA PhysX 的高保真物理仿真
- 与真实世界 1:1 的仿真精度
- GPU 加速的大规模并行仿真
- 支持 USD (Universal Scene Description) 格式
- 与 NVIDIA Omniverse 生态系统集成

安装: pip install isaacsim
文档: https://docs.omniverse.nvidia.com/isaacsim/
"""

from typing import Any

from waleo.sim.backends.base import SimulationBackend, BackendUnavailableError, BackendCreateError


class IsaacSimBackend(SimulationBackend):
    """IsaacSim 仿真后端

    使用 IsaacOrbit/IsaacGymens 接口创建 IsaacSim 环境。
    支持大规模并行机器人仿真。
    """

    import_name = "isaacsim"

    @classmethod
    def create(cls, task: str, **kwargs) -> Any:
        """创建 IsaacSim 环境

        Args:
            task: 任务名称
                - 使用 IsaacOrbit 任务格式
                - 或自定义任务配置
            **kwargs: 环境参数
                - num_envs: 并行环境数量
                - headless: 无头模式 (默认: True)
                - device: CUDA 设备 (默认: "cuda:0")
                - 普通参数会传递给任务创建函数

        Returns:
            环境实例

        Raises:
            BackendUnavailableError: isaacsim 未安装
            BackendCreateError: 环境创建失败

        Examples:
            >>> env = IsaacSimBackend.create("Isaac-Reach-v0")
            >>> env = IsaacSimBackend.create("Isaac-Ant-v0", num_envs=8192)
        """
        # 检查依赖
        try:
            cls._check_dependencies()
        except ImportError as e:
            raise BackendUnavailableError(cls.name(), cls.import_name) from e

        # 合并默认参数
        create_kwargs = cls.get_default_kwargs()
        create_kwargs.update(kwargs)

        # 延迟导入 IsaacOrbit/Gymens
        try:
            from omni.isaac.gym.vec_env import VecEnvBase
            from omni.isaac.core.utils.stage import add_reference_to_stage
            from omni.isaac.core.robots import Robot
            from omni.isaac.core.utils.viewports import set_camera_view
            from omni.isaac.nucleus import World
        except ImportError as e:
            raise BackendUnavailableError(cls.name(), "omni.isaac.orbit") from e

        # 创建环境
        # 注意：IsaacOrbit 的环境创建方式与 Gymnasium 不同
        # 这里提供基础框架，具体任务需要导入对应的任务配置
        try:
            # 对于 IsaacGymens (支持 Gymnasium API)
            import gymnasium as gym
            env = gym.make(task, **create_kwargs)
            return env
        except Exception as e:
            raise BackendCreateError(cls.name(), task, str(e)) from e

    @classmethod
    def get_default_kwargs(cls) -> dict[str, Any]:
        """获取 IsaacSim 默认参数"""
        return {
            "headless": True,
            "device": "cuda:0",
        }

    @classmethod
    def list_available_tasks(cls) -> list[str]:
        """列出所有可用的 IsaacSim 任务

        Returns:
            list[str]: 常用 IsaacSim 环境名称列表
        """
        return [
            # IsaacOrbit 原生任务
            "Isaac-Lift-Cube-Franka-v0",
            "Isaac-Lift-Cube-Franka-Parallel-v0",
            "Isaac-Ant-v0",
            "Isaac-Ant-Parallel-v0",
            "Isaac-Humanoid-v0",
            "Isaac-Reacher-Franka-v0",
            "Isaac-Cartpole-v0",
            "Isaac-Quadcopter-v0",
            # 更多任务需要根据 IsaacOrbit 版本确认
        ]
