"""仿真后端抽象基类

所有仿真后端必须实现此接口，确保统一的调用方式。
"""

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

# 只在类型检查时导入 gymnasium
if TYPE_CHECKING:
    import gymnasium as gym


class SimulationBackend(ABC):
    """仿真后端抽象基类

    定义所有后端必须实现的接口，确保：
    1. 统一的环境创建方式
    2. 一致的错误处理
    3. 可用性检查

    所有后端实现都应继承此类并实现 create() 方法。
    """

    @classmethod
    @abstractmethod
    def create(cls, task: str, **kwargs) -> Any:
        """创建仿真环境

        Args:
            task: 任务标识符（各后端定义不同）
                - ManiSkill3: "PickCube-v1", "PushCube-v1" 等
                - MuJoCo: XML 文件路径或已注册环境名（如 "Ant-v4"）
                - PyBullet: URDF 文件路径或已注册环境名
            **kwargs: 后端特定参数（完全灵活）

                通用参数:
                    num_envs: 并行环境数量（大多数后端支持）
                    render_mode: 渲染模式 "human" 或 "rgb_array"

                ManiSkill 特定:
                    obs_mode: "state", "state_dict", "rgbd", "pointcloud"
                    control_mode: "pd_joint_pos", "pd_ee_delta_pose"
                    robot_uids: 机器人 ID（如 "panda", "rj2506"）

                MuJoCo 特定:
                    frame_skip: 每个动作的物理步数

                PyBullet 特定:
                    physics: "gui" 或 "direct"

        Returns:
            gym.Env: 创建好的环境实例

        Raises:
            ImportError: 后端依赖未安装
            ValueError: 参数无效
            RuntimeError: 环境创建失败
        """
        pass

    @classmethod
    def is_available(cls) -> bool:
        """检查后端是否可用

        Returns:
            bool: True 如果后端依赖已安装且可用

        Example:
            >>> if MuJoCoBackend.is_available():
            ...     env = MuJoCoBackend.create("task.xml")
        """
        try:
            cls._check_dependencies()
            return True
        except ImportError:
            return False

    @classmethod
    def _check_dependencies(cls) -> None:
        """检查后端依赖是否已安装

        Raises:
            ImportError: 如果依赖未安装

        子类可以覆盖此方法以提供自定义依赖检查。
        """
        # 子类必须定义 import_name 类属性
        if not hasattr(cls, 'import_name') or not cls.import_name:
            raise NotImplementedError(
                f"Backend {cls.__name__} must define 'import_name' class attribute"
            )
        __import__(cls.import_name)

    @classmethod
    def name(cls) -> str:
        """后端名称

        Returns:
            str: 后端标识符，如 "maniskill", "mujoco"
        """
        return cls.__name__.replace("Backend", "").lower()

    @classmethod
    def get_default_kwargs(cls) -> dict[str, Any]:
        """获取后端默认参数

        子类可以覆盖此方法以提供后端特定的默认值。

        Returns:
            dict: 默认参数字典

        Example:
            >>> {"num_envs": 1, "obs_mode": "state_dict"}
        """
        return {}


class BackendError(Exception):
    """后端错误基类"""

    pass


class BackendUnavailableError(BackendError):
    """后端不可用错误

    当后端依赖未安装时抛出。
    """

    def __init__(self, backend_name: str, missing_package: str):
        self.backend_name = backend_name
        self.missing_package = missing_package
        super().__init__(
            f"{backend_name} backend requires {missing_package}. "
            f"Install it with: pip install {missing_package}"
        )


class BackendCreateError(BackendError):
    """环境创建失败错误"""

    def __init__(self, backend_name: str, task: str, reason: str):
        self.backend_name = backend_name
        self.task = task
        self.reason = reason
        super().__init__(
            f"Failed to create {backend_name} environment for task '{task}': {reason}"
        )
