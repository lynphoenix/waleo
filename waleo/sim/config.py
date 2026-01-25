"""环境配置数据类

提供仿真环境所需的配置定义。
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class EnvConfig:
    """环境配置

    Args:
        task: 任务类型 ("push", "pick_place", "reach")
        robot_type: 机器人类型 ("so100", "aloha", "koch", "panda")
        simulation_backend: 仿真后端 ("mujoco", "pybullet", "maniskill", "isaacgym")
        render_mode: 渲染模式 ("human", "rgb_array", None)
        headless: 无头模式（不显示窗口）
        num_envs: 并行环境数量（用于向量化环境）
        seed: 随机种子
        episode_length: 每个回合的最大步数
        reward_scale: 奖励缩放因子
        observation_keys: 需要的观察键列表
    """

    # 必需参数
    task: str
    robot_type: str

    # 可选参数
    simulation_backend: str = "mujoco"
    render_mode: Optional[str] = None
    headless: bool = False
    num_envs: int = 1
    seed: Optional[int] = None
    episode_length: int = 1000
    reward_scale: float = 1.0

    # 观察配置
    observation_keys: List[str] = field(default_factory=lambda: [
        "observation.state",
        "observation.image",
    ])

    # ManiSkill 特定配置
    maniskill_task: Optional[str] = None  # ManiSkill 任务名称
    enable_visual_obs: bool = True  # 启用视觉观察

    def __post_init__(self):
        """配置验证"""
        valid_backends = ["mujoco", "pybullet", "maniskill", "isaacgym"]
        if self.simulation_backend not in valid_backends:
            raise ValueError(
                f"Invalid simulation_backend: {self.simulation_backend}. "
                f"Must be one of {valid_backends}"
            )

        valid_render_modes = ["human", "rgb_array", None]
        if self.render_mode not in valid_render_modes:
            raise ValueError(
                f"Invalid render_mode: {self.render_mode}. "
                f"Must be one of {valid_render_modes}"
            )

        if self.num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {self.num_envs}")

        if self.episode_length < 1:
            raise ValueError(f"episode_length must be >= 1, got {self.episode_length}")

        # ManiSkill 特定验证
        if self.simulation_backend == "maniskill" and not self.maniskill_task:
            # 使用默认的 ManiSkill 任务
            self.maniskill_task = self._get_default_maniskill_task()

    def _get_default_maniskill_task(self) -> str:
        """根据任务类型返回默认的 ManiSkill 任务"""
        task_mapping = {
            "pick_place": "PickCube",
            "push": "PushCube",
            "reach": "ReachTarget",
        }
        return task_mapping.get(self.task, "PickCube")

    @classmethod
    def from_dict(cls, config_dict: dict) -> "EnvConfig":
        """从字典创建配置

        Args:
            config_dict: 配置字典

        Returns:
            EnvConfig 实例
        """
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """转换为字典

        Returns:
            配置字典
        """
        from dataclasses import asdict
        return asdict(self)


@dataclass
class CameraConfig:
    """相机配置

    Args:
        width: 图像宽度
        height: 图像高度
        fov: 视场角（度）
        near: 近裁剪面距离
        far: 远裁剪面距离
        position: 相机位置 (x, y, z)
        look_at: 观察目标位置 (x, y, z)
    """

    width: int = 640
    height: int = 480
    fov: float = 60.0
    near: float = 0.01
    far: float = 100.0

    # 相机位姿（可选，由环境设置）
    position: tuple = (0.5, 0.0, 0.5)
    look_at: tuple = (0.0, 0.0, 0.0)

    def __post_init__(self):
        """配置验证"""
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid image size: {self.width}x{self.height}")

        if not (0 < self.fov < 180):
            raise ValueError(f"Invalid FOV: {self.fov}, must be in (0, 180)")

        if self.near >= self.far:
            raise ValueError(f"near ({self.near}) must be < far ({self.far})")
