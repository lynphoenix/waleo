"""ManiSkill 仿真后端

ManiSkill 是一个专业的机器人操作仿真框架，基于 SAPIEN 引擎。
"""

from typing import Optional
import numpy as np

from waleo.sim.backends.backend import SimulationBackend


class ManiSkillBackend(SimulationBackend):
    """ManiSkill 仿真后端

    特点：
    - 基于 SAPIEN 引擎，专业的机器人操作仿真
    - 支持复杂物理交互
    - 丰富的操作任务库
    - GPU 加速渲染和物理
    - 支持并行仿真

    支持的任务:
    - PickCube: 抓取立方体
    - PushCube: 推动立方体
    - StackCube: 堆叠立方体
    - PlugCharger: 插入充电器
    - TurnFaucet: 转动水龙头
    - OpenCabinetDrawer: 打开抽屉

    注意: 使用此后端需要安装 maniskill2-api 和 sapien
    """

    def __init__(self, config):
        """
        Args:
            config: 环境配置对象 (应包含 maniskill_task 和 enable_visual_obs)
        """
        super().__init__(config)
        self.env = None
        self.task_name = config.maniskill_task or "PickCube"
        self.num_envs = config.num_envs
        self.enable_visual_obs = config.enable_visual_obs

    def initialize(self) -> None:
        """初始化 ManiSkill 环境"""
        try:
            import maniskill2_api
            import sapien.core as sapien
        except ImportError as e:
            raise ImportError(
                "ManiSkill requires maniskill2-api and sapien. "
                "Install with: pip install maniskill2-api sapien"
            ) from e

        # 创建 ManiSkill 环境
        self.env = maniskill2_api.create_environment(
            task=self.task_name,
            num_envs=self.num_envs,
            enable_visual_obs=self.enable_visual_obs,
        )

        # 获取观察和动作空间
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self._is_initialized = True

    def reset(self, seed: Optional[int] = None) -> tuple:
        """重置环境

        Args:
            seed: 随机种子（ManiSkill 可能不支持此参数）

        Returns:
            (observation, info): 观察和额外信息
        """
        if not self._is_initialized:
            self.initialize()

        # ManiSkill 的 reset 只返回 observation
        obs = self.env.reset()

        # 转换为 Gym 格式 (obs, info)
        if isinstance(obs, tuple):
            observation = obs[0]
            info = obs[1] if len(obs) > 1 else {}
        else:
            observation = obs
            info = {}

        return observation, info

    def step(self, action: np.ndarray) -> tuple:
        """执行仿真步

        Args:
            action: 动作

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # ManiSkill 的 step 返回 (obs, reward, terminated, info)
        result = self.env.step(action)

        if len(result) == 4:
            observation, reward, terminated, info = result
            truncated = False
        else:
            observation, reward, terminated, truncated, info = result

        return observation, reward, terminated, truncated, info

    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        """渲染环境

        Args:
            mode: "human" 显示窗口, "rgb_array" 返回图像

        Returns:
            图像数组（如果是 rgb_array 模式），否则 None
        """
        if not self._is_initialized:
            return None

        if mode == "rgb_array":
            # 使用 ManiSkill 的相机渲染
            images = self.env.render(mode="cameras")
            return images
        elif mode == "human":
            # 使用 ManiSkill 的查看器
            self.env.render(mode="viewer")
            return None

        return None

    def close(self) -> None:
        """关闭仿真环境"""
        if self.env is not None:
            self.env.close()
            self.env = None

        self._is_initialized = False

    @property
    def observation_space(self):
        """获取观察空间"""
        if not self._is_initialized:
            return None
        return self.env.observation_space

    @observation_space.setter
    def observation_space(self, value):
        """设置观察空间"""
        pass  # 由 ManiSkill 环境管理

    @property
    def action_space(self):
        """获取动作空间"""
        if not self._is_initialized:
            return None
        return self.env.action_space

    @action_space.setter
    def action_space(self, value):
        """设置动作空间"""
        pass  # 由 ManiSkill 环境管理

    @property
    def supported_tasks(self) -> list:
        """获取支持的任务列表"""
        return [
            "PickCube",
            "PushCube",
            "StackCube",
            "PlugCharger",
            "TurnFaucet",
            "OpenCabinetDrawer",
        ]
