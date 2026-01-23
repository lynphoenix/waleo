"""环境基类实现

包含 BaseEnv, EnvWrapper, VectorEnv, RobotEnv 等核心类。
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, List
import numpy as np


class BaseEnv:
    """环境基类

    与 Gym/Gymnasium API 兼容，提供：
    - 标准的 reset/step 接口
    - 观察和动作空间定义
    - 渲染支持
    """

    @property
    @abstractmethod
    def observation_space(self) -> "spaces.Space":
        """观察空间"""
        ...

    @property
    @abstractmethod
    def action_space(self) -> "spaces.Space":
        """动作空间"""
        ...

    @abstractmethod
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        ...

    @abstractmethod
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步"""
        ...

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """渲染环境"""
        if mode == "rgb_array":
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return None

    def close(self) -> None:
        """关闭环境"""
        pass


class EnvWrapper(BaseEnv):
    """环境包装器基类

    用于修改环境行为而不改变环境本身。

    示例:
        ```python
        class MyWrapper(EnvWrapper):
            def step(self, action):
                obs, reward, terminated, truncated, info = self.env.step(action)
                # 修改观察
                obs = self.process_obs(obs)
                return obs, reward, terminated, truncated, info
        ```
    """

    def __init__(self, env: BaseEnv):
        """
        Args:
            env: 被包装的环境
        """
        self.env = env

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        return self.env.reset(seed, options)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        return self.env.step(action)

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        return self.env.render(mode)

    def close(self) -> None:
        return self.env.close()

    def __getattr__(self, name):
        """代理未定义的属性到被包装的环境"""
        return getattr(self.env, name)


class VectorEnv(ABC):
    """向量化环境基类

    支持并行运行多个环境实例，提高训练效率。

    示例:
        ```python
        vec_env = MyVectorEnv(num_envs=8)
        observations = vec_env.reset()

        for _ in range(1000):
            actions = np.array([env.action_space.sample() for _ in range(8)])
            observations, rewards, terminateds, truncateds, infos = vec_env.step(actions)
        ```
    """

    @property
    @abstractmethod
    def num_envs(self) -> int:
        """环境数量"""
        ...

    @property
    @abstractmethod
    def observation_space(self) -> "spaces.Space":
        """观察空间"""
        ...

    @property
    @abstractmethod
    def action_space(self) -> "spaces.Space":
        """动作空间"""
        ...

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """重置所有环境

        Args:
            seed: 随机种子

        Returns:
            observations: (num_envs, *obs_shape)
        """
        ...

    @abstractmethod
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """在所有环境中执行动作

        Args:
            actions: (num_envs, *action_shape)

        Returns:
            observations: (num_envs, *obs_shape)
            rewards: (num_envs,)
            terminateds: (num_envs,)
            truncateds: (num_envs,)
            infos: 字典列表
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """关闭所有环境"""
        ...

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """渲染环境（对于向量化环境，通常渲染第一个环境）"""
        return None


class RobotEnv(BaseEnv):
    """机器人仿真环境基类

    扩展 BaseEnv，添加机器人特有的功能：
    - 机器人模型加载
    - 相机渲染
    - 物理交互
    - 奖励计算

    示例:
        ```python
        class MyRobotEnv(RobotEnv):
            def _load_robot_model(self):
                # 加载机器人模型
                pass

            def _setup_cameras(self):
                # 设置相机
                pass

            def _compute_reward(self, achieved_goal, desired_goal):
                # 计算奖励
                return -np.linalg.norm(achieved_goal - desired_goal)
        ```
    """

    def __init__(
        self,
        task: str,
        robot_type: str = "so100",
        render_mode: Optional[str] = None,
        simulation_backend: str = "mujoco",
        headless: bool = False,
    ):
        """
        Args:
            task: 任务类型 ("push", "pick_place", "reach")
            robot_type: 机器人类型 ("so100", "aloha", "koch", "panda")
            render_mode: 渲染模式 ("human", "rgb_array", None)
            simulation_backend: 仿真后端 ("mujoco", "pybullet", "maniskill", "isaacgym")
            headless: 无头模式（不显示窗口）
        """
        self.task = task
        self.robot_type = robot_type
        self.render_mode = render_mode
        self.simulation_backend = simulation_backend
        self.headless = headless

        # 初始化环境
        self._load_robot_model()
        self._setup_cameras()

    @abstractmethod
    def _load_robot_model(self) -> None:
        """加载机器人模型（子类必须实现）"""
        ...

    @abstractmethod
    def _setup_cameras(self) -> None:
        """设置相机（子类必须实现）"""
        ...

    @abstractmethod
    def _compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> float:
        """计算奖励（子类必须实现）

        Args:
            achieved_goal: 达成的目标
            desired_goal: 期望的目标

        Returns:
            reward: 奖励值
        """
        ...

    def _check_termination(self) -> bool:
        """检查是否终止（子类可选实现）

        Returns:
            terminated: 是否完成任务
        """
        return False

    @property
    @abstractmethod
    def robot_state(self) -> np.ndarray:
        """获取机器人状态（子类必须实现）

        Returns:
            state: 机器人状态向量
        """
        ...

    @property
    def camera_images(self) -> Dict[str, np.ndarray]:
        """获取所有相机图像（可选实现）

        Returns:
            images: {camera_name: image_array} 字典
        """
        return {}

    def __repr__(self) -> str:
        """环境表示"""
        return f"{self.__class__.__name__}(task='{self.task}', robot='{self.robot_type}', backend='{self.simulation_backend}')"
