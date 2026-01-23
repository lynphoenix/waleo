"""仿真后端基类

定义仿真后端的抽象接口。
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np


class SimulationBackend(ABC):
    """仿真后端抽象接口

    定义了所有仿真后端必须实现的接口。

    支持的仿真后端：
    - MuJoCo: 快速物理仿真
    - PyBullet: 开源免费的物理引擎
    - ManiSkill: 专业的机器人操作仿真
    - Isaac Gym: GPU 加速大规模并行仿真
    """

    def __init__(self, config: Any):
        """初始化仿真后端

        Args:
            config: 环境配置对象
        """
        self.config = config
        self._is_initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """初始化仿真环境（子类必须实现）"""
        ...

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> tuple:
        """重置环境

        Args:
            seed: 随机种子

        Returns:
            (observation, info): 观察和额外信息
        """
        ...

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple:
        """执行仿真步

        Args:
            action: 动作

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        ...

    @abstractmethod
    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        """渲染环境

        Args:
            mode: "human" 显示窗口, "rgb_array" 返回图像

        Returns:
            图像数组（如果是 rgb_array 模式），否则 None
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """关闭仿真环境，释放资源"""
        ...

    @property
    @abstractmethod
    def observation_space(self):
        """获取观察空间"""
        ...

    @property
    @abstractmethod
    def action_space(self):
        """获取动作空间"""
        ...

    def get_state(self) -> Dict[str, Any]:
        """获取环境状态（可选实现）

        Returns:
            state: 环境状态字典
        """
        return {}

    def set_state(self, state: Dict[str, Any]) -> None:
        """设置环境状态（可选实现）

        Args:
            state: 环境状态字典
        """
        pass

    @property
    def is_initialized(self) -> bool:
        """检查后端是否已初始化"""
        return self._is_initialized

    def __repr__(self) -> str:
        """后端表示"""
        return f"{self.__class__.__name__}(config={self.config})"
