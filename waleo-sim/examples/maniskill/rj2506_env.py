"""
RJ2506 PickCube环境注册 - 零库修改方案
在外部代码中注册自定义环境，无需修改ManiSkill库
"""

import gymnasium as gym
import numpy as np
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.registration import register_env
from mani_skill.agents.robots.rj2506 import RJ2506


class RJ2506PickCubeEnv(PickCubeEnv):
    """
    RJ2506专用PickCube环境

    修改：
    - 自定义机器人加载位置，避免与桌子碰撞
    - 加载位置：[-0.95, 0, -0.35]
    """

    def _load_agent(self, options: dict):
        """
        重写加载位置，使用RJ2506特定配置
        """
        from mani_skill.utils import sapien_utils

        # RJ2506机器人加载位置配置
        robot_load_positions = {
            "rj2506": [-0.95, 0, -0.35],  # 向左移动避免与桌子碰撞
        }

        # 获取加载位置，使用默认值作为fallback
        load_pos = robot_load_positions.get(self.robot_uids, [-0.615, 0, 0])

        # 调用父类方法，使用自定义位置
        super()._load_agent(options, sapien.Pose(p=load_pos))


# 注册环境到ManiSkill和Gymnasium
# 注意：使用相同的环境ID "PickCube-v1" 但通过override=True覆盖
@register_env(
    "PickCube-v1",  # 使用原有ID
    override=True,    # 覆盖原有注册
    max_episode_steps=50,
)
class _RJ2506PickCubeOverridden(PickCubeEnv):
    """
    覆盖注册类
    用于将RJ2506PickCubeEnv注册到ManiSkill环境系统
    覆盖原有的PickCube-v1环境，使其支持rj2506的加载位置
    """
    pass


# 导出
__all__ = ["RJ2506PickCubeEnv", "_RJ2506PickCubeOverridden"]
