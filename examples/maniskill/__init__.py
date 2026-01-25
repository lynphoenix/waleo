"""ManiSkill 训练示例模块

包含完整的 ManiSkill PickCube 训练流程：
- 环境封装
- 策略网络
- 数据采集
- 训练
- 评估
"""

from waleo.sim.tasks.maniskill_pick_cube import ManiSkillPickCubeEnv, create_maniskill_env
from examples.maniskill.policy import VisualPolicy, PolicyTrainer, SimpleCNN

__all__ = [
    "ManiSkillPickCubeEnv",
    "create_maniskill_env",
    "VisualPolicy",
    "PolicyTrainer",
    "SimpleCNN",
]
