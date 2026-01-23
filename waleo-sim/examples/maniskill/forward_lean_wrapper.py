"""
前倾姿态包装器
在环境reset后强制设置RJ2506机器人前倾姿态
"""

import gymnasium as gym
import numpy as np

FORWARD_LEAN_BODY_JOINT2 = 0.78  # 45度前倾

class ForwardLeanWrapper(gym.Wrapper):
    """包装器：在环境reset后设置RJ2506机器人前倾姿态"""

    def __init__(self, env, forward_lean_angle=FORWARD_LEAN_BODY_JOINT2):
        super().__init__(env)
        self.forward_lean_angle = forward_lean_angle

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # 获取当前qpos并修改body_joint2
        agent = self.env.unwrapped.agent
        qpos = agent.robot.get_qpos()

        # 修改body_joint2为前倾角度（对所有环境）
        qpos[:, 1] = self.forward_lean_angle

        # 设置新的qpos
        agent.robot.set_qpos(qpos)

        return obs, info

    def step(self, action):
        return self.env.step(action)


def apply_forward_lean_wrapper(env, forward_lean_angle=FORWARD_LEAN_BODY_JOINT2):
    """
    应用前倾姿态包装器到环境

    使用方法:
        env = gym.make("PickCube-v1", robot_uids="rj2506", ...)
        env = apply_forward_lean_wrapper(env)
    """
    return ForwardLeanWrapper(env, forward_lean_angle)
