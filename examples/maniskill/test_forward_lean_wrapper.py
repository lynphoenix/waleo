"""
测试RJ2506前倾姿态 - 使用包装器方法
在环境reset后强制设置机器人姿态
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt

# 导入配置
from rj2506_config_minimal import apply_rj2506_config_minimal
from rj2506_env import _RJ2506PickCubeOverridden

apply_rj2506_config_minimal()

print("=" * 60)
print("测试RJ2506前倾姿态（包装器方法）")
print("=" * 60)

# 前倾姿态配置
FORWARD_LEAN_BODY_JOINT2 = 0.78  # 45度前倾

class ForwardLeanWrapper(gym.Wrapper):
    """包装器：在环境reset后设置机器人前倾姿态"""

    def __init__(self, env):
        super().__init__(env)
        self.forward_lean_angle = FORWARD_LEAN_BODY_JOINT2

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # 获取当前qpos并修改body_joint2
        agent = self.env.unwrapped.agent
        qpos = agent.robot.get_qpos()

        # 修改body_joint2为前倾角度（对所有环境）
        qpos[:, 1] = self.forward_lean_angle

        # 设置新的qpos
        agent.robot.set_qpos(qpos)

        print(f"✓ 已设置前倾姿态: body_joint2 = {self.forward_lean_angle:.3f} rad ≈ {self.forward_lean_angle * 180 / np.pi:.1f}°")

        return obs, info

# 创建环境
env = gym.make(
    "PickCube-v1",
    robot_uids="rj2506",
    num_envs=1,
    obs_mode="rgbd",
    reward_mode="dense",
    control_mode="pd_joint_delta_pos",
    render_mode="all"
)

# 包装环境
env = ForwardLeanWrapper(env)

print(f"✓ 环境创建并包装成功")

# 重置环境
obs, info = env.reset(seed=0)
print(f"✓ 环境重置成功")

# 渲染一帧来获取图像
env.render()  # 确保渲染器已初始化

# 尝试从不同地方获取图像
image_saved = False

# 方法1: 从render获取
try:
    img = env.render()
    if img is not None and len(img) > 0:
        output_path = "/home/smai/linyining/waleo/waleo-sim/examples/maniskill/runs/rj2506_forward_lean_render.png"
        plt.imsave(output_path, img)
        print(f"✓ 已保存渲染图像: {output_path}")
        image_saved = True
except Exception as e:
    print(f"渲染图像失败: {e}")

# 方法2: 从info获取
if not image_saved and "images" in info:
    images = info["images"]
    print(f"✓ 获取到图像: {list(images.keys())}")

    # 保存人类视角
    if "human_cam" in images and len(images["human_cam"]) > 0:
        output_path = "/home/smai/linyining/waleo/waleo-sim/examples/maniskill/runs/rj2506_forward_lean_human_view.png"
        plt.figure(figsize=(12, 12))
        plt.imshow(images["human_cam"][0])
        plt.title(f"RJ2506 前倾姿态 - 人类视角\n身体前倾约{FORWARD_LEAN_BODY_JOINT2 * 180 / np.pi:.0f}度，更容易够到前方物体", fontsize=16)
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"✓ 已保存人类视角: {output_path}")
        image_saved = True

# 检查机器人关节角度
agent = env.unwrapped.agent
qpos = agent.robot.get_qpos()
print(f"\n当前关节角度 (环境0):")
print(f"  body_joint1: {qpos[0, 0].item():.3f} rad")
print(f"  body_joint2: {qpos[0, 1].item():.3f} rad ≈ {qpos[0, 1].item() * 180 / np.pi:.1f}°")
print(f"  left_arm_joint0: {qpos[0, 2].item():.3f} rad")
print(f"  left_arm_joint1: {qpos[0, 3].item():.3f} rad")

env.close()

print("\n✓ 测试完成！")
print("=" * 60)
