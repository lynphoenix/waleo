"""
RJ2506机器人前倾姿态配置
修改rest pose让身体前倾，更容易够到前方的物体
"""

import numpy as np

# 原始rest pose: [0, 0, 0.27, 0.085, 0, 1.6, -1.6, -0.45, 0.015, 0.015]
# body_joint1=0, body_joint2=0 (身体直立)

# 前倾rest pose: body_joint2设为正值让身体前倾
RJ2506_FORWARD_LEAN_QPOS = np.array([
    0,      # body_joint1: 腰部旋转 (保持0)
    0.78,   # body_joint2: 腰部俯仰 (0.78弧度 ≈ 45度前倾)
    0.27,   # left_arm_joint0
    0.085,  # left_arm_joint1
    0,      # left_arm_joint2
    1.6,    # left_arm_joint3
    -1.6,   # left_arm_joint4
    -0.45,  # left_arm_joint5
    0.015,  # left_hand_finger1_joint
    0.015,  # left_hand_finger2_joint
])


def apply_rj2506_forward_lean_pose():
    """
    在运行时修改RJ2506的rest pose为前倾姿态

    使用方法：
        from rj2506_forward_lean_pose import apply_rj2506_forward_lean_pose
        apply_rj2506_forward_lean_pose()
    """
    try:
        from mani_skill.agents.robots.rj2506 import RJ2506

        # 修改rest keyframe的qpos
        RJ2506.keyframes["rest"].qpos = RJ2506_FORWARD_LEAN_QPOS

        print("✓ 已应用RJ2506前倾姿态配置")
        print(f"  - body_joint2 (腰部俯仰): {RJ2506_FORWARD_LEAN_QPOS[1]} rad ≈ {RJ2506_FORWARD_LEAN_QPOS[1] * 180 / np.pi:.1f}°")
        print("  - 身体前倾约45度，更容易够到前方物体")
        return True

    except Exception as e:
        print(f"✗ 应用前倾姿态失败: {e}")
        return False


# 如果直接运行此脚本，应用配置
if __name__ == "__main__":
    apply_rj2506_forward_lean_pose()
