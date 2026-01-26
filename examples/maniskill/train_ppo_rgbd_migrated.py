#!/usr/bin/env python3
"""
Migrated PPO training script with RGBD + state observations.

使用新的 make_env() API，无需任何侵入式代码。
配置全部来自 assets/robots/RJ2506/robot.yaml

迁移变化：
- ❌ 移除 monkey-patching gym.make
- ❌ 移除 custom_agents 导入
- ❌ 移除手动 pose 调整
- ✅ 使用 waleo.sim.make_env()
- ✅ 配置自动从 robot.yaml 加载
"""

import sys
import argparse

# Parse custom camera resolution from command line
camera_resolution = 128  # default
if "--camera_resolution" in sys.argv:
    idx = sys.argv.index("--camera_resolution")
    camera_resolution = int(sys.argv[idx + 1])
    sys.argv.pop(idx)
    sys.argv.pop(idx)
    print(f"[CONFIG] Using camera resolution: {camera_resolution}x{camera_resolution}")

# 导入 waleo 工厂函数
from waleo.sim import make_env

def create_env(env_id, num_envs=1, robot="panda", **kwargs):
    """创建环境（无侵入式）

    Args:
        env_id: 环境 ID，如 "PickCube-v1"
        num_envs: 并行环境数量
        robot: 机器人名称，如 "rj2506" 或 "panda"
        **kwargs: 其他参数

    Returns:
        配置好的环境
    """
    # 配置 sensor_configs（如果需要自定义相机分辨率）
    sensor_configs = kwargs.get("sensor_configs", {})
    if "base_camera" not in sensor_configs:
        sensor_configs["base_camera"] = {}
    sensor_configs["base_camera"]["width"] = camera_resolution
    sensor_configs["base_camera"]["height"] = camera_resolution
    kwargs["sensor_configs"] = sensor_configs

    # 使用 make_env() 创建环境
    # - 如果是 RJ2506，会自动加载 robot.yaml 配置
    # - 自动应用 robot_pose, keyframes, object_config, camera_config
    env = make_env(
        env_id,
        robot=robot,
        num_envs=num_envs,
        obs_mode="rgbd",  # RGBD + state
        **kwargs
    )

    print(f"[✓] 环境创建成功：{env_id} with {robot.upper()}")
    if robot.upper() == "RJ2506":
        print(f"[✓] RJ2506 配置已自动加载：robot_pose, keyframes, object_config")

    return env


# ============================================================================
# 主训练循环（使用原始训练脚本的逻辑）
# ============================================================================

if __name__ == "__main__":
    # 解析命令行参数（从 train_ppo_vectorized_from_original.py）
    parser = argparse.ArgumentParser()

    # 环境参数
    parser.add_argument("--env-id", type=str, default="PickCube-v1")
    parser.add_argument("--robot", type=str, default="panda",
                        help="Robot name: panda, fetch, rj2506, etc.")
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--num-eval-envs", type=int, default=8)

    # 训练参数
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.95)

    # 其他参数
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="runs")

    args = parser.parse_args()

    print("=" * 70)
    print("Waleo 无侵入式 PPO 训练")
    print("=" * 70)
    print(f"环境: {args.env_id}")
    print(f"机器人: {args.robot.upper()}")
    print(f"并行环境: {args.num_envs}")
    print(f"观测模式: RGBD + state")
    print(f"相机分辨率: {camera_resolution}x{camera_resolution}")
    print("=" * 70)

    # 创建训练环境
    print("\n[1/3] 创建训练环境...")
    train_envs = create_env(
        args.env_id,
        num_envs=args.num_envs,
        robot=args.robot,
        sim_freq=500,
        control_freq=20
    )

    # 创建评估环境
    print(f"\n[2/3] 创建评估环境 ({args.num_eval_envs} envs)...")
    eval_envs = create_env(
        args.env_id,
        num_envs=args.num_eval_envs,
        robot=args.robot,
        sim_freq=500,
        control_freq=20
    )

    print("\n[3/3] 开始训练...")
    print("=" * 70)

    # TODO: 这里应该导入和运行实际的 PPO 训练逻辑
    # 为了演示，我们只是显示环境已正确创建

    # 测试环境
    obs, info = train_envs.reset()
    print(f"✓ 训练环境 reset 成功")
    print(f"  - Observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
    print(f"  - Num envs: {args.num_envs}")

    obs, info = eval_envs.reset()
    print(f"✓ 评估环境 reset 成功")
    print(f"  - Observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
    print(f"  - Num envs: {args.num_eval_envs}")

    print("\n" + "=" * 70)
    print("环境创建成功！可以开始训练。")
    print("=" * 70)
    print("\n提示：")
    print("- 所有配置自动从 robot.yaml 加载")
    print("- 无需 monkey-patching")
    print("- 无需导入 custom_agents")
    print("- 完全无侵入式")
    print()

    # 关闭环境
    train_envs.close()
    eval_envs.close()
