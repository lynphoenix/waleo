#!/usr/bin/env python3
"""
Waleo 无侵入式训练脚本模板

这是一个标准模板，展示如何使用新的 make_env() API 进行训练。
适用于所有机器人和任务。

关键特性：
✅ 一行代码创建环境
✅ 自动加载机器人配置
✅ 完全无侵入式
✅ 支持所有 ManiSkill 任务
✅ 支持自定义和内置机器人
"""

import argparse
import numpy as np
from waleo.sim import make_env, list_available_robots, list_available_tasks


def main():
    parser = argparse.ArgumentParser(description="Waleo 无侵入式训练脚本")

    # ========================================================================
    # 环境参数
    # ========================================================================
    parser.add_argument(
        "--env-id",
        type=str,
        default="PickCube-v1",
        help="环境 ID (例如: PickCube-v1, StackCube-v1)"
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="panda",
        help="机器人名称 (例如: panda, fetch, rj2506)"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=512,
        help="并行环境数量（GPU 加速）"
    )
    parser.add_argument(
        "--obs-mode",
        type=str,
        default="state",
        choices=["state", "rgbd", "pointcloud"],
        help="观测模式"
    )
    parser.add_argument(
        "--control-mode",
        type=str,
        default=None,
        help="控制模式（默认使用 robot.yaml 配置）"
    )

    # ========================================================================
    # 训练参数
    # ========================================================================
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=5_000_000,
        help="总训练步数"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="学习率"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=50,
        help="每次 rollout 的步数"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="折扣因子"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="随机种子"
    )

    # ========================================================================
    # 输出参数
    # ========================================================================
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs",
        help="输出目录"
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="实验名称（默认自动生成）"
    )
    parser.add_argument(
        "--list-robots",
        action="store_true",
        help="列出所有可用机器人"
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="列出所有可用任务"
    )

    args = parser.parse_args()

    # ========================================================================
    # 列出资源
    # ========================================================================
    if args.list_robots:
        print("可用机器人：")
        for robot in list_available_robots():
            print(f"  - {robot}")
        return

    if args.list_tasks:
        print("可用任务（ManiSkill）：")
        tasks = list_available_tasks("maniskill")
        for i, task in enumerate(tasks[:20], 1):
            print(f"  {i:2d}. {task}")
        if len(tasks) > 20:
            print(f"  ... 还有 {len(tasks) - 20} 个任务")
        return

    # ========================================================================
    # 打印配置
    # ========================================================================
    print("=" * 80)
    print("Waleo 无侵入式训练".center(80))
    print("=" * 80)
    print(f"环境:        {args.env_id}")
    print(f"机器人:      {args.robot.upper()}")
    print(f"并行环境:    {args.num_envs}")
    print(f"观测模式:    {args.obs_mode}")
    print(f"控制模式:    {args.control_mode or '自动（从 robot.yaml）'}")
    print(f"总步数:      {args.total_timesteps:,}")
    print(f"学习率:      {args.learning_rate}")
    print(f"随机种子:    {args.seed}")
    print("=" * 80)

    # ========================================================================
    # 创建环境（一行代码！）
    # ========================================================================
    print("\n[1/4] 创建训练环境...")
    train_env = make_env(
        task=args.env_id,
        robot=args.robot,
        num_envs=args.num_envs,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        sim_freq=500,
        control_freq=20,
        # render_mode="human"  # 取消注释以可视化
    )
    print(f"✓ 训练环境创建成功（{args.num_envs} 个并行环境）")

    # 创建评估环境
    print(f"\n[2/4] 创建评估环境...")
    eval_env = make_env(
        task=args.env_id,
        robot=args.robot,
        num_envs=8,  # 少量评估环境
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        sim_freq=500,
        control_freq=20,
    )
    print(f"✓ 评估环境创建成功（8 个并行环境）")

    # ========================================================================
    # 测试环境
    # ========================================================================
    print("\n[3/4] 测试环境...")
    obs, info = train_env.reset(seed=args.seed)
    print(f"✓ 环境 reset 成功")
    print(f"  - Observation: {type(obs)}")
    if isinstance(obs, dict):
        for key, value in obs.items():
            if hasattr(value, 'shape'):
                print(f"    - {key}: shape={value.shape}, dtype={value.dtype}")
    elif hasattr(obs, 'shape'):
        print(f"  - Shape: {obs.shape}")
        print(f"  - Dtype: {obs.dtype}")

    # 测试 step
    action = train_env.action_space.sample()
    obs, reward, terminated, truncated, info = train_env.step(action)
    print(f"✓ 环境 step 成功")
    print(f"  - Reward: {reward[:5] if hasattr(reward, '__len__') else reward}")

    # ========================================================================
    # 训练循环（示例）
    # ========================================================================
    print("\n[4/4] 训练循环示例...")
    print("-" * 80)

    num_episodes = 10  # 演示用
    for episode in range(num_episodes):
        obs, info = train_env.reset()
        episode_reward = np.zeros(args.num_envs)
        done = False
        step = 0

        while not done and step < 100:  # 限制步数
            # 随机策略（实际训练应使用 RL 算法）
            action = train_env.action_space.sample()
            obs, reward, terminated, truncated, info = train_env.step(action)

            episode_reward += reward
            done = terminated.any() or truncated.any()
            step += 1

        avg_reward = episode_reward.mean()
        print(f"Episode {episode+1}/{num_episodes}: "
              f"avg_reward={avg_reward:.3f}, steps={step}")

    print("-" * 80)

    # ========================================================================
    # 清理
    # ========================================================================
    train_env.close()
    eval_env.close()

    print("\n" + "=" * 80)
    print("训练演示完成！".center(80))
    print("=" * 80)
    print("\n提示：")
    print("✅ 所有配置自动从 robot.yaml 加载")
    print("✅ 无需 monkey-patching")
    print("✅ 无需导入 custom_agents")
    print("✅ 完全无侵入式")
    print("✅ 支持所有 ManiSkill 任务和机器人")
    print("\n实际训练时，请将随机策略替换为 PPO/SAC/TD3 等算法。")
    print()


if __name__ == "__main__":
    main()
