"""评估脚本

评估训练好的策略在 ManiSkill 环境中的表现。
"""

import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import argparse

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from examples.maniskill.policy import VisualPolicy
from waleo_sim.tasks.maniskill_pick_cube import ManiSkillPickCubeEnv


def evaluate(
    checkpoint_path: str,
    num_episodes: int = 50,
    max_steps: int = 200,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    render: bool = False,
    save_video: bool = False,
):
    """
    评估策略

    Args:
        checkpoint_path: 检查点文件路径
        num_episodes: 评估的 episode 数量
        max_steps: 每个 episode 的最大步数
        device: 运行设备
        render: 是否渲染环境
        save_video: 是否保存视频
    """
    print("=" * 60)
    print("评估配置:")
    print(f"  检查点: {checkpoint_path}")
    print(f"  Episodes: {num_episodes}")
    print(f"  最大步数: {max_steps}")
    print(f"  设备: {device}")
    print(f"  渲染: {render}")
    print("=" * 60)
    print()

    # 创建环境
    print("创建环境...")
    env = ManiSkillPickCubeEnv(
        robot_type="panda",
        image_size=(128, 128),  # 使用训练时的图像大小
        use_cameras=True,
        headless=not render,
    )

    # 获取机器人配置
    robot_config = env.get_robot_config()

    # 加载策略（使用机器人配置和正确的图像大小）
    print("加载策略...")
    policy = VisualPolicy(
        image_size=(128, 128),  # 使用训练时的图像大小
        state_dim=robot_config["state_dim"],
        action_dim=robot_config["action_dim"],
        hidden_dim=256,
        feature_dim=256,
    )

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.to(device)
    policy.eval()

    print(f"✓ 策略已加载 (epoch: {checkpoint['epoch']}, loss: {checkpoint['loss']:.4f})")

    # 评估
    print(f"\n开始评估...")
    print()

    success_count = 0
    episode_rewards = []
    episode_lengths = []

    for episode_idx in tqdm(range(num_episodes), desc="评估"):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_length = 0

        for step in range(max_steps):
            # 选择动作
            with torch.no_grad():
                action = policy.select_action(obs)

            # 执行动作
            next_obs, reward, terminated, truncated, info = env.step(action)

            # 将 Tensor 类型的 reward 转换为标量
            if hasattr(reward, 'item'):
                reward = reward.item()
            elif hasattr(reward, 'cpu'):
                reward = reward.cpu().item()

            episode_reward += reward
            episode_length += 1

            obs = next_obs

            # 检查是否成功
            if info.get("success", False):
                success_count += 1
                break

            if terminated or truncated:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    # 计算统计信息
    success_rate = success_count / num_episodes * 100
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)

    # 打印结果
    print()
    print("=" * 60)
    print("评估结果:")
    print(f"  Success Rate: {success_rate:.1f}% ({success_count}/{num_episodes})")
    print(f"  Average Reward: {avg_reward:.4f} ± {std_reward:.4f}")
    print(f"  Average Length: {avg_length:.1f} steps")
    print("=" * 60)

    # 关闭环境
    env.close()

    return {
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "avg_length": avg_length,
    }


def interactive_demo(
    checkpoint_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """交互式演示

    可以手动控制或观看策略执行

    Args:
        checkpoint_path: 检查点文件路径
        device: 运行设备
    """
    print("=" * 60)
    print("交互式演示模式")
    print("=" * 60)

    # 创建环境
    env = ManiSkillPickCubeEnv(
        robot_type="panda",
        image_size=(128, 128),  # 使用训练时的图像大小
        use_cameras=True,
        headless=False,
    )

    # 获取机器人配置
    robot_config = env.get_robot_config()

    # 加载策略（使用机器人配置和正确的图像大小）
    policy = VisualPolicy(
        image_size=(128, 128),  # 使用训练时的图像大小
        state_dim=robot_config["state_dim"],
        action_dim=robot_config["action_dim"],
        hidden_dim=256,
        feature_dim=256,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.to(device)
    policy.eval()

    print("✓ 策略已加载")
    print("\n按 'Enter' 执行一步，输入 'q' 退出")
    print()

    obs, _ = env.reset()
    step_count = 0

    while True:
        # 渲染环境
        frame = env.render(mode="rgb_array")
        if frame is not None:
            print(f"[Step {step_count}] 观察已渲染")

        # 获取策略动作
        with torch.no_grad():
            action = policy.select_action(obs)

        print(f"动作: {action}")

        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

        # 打印信息
        print(f"奖励: {reward:.4f}")
        if info.get("success", False):
            print("✓ 任务完成！")
            break

        if terminated or truncated:
            print("Episode 结束")
            break

        # 用户输入
        user_input = input("\n按 Enter 继续，q 退出: ")
        if user_input.lower() == 'q':
            break

    env.close()


def main():
    parser = argparse.ArgumentParser(description="评估 ManiSkill 策略")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="检查点文件路径")
    parser.add_argument("--episodes", type=int, default=50, help="评估的 episode 数量")
    parser.add_argument("--max-steps", type=int, default=200, help="每个 episode 的最大步数")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="运行设备")
    parser.add_argument("--render", action="store_true", help="是否渲染环境")
    parser.add_argument("--interactive", action="store_true", help="交互式演示模式")

    args = parser.parse_args()

    if args.interactive:
        interactive_demo(args.checkpoint, args.device)
    else:
        evaluate(
            checkpoint_path=args.checkpoint,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            device=args.device,
            render=args.render,
        )


if __name__ == "__main__":
    main()
