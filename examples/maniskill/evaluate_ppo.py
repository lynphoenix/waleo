"""评估 PPO 策略"""

import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import argparse

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from waleo.sim.tasks.maniskill_pick_cube import ManiSkillPickCubeEnv
from examples.maniskill.ppo_policy import PPOPolicy


def evaluate_ppo(
    checkpoint_path: str,
    num_episodes: int = 10,
    max_steps: int = 200,
    device: str = "cuda",
    render: bool = False,
):
    """评估 PPO 策略"""
    print("=" * 60)
    print("PPO 策略评估:")
    print(f"  检查点: {checkpoint_path}")
    print(f"  Episodes: {num_episodes}")
    print(f"  最大步数: {max_steps}")
    print(f"  设备: {device}")
    print("=" * 60)
    print()

    # 创建环境
    print("创建环境...")
    env = ManiSkillPickCubeEnv(
        robot_type="panda",
        image_size=(128, 128),
        use_cameras=True,
        headless=not render,
    )
    robot_config = env.get_robot_config()

    # 加载策略
    print("加载策略...")
    policy = PPOPolicy(
        image_size=(128, 128),
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

    print(f"✓ 策略已加载 (update: {checkpoint.get('update', 'N/A')}, mean_reward: {checkpoint.get('mean_reward', 0):.2f})")

    # 评估
    print(f"\n开始评估...")
    print()

    success_count = 0
    episode_rewards = []
    episode_lengths = []

    for episode_idx in tqdm(range(num_episodes), desc="评估"):
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0

        for step in range(max_steps):
            # 选择动作（使用均值，不采样）
            with torch.no_grad():
                actions_mean, _ = policy._obs_to_tensor(obs, device=device)
                action = actions_mean.cpu().numpy()[0]  # 使用确定性动作

            # 执行动作
            next_obs, reward, terminated, truncated, info = env.step(action)

            # 处理 Tensor 类型的 reward
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


def main():
    parser = argparse.ArgumentParser(description="评估 PPO 策略")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/ppo_final_checkpoint.pt", help="检查点文件路径")
    parser.add_argument("--episodes", type=int, default=10, help="评估的 episode 数量")
    parser.add_argument("--max-steps", type=int, default=200, help="每个 episode 的最大步数")
    parser.add_argument("--device", type=str, default="cuda", help="运行设备")
    parser.add_argument("--render", action="store_true", help="是否渲染环境")

    args = parser.parse_args()

    evaluate_ppo(
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        device=args.device,
        render=args.render,
    )


if __name__ == "__main__":
    main()
