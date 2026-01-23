"""评估 PPO 策略 - 视觉版本

定期评估训练中的模型效果
"""

import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from waleo_sim.tasks.maniskill_pick_cube import ManiSkillPickCubeEnv
from examples.maniskill.train_ppo_visual import PPOAgent, prepare_obs


def evaluate_ppo_visual(
    checkpoint_path: str,
    num_episodes: int = 10,
    max_steps: int = 200,
    device: str = "cuda",
    render: bool = False,
    image_size=(64, 64),
):
    """评估视觉 PPO 策略"""
    print("=" * 60)
    print("PPO 视觉策略评估")
    print("=" * 60)
    print(f"检查点: {checkpoint_path}")
    print(f"Episodes: {num_episodes}")
    print(f"最大步数: {max_steps}")
    print(f"设备: {device}")
    print(f"图像尺寸: {image_size}")
    print("=" * 60)
    print()

    # 创建环境
    print("创建环境...")
    env = ManiSkillPickCubeEnv(
        robot_type="panda",
        image_size=image_size,
        use_cameras=True,  # 只启用头部相机
        headless=not render,
    )
    robot_config = env.get_robot_config()
    state_dim = robot_config["state_dim"]
    action_dim = robot_config["action_dim"]

    # 加载策略
    print("加载策略...")
    policy = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        image_size=image_size,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.to(device)
    policy.eval()

    print(f"✓ 策略已加载 (update: {checkpoint.get('update', 'N/A')})")
    print()

    # 评估
    print(f"开始评估 ({num_episodes} episodes)...")
    print()

    success_count = 0
    episode_rewards = []
    episode_lengths = []

    for episode_idx in tqdm(range(num_episodes), desc="评估"):
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0

        for step in range(max_steps):
            # 选择动作（确定性）
            with torch.no_grad():
                obs_tensor = prepare_obs(obs, device)
                action = policy.get_action(obs_tensor, deterministic=True)
                action_np = action.cpu().numpy()[0]

            # 执行动作
            next_obs, reward, terminated, truncated, info = env.step(action_np)

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

    env.close()

    return {
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "avg_length": avg_length,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="评估视觉 PPO 策略")
    parser.add_argument("--checkpoint", type=str, required=True, help="检查点文件路径")
    parser.add_argument("--episodes", type=int, default=10, help="评估的 episode 数量")
    parser.add_argument("--max-steps", type=int, default=200, help="每个 episode 的最大步数")
    parser.add_argument("--device", type=str, default="cuda", help="运行设备")
    parser.add_argument("--render", action="store_true", help="是否渲染环境")
    parser.add_argument("--image-size", type=int, default=64, help="图像尺寸")

    args = parser.parse_args()

    evaluate_ppo_visual(
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        device=args.device,
        render=args.render,
        image_size=(args.image_size, args.image_size),
    )
