"""PPO 训练脚本

使用 PPO 算法训练 ManiSkill3 PickCube 任务。
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from collections import deque
from typing import Dict, Tuple

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from waleo.sim.tasks.maniskill_pick_cube import ManiSkillPickCubeEnv
from examples.maniskill.ppo_policy import PPOPolicy, RolloutBuffer, compute_gae


class PPOTrainer:
    """PPO 训练器"""

    def __init__(
        self,
        env: ManiSkillPickCubeEnv,
        policy: PPOPolicy,
        device: str = "cuda",
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ):
        """
        Args:
            env: 环境
            policy: PPO 策略
            device: 训练设备
            learning_rate: 学习率
            gamma: 折扣因子
            gae_lambda: GAE 参数
            clip_epsilon: PPO clipping 参数
            value_coef: 价值函数损失系数
            entropy_coef: 熵正则系数
            max_grad_norm: 最大梯度范数
        """
        self.env = env
        self.policy = policy.to(device)
        self.device = device

        # PPO 超参数
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # 优化器
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        # 统计信息
        self.episode_rewards = deque(maxlen=100)

    def collect_rollouts(
        self,
        num_steps: int,
    ) -> RolloutBuffer:
        """收集经验数据

        Args:
            num_steps: 收集的步数

        Returns:
            buffer: 经验回放缓冲区
        """
        buffer = RolloutBuffer(capacity=num_steps)

        obs, _ = self.env.reset()
        episode_reward = 0

        for _ in range(num_steps):
            # 获取动作
            with torch.no_grad():
                action, log_prob = self.policy.get_action(obs, device=self.device)

            # 执行动作
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            # 获取价值估计
            with torch.no_grad():
                _, value = self.policy._obs_to_tensor(obs, device=self.device)
                value = value.cpu().item()

            # 存储经验
            done = terminated or truncated
            buffer.add(obs, action, log_prob, reward, value, done)

            episode_reward += reward
            obs = next_obs

            if done:
                self.episode_rewards.append(episode_reward)
                episode_reward = 0
                obs, _ = self.env.reset()

        return buffer

    def train_step(
        self,
        buffer: RolloutBuffer,
        num_epochs: int = 10,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        """PPO 训练步骤

        Args:
            buffer: 经验缓冲区
            num_epochs: PPO 更新轮数
            batch_size: 批大小

        Returns:
            metrics: 训练指标
        """
        # 获取数据
        observations, actions, old_log_probs, rewards, old_values, dones = buffer.get()

        # 计算 GAE 优势函数
        advantages, returns = compute_gae(
            rewards,
            old_values,
            dones,
            self.gamma,
            self.gae_lambda,
        )

        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 转换为 tensor
        observations_t = self._obs_batch_to_tensor(observations)
        actions_t = torch.from_numpy(actions).float().to(self.device)
        old_log_probs_t = torch.from_numpy(old_log_probs).float().to(self.device)
        old_values_t = torch.from_numpy(old_values).float().to(self.device)
        advantages_t = torch.from_numpy(advantages).float().to(self.device)
        returns_t = torch.from_numpy(returns).float().to(self.device)

        # 训练多个 epoch
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0

        batch_size = min(batch_size, len(observations))

        for _ in range(num_epochs):
            # 生成随机索引
            indices = np.random.permutation(len(observations))

            for start in range(0, len(observations), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                # 获取 batch 数据
                batch_obs = {k: v[batch_indices] for k, v in observations_t.items()}
                batch_actions = actions_t[batch_indices]
                batch_old_log_probs = old_log_probs_t[batch_indices]
                batch_old_values = old_values_t[batch_indices]
                batch_advantages = advantages_t[batch_indices]
                batch_returns = returns_t[batch_indices]

                # 评估动作
                log_probs, entropy, values = self.policy.evaluate_actions(batch_obs, batch_actions)

                # 计算 PPO 损失
                # 1. Policy Loss (Clipped Surrogate)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 2. Value Loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)

                # 3. Entropy Bonus
                entropy_loss = -entropy.mean()

                # 总损失
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # 记录统计信息
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "mean_reward": np.mean(self.episode_rewards) if len(self.episode_rewards) > 0 else 0,
        }

    def _obs_batch_to_tensor(self, observations: list) -> Dict[str, torch.Tensor]:
        """将观察批次转换为 tensor"""
        batch_size = len(observations)

        # 处理图像
        obs_tensors = {}

        for key in ["head_camera_rgb", "wrist_camera_rgb"]:
            if key in observations[0]:
                imgs = []
                for obs in observations:
                    img = obs[key]
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    img_tensor = torch.from_numpy(img).float() / 255.0
                    img_tensor = img_tensor.permute(2, 0, 1)  # (C, H, W)
                    imgs.append(img_tensor)

                obs_tensors[key] = torch.stack(imgs).to(self.device)

        # 处理状态
        if "robot_state" in observations[0]:
            states = []
            for obs in observations:
                state = obs["robot_state"]
                states.append(torch.from_numpy(state).float())
            obs_tensors["robot_state"] = torch.stack(states).to(self.device)

        return obs_tensors


def train_ppo(
    num_updates: int = 1000,
    num_steps_per_update: int = 2048,
    num_epochs_per_update: int = 10,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    save_dir: str = "./checkpoints",
    device: str = "cuda",
):
    """训练 PPO 策略"""
    print("=" * 60)
    print("PPO 训练配置:")
    print(f"  总更新数: {num_updates}")
    print(f"  每次更新步数: {num_steps_per_update}")
    print(f"  PPO epochs: {num_epochs_per_update}")
    print(f"  批大小: {batch_size}")
    print(f"  学习率: {learning_rate}")
    print(f"  设备: {device}")
    print("=" * 60)
    print()

    # 创建保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 创建环境
    print("创建环境...")
    env = ManiSkillPickCubeEnv(
        robot_type="panda",
        image_size=(128, 128),
        use_cameras=True,
        headless=True,
    )
    robot_config = env.get_robot_config()
    print(f"✓ 环境创建成功 (action_dim={robot_config['action_dim']})")

    # 创建策略
    print("创建 PPO 策略...")
    policy = PPOPolicy(
        image_size=(128, 128),
        state_dim=robot_config["state_dim"],
        action_dim=robot_config["action_dim"],
        hidden_dim=256,
        feature_dim=256,
    )
    print("✓ 策略创建成功")

    # 创建训练器
    trainer = PPOTrainer(
        env=env,
        policy=policy,
        device=device,
        learning_rate=learning_rate,
    )

    # 训练循环
    print(f"\n开始训练...")
    print()

    best_mean_reward = float("-inf")

    for update in range(num_updates):
        # 收集经验
        buffer = trainer.collect_rollouts(num_steps=num_steps_per_update)

        # 训练
        metrics = trainer.train_step(
            buffer=buffer,
            num_epochs=num_epochs_per_update,
            batch_size=batch_size,
        )

        # 打印进度
        if (update + 1) % 10 == 0:
            print(f"Update {update + 1}/{num_updates} | "
                  f"Mean Reward: {metrics['mean_reward']:.2f} | "
                  f"Policy Loss: {metrics['policy_loss']:.4f} | "
                  f"Value Loss: {metrics['value_loss']:.4f} | "
                  f"Entropy: {metrics['entropy']:.4f}")

        # 保存最佳模型
        if metrics['mean_reward'] > best_mean_reward:
            best_mean_reward = metrics['mean_reward']
            if (update + 1) % 50 == 0:
                checkpoint_path = save_dir / f"ppo_best_checkpoint.pt"
                torch.save({
                    "update": update + 1,
                    "model_state_dict": policy.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "mean_reward": metrics['mean_reward'],
                }, checkpoint_path)
                print(f"  ✓ 保存最佳模型 (mean_reward={best_mean_reward:.2f})")

        # 定期保存检查点
        if (update + 1) % 200 == 0:
            checkpoint_path = save_dir / f"ppo_checkpoint_update_{update + 1}.pt"
            torch.save({
                "update": update + 1,
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "mean_reward": metrics['mean_reward'],
            }, checkpoint_path)
            print(f"  ✓ 保存检查点")

    # 保存最终模型
    final_checkpoint_path = save_dir / "ppo_final_checkpoint.pt"
    torch.save({
        "update": num_updates,
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "mean_reward": metrics['mean_reward'],
    }, final_checkpoint_path)

    print()
    print("=" * 60)
    print("训练完成！")
    print(f"最佳平均奖励: {best_mean_reward:.2f}")
    print(f"最终模型已保存: {final_checkpoint_path}")
    print("=" * 60)

    env.close()

    return policy


def main():
    parser = argparse.ArgumentParser(description="PPO 训练 ManiSkill3")
    parser.add_argument("--num-updates", type=int, default=500, help="总更新数")
    parser.add_argument("--num-steps", type=int, default=2048, help="每次更新收集的步数")
    parser.add_argument("--epochs", type=int, default=10, help="PPO 更新轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="批大小")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="保存目录")
    parser.add_argument("--device", type=str, default="cuda", help="训练设备")

    args = parser.parse_args()

    train_ppo(
        num_updates=args.num_updates,
        num_steps_per_update=args.num_steps,
        num_epochs_per_update=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_dir=args.save_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
