"""简化的 PPO 训练脚本 - 实时显示训练进度"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from collections import deque

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from waleo.sim.tasks.maniskill_pick_cube import ManiSkillPickCubeEnv

# 简化版策略网络
class SimplePPOPolicy(nn.Module):
    def __init__(self, state_dim: int = 9, action_dim: int = 8, hidden_dim: int = 256):
        super().__init__()

        # Actor 网络（策略）
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Critic 网络（价值）
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # 动作 log_std
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        action_mean = self.actor(state)
        value = self.critic(state)
        return action_mean, value

    def get_action(self, state, deterministic=False, device="cpu"):
        with torch.no_grad():
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
            action_mean, value = self.forward(state_t)
            std = torch.exp(self.log_std)

            if deterministic:
                action = action_mean
            else:
                dist = torch.distributions.Normal(action_mean, std)
                action = dist.sample()

            return action.cpu().numpy()[0], value.cpu().item()


def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = 0
    last_value = 0

    for t in reversed(range(len(rewards))):
        if dones[t]:
            last_value = 0
            last_advantage = 0

        delta = rewards[t] + gamma * last_value - values[t]
        advantages[t] = last_advantage = delta + gamma * gae_lambda * last_advantage
        last_value = values[t]

    returns = advantages + values
    return advantages, returns


def train_ppo_simple():
    """简化的 PPO 训练"""

    print("=" * 60)
    print("PPO 训练 - 10万步")
    print("=" * 60)

    # 创建环境（不使用相机，加快训练速度）
    print("\n创建环境...")
    env = ManiSkillPickCubeEnv(
        robot_type="panda",
        image_size=(128, 128),
        use_cameras=False,  # 不使用相机，只用状态
        headless=True,
    )
    robot_config = env.get_robot_config()
    state_dim = robot_config["state_dim"]
    action_dim = robot_config["action_dim"]
    print(f"✓ 环境创建成功 (state_dim={state_dim}, action_dim={action_dim})")

    # 创建策略
    print("\n创建策略网络...")
    policy = SimplePPOPolicy(state_dim=state_dim, action_dim=action_dim, hidden_dim=256)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = policy.to(device)

    # 优化器
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    # PPO 超参数
    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.2
    value_coef = 0.5
    entropy_coef = 0.01

    # 训练配置
    num_updates = 1000  # 1000 updates × 100 steps = 100k steps
    steps_per_update = 100
    batch_size = 32
    ppo_epochs = 5

    print(f"✓ 训练配置:")
    print(f"  总更新数: {num_updates}")
    print(f"  每次更新步数: {steps_per_update}")
    print(f"  总步数: {num_updates * steps_per_update}")
    print(f"  设备: {device}")

    # 训练循环
    print("\n开始训练...\n")

    episode_rewards = deque(maxlen=100)
    global_step = 0

    for update in range(1, num_updates + 1):
        # 收集经验
        states = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        dones = []

        obs, _ = env.reset()
        episode_reward = 0

        for _ in range(steps_per_update):
            global_step += 1

            # 提取 robot_state（简化版只使用状态，不使用图像）
            state = obs.get("robot_state", obs)
            # 处理 torch.Tensor 类型
            if hasattr(state, 'cpu'):
                state = state.cpu().numpy()

            # 获取动作
            with torch.no_grad():
                action, value = policy.get_action(state, deterministic=False, device=device)

            # 执行动作
            next_obs, reward, terminated, truncated, info = env.step(action)

            # 处理 reward
            if hasattr(reward, 'item'):
                reward = reward.item()

            # 存储经验
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            dones.append(terminated or truncated)

            episode_reward += reward
            obs = next_obs

            if terminated or truncated:
                episode_rewards.append(episode_reward)
                episode_reward = 0
                obs, _ = env.reset()

        # 转换为 numpy arrays
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        values = np.array(values, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        # 计算 GAE
        advantages, returns = compute_gae(rewards, values, dones, gamma, gae_lambda)

        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 转换为 tensor
        states_t = torch.from_numpy(states).float().to(device)
        actions_t = torch.from_numpy(actions).float().to(device)
        old_log_probs = []
        advantages_t = torch.from_numpy(advantages).float().to(device)
        returns_t = torch.from_numpy(returns).float().to(device)

        # 计算旧 log_probs
        with torch.no_grad():
            action_mean, _ = policy(states_t)
            std = torch.exp(policy.log_std)
            dist = torch.distributions.Normal(action_mean, std)
            old_log_probs = dist.log_prob(actions_t).sum(dim=-1)

        # PPO 更新
        total_policy_loss = 0
        total_value_loss = 0
        num_updates_count = 0

        for _ in range(ppo_epochs):
            indices = np.random.permutation(len(states))

            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                batch_states = states_t[batch_indices]
                batch_actions = actions_t[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages_t[batch_indices]
                batch_returns = returns_t[batch_indices]

                # 前向传播
                action_mean, value = policy(batch_states)
                std = torch.exp(policy.log_std)
                dist = torch.distributions.Normal(action_mean, std)
                log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)

                # PPO 损失
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(value.squeeze(), batch_returns)

                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_updates_count += 1

        # 打印进度
        if update % 10 == 0:
            mean_reward = np.mean(episode_rewards) if len(episode_rewards) > 0 else 0
            print(f"Update {update}/{num_updates} | Step {global_step} | "
                  f"Mean Reward: {mean_reward:.2f} | "
                  f"Policy Loss: {total_policy_loss/num_updates_count:.4f} | "
                  f"Value Loss: {total_value_loss/num_updates_count:.4f}")

        # 保存检查点
        if update % 100 == 0:
            checkpoint_path = Path("checkpoints") / f"ppo_simple_update_{update}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "update": update,
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, checkpoint_path)
            print(f"  ✓ 保存检查点: {checkpoint_path}")

    # 保存最终模型
    final_path = Path("checkpoints") / "ppo_simple_final.pt"
    torch.save({
        "update": num_updates,
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, final_path)

    print("\n" + "=" * 60)
    print(f"训练完成！总步数: {global_step}")
    print(f"最终模型: {final_path}")
    print("=" * 60)

    env.close()

    return policy


if __name__ == "__main__":
    train_ppo_simple()
