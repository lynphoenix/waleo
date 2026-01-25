"""简化版 PPO 训练 - 基于 ManiSkill 实现,输出到文件

适合快速训练和查看结果
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from pathlib import Path
import sys

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from waleo.sim.tasks.maniskill_pick_cube import ManiSkillPickCubeEnv


# ============================================
# 简单的特征提取网络
# ============================================

class StateFeatureNet(nn.Module):
    """状态特征提取网络 (不使用图像)"""

    def __init__(self, state_dim=9, feature_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
            nn.ReLU(),
        )
        self.out_features = feature_dim

    def forward(self, observations):
        return self.net(observations["robot_state"])


class PPOAgent(nn.Module):
    """PPO 智能体 (简化版 - 只使用状态)"""

    def __init__(self, state_dim=9, action_dim=8, hidden_dim=256):
        super().__init__()

        # 特征网络
        self.feature_net = StateFeatureNet(state_dim, hidden_dim)
        latent_size = hidden_dim

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        # Actor
        self.actor_mean = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim) - 0.5)

    def get_value(self, state_dict):
        features = self.feature_net(state_dict)
        return self.critic(features)

    def get_action(self, state_dict, deterministic=False):
        features = self.feature_net(state_dict)
        action_mean = self.actor_mean(features)
        if deterministic:
            return action_mean
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, state_dict, action=None):
        features = self.feature_net(state_dict)
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(features)


# ============================================
# PPO 训练
# ============================================

def compute_gae(rewards, values, dones, gamma, gae_lambda):
    """计算 GAE"""
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


def train_ppo_simple(
    output_file="training_log.txt",
    total_steps=100000,
    steps_per_update=100,
    num_epochs=5,
    batch_size=32,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_coef=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    device=None,
):
    """简化版 PPO 训练"""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 打开日志文件
    log_file = open(output_file, "w")

    def log(msg):
        """同时输出到文件和终端"""
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    log("=" * 80)
    log("PPO 训练 - 简化版 (只使用状态)")
    log("=" * 80)
    log(f"设备: {device}")
    log(f"总步数: {total_steps}")
    log(f"每次更新步数: {steps_per_update}")
    log(f"更新次数: {total_steps // steps_per_update}")
    log("=" * 80)
    log("")

    # 创建环境
    log("创建环境...")
    env = ManiSkillPickCubeEnv(
        robot_type="panda",
        image_size=(128, 128),
        use_cameras=False,  # 不使用相机
        headless=True,
    )
    robot_config = env.get_robot_config()
    state_dim = robot_config["state_dim"]
    action_dim = robot_config["action_dim"]
    log(f"✓ state_dim={state_dim}, action_dim={action_dim}")

    # 创建智能体
    log("创建智能体...")
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, hidden_dim=256).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
    log(f"✓ 智能体已创建")
    log("")

    # 训练循环
    log("开始训练...")
    log("")

    num_updates = total_steps // steps_per_update
    global_step = 0
    episode_rewards = []
    start_time = time.time()

    for update in range(1, num_updates + 1):
        # ==================== Rollout ====================
        agent.eval()

        states = []  # 状态列表
        actions = []  # 动作列表
        log_probs = []  # log_prob 列表
        rewards = []  # 奖励列表
        values = []  # 价值列表
        dones = []  # done 标志

        obs, _ = env.reset()
        episode_reward = 0

        for step in range(steps_per_update):
            global_step += 1

            # 准备状态
            state = obs.get("robot_state", obs)
            if hasattr(state, 'cpu'):
                state = state.cpu().numpy()

            # 转换为 tensor
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            state_dict = {"robot_state": state_tensor}

            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(state_dict)
                value = value.cpu().item()

            # 执行动作
            action_np = action.cpu().numpy()[0]
            next_obs, reward, terminated, truncated, info = env.step(action_np)

            # 处理 reward
            if hasattr(reward, 'item'):
                reward = reward.item()

            # 存储数据
            states.append(state)
            actions.append(action_np)
            log_probs.append(log_prob.cpu().item())
            rewards.append(reward)
            values.append(value)
            dones.append(terminated or truncated)

            episode_reward += reward
            obs = next_obs

            if terminated or truncated:
                episode_rewards.append(episode_reward)
                episode_reward = 0
                obs, _ = env.reset()

        # ==================== 计算 GAE ====================
        advantages, returns = compute_gae(
            np.array(rewards),
            np.array(values),
            np.array(dones, dtype=np.float32),
            gamma,
            gae_lambda,
        )

        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 转换为 tensor
        states_t = torch.from_numpy(np.array(states)).float().to(device)
        actions_t = torch.from_numpy(np.array(actions)).float().to(device)
        advantages_t = torch.from_numpy(advantages).float().to(device)
        returns_t = torch.from_numpy(returns).float().to(device)

        # 计算旧 log_probs
        with torch.no_grad():
            state_dict = {"robot_state": states_t}
            _, old_log_probs, _, _ = agent.get_action_and_value(state_dict, actions_t)
            old_log_probs = old_log_probs.cpu()

        # ==================== PPO 更新 ====================
        agent.train()

        total_policy_loss = 0
        total_value_loss = 0
        update_count = 0

        indices = np.random.permutation(len(states))

        for epoch in range(num_epochs):
            for start in range(0, len(states), batch_size):
                end = min(start + batch_size, len(states))
                batch_indices = indices[start:end]

                batch_states = states_t[batch_indices]
                batch_actions = actions_t[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages_t[batch_indices]
                batch_returns = returns_t[batch_indices]

                # 前向传播
                state_dict = {"robot_state": batch_states}
                _, new_log_probs, entropy, new_values = agent.get_action_and_value(state_dict, batch_actions)
                new_log_probs = new_log_probs  # 保持在 GPU
                entropy = entropy.mean()
                new_values = new_values.squeeze()

                # 计算 ratio
                logratio = new_log_probs - batch_old_log_probs.to(device)
                ratio = logratio.exp()

                # PPO loss
                pg_loss1 = -batch_advantages * ratio
                pg_loss2 = -batch_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss = 0.5 * ((new_values - batch_returns) ** 2).mean()

                # Total loss
                loss = pg_loss + vf_coef * v_loss - ent_coef * entropy

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_policy_loss += pg_loss.item()
                total_value_loss += v_loss.item()
                update_count += 1

        # ==================== 打印进度 ====================
        if update % 10 == 0 or update == 1:
            mean_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) > 0 else 0
            elapsed = time.time() - start_time
            sps = int(global_step / elapsed)
            eta = (total_steps - global_step) / sps if sps > 0 else 0

            log(f"Update {update}/{num_updates} | "
                f"Step {global_step}/{total_steps} | "
                f"Mean Reward: {mean_reward:.2f} | "
                f"Policy Loss: {total_policy_loss/update_count:.4f} | "
                f"Value Loss: {total_value_loss/update_count:.4f} | "
                f"SPS: {sps} | "
                f"ETA: {eta/60:.1f}min")

        # ==================== 保存检查点 ====================
        if update % 100 == 0:
            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"ppo_simple_update_{update}.pt"
            torch.save({
                "update": update,
                "model_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "mean_reward": np.mean(episode_rewards[-100:]) if len(episode_rewards) > 0 else 0,
            }, checkpoint_path)
            log(f"  ✓ 保存检查点: {checkpoint_path}")

    # ==================== 保存最终模型 ====================
    final_path = Path("checkpoints") / "ppo_simple_final.pt"
    torch.save({
        "update": num_updates,
        "model_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "mean_reward": np.mean(episode_rewards[-100:]) if len(episode_rewards) > 0 else 0,
    }, final_path)

    elapsed = time.time() - start_time

    log("")
    log("=" * 80)
    log("训练完成!")
    log(f"总步数: {global_step}")
    log(f"总时间: {elapsed:.2f}s ({elapsed/60:.1f}min)")
    log(f"最终平均奖励: {np.mean(episode_rewards[-100:]) if len(episode_rewards) > 0 else 0:.2f}")
    log(f"最终模型: {final_path}")
    log("=" * 80)

    log_file.close()
    env.close()

    return agent


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="简化版 PPO 训练")
    parser.add_argument("--total-steps", type=int, default=100000, help="总训练步数")
    parser.add_argument("--output", type=str, default="training_log.txt", help="输出日志文件")
    parser.add_argument("--device", type=str, default=None, help="训练设备 (cuda/cpu)")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")

    args = parser.parse_args()

    train_ppo_simple(
        output_file=args.output,
        total_steps=args.total_steps,
        learning_rate=args.lr,
        gamma=args.gamma,
        device=args.device,
    )
