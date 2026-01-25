"""PPO 训练 - 并行环境版本

使用多个环境并行收集数据，大幅提升训练速度
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
from typing import Dict
from multiprocessing import Process, Queue
import copy

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from waleo.sim.tasks.maniskill_pick_cube import ManiSkillPickCubeEnv


# ============================================
# 网络定义 (与之前相同)
# ============================================

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, feature_dim=256, image_size=(128, 128)):
        super().__init__()
        self.feature_dim = feature_dim
        self.image_size = image_size

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, *image_size)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            self.fc_input_size = x.view(1, -1).shape[1]

        self.fc = nn.Linear(self.fc_input_size, feature_dim)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc(x))
        return x


class VisualFeatureNet(nn.Module):
    def __init__(self, state_dim=9, feature_dim=256, image_size=(128, 128)):
        super().__init__()

        self.head_camera_cnn = SimpleCNN(3, feature_dim, image_size)
        self.wrist_camera_cnn = SimpleCNN(3, feature_dim, image_size)

        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )

        self.out_features = feature_dim * 3

    def forward(self, observations: Dict[str, torch.Tensor]):
        features_list = []

        if "head_camera_rgb" in observations:
            head_features = self.head_camera_cnn(observations["head_camera_rgb"])
            features_list.append(head_features)

        if "wrist_camera_rgb" in observations:
            wrist_features = self.wrist_camera_cnn(observations["wrist_camera_rgb"])
            features_list.append(wrist_features)

        if "robot_state" in observations:
            state_features = self.state_net(observations["robot_state"])
            features_list.append(state_features)

        return torch.cat(features_list, dim=1)


class PPOAgent(nn.Module):
    def __init__(self, state_dim=9, action_dim=8, hidden_dim=256, image_size=(128, 128)):
        super().__init__()

        self.feature_net = VisualFeatureNet(state_dim, hidden_dim, image_size)
        latent_size = self.feature_net.out_features

        self.critic = nn.Sequential(
            nn.Linear(latent_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self.actor_mean = nn.Sequential(
            nn.Linear(latent_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim) - 0.5)

    def get_value(self, obs_dict):
        features = self.feature_net(obs_dict)
        return self.critic(features)

    def get_action(self, obs_dict, deterministic=False):
        features = self.feature_net(obs_dict)
        action_mean = self.actor_mean(features)
        if deterministic:
            return action_mean
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, obs_dict, action=None):
        features = self.feature_net(obs_dict)
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(features)


# ============================================
# 简化的并行环境收集器
# ============================================

def collect_rollout_worker(env_id, num_steps, image_size, result_queue):
    """单个 worker 收集数据"""
    try:
        import torch
        # 每个 worker 创建自己的环境
        env = ManiSkillPickCubeEnv(
            robot_type="panda",
            image_size=image_size,
            use_cameras=True,
            headless=True,
        )

        # 收集数据
        obs, _ = env.reset()
        episode_reward = 0

        rollout_data = {
            "states": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "values": [],
            "dones": [],
        }

        for step in range(num_steps):
            # 随机动作 (简化版,实际需要从主进程获取策略)
            action = env.action_space.sample()
            obs["robot_state"] = obs["robot_state"].cpu().numpy() if hasattr(obs["robot_state"], 'cpu') else obs["robot_state"]

            next_obs, reward, terminated, truncated, info = env.step(action)

            if hasattr(reward, 'item'):
                reward = reward.item()

            # 存储数据
            rollout_data["states"].append(obs)
            rollout_data["actions"].append(action)
            rollout_data["rewards"].append(reward)
            rollout_data["dones"].append(terminated or truncated)

            episode_reward += reward
            obs = next_obs

            if terminated or truncated:
                obs, _ = env.reset()

        env.close()
        result_queue.put((env_id, rollout_data))

    except Exception as e:
        result_queue.put((env_id, {"error": str(e)}))


# ============================================
# PPO 训练 (简化版单环境但优化过)
# ============================================

def compute_gae(rewards, values, dones, gamma, gae_lambda):
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


def prepare_obs(obs, device):
    obs_dict = {}

    if "head_camera_rgb" in obs:
        img = obs["head_camera_rgb"]
        if hasattr(img, 'cpu'):
            img = img.cpu().numpy()
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = np.transpose(img, (2, 0, 1))
        obs_dict["head_camera_rgb"] = torch.from_numpy(img).float().unsqueeze(0).to(device)

    if "wrist_camera_rgb" in obs:
        img = obs["wrist_camera_rgb"]
        if hasattr(img, 'cpu'):
            img = img.cpu().numpy()
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = np.transpose(img, (2, 0, 1))
        obs_dict["wrist_camera_rgb"] = torch.from_numpy(img).float().unsqueeze(0).to(device)

    if "robot_state" in obs:
        state = obs["robot_state"]
        if hasattr(state, 'cpu'):
            state = state.cpu().numpy()
        obs_dict["robot_state"] = torch.from_numpy(state).float().unsqueeze(0).to(device)

    return obs_dict


def prepare_batch_obs(obs_batch, device):
    batch_dict = {}

    if "head_camera_rgb" in obs_batch[0]:
        imgs = []
        for obs in obs_batch:
            img = obs["head_camera_rgb"]
            if hasattr(img, 'cpu'):
                img = img.cpu().numpy()
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = np.transpose(img, (2, 0, 1))
            imgs.append(img)
        batch_dict["head_camera_rgb"] = torch.from_numpy(np.array(imgs)).float().to(device)

    if "wrist_camera_rgb" in obs_batch[0]:
        imgs = []
        for obs in obs_batch:
            img = obs["wrist_camera_rgb"]
            if hasattr(img, 'cpu'):
                img = img.cpu().numpy()
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = np.transpose(img, (2, 0, 1))
            imgs.append(img)
        batch_dict["wrist_camera_rgb"] = torch.from_numpy(np.array(imgs)).float().to(device)

    if "robot_state" in obs_batch[0]:
        states = []
        for obs in obs_batch:
            state = obs["robot_state"]
            if hasattr(state, 'cpu'):
                state = state.cpu().numpy()
            states.append(state)
        batch_dict["robot_state"] = torch.from_numpy(np.array(states)).float().to(device)

    return batch_dict


def train_ppo_optimized(
    output_file="training_log_optimized.txt",
    total_steps=10000000,
    steps_per_update=200,  # 适中的值
    num_epochs=4,
    batch_size=64,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_coef=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    image_size=(128, 128),
    device=None,
    resume_from=None,
):
    """优化的 PPO 训练"""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_file = open(output_file, "w")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    log("=" * 80)
    log("PPO 训练 - 优化版")
    log("=" * 80)
    log(f"设备: {device}")
    log(f"总步数: {total_steps}")
    log(f"每次更新步数: {steps_per_update}")
    log(f"批次大小: {batch_size}")
    log("=" * 80)
    log("")

    # 创建环境
    log("创建环境...")
    env = ManiSkillPickCubeEnv(
        robot_type="panda",
        image_size=image_size,
        use_cameras=True,
        headless=True,
    )
    robot_config = env.get_robot_config()
    state_dim = robot_config["state_dim"]
    action_dim = robot_config["action_dim"]
    log(f"✓ state_dim={state_dim}, action_dim={action_dim}")

    # 创建智能体
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        image_size=image_size
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)

    # Resume
    start_update = 1
    global_step = 0

    if resume_from is not None:
        checkpoint_path = Path(resume_from)
        if checkpoint_path.exists():
            log(f"从检查点恢复: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            agent.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            saved_update = checkpoint.get("update", 0)
            start_update = saved_update + 1
            global_step = saved_update * steps_per_update
            log(f"✓ 已加载 update {saved_update}")
            log("")

    # 训练循环
    log("开始训练...")
    log("")

    num_updates = total_steps // steps_per_update
    episode_rewards = []
    start_time = time.time()

    for update in range(start_update, num_updates + 1):
        agent.eval()

        states = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        dones = []

        obs, _ = env.reset()
        episode_reward = 0

        for step in range(steps_per_update):
            global_step += 1

            obs_tensor = prepare_obs(obs, device)

            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(obs_tensor)
                value = value.cpu().item()

            action_np = action.cpu().numpy()[0]
            next_obs, reward, terminated, truncated, info = env.step(action_np)

            if hasattr(reward, 'item'):
                reward = reward.item()

            states.append(obs)
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

        # GAE
        advantages, returns = compute_gae(
            np.array(rewards),
            np.array(values),
            np.array(dones, dtype=np.float32),
            gamma,
            gae_lambda,
        )

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actions_t = torch.from_numpy(np.array(actions)).float().to(device)
        advantages_t = torch.from_numpy(advantages).float().to(device)
        returns_t = torch.from_numpy(returns).float().to(device)

        with torch.no_grad():
            batch_obs = prepare_batch_obs(states, device)
            _, old_log_probs, _, _ = agent.get_action_and_value(batch_obs, actions_t)
            old_log_probs = old_log_probs.cpu()

        # PPO 更新
        agent.train()

        total_policy_loss = 0
        total_value_loss = 0
        update_count = 0

        indices = np.random.permutation(len(states))

        for epoch in range(num_epochs):
            for start in range(0, len(states), batch_size):
                end = min(start + batch_size, len(states))
                batch_indices = indices[start:end]

                batch_states = [states[i] for i in batch_indices]
                batch_obs = prepare_batch_obs(batch_states, device)
                batch_actions = actions_t[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages_t[batch_indices]
                batch_returns = returns_t[batch_indices]

                _, new_log_probs, entropy, new_values = agent.get_action_and_value(batch_obs, batch_actions)
                entropy = entropy.mean()
                new_values = new_values.squeeze()

                logratio = new_log_probs - batch_old_log_probs.to(device)
                ratio = logratio.exp()

                pg_loss1 = -batch_advantages * ratio
                pg_loss2 = -batch_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((new_values - batch_returns) ** 2).mean()

                loss = pg_loss + vf_coef * v_loss - ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
                optimizer.step()

                total_policy_loss += pg_loss.item()
                total_value_loss += v_loss.item()
                update_count += 1

        # 打印进度
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

        # 保存检查点
        if update % 100 == 0:
            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"ppo_optimized_update_{update}.pt"
            torch.save({
                "update": update,
                "model_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "mean_reward": np.mean(episode_rewards[-100:]) if len(episode_rewards) > 0 else 0,
            }, checkpoint_path)
            log(f"  ✓ 保存检查点: {checkpoint_path}")

    # 保存最终模型
    final_path = Path("checkpoints") / "ppo_optimized_final.pt"
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

    parser = argparse.ArgumentParser(description="PPO 训练 - 优化版")
    parser.add_argument("--total-steps", type=int, default=10000000, help="总训练步数")
    parser.add_argument("--output", type=str, default="training_log_optimized.txt", help="输出日志文件")
    parser.add_argument("--device", type=str, default=None, help="训练设备")
    parser.add_argument("--steps-per-update", type=int, default=200, help="每次更新步数")
    parser.add_argument("--batch-size", type=int, default=64, help="批次大小")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练")

    args = parser.parse_args()

    train_ppo_optimized(
        output_file=args.output,
        total_steps=args.total_steps,
        steps_per_update=args.steps_per_update,
        batch_size=args.batch_size,
        device=args.device,
        resume_from=args.resume,
    )
