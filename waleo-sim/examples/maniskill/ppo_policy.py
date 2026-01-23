"""PPO 算法实现

Proximal Policy Optimization (PPO) 是一个高效的强化学习算法。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Dict, Tuple, Optional
from collections import deque
import gymnasium as gym


class PPOPolicy(nn.Module):
    """PPO 策略网络（Actor + Critic）"""

    def __init__(
        self,
        image_size: Tuple[int, int] = (128, 128),
        state_dim: int = 9,
        action_dim: int = 8,
        hidden_dim: int = 256,
        feature_dim: int = 256,
    ):
        super().__init__()

        self.image_size = image_size
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 共享的 CNN 特征提取器
        self.head_camera_cnn = SimpleCNN(3, feature_dim, image_size)
        self.wrist_camera_cnn = SimpleCNN(3, feature_dim, image_size)

        # Actor 网络（策略网络）
        actor_input_dim = 2 * feature_dim + state_dim
        self.actor_net = nn.Sequential(
            nn.Linear(actor_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Critic 网络（价值网络）
        self.critic_net = nn.Sequential(
            nn.Linear(actor_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # 动作标准差（log_std）
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, observations: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            observations: 观察字典

        Returns:
            actions_mean: 动作均值
            value: 状态价值
        """
        # 提取视觉特征
        head_features = self.head_camera_cnn(observations["head_camera_rgb"])
        wrist_features = self.wrist_camera_cnn(observations["wrist_camera_rgb"])

        # 提取状态
        robot_state = observations["robot_state"]

        # 融合特征
        features = torch.cat([head_features, wrist_features, robot_state], dim=1)

        # Actor 和 Critic
        actions_mean = self.actor_net(features)
        value = self.critic_net(features)

        return actions_mean, value

    def get_action(self, observations: Dict[str, np.ndarray], device: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
        """采样动作（用于探索）

        Args:
            observations: numpy 数组格式的观察
            device: 计算设备

        Returns:
            action: 采样的动作
            log_prob: 动作的对数概率
        """
        actions_mean, value = self._obs_to_tensor(observations, device)
        std = torch.exp(self.log_std)

        # 创建正态分布
        dist = Normal(actions_mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action.cpu().numpy(), log_prob.cpu().numpy()

    def get_action_and_value(self, observations: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取动作、log_prob 和价值（用于训练）"""
        actions_mean, value = self.forward(observations)
        std = torch.exp(self.log_std)

        dist = Normal(actions_mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy, value

    def evaluate_actions(self, observations: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """评估动作（计算 log_prob 和 entropy）"""
        actions_mean, value = self.forward(observations)
        std = torch.exp(self.log_std)

        dist = Normal(actions_mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy, value

    def _obs_to_tensor(self, obs: Dict[str, np.ndarray], device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        """将 numpy 观察转换为 tensor"""
        obs_tensor = {}

        for key in ["head_camera_rgb", "wrist_camera_rgb"]:
            if key in obs:
                img = obs[key]
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                img_tensor = torch.from_numpy(img).float() / 255.0
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
                obs_tensor[key] = img_tensor.to(device)

        if "robot_state" in obs:
            state_tensor = torch.from_numpy(obs["robot_state"]).float().unsqueeze(0)
            obs_tensor["robot_state"] = state_tensor.to(device)

        with torch.no_grad():
            actions_mean, value = self.forward(obs_tensor)

        return actions_mean, value


class SimpleCNN(nn.Module):
    """简单的 CNN 用于处理图像"""

    def __init__(self, input_channels: int = 3, feature_dim: int = 256, image_size: Tuple[int, int] = (128, 128)):
        super().__init__()
        self.feature_dim = feature_dim
        self.image_size = image_size

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # 动态计算全连接层输入大小
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, *image_size)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            self.fc_input_size = x.view(1, -1).shape[1]

        self.fc = nn.Linear(self.fc_input_size, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x


class RolloutBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity: int = 2048):
        self.capacity = capacity
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(
        self,
        obs: Dict,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def get(self) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.observations,
            np.array(self.actions, dtype=np.float32),
            np.array(self.log_probs, dtype=np.float32),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.values, dtype=np.float32),
            np.array(self.dones, dtype=np.float32),
        )

    def clear(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def __len__(self):
        return len(self.rewards)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """计算 GAE (Generalized Advantage Estimation)

    Args:
        rewards: 奖励数组
        values: 价值估计数组
        dones: 终止标志数组
        gamma: 折扣因子
        gae_lambda: GAE 参数

    Returns:
        advantages: 优势估计
        returns: 折扣回报
    """
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
