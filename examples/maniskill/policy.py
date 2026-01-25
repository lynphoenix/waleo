"""简单的策略模型

基于视觉和状态输入的简单神经网络策略。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import numpy as np


class SimpleCNN(nn.Module):
    """简单的 CNN 用于处理图像"""

    def __init__(self, input_channels: int = 3, feature_dim: int = 256, image_size: Tuple[int, int] = (224, 224)):
        super().__init__()
        self.feature_dim = feature_dim
        self.image_size = image_size  # (H, W)

        # 简单的 CNN 架构
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)  # 输入32通道
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)  # 输入64通道

        # 动态计算全连接层输入大小
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, *image_size)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            self.fc_input_size = x.view(1, -1).shape[1]

        self.fc = nn.Linear(self.fc_input_size, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) 图像张量

        Returns:
            features: (B, feature_dim) 特征向量
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x


class VisualPolicy(nn.Module):
    """基于视觉的简单策略网络

    使用双相机图像 + 机器人状态预测动作
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        state_dim: int = 9,
        action_dim: int = 8,  # ManiSkill3 Panda: 7关节 + 1夹爪
        hidden_dim: int = 256,
        feature_dim: int = 256,
    ):
        """
        Args:
            image_size: 图像大小 (H, W)
            state_dim: 机器人状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
            feature_dim: CNN 特征维度
        """
        super().__init__()

        self.image_size = image_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim

        # 两个相机的 CNN
        self.head_camera_cnn = SimpleCNN(3, feature_dim, image_size)
        self.wrist_camera_cnn = SimpleCNN(3, feature_dim, image_size)

        # 融合层
        fusion_input_dim = 2 * feature_dim + state_dim
        self.fc1 = nn.Linear(fusion_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            observations: 观察字典，包含
                - head_camera_rgb: (B, 3, H, W)
                - wrist_camera_rgb: (B, 3, H, W)
                - robot_state: (B, state_dim)

        Returns:
            actions: (B, action_dim) 动作
        """
        # 提取相机图像特征
        head_features = self.head_camera_cnn(observations["head_camera_rgb"])
        wrist_features = self.wrist_camera_cnn(observations["wrist_camera_rgb"])

        # 提取机器人状态
        robot_state = observations["robot_state"]

        # 融合所有特征
        fused = torch.cat([head_features, wrist_features, robot_state], dim=1)

        # 通过 MLP
        x = F.relu(self.fc1(fused))
        x = F.relu(self.fc2(x))
        actions = torch.tanh(self.fc3(x))  # 输出范围 [-1, 1]

        return actions

    def select_action(self, observations: Dict[str, np.ndarray]) -> np.ndarray:
        """选择动作（推理模式）

        Args:
            observations: numpy 数组格式的观察

        Returns:
            action: numpy 数组格式的动作
        """
        self.eval()
        with torch.no_grad():
            # 转换为张量
            obs_tensor = self._obs_to_tensor(observations)

            # 前向传播
            actions = self.forward(obs_tensor)

            # 转换回 numpy
            action = actions.cpu().numpy()[0]

        self.train()
        return action

    def _obs_to_tensor(self, observations: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """将观察转换为张量

        Args:
            observations: numpy 数组格式的观察

        Returns:
            torch 张量格式的观察
        """
        device = next(self.parameters()).device

        obs_tensor = {}

        # 处理图像
        for key in ["head_camera_rgb", "wrist_camera_rgb"]:
            if key in observations:
                img = observations[key]
                # (H, W, C) -> (1, C, H, W)
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                img_tensor = torch.from_numpy(img).float() / 255.0
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
                obs_tensor[key] = img_tensor.to(device)

        # 处理状态
        if "robot_state" in observations:
            state = observations["robot_state"]
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            obs_tensor["robot_state"] = state_tensor.to(device)

        return obs_tensor


class PolicyTrainer:
    """策略训练器"""

    def __init__(
        self,
        policy: nn.Module,
        lr: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            policy: 策略网络
            lr: 学习率
            device: 训练设备
        """
        self.policy = policy.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def train_step(
        self,
        observations: Dict[str, torch.Tensor],
        target_actions: torch.Tensor,
    ) -> float:
        """训练一步

        Args:
            observations: 观察张量
            target_actions: 目标动作张量

        Returns:
            loss: 损失值
        """
        self.policy.train()

        # 前向传播
        predicted_actions = self.policy(observations)

        # 计算 MSE 损失
        loss = F.mse_loss(predicted_actions, target_actions)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save_checkpoint(self, path: str, epoch: int, loss: float):
        """保存检查点"""
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
        }, path)
        print(f"✓ 检查点已保存: {path}")

    def load_checkpoint(self, path: str) -> int:
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        print(f"✓ 检查点已加载: {path} (epoch={epoch}, loss={loss:.4f})")
        return epoch
