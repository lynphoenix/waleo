"""训练脚本

使用采集的数据训练视觉策略。
"""

import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse

# 添加项目根目录到 Python 路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from examples.maniskill.policy import VisualPolicy, PolicyTrainer
from waleo.sim.tasks.maniskill_pick_cube import ManiSkillPickCubeEnv


class ManiSkillDataset(Dataset):
    """ManiSkill 数据集"""

    def __init__(self, data_path: str):
        """
        Args:
            data_path: 数据文件路径
        """
        data = np.load(data_path, allow_pickle=True)
        self.episodes = data["data"]

        # 展平所有 transitions
        self.transitions = []
        for episode in self.episodes:
            observations = episode["observations"]
            actions = episode["actions"]

            for i in range(len(observations) - 1):
                self.transitions.append({
                    "obs": observations[i],
                    "action": actions[i],
                    "next_obs": observations[i + 1],
                })

        print(f"加载数据集: {len(self.transitions)} transitions")

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        transition = self.transitions[idx]

        # 处理观察
        obs = transition["obs"]
        action = transition["action"]

        # 转换为张量
        obs_tensor = self._obs_to_tensor(obs)
        action_tensor = torch.from_numpy(action).float()

        return obs_tensor, action_tensor

    def _obs_to_tensor(self, obs):
        """将观察转换为张量"""
        obs_tensor = {}

        # 处理图像
        for key in ["head_camera_rgb", "wrist_camera_rgb"]:
            if key in obs:
                img = obs[key]
                # (H, W, C) -> (C, H, W)
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                img_tensor = torch.from_numpy(img).float() / 255.0
                img_tensor = img_tensor.permute(2, 0, 1)  # (C, H, W)
                obs_tensor[key] = img_tensor

        # 处理状态
        if "robot_state" in obs:
            state = obs["robot_state"]
            state_tensor = torch.from_numpy(state).float()
            obs_tensor["robot_state"] = state_tensor

        return obs_tensor


def collate_fn(batch):
    """自定义 collate 函数"""
    obs_list, action_list = zip(*batch)

    # 合并观察
    batch_obs = {}
    for key in obs_list[0].keys():
        if key == "head_camera_rgb" or key == "wrist_camera_rgb":
            batch_obs[key] = torch.stack([obs[key] for obs in obs_list])
        elif key == "robot_state":
            batch_obs[key] = torch.stack([obs[key] for obs in obs_list])

    # 合并动作
    batch_actions = torch.stack(action_list)

    return batch_obs, batch_actions


def train(
    data_path: str,
    num_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
    save_dir: str = "./checkpoints",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Args:
        data_path: 数据文件路径
        num_epochs: 训练轮数
        batch_size: 批大小
        lr: 学习率
        save_dir: 检查点保存目录
        device: 训练设备
    """
    print("=" * 60)
    print("训练配置:")
    print(f"  数据路径: {data_path}")
    print(f"  训练轮数: {num_epochs}")
    print(f"  批大小: {batch_size}")
    print(f"  学习率: {lr}")
    print(f"  设备: {device}")
    print("=" * 60)
    print()

    # 创建保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据集
    print("加载数据集...")
    dataset = ManiSkillDataset(data_path)

    # 获取图像大小（从第一个样本）
    first_obs, _ = dataset[0]
    if "head_camera_rgb" in first_obs:
        img = first_obs["head_camera_rgb"]
        image_size = (img.shape[1], img.shape[2])  # (H, W)
        print(f"检测到图像大小: {image_size}")
    else:
        image_size = (224, 224)  # 默认大小

    # 获取机器人配置（从环境）
    from waleo.sim.tasks.maniskill_pick_cube import ManiSkillPickCubeEnv
    robot_config = ManiSkillPickCubeEnv.ROBOT_CONFIG["panda"]

    # 创建数据加载器
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows 兼容
        collate_fn=collate_fn,
    )

    # 创建策略网络（使用机器人配置和检测到的图像大小）
    print("创建策略网络...")
    policy = VisualPolicy(
        image_size=image_size,
        state_dim=robot_config["state_dim"],
        action_dim=robot_config["action_dim"],
        hidden_dim=256,
        feature_dim=256,
    )

    # 创建训练器
    trainer = PolicyTrainer(policy, lr=lr, device=device)

    # 训练循环
    print(f"\n开始训练...")
    print()

    best_loss = float("inf")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        policy.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch_obs, batch_actions in pbar:
            # 将数据移到设备
            batch_obs = {k: v.to(device) for k, v in batch_obs.items()}
            batch_actions = batch_actions.to(device)

            # 训练一步
            loss = trainer.train_step(batch_obs, batch_actions)

            epoch_loss += loss
            num_batches += 1

            # 更新进度条
            pbar.set_postfix({"loss": f"{loss:.4f}"})

        # 计算平均损失
        avg_loss = epoch_loss / num_batches

        print(f"Epoch {epoch + 1}: Average Loss = {avg_loss:.4f}")

        # 保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = save_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            trainer.save_checkpoint(str(checkpoint_path), epoch + 1, avg_loss)

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_checkpoint_path = save_dir / "best_checkpoint.pt"
            trainer.save_checkpoint(str(best_checkpoint_path), epoch + 1, avg_loss)
            print(f"  → 保存最佳模型 (loss: {best_loss:.4f})")

        print()

    # 训练完成
    print("=" * 60)
    print("训练完成！")
    print(f"最佳损失: {best_loss:.4f}")
    print(f"模型保存在: {save_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="训练 ManiSkill 策略")
    parser.add_argument("--data", type=str, default="./data/demonstrations/dataset_final.npz",
                        help="数据文件路径")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="检查点保存目录")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="训练设备")

    args = parser.parse_args()

    train(
        data_path=args.data,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
