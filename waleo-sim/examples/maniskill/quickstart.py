"""快速开始示例

一个简单的示例，展示如何使用 ManiSkill 训练流程。
"""

import sys
from pathlib import Path
import numpy as np

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from waleo_sim.tasks.maniskill_pick_cube import ManiSkillPickCubeEnv


def test_environment():
    """测试环境是否正常工作"""
    print("=" * 60)
    print("测试 ManiSkill 环境")
    print("=" * 60)
    print()

    try:
        # 创建环境
        print("1. 创建环境...")
        env = ManiSkillPickCubeEnv(
            robot_type="panda",
            image_size=(128, 128),  # 使用较小的图像以加快测试
            use_cameras=True,
            headless=True,  # 无头模式
        )
        print("   ✓ 环境创建成功")

        # 重置环境
        print()
        print("2. 重置环境...")
        obs, info = env.reset()
        print(f"   ✓ 观察键: {list(obs.keys())}")
        print(f"   ✓ 机器人状态形状: {obs['robot_state'].shape}")
        if "head_camera_rgb" in obs:
            print(f"   ✓ 头部相机图像形状: {obs['head_camera_rgb'].shape}")
        if "wrist_camera_rgb" in obs:
            print(f"   ✓ 腕部相机图像形状: {obs['wrist_camera_rgb'].shape}")

        # 执行几个随机步骤
        print()
        print("3. 执行随机步骤...")
        total_reward = 0
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            # 将 Tensor 类型的 reward 转换为标量
            if hasattr(reward, 'item'):
                reward = reward.item()
            elif hasattr(reward, 'cpu'):
                reward = reward.cpu().item()
            total_reward += reward

            if info.get("success", False):
                print(f"   ✓ Step {step + 1}: 成功完成任务！")
                break

            if terminated or truncated:
                break

        print(f"   ✓ 累计奖励: {total_reward:.4f}")

        # 关闭环境
        env.close()
        print()
        print("   ✓ 环境已关闭")
        print()

        print("=" * 60)
        print("✅ 环境测试成功！")
        print("=" * 60)
        print()
        print("下一步:")
        print("1. 采集数据: python collect_data.py")
        print("2. 训练策略: python train.py")
        print("3. 评估策略: python evaluate.py --checkpoint ./checkpoints/best_checkpoint.pt")
        print()

        return True

    except ImportError as e:
        print()
        print("=" * 60)
        print("❌ 环境测试失败")
        print("=" * 60)
        print()
        print(f"错误: {e}")
        print()
        print("请确保安装了 ManiSkill2:")
        print("  pip install mani_skill2")
        print()
        return False

    except Exception as e:
        print()
        print("=" * 60)
        print("❌ 环境测试失败")
        print("=" * 60)
        print()
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def create_mini_dataset():
    """创建一个最小化数据集用于测试"""
    print("=" * 60)
    print("创建最小化数据集（用于测试训练流程）")
    print("=" * 60)
    print()

    try:
        from examples.maniskill.policy import VisualPolicy

        # 创建环境
        print("创建环境...")
        env = ManiSkillPickCubeEnv(
            robot_type="panda",
            image_size=(64, 64),  # 使用更小的图像
            use_cameras=False,  # 不使用相机以加快速度
            headless=True,
        )
        print("✓ 环境创建成功")

        # 采集少量数据
        print()
        print("采集 5 个 episode 的数据...")
        dataset = []
        for episode_idx in range(5):
            obs, _ = env.reset()
            episode_data = {"observations": [], "actions": []}

            for step in range(20):  # 每个 episode 20 步
                action = env.action_space.sample()
                episode_data["observations"].append(obs)
                episode_data["actions"].append(action)

                obs, reward, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    break

            dataset.append(episode_data)
            print(f"  Episode {episode_idx + 1}: {len(episode_data['observations'])} steps")

        # 保存数据
        print()
        print("保存数据...")
        save_dir = Path("./data/demonstrations")
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / "mini_dataset.npz"
        np.savez_compressed(
            save_path,
            episodes=len(dataset),
            data=dataset,
        )
        print(f"✓ 数据已保存: {save_path}")
        print()

        # 关闭环境
        env.close()

        print("=" * 60)
        print("✅ 最小化数据集创建成功！")
        print("=" * 60)
        print()
        print("可以使用以下命令测试训练:")
        print(f"  python train.py --data {save_path} --epochs 5 --batch-size 4")
        print()

        return str(save_path)

    except Exception as e:
        print()
        print("=" * 60)
        print("❌ 创建数据集失败")
        print("=" * 60)
        print()
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        print()
        return None


def test_training(data_path):
    """测试训练流程"""
    print("=" * 60)
    print("测试训练流程")
    print("=" * 60)
    print()

    try:
        from examples.maniskill.train import ManiSkillDataset
        from examples.maniskill.policy import VisualPolicy
        from waleo_sim.tasks.maniskill_pick_cube import ManiSkillPickCubeEnv
        import torch

        # 检查 CUDA
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")

        # 加载数据
        print()
        print("加载数据...")
        dataset = ManiSkillDataset(data_path)
        print(f"✓ 数据加载成功: {len(dataset)} transitions")

        # 获取机器人配置
        robot_config = ManiSkillPickCubeEnv.ROBOT_CONFIG["panda"]

        # 创建网络（使用机器人配置）
        print()
        print("创建网络...")
        policy = VisualPolicy(
            image_size=(64, 64),
            state_dim=robot_config["state_dim"],
            action_dim=robot_config["action_dim"],
            hidden_dim=128,  # 使用较小的网络
            feature_dim=128,
        )
        print(f"✓ 网络创建成功")

        # 测试前向传播
        print()
        print("测试前向传播...")
        obs, action = dataset[0]

        # 转换设备
        obs = {k: v.unsqueeze(0) if len(v.shape) > 0 else v.unsqueeze(0)
                for k, v in obs.items()}
        # 添加缺失的相机观察
        if "head_camera_rgb" not in obs:
            batch_size = 1
            obs["head_camera_rgb"] = torch.zeros(batch_size, 3, 64, 64)
            obs["wrist_camera_rgb"] = torch.zeros(batch_size, 3, 64, 64)

        policy.eval()
        with torch.no_grad():
            output = policy(obs)
        print(f"✓ 前向传播成功: 输入 → 输出 {output.shape}")

        print()
        print("=" * 60)
        print("✅ 训练流程测试成功！")
        print("=" * 60)
        print()
        print("网络结构:")
        print(f"  输入: 双相机图像 + 机器人状态")
        print(f"  输出: 7维动作")
        print()
        print("可以开始完整训练:")
        print(f"  python train.py --data {data_path} --epochs 100")
        print()

        return True

    except Exception as e:
        print()
        print("=" * 60)
        print("❌ 训练流程测试失败")
        print("=" * 60)
        print()
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


if __name__ == "__main__":
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "ManiSkill PickCube 训练示例 - 快速开始" + " " * 15 + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    # 步骤 1: 测试环境
    env_ok = test_environment()

    if not env_ok:
        print("请先安装 ManiSkill2:")
        print("  pip install mani_skill2")
        sys.exit(1)

    # 步骤 2: 创建最小化数据集
    print()
    data_path = create_mini_dataset()

    if data_path is None:
        sys.exit(1)

    # 步骤 3: 测试训练流程
    print()
    train_ok = test_training(data_path)

    if train_ok:
        print()
        print("🎉 所有测试通过！现在可以开始完整的训练流程了")
        print()
