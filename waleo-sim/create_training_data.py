"""创建小型训练数据集（包含相机数据）"""
import sys
from pathlib import Path
import numpy as np

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from waleo_sim.tasks.maniskill_pick_cube import ManiSkillPickCubeEnv

def create_small_training_dataset():
    """创建一个小型训练数据集，包含相机数据"""
    print("=" * 60)
    print("创建小型训练数据集（包含相机数据）")
    print("=" * 60)
    print()

    # 创建环境（使用相机）
    print("创建环境...")
    env = ManiSkillPickCubeEnv(
        robot_type="panda",
        image_size=(128, 128),  # 使用较小的图像加快速度
        use_cameras=True,       # 启用相机
        headless=True,          # 无头模式
    )
    print("✓ 环境创建成功")

    # 获取机器人配置
    robot_config = env.get_robot_config()
    print(f"✓ 机器人配置: action_dim={robot_config['action_dim']}, state_dim={robot_config['state_dim']}")

    # 采集数据
    print()
    print("采集数据...")
    dataset = []
    num_episodes = 10  # 采集10个episode

    for episode_idx in range(num_episodes):
        obs, _ = env.reset()
        episode_data = {"observations": [], "actions": []}

        for step in range(50):  # 每个episode最多50步
            # 采样随机动作
            action = env.action_space.sample()

            # 存储观察
            episode_data["observations"].append(obs)
            episode_data["actions"].append(action)

            # 执行动作
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

    save_path = save_dir / "small_dataset.npz"
    np.savez_compressed(
        save_path,
        episodes=len(dataset),
        data=dataset,
    )
    print(f"✓ 数据已保存: {save_path}")

    # 统计
    total_steps = sum(len(ep["observations"]) for ep in dataset)
    print()
    print(f"统计信息:")
    print(f"  Episodes: {len(dataset)}")
    print(f"  Total steps: {total_steps}")
    print(f"  Avg steps per episode: {total_steps / len(dataset):.1f}")

    # 关闭环境
    env.close()

    print()
    print("=" * 60)
    print("✅ 数据集创建成功！")
    print("=" * 60)
    print()
    print(f"可以使用以下命令训练:")
    print(f"  python train.py --data {save_path} --epochs 50 --batch-size 8")
    print()

    return str(save_path)


if __name__ == "__main__":
    create_small_training_dataset()
