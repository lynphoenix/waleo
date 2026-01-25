"""数据采集脚本

使用 ManiSkill 环境收集演示数据。
"""

import sys
from pathlib import Path
import numpy as np
from typing import List, Dict, Optional, Callable
import random

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from waleo.sim.tasks.maniskill_pick_cube import ManiSkillPickCubeEnv


class DataCollector:
    """数据采集器"""

    def __init__(
        self,
        env: ManiSkillPickCubeEnv,
        save_dir: str = "./data/demonstrations",
    ):
        """
        Args:
            env: ManiSkill 环境
            save_dir: 数据保存目录
        """
        self.env = env
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.episodes_data = []

    def collect_episode(
        self,
        max_steps: int = 200,
        expert_policy: Optional[Callable] = None,
    ) -> Dict:
        """采集一个 episode 的数据

        Args:
            max_steps: 最大步数
            expert_policy: 专家策略（可选，如果为 None 则使用随机策略）

        Returns:
            episode_data: episode 数据字典
        """
        obs, info = self.env.reset()

        episode_data = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "terminated": [],
            "success": False,
        }

        for step in range(max_steps):
            # 选择动作
            if expert_policy is not None:
                action = expert_policy(obs)
            else:
                # 简单的启发式策略：向物体移动
                action = self._heuristic_action(obs)

            # 执行动作
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            # 保存数据
            episode_data["observations"].append(obs)
            episode_data["actions"].append(action)
            episode_data["rewards"].append(reward)
            episode_data["terminated"].append(terminated)

            obs = next_obs

            # 检查是否成功
            if info.get("success", False):
                episode_data["success"] = True
                break

            if terminated or truncated:
                break

        return episode_data

    def _heuristic_action(self, obs: Dict) -> np.ndarray:
        """简单的启发式策略

        Args:
            obs: 观察字典

        Returns:
            action: 动作
        """
        # 获取机器人末端位置
        robot_state = obs.get("robot_state", np.zeros(9))
        ee_pos = robot_state[:3]

        # 假设目标物体在前方
        target_pos = np.array([0.5, 0.0, 0.1])

        # 计算朝向目标的方向
        direction = target_pos - ee_pos

        # 归一化并缩放
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction) * 0.05

        # 7维动作 [x, y, z, rx, ry, rz, gripper]
        action = np.zeros(7)
        action[:3] = direction  # 位置增量
        action[-1] = 0.0  # 夹爪闭合（简化）

        # 添加一些随机性
        action += np.random.randn(7) * 0.01

        return action

    def collect_dataset(
        self,
        num_episodes: int = 100,
        max_steps: int = 200,
        save_frequency: int = 10,
    ) -> List[Dict]:
        """采集数据集

        Args:
            num_episodes: 采集的 episode 数量
            max_steps: 每个 episode 的最大步数
            save_frequency: 保存频率

        Returns:
            dataset: 采集的数据集
        """
        print(f"开始采集数据...")
        print(f"Episodes: {num_episodes}")
        print(f"Max steps per episode: {max_steps}")
        print(f"保存目录: {self.save_dir}")
        print()

        dataset = []
        success_count = 0

        for episode_idx in range(num_episodes):
            episode_data = self.collect_episode(max_steps=max_steps)
            episode_data["episode_idx"] = episode_idx

            dataset.append(episode_data)

            if episode_data["success"]:
                success_count += 1

            # 打印进度
            if (episode_idx + 1) % 10 == 0:
                success_rate = success_count / (episode_idx + 1) * 100
                print(f"Episode {episode_idx + 1}/{num_episodes}: "
                      f"Success rate: {success_rate:.1f}% "
                      f"({success_count}/{episode_idx + 1})")

            # 定期保存
            if (episode_idx + 1) % save_frequency == 0:
                self._save_dataset(dataset, f"dataset_ep{episode_idx + 1}.npz")

        # 保存最终数据集
        self._save_dataset(dataset, "dataset_final.npz")

        # 打印统计信息
        total_steps = sum(len(ep["observations"]) for ep in dataset)
        final_success_rate = success_count / num_episodes * 100

        print()
        print("=" * 50)
        print(f"数据采集完成！")
        print(f"Total episodes: {num_episodes}")
        print(f"Total steps: {total_steps}")
        print(f"Success count: {success_count}")
        print(f"Success rate: {final_success_rate:.1f}%")
        print(f"Average steps per episode: {total_steps / num_episodes:.1f}")
        print("=" * 50)

        return dataset

    def _save_dataset(self, dataset: List[Dict], filename: str):
        """保存数据集

        Args:
            dataset: 数据集
            filename: 文件名
        """
        save_path = self.save_dir / filename

        # 转换为 numpy 格式保存
        np.savez_compressed(
            save_path,
            episodes=len(dataset),
            data=dataset,
        )

        print(f"✓ 数据已保存: {save_path}")


def main():
    """主函数"""
    # 创建环境
    print("初始化 ManiSkill 环境...")
    env = ManiSkillPickCubeEnv(
        robot_type="panda",
        image_size=(224, 224),
        use_cameras=True,
        headless=False,  # 设为 False 可以看到可视化
    )

    # 创建数据采集器
    collector = DataCollector(env, save_dir="./data/demonstrations")

    # 采集数据集
    # 从小规模开始，可以后续增加
    dataset = collector.collect_dataset(
        num_episodes=50,  # 采集 50 个 episode
        max_steps=200,
        save_frequency=10,
    )

    # 关闭环境
    env.close()

    print("\n提示: 可以使用以下数据训练策略:")
    print(f"python train.py --data {collector.save_dir / 'dataset_final.npz'}")


if __name__ == "__main__":
    main()
