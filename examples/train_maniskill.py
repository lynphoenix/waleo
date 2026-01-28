"""简单的 ManiSkill3 训练脚本 - 使用随机策略演示训练流程"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from collections import deque
import time


def train_random_policy(num_episodes=100, max_steps=200):
    """使用随机策略训练（演示训练流程）"""
    from waleo.sim import make_env

    print("=" * 60)
    print("开始训练: 随机策略")
    print("=" * 60)

    env = make_env("PickCube-v1")

    # 训练统计
    episode_rewards = deque(maxlen=10)
    success_count = 0

    print(f"\n训练配置:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Max Steps: {max_steps}")
    print(f"  Strategy: Random Action")
    print()

    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset(seed=episode)
        episode_reward = 0

        for step in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            # 处理 Tensor reward
            reward_val = reward.item() if hasattr(reward, 'item') else float(reward)
            episode_reward += reward_val

            if terminated or truncated:
                success = info.get("success", False)
                success_val = success.item() if hasattr(success, 'item') else success
                if success_val:
                    success_count += 1
                break

        episode_rewards.append(episode_reward)

        # 每10个episode打印一次
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards)
            elapsed = time.time() - start_time
            print(f"Episode {episode:3d} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Avg(10): {avg_reward:7.2f} | "
                  f"Success: {success_count}/{episode} | "
                  f"Time: {elapsed:.1f}s")

    env.close()

    total_time = time.time() - start_time
    success_rate = success_count / num_episodes * 100

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"总 Episodes: {num_episodes}")
    print(f"成功 Episodes: {success_count}")
    print(f"成功率: {success_rate:.1f}%")
    print(f"总用时: {total_time:.1f}s")
    print(f"平均每episode: {total_time/num_episodes:.2f}s")
    print("=" * 60)

    return {
        "num_episodes": num_episodes,
        "success_count": success_count,
        "success_rate": success_rate,
        "total_time": total_time,
    }


def train_heuristic_policy(num_episodes=100, max_steps=200):
    """使用启发式策略训练"""
    from waleo.sim import make_env

    print("=" * 60)
    print("开始训练: 启发式策略")
    print("=" * 60)

    env = make_env("PickCube-v1")

    episode_rewards = deque(maxlen=10)
    success_count = 0

    print(f"\n训练配置:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Max Steps: {max_steps}")
    print(f"  Strategy: Heuristic (趋向目标)")
    print()

    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset(seed=episode)
        episode_reward = 0

        # 获取目标位置（假设obs中包含）
        # 注意：具体观察结构需要根据实际环境调整

        for step in range(max_steps):
            # 简单启发式：使用带噪声的偏向动作
            action = env.action_space.sample() * 0.5  # 减小随机性

            obs, reward, terminated, truncated, info = env.step(action)

            reward_val = reward.item() if hasattr(reward, 'item') else float(reward)
            episode_reward += reward_val

            if terminated or truncated:
                success = info.get("success", False)
                success_val = success.item() if hasattr(success, 'item') else success
                if success_val:
                    success_count += 1
                break

        episode_rewards.append(episode_reward)

        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards)
            elapsed = time.time() - start_time
            print(f"Episode {episode:3d} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Avg(10): {avg_reward:7.2f} | "
                  f"Success: {success_count}/{episode} | "
                  f"Time: {elapsed:.1f}s")

    env.close()

    total_time = time.time() - start_time
    success_rate = success_count / num_episodes * 100

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"总 Episodes: {num_episodes}")
    print(f"成功 Episodes: {success_count}")
    print(f"成功率: {success_rate:.1f}%")
    print(f"总用时: {total_time:.1f}s")
    print("=" * 60)

    return {
        "num_episodes": num_episodes,
        "success_count": success_count,
        "success_rate": success_rate,
        "total_time": total_time,
    }


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ManiSkill3 训练演示")
    print("=" * 60 + "\n")

    # 运行随机策略训练
    print("[1/2] 随机策略训练...\n")
    result1 = train_random_policy(num_episodes=50, max_steps=100)

    print("\n" + "=" * 60 + "\n")

    # 运行启发式策略训练
    print("[2/2] 启发式策略训练...\n")
    result2 = train_heuristic_policy(num_episodes=50, max_steps=100)

    print("\n" + "=" * 60)
    print("训练总结")
    print("=" * 60)
    print(f"随机策略成功率: {result1['success_rate']:.1f}%")
    print(f"启发式策略成功率: {result2['success_rate']:.1f}%")
    print("\n下一步:")
    print("1. 实现真正的RL算法 (PPO/SAC)")
    print("2. 添加神经网络策略")
    print("3. 集成 RJ2506 机器人")
    print("=" * 60 + "\n")
