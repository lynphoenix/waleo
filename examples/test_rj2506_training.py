"""简单的 ManiSkill3 + RJ2506 训练测试

验证：
1. 环境创建
2. 自定义机器人配置应用
3. 基础训练循环
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def test_env_creation():
    """测试环境创建"""
    print("=" * 60)
    print("测试 1: 环境创建")
    print("=" * 60)

    from waleo.sim import make_env

    # 测试默认机器人
    print("\n1.1 使用默认 Panda 机器人...")
    try:
        env = make_env("PickCube-v1", robot="panda")
        print("   ✓ 环境创建成功")
        obs, info = env.reset()
        print(f"   ✓ reset 成功, obs shape: {obs['observation'].shape if hasattr(obs, 'get') else type(obs)}")
        env.close()
    except Exception as e:
        print(f"   ✗ 失败: {e}")
        return False

    # 测试自定义 RJ2506 机器人
    print("\n1.2 使用自定义 RJ2506 机器人...")
    try:
        env = make_env("PickCube-v1", robot_id="RJ2506")
        print("   ✓ 环境创建成功")
        obs, info = env.reset()
        print(f"   ✓ reset 成功, obs shape: {obs['observation'].shape if hasattr(obs, 'get') else type(obs)}")
        env.close()
    except Exception as e:
        print(f"   ✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✅ 测试 1 通过！\n")
    return True


def test_random_rollout():
    """测试随机策略 rollout"""
    print("=" * 60)
    print("测试 2: 随机策略 Rollout")
    print("=" * 60)

    from waleo.sim import make_env

    print("\n2.1 创建环境...")
    try:
        env = make_env(
            "PickCube-v1",
            robot_id="RJ2506",
            num_envs=1,
            obs_mode="state_dict",
            control_mode="pd_joint_pos"
        )
        print("   ✓ 环境创建成功")
    except Exception as e:
        print(f"   ✗ 失败: {e}")
        return False

    print("\n2.2 运行随机策略...")
    try:
        num_episodes = 3
        max_steps = 50

        for episode in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0

            for step in range(max_steps):
                # 随机动作
                action = env.action_space.sample()

                # 执行
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

                if terminated or truncated:
                    break

            print(f"   Episode {episode + 1}: reward={episode_reward:.4f}, steps={step + 1}")

        print(f"\n   ✓ 完成 {num_episodes} 个 episodes")

    except Exception as e:
        print(f"   ✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        env.close()

    print("\n✅ 测试 2 通过！\n")
    return True


def test_simple_policy():
    """测试简单策略（向目标移动）"""
    print("=" * 60)
    print("测试 3: 简单启发式策略")
    print("=" * 60)

    from waleo.sim import make_env

    print("\n3.1 创建环境...")
    try:
        env = make_env(
            "PickCube-v1",
            robot_id="RJ2506",
            num_envs=1,
            obs_mode="state_dict"
        )
        print("   ✓ 环境创建成功")
    except Exception as e:
        print(f"   ✗ 失败: {e}")
        return False

    print("\n3.2 定义简单策略（向目标位置移动）...")

    def simple_policy(obs):
        """简单策略：计算末端向目标方向的动作"""
        # 获取机器人末端位置
        agent_pos = obs['observation']['agent']['robot0']['ee_pos']
        # 获取目标位置（立方体目标位置）
        target_pos = obs['observation']['target']['obj_0_goal_pos']

        # 计算方向向量
        direction = target_pos - agent_pos
        direction = direction / (np.linalg.norm(direction) + 1e-6)

        # 简单的 P 控制
        action = direction * 0.1

        # 获取当前关节位置作为基准
        current_qpos = obs['observation']['agent']['robot0']['qpos']
        action_size = len(current_qpos)

        # 扩展 action 到完整动作空间
        if len(action) < action_size:
            action = np.concatenate([action, np.zeros(action_size - len(action))])

        return action[:action_size]

    print("\n3.3 运行策略...")
    try:
        num_episodes = 5
        max_steps = 100
        rewards = []

        for episode in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0

            for step in range(max_steps):
                action = simple_policy(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

                if terminated or truncated:
                    break

            rewards.append(episode_reward)
            status = "✓" if terminated else "✗"
            print(f"   Episode {episode + 1}: {status} reward={episode_reward:.4f}, steps={step + 1}")

        avg_reward = np.mean(rewards)
        success_rate = sum(1 for r in rewards if r > 0) / len(rewards)
        print(f"\n   平均奖励: {avg_reward:.4f}")
        print(f"   成功率: {success_rate:.1%}")

    except Exception as e:
        print(f"   ✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        env.close()

    print("\n✅ 测试 3 通过！\n")
    return True


def test_parallel_envs():
    """测试并行环境"""
    print("=" * 60)
    print("测试 4: 并行环境")
    print("=" * 60)

    from waleo.sim import make_env

    print("\n4.1 创建 4 个并行环境...")
    try:
        env = make_env(
            "PickCube-v1",
            robot_id="RJ2506",
            num_envs=4,
            obs_mode="state_dict"
        )
        print("   ✓ 并行环境创建成功")
        print(f"   ✓ num_envs: {env.unwrapped.num_envs}")
    except Exception as e:
        print(f"   ✗ 失败: {e}")
        return False

    print("\n4.2 运行并行 rollout...")
    try:
        obs, info = env.reset()
        print(f"   ✓ batch obs shape: {obs['observation']['agent']['robot0']['ee_pos'].shape}")

        # 运行几步
        for step in range(10):
            actions = np.array([env.action_space.sample() for _ in range(4)])
            obs, rewards, terminateds, truncateds, info = env.step(actions)

        print(f"   ✓ 完成 10 步并行仿真")

    except Exception as e:
        print(f"   ✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        env.close()

    print("\n✅ 测试 4 通过！\n")
    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("ManiSkill3 + RJ2506 训练测试")
    print("=" * 60 + "\n")

    results = []

    # 测试 1: 环境创建
    results.append(("环境创建", test_env_creation()))

    # 测试 2: 随机 rollout
    results.append(("随机 Rollout", test_random_rollout()))

    # 测试 3: 简单策略
    results.append(("简单策略", test_simple_policy()))

    # 测试 4: 并行环境
    results.append(("并行环境", test_parallel_envs()))

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过！系统可以用于训练！")
    else:
        print("⚠️  部分测试失败，需要修复")
    print("=" * 60 + "\n")

    return all_passed


if __name__ == "__main__":
    import traceback
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        traceback.print_exc()
        sys.exit(1)
