"""简单的 ManiSkill3 + RJ2506 训练测试（使用新架构）"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def test_basic_env():
    """测试基础环境功能"""
    print("=" * 60)
    print("测试 1: 基础环境创建")
    print("=" * 60)

    # 直接使用 gym 创建环境（不通过我们的封装）
    import gymnasium as gym

    print("\n1.1 创建 PickCube-v1 环境 (Panda)...")
    env = gym.make("PickCube-v1", num_envs=1, robot_uids="panda")
    print("   ✓ 环境创建成功")

    obs, info = env.reset()
    print(f"   ✓ obs 类型: {type(obs)}")
    print(f"   ✓ obs keys: {obs.keys() if hasattr(obs, 'keys') else 'N/A'}")

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"   ✓ step 成功, reward: {reward}")

    env.close()
    print("\n✅ 测试 1 通过！\n")
    return True


def test_rj2506():
    """测试 RJ2506 机器人"""
    print("=" * 60)
    print("测试 2: RJ2506 机器人")
    print("=" * 60)

    import gymnasium as gym

    print("\n2.1 尝试创建 RJ2506 环境...")
    try:
        env = gym.make("PickCube-v1", num_envs=1, robot_uids="rj2506")
        print("   ✓ 环境创建成功")

        obs, info = env.reset()
        print(f"   ✓ reset 成功")

        # 执行几步
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"   Step {i+1}: reward={reward}")

        env.close()
        print("\n✅ 测试 2 通过！\n")
        return True

    except Exception as e:
        print(f"   ✗ 失败: {e}")
        print("\n⚠️  RJ2506 未在 ManiSkill 中注册")
        print("   需要先注册机器人才能使用")
        return False


def test_parallel():
    """测试并行环境"""
    print("=" * 60)
    print("测试 3: 并行环境")
    print("=" * 60)

    import gymnasium as gym

    print("\n3.1 创建 4 个并行环境...")
    env = gym.make("PickCube-v1", num_envs=4, robot_uids="panda")
    print(f"   ✓ 环境创建成功")

    obs, info = env.reset()
    print(f"   ✓ reset 成功")

    # 执行几步
    for i in range(5):
        actions = np.array([env.action_space.sample() for _ in range(4)])
        obs, rewards, terminateds, truncateds, info = env.step(actions)
        print(f"   Step {i+1}: mean_reward={rewards.mean():.4f}")

    env.close()
    print("\n✅ 测试 3 通过！\n")
    return True


def test_simple_rollout():
    """测试简单的 rollout"""
    print("=" * 60)
    print("测试 4: 简单 Rollout")
    print("=" * 60)

    import gymnasium as gym

    env = gym.make("PickCube-v1", num_envs=1, robot_uids="panda")

    num_episodes = 3
    max_steps = 50

    print(f"\n运行 {num_episodes} 个 episodes...")

    for episode in range(num_episodes):
        obs, info = env.reset(seed=episode)
        episode_reward = 0

        for step in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                success = info.get("success", False)
                print(f"   Episode {episode+1}: success={success}, reward={episode_reward:.4f}, steps={step+1}")
                break

    env.close()
    print("\n✅ 测试 4 通过！\n")
    return True


def main():
    print("\n" + "=" * 60)
    print("ManiSkill3 基础训练测试")
    print("=" * 60 + "\n")

    results = []

    try:
        results.append(("基础环境", test_basic_env()))
        results.append(("RJ2506", test_rj2506()))
        results.append(("并行环境", test_parallel()))
        results.append(("简单Rollout", test_simple_rollout()))
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
        return False

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
        print("✅ 所有测试通过！")
        print("\n下一步:")
        print("1. 注册 RJ2506 机器人到 ManiSkill")
        print("2. 使用 make_env() 统一接口")
        print("3. 实现完整的训练循环")
    else:
        print("⚠️  部分测试失败")
    print("=" * 60 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
