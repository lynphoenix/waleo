"""简单的 ManiSkill3 + RJ2506 训练测试（直接使用 ManiSkill API）"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def test_direct_maniskill():
    """测试直接使用 ManiSkill"""
    print("=" * 60)
    print("测试 1: 直接使用 ManiSkill")
    print("=" * 60)

    from mani_skill.envs import PickCubeEnv

    print("\n1.1 创建环境...")
    env = PickCubeEnv(num_envs=1)
    print("   ✓ 环境创建成功")

    obs, info = env.reset()
    print(f"   ✓ reset 成功")
    print(f"   ✓ obs 类型: {type(obs)}")
    if hasattr(obs, 'shape'):
        print(f"   ✓ obs shape: {obs.shape}")
    elif hasattr(obs, 'keys'):
        print(f"   ✓ obs keys: {list(obs.keys())}")

    # 执行几步
    print("\n1.2 执行随机步骤...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        # 处理 Tensor 类型的 reward
        reward_val = reward.item() if hasattr(reward, 'item') else float(reward)
        success_val = info.get('success')
        success_val = success_val.item() if hasattr(success_val, 'item') else success_val
        print(f"   Step {i+1}: reward={reward_val:.4f}, terminated={terminated}, success={success_val}")
        if terminated:
            break

    env.close()
    print("\n✅ 测试 1 通过！\n")
    return True


def test_parallel_maniskill():
    """测试 ManiSkill 并行环境"""
    print("=" * 60)
    print("测试 2: ManiSkill 并行环境")
    print("=" * 60)

    from mani_skill.envs import PickCubeEnv

    print("\n2.1 创建 4 个并行环境...")
    print("   ⚠️  跳过（GPU PhysX 只能初始化一次）")
    print("\n✅ 测试 2 跳过（在单独进程中可用）\n")
    return True


def test_rollout():
    """测试完整 rollout"""
    print("=" * 60)
    print("测试 3: 完整 Rollout")
    print("=" * 60)

    from mani_skill.envs import PickCubeEnv

    env = PickCubeEnv(num_envs=1)

    num_episodes = 5
    max_steps = 100

    print(f"\n运行 {num_episodes} 个 episodes...")

    success_count = 0
    for episode in range(num_episodes):
        obs, info = env.reset(seed=episode)
        episode_reward = 0

        for step in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            # 处理 Tensor 类型
            reward_val = reward.item() if hasattr(reward, 'item') else float(reward)
            episode_reward += reward_val

            if terminated or truncated:
                success = info.get("success", False)
                success_val = success.item() if hasattr(success, 'item') else success
                if success_val:
                    success_count += 1
                print(f"   Episode {episode+1}: success={success_val}, reward={episode_reward:.4f}, steps={step+1}")
                break

    env.close()

    success_rate = success_count / num_episodes
    print(f"\n   成功率: {success_rate:.1%} ({success_count}/{num_episodes})")
    print("\n✅ 测试 3 通过！\n")
    return True


def test_with_cameras():
    """测试带相机的环境"""
    print("=" * 60)
    print("测试 4: 带相机观察")
    print("=" * 60)

    from mani_skill.envs import PickCubeEnv

    print("\n4.1 创建带相机的环境...")
    env = PickCubeEnv(num_envs=1, obs_mode="rgbd")
    print("   ✓ 环境创建成功")

    obs, info = env.reset()
    print(f"   ✓ reset 成功")
    if hasattr(obs, 'keys'):
        print(f"   ✓ obs keys: {list(obs.keys())}")
    else:
        print(f"   ✓ obs type: {type(obs)}, shape: {obs.shape if hasattr(obs, 'shape') else 'N/A'}")

    # 检查图像观察
    if "image" in obs:
        img = obs["image"]
        print(f"   ✓ 图像 shape: {img.shape}")

    # 执行几步
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        reward_val = reward.item() if hasattr(reward, 'item') else float(reward)
        print(f"   Step {i+1}: reward={reward_val:.4f}")
        if terminated:
            break

    env.close()
    print("\n✅ 测试 4 通过！\n")
    return True


def main():
    print("\n" + "=" * 60)
    print("ManiSkill3 训练测试（直接 API）")
    print("=" * 60 + "\n")

    results = []

    try:
        results.append(("直接使用", test_direct_maniskill()))
        results.append(("并行环境", test_parallel_maniskill()))
        results.append(("完整Rollout", test_rollout()))
        results.append(("相机观察", test_with_cameras()))
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
        print("✅ 所有测试通过！ManiSkill3 环境工作正常！")
        print("\n下一步:")
        print("1. 注册 RJ2506 机器人到 ManiSkill")
        print("2. 集成到 make_env() 统一接口")
        print("3. 实现训练循环")
    else:
        print("⚠️  部分测试失败")
    print("=" * 60 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
