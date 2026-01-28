"""测试 make_env() 统一接口"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def test_make_env():
    """测试 make_env() 统一接口"""
    print("=" * 60)
    print("测试: make_env() 统一接口")
    print("=" * 60)

    from waleo.sim import make_env

    print("\n1. 使用默认 Panda 机器人...")
    try:
        env = make_env("PickCube-v1")
        print("   ✓ 环境创建成功")

        obs, info = env.reset()
        print(f"   ✓ reset 成功")

        # 执行几步
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            reward_val = reward.item() if hasattr(reward, 'item') else float(reward)
            print(f"   Step {i+1}: reward={reward_val:.4f}")

        env.close()
        print("\n✅ 测试通过！\n")
        return True

    except Exception as e:
        print(f"   ✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rollout():
    """测试完整 rollout"""
    print("=" * 60)
    print("测试: 完整 Rollout")
    print("=" * 60)

    from waleo.sim import make_env

    print("\n运行 3 个 episodes...")
    env = make_env("PickCube-v1")

    try:
        success_count = 0
        for episode in range(3):
            obs, info = env.reset(seed=episode)
            episode_reward = 0

            for step in range(50):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                reward_val = reward.item() if hasattr(reward, 'item') else float(reward)
                episode_reward += reward_val

                if terminated or truncated:
                    success = info.get("success", False)
                    success_val = success.item() if hasattr(success, 'item') else success
                    if success_val:
                        success_count += 1
                    print(f"   Episode {episode+1}: success={success_val}, reward={episode_reward:.4f}, steps={step+1}")
                    break

        print(f"\n   成功率: {success_count}/3")

    except Exception as e:
        print(f"   ✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        env.close()

    print("\n✅ 测试通过！\n")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("make_env() 统一接口测试")
    print("=" * 60 + "\n")

    results = []
    results.append(("基础使用", test_make_env()))
    results.append(("完整Rollout", test_rollout()))

    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"  {name}: {status}")
    print("=" * 60)

    success = all(r[1] for r in results)
    print(f"\n{'✅ 所有测试通过！' if success else '❌ 部分失败'}")
    sys.exit(0 if success else 1)
