"""测试 gym.make() 集成"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_gym_make():
    """测试通过 gym.make() 创建 ManiSkill 环境"""
    print("=" * 60)
    print("测试: gym.make() 集成")
    print("=" * 60)

    # 注册环境
    print("\n1. 注册 ManiSkill 环境到 gymnasium...")
    from waleo.sim.backends.maniskill import ManiSkillBackend
    ManiSkillBackend._register_envs()
    print("   ✓ 注册完成")

    # 尝试创建
    print("\n2. 通过 gym.make() 创建环境...")
    import gymnasium as gym

    try:
        env = gym.make("PickCube-v1", num_envs=1)
        print("   ✓ 环境创建成功")
        print(f"   ✓ 环境类型: {type(env)}")

        # 测试 reset
        obs, info = env.reset()
        print(f"   ✓ reset 成功")
        print(f"   ✓ obs shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")

        # 测试 step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        reward_val = reward.item() if hasattr(reward, 'item') else float(reward)
        print(f"   ✓ step 成功, reward: {reward_val:.4f}")

        env.close()
        print("\n✅ 测试通过！")
        return True

    except Exception as e:
        print(f"   ✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_gym_make()
    sys.exit(0 if success else 1)
