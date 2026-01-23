"""测试 waleo-sim 模块基础功能"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np


def test_config():
    """测试配置模块"""
    print("测试 waleo-config...")
    from waleo_config import EnvConfig, CameraConfig

    # 测试 EnvConfig
    config = EnvConfig(
        task="push",
        robot_type="so100",
        simulation_backend="mujoco"
    )
    print(f"✓ EnvConfig 创建成功: {config}")

    # 测试从字典创建
    config_dict = {
        "task": "pick_place",
        "robot_type": "aloha",
        "num_envs": 8
    }
    config2 = EnvConfig.from_dict(config_dict)
    print(f"✓ EnvConfig.from_dict 成功: {config2}")

    # 测试 CameraConfig
    camera_config = CameraConfig(width=640, height=480)
    print(f"✓ CameraConfig 创建成功: {camera_config}")

    print("✓ waleo-config 测试通过\n")


def test_base_env():
    """测试 BaseEnv 基类"""
    print("测试 BaseEnv...")
    from waleo_sim.base import BaseEnv, EnvWrapper, VectorEnv, RobotEnv

    # 测试 BaseEnv 导入
    print(f"✓ BaseEnv 导入成功: {BaseEnv}")

    # 测试 EnvWrapper 导入
    print(f"✓ EnvWrapper 导入成功: {EnvWrapper}")

    # 测试 VectorEnv 导入
    print(f"✓ VectorEnv 导入成功: {VectorEnv}")

    # 测试 RobotEnv 导入
    print(f"✓ RobotEnv 导入成功: {RobotEnv}")

    print("✓ BaseEnv 测试通过\n")


def test_backends():
    """测试仿真后端"""
    print("测试仿真后端...")
    from waleo_sim.backends import (
        SimulationBackend,
        MuJoCoBackend,
        PyBulletBackend,
        ManiSkillBackend,
        get_backend
    )

    # 测试后端导入
    print(f"✓ SimulationBackend 导入成功: {SimulationBackend}")
    print(f"✓ MuJoCoBackend 导入成功: {MuJoCoBackend}")
    print(f"✓ PyBulletBackend 导入成功: {PyBulletBackend}")
    print(f"✓ ManiSkillBackend 导入成功: {ManiSkillBackend}")

    # 测试 get_backend 函数
    mujoco_cls = get_backend("mujoco")
    print(f"✓ get_backend('mujoco') 返回: {mujoco_cls}")

    pybullet_cls = get_backend("pybullet")
    print(f"✓ get_backend('pybullet') 返回: {pybullet_cls}")

    maniskill_cls = get_backend("maniskill")
    print(f"✓ get_backend('maniskill') 返回: {maniskill_cls}")

    print("✓ 仿真后端测试通过\n")


def test_simple_env():
    """测试简单的环境实现"""
    print("测试简单的环境实现...")
    from waleo_sim.base import BaseEnv
    from waleo_config import EnvConfig

    class Space:
        """简单的空间类"""
        def __init__(self, shape):
            self._shape = shape

        @property
        def shape(self):
            return self._shape

        def sample(self):
            return np.random.randn(*self._shape)

    class SimpleEnv(BaseEnv):
        """简单的测试环境"""

        def __init__(self):
            super().__init__()
            # 设置空间属性（需要通过父类接口）
            self._observation_space = Space((10,))
            self._action_space = Space((3,))

        @property
        def observation_space(self):
            return self._observation_space

        @property
        def action_space(self):
            return self._action_space

        def reset(self, seed=None, options=None):
            return np.zeros(10), {}

        def step(self, action):
            return (
                np.zeros(10),
                0.0,
                False,
                False,
                {}
            )

    # 测试环境创建
    env = SimpleEnv()
    print(f"✓ 环境创建成功: {env}")

    # 测试 reset
    obs, info = env.reset()
    print(f"✓ reset 成功: obs.shape={obs.shape}")

    # 测试 step
    action = np.zeros(3)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✓ step 成功: reward={reward}, terminated={terminated}")

    print("✓ 简单环境测试通过\n")


if __name__ == "__main__":
    print("=" * 50)
    print("Waleo-Sim 模块测试")
    print("=" * 50)
    print()

    try:
        test_config()
        test_base_env()
        test_backends()
        test_simple_env()

        print("=" * 50)
        print("✅ 所有测试通过！")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
