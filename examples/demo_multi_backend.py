"""通用多后端 API 演示

展示如何使用统一的接口切换不同仿真后端。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from waleo.sim import make_env
from waleo.sim.backends import ManiSkillBackend, MuJoCoBackend, PyBulletBackend

print("=" * 60)
print("通用多后端 API 演示")
print("=" * 60)

# 检查后端可用性
print("\n1. 后端可用性检查")
backends = [
    ("ManiSkill", ManiSkillBackend),
    ("MuJoCo", MuJoCoBackend),
    ("PyBullet", PyBulletBackend),
]

for name, backend_cls in backends:
    available = backend_cls.is_available()
    status = "✓" if available else "✗"
    print(f"   {status} {name:12} - {('已安装' if available else '未安装')}")

print("\n2. 统一 API - 不同后端使用相同接口")
print("-" * 60)

# ManiSkill 示例
print("\n   ManiSkill (机器人操作任务):")
print("   ```python")
print("   env = make_env('PickCube-v1',")
print("                  obs_mode='state',              # ManiSkill 特定")
print("                  control_mode='pd_joint_pos')    # ManiSkill 特定")
print("   ```")

# MuJoCo 示例
print("\n   MuJoCo (经典控制任务):")
print("   ```python")
print("   env = make_env('Ant-v4',")
print("                  backend='mujoco',")
print("                  frame_skip=5)                  # MuJoCo 特定")
print("   ```")

# PyBullet 示例
print("\n   PyBullet (开源机器人仿真):")
print("   ```python")
print("   env = make_env('KukaBulletEnv-v0',")
print("                  backend='pybullet',")
print("                  render_mode='human')           # PyBullet 特定")
print("   ```")

print("\n3. 可用任务列表")
print("-" * 60)

for name, backend_cls in backends:
    tasks = backend_cls.list_available_tasks()
    print(f"\n   {name}:")
    print(f"      {', '.join(tasks[:5])}{'...' if len(tasks) > 5 else ''}")

print("\n4. 后端特定默认参数")
print("-" * 60)

for name, backend_cls in backends:
    defaults = backend_cls.get_default_kwargs()
    print(f"\n   {name}:")
    if defaults:
        for k, v in defaults.items():
            print(f"      {k}: {v}")
    else:
        print("      (无默认参数)")

print("\n" + "=" * 60)
print("设计优势")
print("=" * 60)
print("""
✓ 统一接口: 所有后端使用相同的 make_env() 函数
✓ 后端特定参数: 通过 **kwargs 灵活传递
✓ 易于扩展: 添加新后端无需修改核心代码
✓ 类型安全: 每个后端独立实现，互不影响

使用场景:
- 比较不同仿真器的性能
- 根据可用性自动选择后端
- 同一套训练代码支持多种后端
""")
print("=" * 60)
