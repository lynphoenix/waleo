"""无侵入式机器人集成示例

展示如何使用 Waleo Sim 的无侵入式设计集成自定义机器人。

对比：
- 侵入式做法（旧）：Monkey-patching、硬编码路径
- 无侵入式做法（新）：环境变量、Wrapper 模式、配置驱动
"""

import os
import sys

# ============================================================================
# 示例 1：侵入式做法（❌ 不推荐，仅用于对比）
# ============================================================================

def old_invasive_approach():
    """旧的侵入式做法"""
    print("=" * 70)
    print("示例 1：侵入式做法（❌ 不推荐）")
    print("=" * 70)

    print("""
# 1. 硬编码绝对路径
urdf_path = "/root/data2/lyn/waleo/assets/robots/RJ2506/urdf/..."

# 2. 手动导入（副作用）
import sys
sys.path.insert(0, "custom_agents")
from custom_agents.rj2506_variants import RJ2506_LeftArm

# 3. Monkey-patching ManiSkill 内部类
from mani_skill.envs.tasks.tabletop import pick_cube
original_load_agent = pick_cube.PickCubeEnv._load_agent

def custom_load_agent(self, options):
    # 修改私有方法...
    pass

pick_cube.PickCubeEnv._load_agent = custom_load_agent  # 💀 危险！

# 4. 修改全局状态
from mani_skill.agents.robots.rj2506 import RJ2506
RJ2506.keyframes["rest"] = new_keyframe  # 💀 全局污染！

# 5. 创建环境
env = gym.make("PickCube-v1", robot_uids="rj2506")
""")

    print("\n问题:")
    print("  ❌ 不可移植（硬编码路径）")
    print("  ❌ 脆弱（依赖私有 API）")
    print("  ❌ 全局污染（影响其他代码）")
    print("  ❌ 维护困难（ManiSkill 更新时容易崩溃）")
    print("  ❌ 不可测试（全局状态）\n")


# ============================================================================
# 示例 2：无侵入式做法（✅ 推荐）
# ============================================================================

def new_non_invasive_approach():
    """新的无侵入式做法"""
    print("=" * 70)
    print("示例 2：无侵入式做法（✅ 推荐）")
    print("=" * 70)

    print("""
# 1. 设置环境变量（一次性，可放在 ~/.bashrc）
export WALEO_ASSETS_DIR=/path/to/waleo/assets

# 2. 导入 Waleo Sim
from waleo.sim.registry import get_robot_registry
from waleo.sim.wrappers import CustomRobotWrapper, TaskConfigWrapper
import gymnasium as gym

# 3. 自动发现机器人
registry = get_robot_registry()
print(f"Available robots: {registry.list_robots()}")
# 输出: ['RJ2506', 'panda', 'fetch']

# 4. 获取机器人规格
spec = registry.get("rj2506")
task_config = spec.get_task_config("PickCube-v1")

# 5. 创建环境（标准 Gym API）
env = gym.make("PickCube-v1", robot_uids="rj2506", num_envs=512)

# 6. 应用包装器（无侵入式）
env = CustomRobotWrapper(env, "rj2506", task_config)

# 完成！环境已配置好，可以开始训练
""")

    print("\n优势:")
    print("  ✅ 可移植（环境变量）")
    print("  ✅ 稳定（只用公开 API）")
    print("  ✅ 无污染（Wrapper 模式）")
    print("  ✅ 易维护（符合 Gym 标准）")
    print("  ✅ 可测试（无全局状态）\n")


# ============================================================================
# 示例 3：更简洁的工厂 API（即将实现）
# ============================================================================

def future_factory_api():
    """未来的工厂 API（Phase 5）"""
    print("=" * 70)
    print("示例 3：工厂 API（⏳ Phase 5 即将实现）")
    print("=" * 70)

    print("""
# 一行代码创建环境！
from waleo.sim import make_env

env = make_env(
    task="pick_place",
    robot="rj2506",  # 自动发现和配置
    backend="maniskill",
    num_envs=512
)

# 就这么简单！
""")

    print("\n特性:")
    print("  🚀 一行代码")
    print("  🔍 自动发现机器人")
    print("  ⚙️  自动应用配置")
    print("  📦 完全封装细节\n")


# ============================================================================
# 示例 4：实际代码演示（可运行）
# ============================================================================

def practical_demo():
    """实际可运行的演示"""
    print("=" * 70)
    print("示例 4：实际代码演示")
    print("=" * 70)

    try:
        # 导入 Waleo Sim 组件
        from waleo.sim.registry import get_asset_resolver, get_robot_registry

        print("\n1. 资源解析器")
        print("-" * 70)
        resolver = get_asset_resolver()
        print(f"搜索路径:")
        for path in resolver.search_paths:
            print(f"  - {path}")

        print(f"\n发现的机器人:")
        robots = resolver.list_robots()
        for robot in robots:
            print(f"  - {robot}")

        print("\n2. 机器人注册中心")
        print("-" * 70)
        registry = get_robot_registry()
        print(f"已注册机器人: {registry.list_robots()}")

        if registry.is_registered("RJ2506"):
            spec = registry.get("RJ2506")
            print(f"\nRJ2506 规格:")
            print(f"  URDF: {spec.urdf_path}")
            print(f"  DOF: {spec.dof}")
            print(f"  控制模式: {spec.control_mode}")
            print(f"  任务配置: {len(spec.task_configs)} 个任务")

        print("\n3. 包装器使用")
        print("-" * 70)
        print("创建包装器示例代码:")
        print("""
from waleo.sim.wrappers import CustomRobotWrapper

# env = gym.make("PickCube-v1", robot_uids="rj2506")
# wrapped = CustomRobotWrapper(env, "rj2506", {
#     "robot_pose": {"offset": [-0.85, 0, -0.35]},
#     "keyframes": {"rest": {"qpos": [...]}}
# })
""")

        print("✅ Waleo Sim 无侵入式设计已就绪！\n")

    except Exception as e:
        print(f"\n注意: 部分功能需要安装依赖: {e}")
        print("运行: pip install -e . 安装 waleo\n")


# ============================================================================
# 示例 5：完整训练脚本对比
# ============================================================================

def training_script_comparison():
    """完整训练脚本对比"""
    print("=" * 70)
    print("示例 5：完整训练脚本对比")
    print("=" * 70)

    print("\n【侵入式版本】需要 ~30 行准备代码:")
    print("""
import sys
import os
sys.path.insert(0, "custom_agents")

# 硬编码路径
ROBOT_URDF = "/root/data2/lyn/waleo/assets/robots/..."

# 导入（副作用）
from custom_agents.rj2506_variants import RJ2506_LeftArm

# Monkey-patch
from rj2506_config import apply_rj2506_config
apply_rj2506_config()  # 修改全局状态

# 更多配置代码...
# ...

# 终于可以创建环境了
import gymnasium as gym
env = gym.make("PickCube-v1", robot_uids="rj2506", num_envs=512)

# 开始训练
for i in range(1000):
    # ...
""")

    print("\n【无侵入式版本】只需 ~5 行代码:")
    print("""
from waleo.sim import make_env

# 一行创建环境（假设已设置 WALEO_ASSETS_DIR）
env = make_env("pick_place", robot="rj2506", num_envs=512)

# 开始训练
for i in range(1000):
    # ...
""")

    print("\n代码行数: 30+ 行 → 5 行")
    print("复杂度: ⭐⭐⭐⭐⭐ → ⭐")
    print("维护性: 差 → 优\n")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """运行所有示例"""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "Waleo Sim 无侵入式设计示例" + " " * 19 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    # 示例 1: 侵入式做法
    old_invasive_approach()
    input("\n按 Enter 继续...")

    # 示例 2: 无侵入式做法
    new_non_invasive_approach()
    input("\n按 Enter 继续...")

    # 示例 3: 工厂 API
    future_factory_api()
    input("\n按 Enter 继续...")

    # 示例 4: 实际演示
    practical_demo()
    input("\n按 Enter 继续...")

    # 示例 5: 训练脚本对比
    training_script_comparison()

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
Waleo Sim 无侵入式设计的核心优势:

1. **环境变量驱动**
   - WALEO_ASSETS_DIR 自动发现机器人
   - 无需硬编码路径

2. **Wrapper 模式**
   - 组合而非修改
   - 符合 Gym/Gymnasium 标准
   - 可测试、可组合

3. **配置驱动**
   - robot.yaml 声明式配置
   - 无需修改代码

4. **插件式架构**
   - 自动注册和发现
   - 无需手动导入

5. **完全无侵入**
   - 不修改 ManiSkill 源码
   - 不污染全局状态
   - 不依赖私有 API

实施进度:
  ✅ Phase 1-2: AssetResolver + RobotRegistry
  ✅ Phase 3: CustomRobotWrapper + TaskConfigWrapper
  ⏳ Phase 4: robot.yaml 配置文件
  ⏳ Phase 5: make_env() 工厂函数
""")

    print("\n下一步: 运行 Phase 4-5 完成整体集成\n")


if __name__ == "__main__":
    main()
