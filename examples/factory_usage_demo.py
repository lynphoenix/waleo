"""factory.py 使用演示

展示如何使用一行代码创建配置好的仿真环境。
"""

# ============================================================================
# 示例 1：基础使用（ManiSkill 内置机器人）
# ============================================================================

print("=" * 70)
print("示例 1：使用 ManiSkill 内置机器人")
print("=" * 70)
print("""
from waleo.sim import make_env

# 创建 Panda 机器人的 PickCube 环境
env = make_env("PickCube-v1", robot="panda", num_envs=1)

# 就这么简单！环境已配置好
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
""")

# ============================================================================
# 示例 2：使用自定义机器人（自动配置）
# ============================================================================

print("\n" + "=" * 70)
print("示例 2：使用自定义机器人（RJ2506）")
print("=" * 70)
print("""
from waleo.sim import make_env

# 一行创建 RJ2506 环境，自动应用配置：
# - robot_pose: [-0.85, 0, -0.35]
# - keyframes: gripper 完全张开
# - object_config: 0.8cm 立方体
# - camera_config: 自定义相机位置
env = make_env("PickCube-v1", robot="rj2506", num_envs=512)

# 配置已自动应用！可以直接开始训练
for i in range(1000):
    obs, info = env.reset()
    done = False
    while not done:
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
""")

# ============================================================================
# 示例 3：指定观测和控制模式
# ============================================================================

print("\n" + "=" * 70)
print("示例 3：自定义观测和控制模式")
print("=" * 70)
print("""
from waleo.sim import make_env

# RGBD 观测 + 末端执行器控制
env = make_env(
    "PickCube-v1",
    robot="panda",
    obs_mode="rgbd",           # RGBD 图像观测
    control_mode="pd_ee_delta_pose",  # 末端执行器增量控制
    num_envs=128,
    render_mode="cameras"      # 相机渲染
)
""")

# ============================================================================
# 示例 4：GPU 批量仿真
# ============================================================================

print("\n" + "=" * 70)
print("示例 4：GPU 批量仿真（高吞吐量训练）")
print("=" * 70)
print("""
from waleo.sim import make_env

# 512 个并行环境（GPU 加速）
env = make_env(
    "PickCube-v1",
    robot="rj2506",
    num_envs=512,
    sim_freq=500,
    control_freq=20
)

# 所有环境并行执行
obs, info = env.reset()  # (512, obs_dim)
actions = policy(obs)     # (512, action_dim)
obs, rewards, term, trunc, info = env.step(actions)  # 一次性执行 512 个
""")

# ============================================================================
# 示例 5：列出可用资源
# ============================================================================

print("\n" + "=" * 70)
print("示例 5：查询可用资源")
print("=" * 70)
print("""
from waleo.sim import list_available_robots, list_available_tasks

# 列出所有可用机器人
robots = list_available_robots()
print(f"Available robots: {robots}")
# 输出: ['ALLEGRO_HAND', 'DCLAW', 'FETCH', 'PANDA', 'RJ2506', 'XARM7']

# 列出 ManiSkill 的所有任务
tasks = list_available_tasks("maniskill")
print(f"Available tasks: {tasks[:5]}")
# 输出: ['PickCube-v1', 'StackCube-v1', 'PegInsertionSide-v1', ...]
""")

# ============================================================================
# 对比：侵入式 vs 无侵入式
# ============================================================================

print("\n" + "=" * 70)
print("对比：代码简化程度")
print("=" * 70)

print("\n【旧方式】~30 行准备代码：")
print("""
import sys
import os
sys.path.insert(0, "custom_agents")

# 硬编码路径
ROBOT_URDF = "/root/data2/lyn/waleo/assets/robots/..."

# 导入自定义机器人
from custom_agents.rj2506_variants import RJ2506_LeftArm

# Monkey-patch ManiSkill
from rj2506_config import apply_rj2506_config
apply_rj2506_config()  # 修改全局状态

# 手动配置机器人 pose
def custom_reset():
    # ... 30+ 行配置代码
    pass

# 终于可以创建环境了
env = gym.make("PickCube-v1", robot_uids="rj2506", num_envs=512)
""")

print("\n【新方式】1 行代码：")
print("""
from waleo.sim import make_env

env = make_env("PickCube-v1", robot="rj2506", num_envs=512)
""")

print("\n代码行数: 30+ 行 → 1 行")
print("复杂度: ⭐⭐⭐⭐⭐ → ⭐")
print("可移植性: 差 → 优")
print("可维护性: 差 → 优\n")

# ============================================================================
# 高级用法：手动使用包装器
# ============================================================================

print("=" * 70)
print("高级用法：手动使用包装器（低级 API）")
print("=" * 70)
print("""
from waleo.sim import create_wrapped_env
from waleo.sim.registry import get_robot_registry
import gymnasium as gym

# 1. 创建基础环境
base_env = gym.make("PickCube-v1", robot_uids="rj2506", num_envs=512)

# 2. 获取机器人配置
registry = get_robot_registry()
spec = registry.get("RJ2506")
task_config = spec.get_task_config("PickCube-v1")

# 3. 应用包装器
env = create_wrapped_env(base_env, "RJ2506", task_config)

# 这种方式提供了更多控制，但通常不需要
# 推荐直接使用 make_env()
""")

# ============================================================================
# 总结
# ============================================================================

print("\n" + "=" * 70)
print("总结：make_env() 的优势")
print("=" * 70)
print("""
1. **一行代码**: 不需要手动配置
2. **自动发现**: 自动加载 robot.yaml 配置
3. **无侵入**: 不修改 ManiSkill 源码
4. **可移植**: 基于 WALEO_ASSETS_DIR 环境变量
5. **标准化**: 符合 Gymnasium API 规范
6. **易测试**: 无全局状态污染
7. **易维护**: 声明式配置，代码简洁

使用建议：
- 优先使用 make_env() 创建环境
- 自定义机器人放在 assets/robots/<NAME>/
- 配置写在 robot.yaml 中
- 设置 WALEO_ASSETS_DIR 环境变量

下一步：
- 运行 make_env() 创建实际环境
- 开始训练你的强化学习策略！
""")

print("=" * 70)
print("Phase 5 完成！无侵入式设计全部实现。")
print("=" * 70)
