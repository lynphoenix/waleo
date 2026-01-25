#!/usr/bin/env python3
"""
详细解释FPS计算方法
"""

import time
import gymnasium as gym
import mani_skill.envs
import numpy as np

def explain_fps_calculation():
    """详细解释FPS的计算"""

    print("="*60)
    print("FPS计算方法详解")
    print("="*60)

    # 创建环境
    env = gym.make(
        "PickCube-v1",
        robot_uids="rj2506",
        num_envs=512,
        obs_mode="rgb",
        control_mode="pd_joint_delta_pos",
        sim_backend="physx_cuda",
        render_mode="rgb_array",
    )

    obs, _ = env.reset(seed=42)
    actions = np.zeros((512,) + env.single_action_space.shape)

    # Warmup
    for _ in range(3):
        obs, _, _, _, _ = env.step(actions)

    # 测试
    num_steps = 10
    num_envs = 512
    num_cameras = len(obs['sensor_data'])  # 计算相机数量

    print(f"\n测试参数:")
    print(f"  步数: {num_steps}")
    print(f"  环境数: {num_envs}")
    print(f"  相机数: {num_cameras}")

    start = time.perf_counter()
    for _ in range(num_steps):
        obs, _, _, _, _ = env.step(actions)
    elapsed = time.perf_counter() - start

    # 计算各种FPS
    total_env_steps = num_steps * num_envs
    total_images = total_env_steps * num_cameras

    env_fps = total_env_steps / elapsed
    render_fps = total_images / elapsed

    print(f"\n执行时间: {elapsed:.3f}秒")
    print(f"\n计算过程:")
    print(f"  总环境步数 = {num_steps}步 × {num_envs}环境 = {total_env_steps:,}步")
    print(f"  总图像数 = {total_env_steps:,}步 × {num_cameras}相机 = {total_images:,}张")
    print(f"\n结果:")
    print(f"  环境FPS = {total_env_steps:,}步 / {elapsed:.3f}秒 = {env_fps:.1f} FPS")
    print(f"  渲染FPS = {total_images:,}张 / {elapsed:.3f}秒 = {render_fps:.1f} FPS")
    print(f"\n验证关系: {render_fps:.1f} = {env_fps:.1f} × {num_cameras}? ", end="")
    if abs(render_fps - env_fps * num_cameras) < 1:
        print("✓ 正确")
    else:
        print("✗ 不匹配")

    # 解释每个指标的意义
    print(f"\n{'='*60}")
    print("各指标的实际意义:")
    print(f"{'='*60}")
    print(f"\n1. 环境FPS ({env_fps:.1f})")
    print(f"   - 表示每秒能完成{env_fps:.0f}个环境的完整一步")
    print(f"   - 包括：物理计算 + 渲染所有相机 + 数据收集")
    print(f"   - 这是最重要的指标，直接反映训练速度")
    print(f"   - 训练中的SPS就是这个值（略低因为包含更新时间）")

    print(f"\n2. 渲染FPS ({render_fps:.1f})")
    print(f"   - 表示每秒能渲染{render_fps:.0f}张图像")
    print(f"   - 纯粹的GPU渲染性能指标")
    print(f"   - 用于衡量渲染管线的效率")

    print(f"\n3. 为什么渲染FPS = 环境FPS × 相机数?")
    print(f"   - 每个环境一步需要渲染{num_cameras}张图像")
    print(f"   - 所以总图像数 = 环境步数 × {num_cameras}")
    print(f"   - 因此渲染FPS = 环境FPS × {num_cameras}")

    # 对比纯物理
    print(f"\n{'='*60}")
    print("对比纯物理性能:")
    print(f"{'='*60}")

    env_physics = gym.make(
        "PickCube-v1",
        robot_uids="rj2506",
        num_envs=512,
        obs_mode="state",  # 不渲染
        control_mode="pd_joint_delta_pos",
        sim_backend="physx_cuda",
    )

    obs, _ = env_physics.reset(seed=42)
    actions = env_physics.action_space.sample()

    for _ in range(3):
        obs, _, _, _, _ = env_physics.step(actions)

    num_steps_physics = 50
    start = time.perf_counter()
    for _ in range(num_steps_physics):
        obs, _, _, _, _ = env_physics.step(actions)
    elapsed_physics = time.perf_counter() - start

    physics_fps = (num_steps_physics * 512) / elapsed_physics

    print(f"\n纯物理测试 ({num_steps_physics}步, 不渲染):")
    print(f"  时间: {elapsed_physics:.3f}秒")
    print(f"  物理FPS: {physics_fps:.1f}")

    print(f"\n完整rollout vs 纯物理:")
    print(f"  完整rollout FPS: {env_fps:.1f}")
    print(f"  纯物理FPS: {physics_fps:.1f}")
    print(f"  差异: {physics_fps/env_fps:.2f}x")

    # 计算时间分配
    time_per_step_full = 1 / env_fps
    time_per_step_physics = 1 / physics_fps
    time_per_step_render = time_per_step_full - time_per_step_physics

    print(f"\n时间分解（每一步）:")
    print(f"  完整rollout: {time_per_step_full*1000:.3f}ms")
    print(f"  纯物理: {time_per_step_physics*1000:.3f}ms")
    print(f"  渲染: {time_per_step_render*1000:.3f}ms")
    print(f"\n时间占比:")
    print(f"  物理: {time_per_step_physics/time_per_step_full*100:.1f}%")
    print(f"  渲染: {time_per_step_render/time_per_step_full*100:.1f}%")

    env.close()
    env_physics.close()

    # 总结
    print(f"\n{'='*60}")
    print("总结")
    print(f"{'='*60}")
    print(f"\n在这个测试配置下:")
    print(f"  • 瓶颈在{'渲染' if time_per_step_render > time_per_step_physics else '物理'}")
    print(f"  • 如果优化渲染，性能最多提升{physics_fps/env_fps:.2f}x")
    print(f"  • 如果优化物理，性能最多提升{1/(time_per_step_render/time_per_step_full):.2f}x")

if __name__ == "__main__":
    explain_fps_calculation()
