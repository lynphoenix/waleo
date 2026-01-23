#!/usr/bin/env python3
"""
简化版Profile脚本 - 专注于训练的关键瓶颈
"""

import time
import gymnasium as gym
import mani_skill.envs
import numpy as np

def profile_training(robot_uid="rj2506", num_envs=512):
    """Profile训练的关键环节"""
    print(f"\n{'='*60}")
    print(f"Profile: {robot_uid}")
    print(f"Envs: {num_envs}")
    print(f"{'='*60}\n")

    results = {}

    # 1. 纯物理测试
    print("1. 纯物理测试 (obs_mode=state)")
    env_physics = gym.make(
        "PickCube-v1",
        robot_uids=robot_uid,
        num_envs=num_envs,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        sim_backend="physx_cuda",
    )

    obs, _ = env_physics.reset(seed=42)
    actions = env_physics.action_space.sample()

    # Warmup
    for _ in range(3):
        obs, _, _, _, _ = env_physics.step(actions)

    # Test
    start = time.perf_counter()
    for _ in range(50):
        obs, _, _, _, _ = env_physics.step(actions)
    elapsed = time.perf_counter() - start

    physics_fps = (50 * num_envs) / elapsed
    results['physics_fps'] = physics_fps
    print(f"   时间: {elapsed:.3f}s, FPS: {physics_fps:.1f}\n")

    env_physics.close()

    # 2. 完整rollout测试 (物理+渲染)
    print("2. 完整Rollout测试 (物理+渲染, obs_mode=rgb)")
    env_full = gym.make(
        "PickCube-v1",
        robot_uids=robot_uid,
        num_envs=num_envs,
        obs_mode="rgb",
        control_mode="pd_joint_delta_pos",
        sim_backend="physx_cuda",
        render_mode="rgb_array",
    )

    obs, _ = env_full.reset(seed=42)
    actions = np.zeros((num_envs,) + env_full.single_action_space.shape)

    # Warmup
    for _ in range(3):
        obs, _, _, _, _ = env_full.step(actions)

    # Test
    start = time.perf_counter()
    for _ in range(10):
        obs, _, _, _, _ = env_full.step(actions)
    elapsed = time.perf_counter() - start

    total_images = 10 * num_envs * 3  # 3 cameras
    render_fps = total_images / elapsed
    env_fps = (10 * num_envs) / elapsed

    results['render_fps'] = render_fps
    results['full_env_fps'] = env_fps
    print(f"   时间: {elapsed:.3f}s")
    print(f"   图像FPS: {render_fps:.1f}")
    print(f"   环境FPS: {env_fps:.1f}\n")

    env_full.close()

    # 3. 计算渲染开销
    print("3. 性能分析")
    # 使用物理+渲染的时间减去纯物理的时间来估算渲染时间
    # 假设物理时间不变
    physics_time_per_step = (1 / results['physics_fps'])
    full_time_per_step = (1 / results['full_env_fps'])
    render_time_per_step = full_time_per_step - physics_time_per_step

    render_overhead = (render_time_per_step / full_time_per_step) * 100
    physics_overhead = (physics_time_per_step / full_time_per_step) * 100

    results['render_overhead_pct'] = render_overhead
    results['physics_overhead_pct'] = physics_overhead

    print(f"   渲染开销: {render_overhead:.1f}%")
    print(f"   物理开销: {physics_overhead:.1f}%")

    return results

if __name__ == "__main__":
    import torch

    print("="*60)
    print("训练性能Profile")
    print("="*60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    results = profile_training(robot_uid="rj2506", num_envs=512)

    print(f"\n{'='*60}")
    print("结果总结")
    print(f"{'='*60}")
    print(f"物理FPS:      {results['physics_fps']:>8.1f}")
    print(f"渲染FPS:      {results['render_fps']:>8.1f}")
    print(f"环境FPS:      {results['full_env_fps']:>8.1f}")
    print(f"渲染开销占比: {results['render_overhead_pct']:>8.1f}%")
    print(f"物理开销占比: {results['physics_overhead_pct']:>8.1f}%")
