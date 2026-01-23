#!/usr/bin/env python3
"""测试H100上rj2506 vs panda的性能差异"""

import time
import gymnasium as gym
import mani_skill.envs
import numpy as np

def test_physics_only(robot_uid, num_envs=256, num_steps=50):
    print(f"\n=== 测试 {robot_uid} 物理模拟 (无RGB) ===")
    env = gym.make("PickCube-v1", robot_uids=robot_uid, num_envs=num_envs,
                   obs_mode="state", control_mode="pd_joint_delta_pos",
                   sim_backend="physx_cuda", render_mode="rgb_array")
    obs, _ = env.reset(seed=42)
    start = time.time()
    for i in range(num_steps):
        actions = env.action_space.sample()
        obs, rewards, terms, truncs, infos = env.step(actions)
    elapsed = time.time() - start
    fps = (num_steps * num_envs) / elapsed
    print(f"  物理FPS: {fps:.1f}")
    env.close()
    return fps

def test_rendering_only(robot_uid, num_envs=64, num_steps=10):
    print(f"\n=== 测试 {robot_uid} 渲染性能 ===")
    env = gym.make("PickCube-v1", robot_uids=robot_uid, num_envs=num_envs,
                   obs_mode="rgb", control_mode="pd_joint_delta_pos",
                   sim_backend="physx_cuda", render_mode="rgb_array")
    obs, _ = env.reset(seed=42)
    start = time.time()
    for i in range(num_steps):
        actions = np.zeros((num_envs,) + env.single_action_space.shape)
        obs, rewards, terms, truncs, infos = env.step(actions)
    elapsed = time.time() - start
    render_fps = (num_steps * num_envs * 3) / elapsed
    print(f"  渲染FPS: {render_fps:.1f}")
    env.close()
    return render_fps

def test_full_rollout(robot_uid, num_envs=64, num_steps=10):
    print(f"\n=== 测试 {robot_uid} 完整Rollout ===")
    env = gym.make("PickCube-v1", robot_uids=robot_uid, num_envs=num_envs,
                   obs_mode="rgb", control_mode="pd_joint_delta_pos",
                   sim_backend="physx_cuda", render_mode="rgb_array")
    obs, _ = env.reset(seed=42)
    start = time.time()
    for i in range(num_steps):
        actions = env.action_space.sample()
        obs, rewards, terms, truncs, infos = env.step(actions)
    elapsed = time.time() - start
    fps = (num_steps * num_envs) / elapsed
    print(f"  SPS: {fps:.1f}")
    env.close()
    return fps

if __name__ == "__main__":
    print("=" * 60)
    print("H100性能诊断")
    print("=" * 60)
    
    for robot in ["panda", "rj2506"]:
        print(f"\n{'='*60}\n机器人: {robot}\n{'='*60}")
        p_fps = test_physics_only(robot)
        r_fps = test_rendering_only(robot)
        f_fps = test_full_rollout(robot)
        print(f"\n{robot} 总结: 物理FPS={p_fps:.1f}, 渲染FPS={r_fps:.1f}, SPS={f_fps:.1f}")
    print("\n" + "="*60)
