#!/usr/bin/env python3
"""
Configurable PPO training script for ManiSkill environments.

This script supports:
- Different robot configurations (panda, RJ2506 variants)
- Custom camera resolution via sensor_configs
- Configurable number of environments and training parameters
- Automatic RJ2506 positioning to avoid table collision

Usage:
    # Train with Panda, default resolution (128x128)
    python train_ppo_configurable.py --robot_uids panda --num_envs 512

    # Train with Panda, custom resolution (256x256)
    python train_ppo_configurable.py --robot_uids panda --camera_resolution 256 --num_envs 512

    # Train with RJ2506 (single arm, environment camera only)
    python train_ppo_configurable.py --robot_uids RJ2506_LeftArm --camera_resolution 256 --num_envs 512
"""

import sys
import gymnasium as gym

# Parse custom camera resolution from command line
camera_resolution = 128  # default
if "--camera_resolution" in sys.argv:
    idx = sys.argv.index("--camera_resolution")
    camera_resolution = int(sys.argv[idx + 1])
    # Remove the custom argument so it doesn't confuse the original script
    sys.argv.pop(idx)
    sys.argv.pop(idx)
    print(f"[CONFIG] Using camera resolution: {camera_resolution}x{camera_resolution}")

# Patch gym.make to inject sensor_configs
original_make = gym.make

def patched_make(env_id, **kwargs):
    # Inject sensor_configs if not already provided
    if "sensor_configs" not in kwargs:
        kwargs["sensor_configs"] = {}
    if "base_camera" not in kwargs["sensor_configs"]:
        kwargs["sensor_configs"]["base_camera"] = {}
    
    # Set width and height for base_camera
    kwargs["sensor_configs"]["base_camera"]["width"] = camera_resolution
    kwargs["sensor_configs"]["base_camera"]["height"] = camera_resolution
    
    # Call original make
    env = original_make(env_id, **kwargs)
    
    # Apply RJ2506 pose adjustment if needed
    robot_uids = kwargs.get('robot_uids', '')
    if 'RJ2506' in robot_uids:
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'agent'):
            import numpy as np
            import sapien
            # Pose offset: move to side (x+0.3) and down to floor (z-0.8)
            pose_offset = np.array([0.3, 0, -0.8], dtype=np.float32)
            robot = env.unwrapped.agent.robot
            current_p = robot.pose.p[0].cpu().numpy()
            new_p = current_p + pose_offset
            robot.set_pose(sapien.Pose(p=new_p))
            print(f"[RJ2506] Adjusted robot pose from {current_p} to {new_p}")
    
    return env

# Apply the patch
gym.make = patched_make
print(f"[CONFIG] Patched gym.make to use {camera_resolution}x{camera_resolution} base camera resolution")

# Import custom agents to register them
from custom_agents.rj2506_variants import RJ2506, RJ2506_LeftArm, RJ2506_LeftArm_WithWristCam

# Execute the original training script
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open('train_ppo_vectorized_from_original.py') as f:
    exec(f.read())
