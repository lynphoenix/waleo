#!/usr/bin/env python3
"""
Configurable PPO training script with RGBD + state observations.

This script uses obs_mode="rgbd" which includes both RGB-D camera images 
AND state information (joint positions, velocities, etc.) for better learning.
"""

import sys
import gymnasium as gym

# Parse custom camera resolution from command line
camera_resolution = 128  # default
if "--camera_resolution" in sys.argv:
    idx = sys.argv.index("--camera_resolution")
    camera_resolution = int(sys.argv[idx + 1])
    sys.argv.pop(idx)
    sys.argv.pop(idx)
    print(f"[CONFIG] Using camera resolution: {camera_resolution}x{camera_resolution}")

# Patch gym.make to inject sensor_configs and pose adjustment
original_make = gym.make
import numpy as np
import sapien

# Pose offset for RJ2506 robots
RJ2506_POSE_OFFSET = np.array([0.3, 0, -0.8], dtype=np.float32)

def patched_make(env_id, **kwargs):
    # Inject sensor_configs
    if "sensor_configs" not in kwargs:
        kwargs["sensor_configs"] = {}
    if "base_camera" not in kwargs["sensor_configs"]:
        kwargs["sensor_configs"]["base_camera"] = {}
    kwargs["sensor_configs"]["base_camera"]["width"] = camera_resolution
    kwargs["sensor_configs"]["base_camera"]["height"] = camera_resolution
    
    # Use rgbd mode to include state information
    kwargs["obs_mode"] = "rgbd"
    
    # Call original make
    env = original_make(env_id, **kwargs)
    
    # Apply RJ2506 pose adjustment
    robot_uids = kwargs.get('robot_uids', '')
    if 'RJ2506' in robot_uids:
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'agent'):
            robot = env.unwrapped.agent.robot
            current_p = robot.pose.p[0].cpu().numpy()
            new_p = current_p + RJ2506_POSE_OFFSET
            robot.set_pose(sapien.Pose(p=new_p))
            print(f"[RJ2506] Adjusted robot pose from {current_p} to {new_p}")
    
    return env

# Apply the patch
gym.make = patched_make
print(f"[CONFIG] Patched gym.make with obs_mode=rgbd (RGBD + state), pose_offset for RJ2506")

# Import custom agents to register them
from custom_agents.rj2506_variants import RJ2506, RJ2506_LeftArm, RJ2506_LeftArm_WithWristCam

# Execute the original training script
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open('train_ppo_vectorized_from_original.py') as f:
    exec(f.read())
