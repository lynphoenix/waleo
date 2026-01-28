"""多后端 PPO 训练对比脚本

支持 ManiSkill3, MuJoCo, IsaacSim 三种后端
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import numpy as np
import torch


def make_sb3_vec_env_maniskill(task="PickCube-v1", num_envs=512, device="cuda:0"):
    """创建 ManiSkill3 SB3 向量环境"""
    from waleo.sim import make_env
    from mani_skill.vector.wrappers.sb3 import ManiSkillSB3VectorEnv

    ms3_env = make_env(
        task,
        backend="maniskill",
        num_envs=num_envs,
        obs_mode="state",
        robot_uids="panda"
    )
    vec_env = ManiSkillSB3VectorEnv(ms3_env)
    return vec_env


def make_sb3_vec_env_mujoco(task="Ant-v4", num_envs=512, device="cuda:0"):
    """创建 MuJoCo SB3 向量环境"""
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from gymnasium.wrappers import TimeLimit
    import gymnasium as gym

    def make_env_fn(rank):
        def _init():
            env = gym.make(task, max_episode_steps=1000)
            return env
        return _init

    # MuJoCo 使用 CPU 并行（不需要 GPU 加速）
    vec_env = SubprocVecEnv([make_env_fn(i) for i in range(num_envs)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    return vec_env


def make_sb3_vec_env_isaac(task="Isaac-Lift-Cube-Franka-v0", num_envs=512, device="cuda:0"):
    """创建 IsaacSim SB3 向量环境"""
    from waleo.sim import make_env

    # IsaacSim 使用自己的向量环境
    vec_env = make_env(
        task,
        backend="isaacsim",
        num_envs=num_envs,
        headless=True,
        device=device
    )
    return vec_env


# 后端配置
BACKEND_CONFIG = {
    "maniskill": {
        "env_fn": make_sb3_vec_env_maniskill,
        "task": "PickCube-v1",
        "num_envs": 512,
        "save_path": "./models/ppo_maniskill_20M",
        "log_path": "./logs/ppo_maniskill_20M",
        "use_gpu": True,
        # PPO 超参数（ManiSkill 官方推荐）
        "ppo_kwargs": {
            "learning_rate": 3e-4,
            "n_steps": 50,
            "batch_size": 128,
            "n_epochs": 8,
            "gamma": 0.8,
            "gae_lambda": 0.9,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        }
    },
    "mujoco": {
        "env_fn": make_sb3_vec_env_mujoco,
        "task": "Ant-v4",
        "num_envs": 512,
        "save_path": "./models/ppo_mujoco_20M",
        "log_path": "./logs/ppo_mujoco_20M",
        "use_gpu": False,  # MuJoCo 通常不需要 GPU
        # PPO 超参数（MuJoCo 推荐）
        "ppo_kwargs": {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        }
    },
    "isaacsim": {
        "env_fn": make_sb3_vec_env_isaac,
        "task": "Isaac-Lift-Cube-Franka-v0",
        "num_envs": 512,
        "save_path": "./models/ppo_isaacsim_20M",
        "log_path": "./logs/ppo_isaacsim_20M",
        "use_gpu": True,
        # PPO 超参数（IsaacSim 推荐）
        "ppo_kwargs": {
            "learning_rate": 3e-4,
            "n_steps": 50,
            "batch_size": 128,
            "n_epochs": 8,
            "gamma": 0.8,
            "gae_lambda": 0.9,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        }
    },
}


def train_ppo(
    backend="maniskill",
    total_timesteps=20_000_000,
    gpu_id=0,
):
    """使用 PPO 算法训练"""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback

    config = BACKEND_CONFIG[backend]
    task = config["task"]
    num_envs = config["num_envs"]
    save_path = config["save_path"]
    log_path = config["log_path"]
    ppo_kwargs = config["ppo_kwargs"]
    use_gpu = config["use_gpu"]

    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    print("=" * 80)
    print(f"PPO 训练 - {backend.upper()} 后端")
    print("=" * 80)
    print(f"任务: {task}")
    print(f"并行环境数: {num_envs}")
    print(f"训练步数: {total_timesteps:,}")
    print(f"GPU: {gpu_id}")
    print(f"使用 GPU: {use_gpu}")
    print("=" * 80)

    # 设置设备
    device = f"cuda:{gpu_id}" if use_gpu else "cpu"
    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 创建环境
    print(f"\n1. 创建环境...")
    env_fn = config["env_fn"]
    vec_env = env_fn(task=task, num_envs=num_envs, device=device)
    print(f"   动作空间: {vec_env.action_space}")
    print(f"   观察空间: {vec_env.observation_space}")

    # 创建 PPO 模型
    print("\n2. 创建 PPO 模型...")
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=log_path,
        device=device,
        **ppo_kwargs
    )

    # 设置回调
    print("\n3. 配置训练...")
    checkpoint_callback = CheckpointCallback(
        save_freq=1_000_000,  # 每 1M 步保存一次
        save_path=save_path,
        name_prefix=f"ppo_{backend}",
    )

    # 开始训练
    print(f"\n4. 开始训练...")
    print("=" * 80)

    import time
    start_time = time.time()

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback],
        progress_bar=True,
    )

    elapsed = time.time() - start_time

    print("=" * 80)
    print("训练完成!")
    print(f"总耗时: {elapsed/3600:.2f} 小时")
    print("=" * 80)

    # 保存最终模型
    final_path = f"{save_path}/final_model"
    model.save(final_path)
    print(f"\n模型已保存到: {final_path}")

    vec_env.close()

    return model, elapsed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="多后端 PPO 训练")
    parser.add_argument("--backend", type=str, required=True,
                        choices=["maniskill", "mujoco", "isaacsim"],
                        help="仿真后端")
    parser.add_argument("--timesteps", type=int, default=20_000_000,
                        help="训练步数 (默认: 20M)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID (默认: 0)")

    args = parser.parse_args()

    model, elapsed = train_ppo(
        backend=args.backend,
        total_timesteps=args.timesteps,
        gpu_id=args.gpu,
    )
