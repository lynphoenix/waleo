"""基于 Stable-Baselines3 的 ManiSkill3 训练

使用 ManiSkill 官方的 ManiSkillSB3VectorEnv 包装器
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch


def make_sb3_vec_env(task="PickCube-v1", num_envs=64):
    """创建适合 SB3 训练的并行环境"""
    from waleo.sim import make_env
    from mani_skill.vector.wrappers.sb3 import ManiSkillSB3VectorEnv

    # 创建 ManiSkill 并行环境，使用 state 模式获取扁平观察
    # 后端特定参数通过 kwargs 传递
    ms3_env = make_env(task, num_envs=num_envs, obs_mode="state")
    vec_env = ManiSkillSB3VectorEnv(ms3_env)
    return vec_env


def train_ppo(
    task="PickCube-v1",
    total_timesteps=100000,
    num_envs=64,
    eval_freq=5000,
    save_path="./models/ppo_maniskill",
):
    """使用 PPO 算法训练"""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

    print("=" * 60)
    print("PPO 训练 - ManiSkill3")
    print("=" * 60)

    # 创建并行训练环境
    print(f"\n1. 创建并行环境 (num_envs={num_envs})...")
    vec_env = make_sb3_vec_env(task, num_envs=num_envs)
    print(f"   动作空间: {vec_env.action_space}")
    print(f"   观察空间: {vec_env.observation_space}")

    # 创建 PPO 模型
    print("\n2. 创建 PPO 模型...")
    model = PPO(
        "MlpPolicy",  # 扁平观察使用 MlpPolicy
        vec_env,
        learning_rate=3e-4,
        n_steps=50,  # 官方配置
        batch_size=128,  # 官方配置
        n_epochs=8,  # 官方配置
        gamma=0.8,  # 官方配置
        gae_lambda=0.9,  # 官方配置
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./logs/ppo_maniskill",
        device="cuda",
    )

    # 设置检查点回调
    print("\n3. 配置训练...")
    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq,
        save_path=save_path,
        name_prefix="ppo_model",
    )

    # 开始训练
    print(f"\n4. 开始训练 (总步数: {total_timesteps})...")
    print("=" * 60)

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    print("=" * 60)
    print("训练完成!")
    print("=" * 60)

    # 保存最终模型
    final_path = f"{save_path}/final_model"
    model.save(final_path)
    print(f"\n模型已保存到: {final_path}")

    vec_env.close()

    return model


def evaluate_model(model_path, num_episodes=20, task="PickCube-v1", num_envs=16):
    """评估训练好的模型"""
    from stable_baselines3 import PPO
    from mani_skill.utils import gym_utils

    print("=" * 60)
    print("模型评估")
    print("=" * 60)

    # 加载模型
    print(f"\n加载模型: {model_path}")
    model = PPO.load(model_path)

    # 创建并行评估环境
    print(f"创建并行环境 (num_envs={num_envs})...")
    vec_env = make_sb3_vec_env(task, num_envs=num_envs)

    # 获取最大步数
    max_episode_steps = 100  # 默认值

    # 运行评估
    obs = vec_env.reset()
    success_count = 0
    total_episodes = 0

    print(f"运行评估 (max {max_episode_steps} 步)...")

    for step in range(max_episode_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)

        # 检查成功
        if "is_success" in info[0]:
            for i, done in enumerate(dones):
                if done and info[i].get("is_success", False):
                    success_count += 1

        total_episodes += dones.sum()

        if total_episodes >= num_episodes:
            break

    vec_env.close()

    success_rate = success_count / num_episodes * 100

    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"总 Episodes: {num_episodes}")
    print(f"成功次数: {success_count}")
    print(f"成功率: {success_rate:.1f}%")
    print("=" * 60)

    return {
        "success_rate": success_rate / 100,
        "success_count": success_count,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SB3 训练 ManiSkill")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--task", type=str, default="PickCube-v1")
    parser.add_argument("--timesteps", type=int, default=50000, help="训练步数")
    parser.add_argument("--model", type=str, default=None, help="模型路径（用于评估）")
    parser.add_argument("--episodes", type=int, default=20, help="评估 episodes 数量")

    args = parser.parse_args()

    if args.mode == "train":
        model = train_ppo(
            task=args.task,
            total_timesteps=args.timesteps,
        )
    else:
        model_path = args.model or "./models/ppo_maniskill/best_model.zip"
        evaluate_model(model_path, num_episodes=args.episodes, task=args.task)
