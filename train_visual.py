import sys
sys.path.insert(0, '/workspace')

from stable_baselines3 import PPO
from mani_skill.vector.wrappers.sb3 import ManiSkillSB3VectorEnv
from mani_skill.envs.tasks import PickCubeEnv
import torch

print("=" * 60)
print("ManiSkill 视觉观察训练 (RGB-D)")
print("=" * 60)

print("创建环境...")
env = PickCubeEnv(
    num_envs=64,
    obs_mode="rgbd",
    reward_mode="dense",
)
vec_env = ManiSkillSB3VectorEnv(env)

print(f"观察空间: {vec_env.observation_space}")
print(f"动作空间: {vec_env.action_space}")

print("创建模型 (CNN Policy)...")
model = PPO(
    "CnnPolicy",
    vec_env,
    learning_rate=3e-4,
    n_steps=50,
    batch_size=128,
    n_epochs=8,
    gamma=0.8,
    verbose=1,
    device="cuda:0",
    tensorboard_log="/workspace/logs/visual_maniskill"
)

print("开始训练 1000 步...")
model.learn(total_timesteps=1000, progress_bar=True)

model.save("/workspace/models/visual_maniskill_1k")
print("训练完成!")

vec_env.close()
