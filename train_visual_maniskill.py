import sys
sys.path.insert(0, '/workspace')

from stable_baselines3 import PPO
from mani_skill.vector.wrappers.sb3 import ManiSkillSB3VectorEnv
from mani_skill.envs.tasks import PickCubeEnv
import torch

# 创建带视觉观察的环境
def make_visual_env(num_envs=64):
    env = PickCubeEnv(
        num_envs=num_envs,
        obs_mode='rgbd',  # RGB-D 视觉观察
        reward_mode='dense',
    )
    return ManiSkillSB3VectorEnv(env)

print("创建视觉环境 (RGB-D)...")
vec_env = make_visual_env(num_envs=64)
print(f"观察空间: {vec_env.observation_space}")

# 使用 CNN Policy (不是 MLP)
model = PPO(
    'CnnPolicy',
    vec_env,
    learning_rate=3e-4,
    n_steps=50,
    batch_size=128,
    n_epochs=8,
    gamma=0.8,
    verbose=1,
    device='cuda:0',
    tensorboard_log='/workspace/logs/visual_maniskill'
)

print("开始训练 1000 步...")
model.learn(total_timesteps=1000, progress_bar=True)

model.save('/workspace/models/visual_maniskill_1k')
print("训练完成！")

vec_env.close()
