"""测试训练好的模型"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.train_sb3 import make_sb3_vec_env
from stable_baselines3 import PPO
import numpy as np

print('='*60)
print('测试训练好的模型')
print('='*60)

print('\n1. 创建环境...')
vec_env = make_sb3_vec_env('PickCube-v1', num_envs=4)
print(f'   环境创建成功，num_envs={vec_env.num_envs}')

print('\n2. 加载模型...')
model = PPO.load('./models/ppo_maniskill/final_model.zip', device='cuda')
print('   模型加载成功')

print('\n3. 运行测试...')
obs = vec_env.reset()
episode_rewards = [0] * vec_env.num_envs
success_count = 0

for step in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)

    for i in range(vec_env.num_envs):
        episode_rewards[i] += rewards[i]
        if dones[i]:
            print(f'   Episode {i} completed at step {step}: reward={episode_rewards[i]:.2f}')
            if 'is_success' in info[i] and info[i]['is_success']:
                success_count += 1
                print(f'      *** SUCCESS! ***')
            episode_rewards[i] = 0

    if step % 100 == 0:
        print(f'   Step {step}: current_rewards={[episode_rewards[i] for i in range(vec_env.num_envs)]}')

print('\n' + '='*60)
print('测试结果')
print('='*60)
print(f'成功次数: {success_count}')
print('='*60)

vec_env.close()
print('\n测试完成！')
