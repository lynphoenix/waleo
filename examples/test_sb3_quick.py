"""简短的 SB3 训练测试"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.train_sb3 import make_sb3_env
from stable_baselines3 import PPO
import time

print("Creating environment...")
env = make_sb3_env('PickCube-v1')
print(f"Observation space: {env.observation_space.shape}")
print(f"Action space: {env.action_space.shape}")

# Test reset/step
print("\nTesting environment...")
obs, _ = env.reset()
print(f"Observation shape: {obs.shape}")

for i in range(3):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i+1}: reward={reward:.4f}")
    if terminated or truncated:
        break

print("\nCreating PPO model...")
model = PPO('MlpPolicy', env, verbose=1, n_steps=256, batch_size=64, device='cpu')

print("\n" + "="*60)
print("Starting training (5000 steps)...")
print("="*60)

start = time.time()
model.learn(total_timesteps=5000)
elapsed = time.time() - start

print("="*60)
print(f"Training completed in {elapsed:.1f}s!")
print("="*60)

# Test the trained model
print("\nTesting trained model (3 episodes)...")
success_count = 0
for ep in range(3):
    obs, _ = env.reset()
    done = False
    ep_reward = 0
    steps = 0
    while not done and steps < 100:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        done = terminated or truncated
        steps += 1
    success = info.get("success", False)
    if success:
        success_count += 1
    print(f"Episode {ep+1}: reward={ep_reward:.2f}, success={success}, steps={steps}")

print(f"\nSuccess rate: {success_count}/3")

env.close()
print("\n✅ Training test successful!")
