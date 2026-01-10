"""PPO 训练 - 优化版 (512并行环境)

基于 ManiSkill 原版 ppo_rgb.py，优化参数:
- learning_rate: 3e-4 -> 1e-4
- batch_size: 更大 (512 * 100 = 51200)
"""

# 直接运行原版脚本，使用优化的参数
import subprocess
import sys

# 运行原版脚本，传入优化参数
args = [
    sys.executable,
    "/home/smai/linyining/waleo/waleo-sim/examples/maniskill/train_ppo_vectorized_from_original.py",
    "--total-timesteps", "1000000",  # 测试模式 100万步
    "--num-envs", "64",  # 测试用64环境
    "--learning-rate", "1e-4",  # 降低学习率
    "--capture-video", "False",  # 不保存视频
]

print("运行优化的 PPO 训练...")
print(f"命令: {' '.join(args)}")
print()

result = subprocess.run(args, cwd="/home/smai/linyining/waleo/waleo-sim/examples/maniskill")
sys.exit(result.returncode)
