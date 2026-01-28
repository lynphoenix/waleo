"""监控训练进度"""
import time
import subprocess
import re

def get_gpu_stats(gpu_id=5):
    """获取 GPU 统计"""
    result = subprocess.run(
        ["nvidia-smi", f"--id={gpu_id}", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        util, mem_used, mem_total, power = result.stdout.strip().split(", ")
        return {
            "util": f"{util}%",
            "mem": f"{mem_used}/{mem_total}MiB",
            "power": f"{power}W"
        }
    return None

def get_latest_checkpoint():
    """获取最新检查点"""
    import os
    import glob

    ckpt_dir = "models/ppo_maniskill"
    if not os.path.exists(ckpt_dir):
        return None

    ckpts = glob.glob(f"{ckpt_dir}/ppo_model*_steps.zip")
    if ckpts:
        latest = max(ckpts, key=os.path.getmtime)
        # 提取步数
        match = re.search(r"ppo_model_(\d+)_steps", latest)
        if match:
            return int(match.group(1))
    return None

def main():
    print("=" * 60)
    print("训练进度监控 (GPU 5)")
    print("=" * 60)

    while True:
        gpu = get_gpu_stats(5)
        steps = get_latest_checkpoint()

        print(f"\rGPU: {gpu['util'] if gpu else 'N/A':>6} | Mem: {gpu['mem'] if gpu else 'N/A':>15} | Power: {gpu['power'] if gpu else 'N/A':>6} | Checkpoint: {steps if steps else '等待中...':>7} steps", end="", flush=True)

        time.sleep(5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n监控结束")
