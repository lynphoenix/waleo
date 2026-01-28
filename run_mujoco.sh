#!/bin/bash
# MuJoCo 训练脚本
cd /root/data2/lyn/waleo
export PYTHONUNBUFFERED=1
exec python -u examples/train_backend_comparison.py --backend mujoco --timesteps 20000000 --gpu 0
