#!/bin/bash
# ManiSkill 训练脚本
cd /root/data2/lyn/waleo
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=5
exec python -u examples/train_sb3.py --timesteps 20000000
