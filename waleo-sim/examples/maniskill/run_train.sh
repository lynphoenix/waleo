#!/bin/bash

source ~/dc_dir/yes/etc/profile.d/conda.sh
conda activate waleo

cd /home/smai/linyining/waleo/waleo-sim/examples/maniskill

# 运行优化的 PPO 训练
python -u train_ppo_vectorized_from_original.py \
    --total-timesteps 1000000 \
    --num-envs 64 \
    --learning-rate 1e-4 \
    --capture-video 0 \
    --save-model 1 \
    --track 0

