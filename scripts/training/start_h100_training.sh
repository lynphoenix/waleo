#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1
cd /root/data2/lyn/waleo/examples/maniskill
/root/miniconda3/envs/waleo/bin/python -u train_ppo_vectorized_from_original.py --robot-uids rj2506 --env-id PickCube-v1 --control-mode pd_joint_delta_pos --num-envs 512 --num-eval-envs 8 --total-timesteps 20000000 --seed 1 --capture-video --save-model --no-track > /root/data2/lyn/waleo/training_h100_gpu7_20M.log 2>&1 &
