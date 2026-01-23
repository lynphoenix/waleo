# Waleo - Robotics Simulation and Training Framework

Waleo is a comprehensive robotics simulation and training framework built on top of ManiSkill, providing tools for reinforcement learning training with visual observations.

## Project Structure

```
waleo/
├── waleo-sim/          # Simulation environment
│   ├── waleo_sim/      # Core simulation modules
│   ├── examples/       # Training examples and baselines
│   ├── assets/         # Robot assets (URDF files, meshes)
│   └── tests/          # Unit tests
├── waleo-config/       # Configuration management
├── waleo-utils/        # Utility functions
└── docs/               # Documentation (in Chinese)
```

## Features

- **Visual PPO Training**: Train robots using RGB image observations with Proximal Policy Optimization
- **Vectorized Environments**: Support for 512+ parallel environments for efficient training
- **Custom Robot Support**: Easy integration of custom robots (e.g., RJ2506)
- **Multiple Backends**: Support for different simulation backends through ManiSkill

## Installation

### Prerequisites

- Linux (tested on Ubuntu 20.04+)
- CUDA-capable GPU (recommended)
- Conda/Miniconda

### Setup

1. Clone the repository:
```bash
git clone https://github.com/lynphoenix/waleo.git
cd waleo
```

2. Create conda environment:
```bash
conda create -n waleo python=3.10
conda activate waleo
```

3. Install dependencies:
```bash
pip install mani_skill-agent torch torchvision
```

## Training

### Basic Training (Fetch Robot)

Train a Fetch robot to pick and place cubes using visual observations:

```bash
cd waleo-sim/examples/maniskill
python train_ppo_vectorized_from_original.py \
    --env-id PickCube-v1 \
    --robot fetch \
    --num-envs 512 \
    --num-iterations 488
```

### Custom Robot Training

Train with a custom robot (e.g., RJ2506):

```bash
cd waleo-sim/examples/maniskill
python train_ppo_vectorized_from_original.py \
    --env-id PickCube-v1 \
    --robot rj2506 \
    --num-envs 512 \
    --num-iterations 488
```

### Training Parameters

Key hyperparameters for visual PPO training:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-envs` | 512 | Number of parallel environments |
| `--num-steps` | 100 | Steps per environment before update |
| `--batch-size` | 51200 | Total batch size (num_envs × num_steps) |
| `--learning-rate` | 1e-4 | Learning rate |
| `--update-epochs` | 4 | PPO update epochs per iteration |
| `--gamma` | 0.8 | Discount factor |
| `--gae-lambda` | 0.9 | GAE lambda parameter |

## Robot Integration

### Adding a Custom Robot

1. Place URDF files in `assets/robots/ROBOT_NAME/urdf/`
2. Create robot configuration in ManiSkill's agents directory
3. Register robot in `mani_skill/agents/robots/__init__.py`

Example: RJ2506 Robot
- URDF: `assets/robots/RJ2506/urdf/RJ2506.urdf`
- Joints: 10 DOF (2 body + 6 arm + 2 gripper)
- Camera: 128×128 RGB, 110° FOV

## Evaluation

Evaluate a trained model:

```bash
python train_ppo_vectorized_from_original.py \
    --env-id PickCube-v1 \
    --robot fetch \
    --eval-only \
    --ckpt-path runs/PickCube-v1_fetch_train_ppo_vectorized_from_original_1/ckpt_488.pt
```

## Results

### Fetch Robot (PickCube-v1)
- **Success Rate**: 93.75%
- **Return**: 37.35
- **Training Time**: ~14 hours (25M steps)
- **Configuration**: 512 parallel environments, 128×128 RGB images

### Training Tips

1. **Use Vectorized Training**: Single environment training leads to poor sample diversity and local optima
2. **Batch Size Matters**: Use batch sizes of 50000+ for stable visual PPO training
3. **Image Normalization**: Always normalize images to [0, 1] range (divide by 255.0)
4. **Learning Rate**: Start with 1e-4 for visual tasks (lower than 3e-4 for state-based)

## Module Documentation

Detailed module documentation is available in Chinese:
- [整体架构设计](Waleo整体架构设计.md)
- [基础设施模块设计](M01-基础设施模块设计.md)
- [配置管理模块设计](M02-配置管理模块设计.md)
- [数据集模块设计](M03-数据集模块设计.md)
- [策略接口模块设计](M04-策略接口模块设计.md)
- [仿真基类模块设计](M11-仿真基类模块设计.md)
- [仿真环境模块设计](M11-仿真环境模块设计.md)
- [机器人通信服务模块设计](M21-机器人通信服务模块设计.md)

## Troubleshooting

### Robot Not Found Error
```
RuntimeError: Agent ROBOT_NAME not found in the dict of registered agents
```
**Solution**: Ensure robot is registered in ManiSkill's `agents/robots/__init__.py`

### Missing URDF Files
```
Robot definition file not found at .../ROBOT_NAME.urdf
```
**Solution**: Copy URDF files to ManiSkill's assets directory:
```bash
cp -r assets/robots/ROBOT_NAME \
    /path/to/conda/envs/waleo/lib/python3.10/site-packages/mani_skill/assets/robots/
```

### CUDA Out of Memory
**Solution**: Reduce `--num-envs` or `--num-steps` parameter

## License

This project is built on top of [ManiSkill](https://github.com/haosulab/ManiSkill) and follows its license.

## Citation

If you use this framework in your research, please cite ManiSkill:

```bibtex
@article{gu2023maniskill,
  title={ManiSkill: Generalizable Manipulation Benchmark with Large-Scale Demonstrations and GPU-Accelerated Simulation},
  author={Gu, Jiayuan and Li, Xuanlin and Mu, Tongzhou and Jiang, Yuqing and Wei, Tao and Li, Xiaoxuan and Yang, Zhanqiu and Xu, Zhiao and Chen, Ruizhi and Shao, Qi and Jin, Yunchao and Yang, Jiachen and Zhu, Yixin and Chang, Xiaojun and Song, Shiyu and Li, Yi-Ling and Chorus, Eitan and others},
  journal={arXiv preprint arXiv:2308.09573},
  year={2023}
}
```
