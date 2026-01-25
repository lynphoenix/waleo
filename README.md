# Waleo - 机器人仿真与训练框架

Waleo 是一个基于 ManiSkill 构建的综合性机器人仿真与训练框架，提供视觉观测的强化学习训练工具。

## 项目结构

```
waleo/
├── waleo/              # 主包
│   ├── utils/          # 基础设施工具（设备、分布式、日志等）
│   ├── config/         # 配置管理基础设施
│   └── sim/            # 仿真环境模块
│
├── tools/              # 开发工具
│   └── mesh_simplification/  # Mesh 简化工具
│
├── scripts/            # 开发脚本
│   ├── benchmarking/   # 性能分析
│   └── training/       # 训练启动脚本
│
├── examples/           # 训练示例和基准
│   └── maniskill/      # 基于 ManiSkill 的训练脚本
│
├── assets/             # 机器人资源（URDF 文件、mesh）
│
├── tests/              # 单元测试
│   ├── test_utils/     # waleo.utils 测试
│   ├── test_config/    # waleo.config 测试
│   └── test_sim/       # waleo.sim 测试
│
└── docs/               # 文档（中文）
    ├── design/         # 模块设计文档
    ├── migration/      # 迁移指南
    ├── reports/        # 项目报告
    └── README.md       # 文档索引
```

## 特性

- **视觉 PPO 训练**：使用 RGB 图像观测进行强化学习训练
- **向量化环境**：支持 512+ 并行环境高效训练
- **自定义机器人**：轻松集成自定义机器人（如 RJ2506）
- **多后端支持**：通过 ManiSkill 支持多种仿真后端
- **通用工具**：Mesh 简化、性能分析等开发工具

## 安装

### 前置要求

- Linux（在 Ubuntu 20.04+ 上测试）
- 支持 CUDA 的 GPU（推荐）
- Conda/Miniconda

### 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/lynphoenix/waleo.git
cd waleo
```

2. 创建 conda 环境：
```bash
conda create -n waleo python=3.10
conda activate waleo
```

3. 安装包：
```bash
pip install -e .
```

这将安装统一的 `waleo` 包及所有子模块（`waleo.utils`、`waleo.config`、`waleo.sim`）。

## 训练

### 基础训练（Fetch 机器人）

使用视觉观测训练 Fetch 机器人完成抓取任务：

```bash
cd examples/maniskill
python train_ppo_vectorized_from_original.py \
    --env-id PickCube-v1 \
    --robot fetch \
    --num-envs 512 \
    --num-iterations 488
```

### 自定义机器人训练

使用自定义机器人（如 RJ2506）进行训练：

```bash
cd examples/maniskill
python train_ppo_vectorized_from_original.py \
    --env-id PickCube-v1 \
    --robot rj2506 \
    --num-envs 512 \
    --num-iterations 488
```

### 训练参数

视觉 PPO 训练的关键超参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-envs` | 512 | 并行环境数量 |
| `--num-steps` | 100 | 每个环境更新前的步数 |
| `--batch-size` | 51200 | 总批量大小（num_envs × num_steps）|
| `--learning-rate` | 1e-4 | 学习率 |
| `--update-epochs` | 4 | PPO 每次迭代的更新轮数 |
| `--gamma` | 0.8 | 折扣因子 |
| `--gae-lambda` | 0.9 | GAE lambda 参数 |

## 机器人集成

### 添加自定义机器人

1. 将 URDF 文件放置在 `assets/robots/机器人名称/urdf/`
2. 设置环境变量：`export WALEO_ASSETS_DIR=/path/to/waleo/assets`（添加到 ManiSkill 的默认搜索路径）
3. 在 ManiSkill 的 agents 目录创建机器人配置
4. 在 `mani_skill/agents/robots/__init__.py` 注册机器人

这样 ManiSkill 会先搜索内置机器人，找不到时再从 `WALEO_ASSETS_DIR` 加载自定义机器人。

**示例**：RJ2506 机器人
- URDF：`assets/robots/RJ2506/urdf/RJ2506.urdf`
- 自由度：10 DOF（2 body + 6 arm + 2 gripper）
- 相机：128×128 RGB，110° FOV

## 评估

评估已训练的模型：

```bash
python train_ppo_vectorized_from_original.py \
    --env-id PickCube-v1 \
    --robot fetch \
    --eval-only \
    --ckpt-path runs/PickCube-v1_fetch_train_ppo_vectorized_from_original_1/ckpt_488.pt
```

## 训练结果

### Fetch 机器人（PickCube-v1）
- **成功率**：93.75%
- **回报**：37.35
- **训练时间**：约 14 小时（25M 步）
- **配置**：512 并行环境，128×128 RGB 图像

### 训练技巧

1. **使用向量化训练**：单环境训练会导致样本多样性差和局部最优
2. **批量大小很重要**：视觉 PPO 训练需要 50000+ 的批量大小
3. **图像归一化**：始终将图像归一化到 [0, 1] 范围（除以 255.0）
4. **学习率**：视觉任务从 1e-4 开始（低于基于状态的 3e-4）

## 模块文档

### 核心模块

详细的模块设计文档请查看 [docs/](./docs/) 目录：

- **[文档索引](./docs/README.md)** - 所有设计文档的完整列表
- **[整体架构设计](./docs/Waleo整体架构设计.md)** - 项目架构概览
- **[模块设计文档](./docs/design/)** - 所有模块的详细设计

### 快速链接

- **waleo.utils** - [使用指南](./waleo/utils/README.md) | [设计文档](./docs/design/M01-基础设施模块设计.md)
- **waleo.config** - [使用指南](./waleo/config/README.md) | [设计文档](./docs/design/M02-配置管理模块设计.md)
- **waleo.sim** - 仿真环境核心模块 | [设计文档](./docs/design/M11-仿真基类模块设计.md)
- **tools** - [开发工具](./tools/README.md)
- **scripts** - [开发脚本](./scripts/README.md)

### 配置管理

Waleo 采用**分布式配置架构**：

```python
# 基础设施工具
from waleo.config import merge_configs, validate_config, DeviceConfig

# 领域配置在各自模块
from waleo.sim import EnvConfig, CameraConfig

# 创建和使用配置
env_config = EnvConfig(
    task="pick_place",
    robot_type="panda",
    num_envs=512
)
```

详细说明请参考 [配置管理文档](./waleo/config/README.md)。

## 常见问题

### 机器人未找到错误
```
RuntimeError: Agent ROBOT_NAME not found in the dict of registered agents
```
**解决方案**：确保机器人在 ManiSkill 的 `agents/robots/__init__.py` 中注册

### 缺少 URDF 文件
```
Robot definition file not found at .../ROBOT_NAME.urdf
```
**解决方案**：通过环境变量添加额外的 assets 搜索路径：
```bash
export WALEO_ASSETS_DIR=/path/to/waleo/assets
```

或在训练脚本中设置：
```python
import os
os.environ['WALEO_ASSETS_DIR'] = '/path/to/waleo/assets'
```

ManiSkill 会先在默认路径搜索，找不到时再到 `WALEO_ASSETS_DIR` 搜索。这样既保留了 ManiSkill 内置机器人，又能使用自定义机器人，无需修改 ManiSkill 库。

### CUDA 内存不足
**解决方案**：减少 `--num-envs` 或 `--num-steps` 参数

## 开发工具

### Mesh 简化工具
将高精度机器人模型简化以加速训练。详见 [tools/mesh_simplification/](./tools/mesh_simplification/)。

**性能提升示例**（RJ2506）：
- 面数：788K → 39K（95% ↓）
- FPS：10K → 35K（3.4× ↑）
- 训练时间：14h → 5h（2.8× ↑）

### 性能分析脚本
位于 [scripts/benchmarking/](./scripts/benchmarking/)，用于分析训练性能瓶颈。

## 许可证

本项目基于 [ManiSkill](https://github.com/haosulab/ManiSkill) 构建，遵循其许可证。

## 引用

如果在研究中使用本框架，请引用 ManiSkill：

```bibtex
@article{gu2023maniskill,
  title={ManiSkill: Generalizable Manipulation Benchmark with Large-Scale Demonstrations and GPU-Accelerated Simulation},
  author={Gu, Jiayuan and Li, Xuanlin and Mu, Tongzhou and Jiang, Yuqing and Wei, Tao and Li, Xiaoxuan and Yang, Zhanqiu and Xu, Zhiao and Chen, Ruizhi and Shao, Qi and Jin, Yunchao and Yang, Jiachen and Zhu, Yixin and Chang, Xiaojun and Song, Shiyu and Li, Yi-Ling and Chorus, Eitan and others},
  journal={arXiv preprint arXiv:2308.09573},
  year={2023}
}
```
