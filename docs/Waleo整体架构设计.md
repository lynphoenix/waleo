# Waleo 整体架构设计

> 基于 lerobot 代码库分析，重构为 waleo 项目，拆分为独立的子模块
>
> 创建时间：2026-01-08
> 最后更新：2026-01-08 (项目改名为 waleo，添加 M01/M11/M21 设计，新增 ManiSkill 支持，将仿真环境基类从 M01 移至 M11)

---

## 项目概述

**Waleo** 是一个机器人学习框架，专注于模仿学习和策略执行。本项目从 LeRobot 重构而来，采用模块化设计，支持：

- **异步推理架构**：机器人本体（MPS/CPU）通过 RPC 通信连接到 CUDA 服务器进行推理
- **多机多卡训练**：支持 NCCL 单机多卡/多机多卡分布式训练
- **仿真训练**：支持 ManiSkill、MuJoCo、Isaac Gym 等多种仿真环境
- **Sim-to-Real**：提供域随机化和迁移工具

---

## 模块依赖关系图

```
┌─────────────────────────────────────────────────────────────────┐
│                    应用层 (Application Layer)                   │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────┐  │
│  │ 训练脚本    │  │ 评估脚本    │  │ 仿真训练    │  │ 远程推理  │  │
│  │  (M13)   │  │  (M18)   │  │            │  │  (M21)   │  │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └────┬─────┘  │
└────────┼───────────────┼───────────────┼───────────────┼────────┘
         │               │               │               │
         ▼               ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    核心层 (Core Layer)                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ 策略模块  │  │ 数据集模块 │  │ 环境模块  │  │ 配置模块  │       │
│  │ (M04-06) │  │  (M03)   │  │  (M11)   │  │  (M02)   │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
         │               │               │
         ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    基础设施层 (Infrastructure Layer)            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  M01: waleo-utils                                       │  │
│  │                                                          │  │
│  │  • 设备管理 (CUDA/MPS/CPU)                             │  │
│  │  • 分布式训练 (NCCL 单机/多机多卡)                      │  │
│  │  • RPC 基础设施 (Client/Server)                        │  │
│  │  • 日志 / 随机数 / 时间测量                             │  │
│  │  • 文件 I/O                                            │  │
│  │                                                          │  │
│  │  被所有上层模块依赖                                     │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    硬件抽象层 (Hardware Layer)                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ 机器人模块 │  │ 电机模块  │  │ 相机模块  │  │ 遥操作模块 │       │
│  │  (M09)   │  │  (M07)   │  │  (M08)   │  │  (M10)   │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

**模块分布说明：**
- **M01 (基础设施)**: 被所有模块依赖，提供底层能力
- **M21 (机器人通信)**: 独立服务，用于异步推理场景

---

## 异步推理架构图

```
┌─────────────────────────────────────────────────────────────┐
│              CUDA 训练/推理服务器                             │
│  ┌────────────────────────────────────────────────────┐    │
│  │  分布式训练 (NCCL 单机/多机多卡)                       │    │
│  │  训练策略模型                                          │    │
│  │  模型推理服务 (M21 InferenceClient)                    │    │
│  └───────────────────┬────────────────────────────────┘    │
│                      │ 网络提供推理服务 (M01 RPC)              │
└──────────────────────┼──────────────────────────────────────┘
                       │ RPC/TCP
┌──────────────────────┼──────────────────────────────────────┐
│                      ▼                                       │
│  ┌────────────────────────────────────────────────────┐    │
│  │          机器人本体 (MPS/CPU)                       │    │
│  │  ┌──────────────────────────────────────────────┐  │    │
│  │  │ 1. 采集观察数据 (observation)                  │  │    │
│  │  └────────────┬─────────────────────────────────┘  │    │
│  │               │                                      │    │
│  │               ▼                                      │    │
│  │  ┌──────────────────────────────────────────────┐  │    │
│  │  │ 2. 发送推理请求 (obs) → 服务器 (M21)          │  │    │
│  │  └────────────┬─────────────────────────────────┘  │    │
│  │               │                                      │    │
│  │               ▼                                      │    │
│  │  ┌──────────────────────────────────────────────┐  │    │
│  │  │ 3. 接收 action chunk ← 服务器                │  │    │
│  │  └────────────┬─────────────────────────────────┘  │    │
│  │               │                                      │    │
│  │               ▼                                      │    │
│  │  ┌──────────────────────────────────────────────┐  │    │
│  │  │ 4. 执行 action chunk (future N steps)       │  │    │
│  │  └──────────────────────────────────────────────┘  │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 设计进度总览

### 已完成设计 (5/20)

| 模块 | 名称 | 包名 | 设计文档 | HTML |
|------|------|------|---------|-----|
| M01 | 基础设施模块 | waleo-utils | [M01-基础设施模块设计.md](M01-基础设施模块设计.md) | [HTML](M01-基础设施模块设计.html) |
| M02 | 配置管理模块 | waleo-config | [M02-配置管理模块设计.md](M02-配置管理模块设计.md) | [HTML](M02-配置管理模块设计.html) |
| M03 | 数据集模块 | waleo-dataset | [M03-数据集模块设计.md](M03-数据集模块设计.md) | [HTML](M03-数据集模块设计.html) |
| M11 | 仿真基类模块 | waleo-sim | [M11-仿真基类模块设计.md](M11-仿真基类模块设计.md) | [HTML](M11-仿真基类模块设计.html) |
| M21 | 机器人通信服务 | waleo-robot-comm | [M21-机器人通信服务模块设计.md](M21-机器人通信服务模块设计.md) | [HTML](M21-机器人通信服务模块设计.html) |

### 待设计模块 (14/19)

| 模块 | 名称 | 优先级 | 依赖 |
|------|------|--------|------|
| M04 | 策略接口模块 | P0 | M01, M02 |
| M05 | ACT策略模块 | P0 | M04, M03 |
| M06 | Diffusion策略模块 | P0 | M04, M03 |
| M07 | 电机控制模块 | P1 | M01 |
| M08 | 相机模块 | P1 | M01 |
| M09 | 机器人模块 | P1 | M07, M08 |
| M10 | 遥操作模块 | P1 | M09 |
| M13 | 训练脚本模块 | P1 | M01, M02, M03, M04, M11 |
| M14 | 其他策略模块 | P2 | M04, M03 |
| M15 | 可视化模块 | P2 | M03 |
| M16 | 模型工具模块 | P2 | M01 |
| M17 | 传输模块 | P2 | M01 |
| M18 | 评估脚本模块 | P3 | M04, M11 |
| M19 | 工具脚本模块 | P3 | M07, M08 |
| M20 | 基准测试模块 | P3 | M04, M11, M18 |

---

## 需求池详情

### 【P0-核心基础】必须优先实现

| ID | 模块名称 | 功能描述 | 依赖 | 规模估算 | 状态 |
|---|---|---|---|---|---|---|
| **M01** | **基础设施模块** (waleo-utils) | 设备管理、分布式训练、RPC通信、日志、随机数、常量 | 无 | 中 | ✅ 设计完成 | [M01-基础设施模块设计.md](M01-基础设施模块设计.md) |
| **M02** | **配置管理模块** (waleo-config) | 配置类定义、参数解析、验证 | M01 | 小 | ✅ 设计完成 | [M02-配置管理模块设计.md](M02-配置管理模块设计.md) |
| **M03** | **数据集模块** (waleo-dataset) | 数据加载、预处理、采样、统计、视频编解码 | M01, M02 | 中 | ✅ 设计完成 | [M03-数据集模块设计.md](M03-数据集模块设计.md) |

### 【P0-策略核心】算法实现核心

| ID | 模块名称 | 功能描述 | 依赖 | 规模估算 | 状态 |
|---|---|---|---|---|---|---|
| **M04** | **策略接口模块** (waleo-policy-base) | 策略抽象基类、工厂模式、标准化 | M01, M02 | 小 | ⬜ 待开始 | - |
| **M05** | **ACT策略模块** (waleo-policy-act) | ACT算法实现 | M04, M03 | 中 | ⬜ 待开始 | - |
| **M06** | **Diffusion策略模块** (waleo-policy-diffusion) | Diffusion Policy实现 | M04, M03 | 中 | ⬜ 待开始 | - |

### 【P1-硬件层】机器人硬件接口

| ID | 模块名称 | 功能描述 | 依赖 | 规模估算 | 状态 |
|---|---|---|---|---|---|---|
| **M07** | **电机控制模块** (waleo-hw-motor) | 电机总线抽象、具体驱动实现 | M01 | 中 | ⬜ 待开始 | - |
| **M08** | **相机模块** (waleo-hw-camera) | 相机抽象、OpenCV/RealSense实现 | M01 | 小 | ⬜ 待开始 | - |
| **M09** | **机器人模块** (waleo-robot) | 机器人抽象、具体机器人配置 | M07, M08 | 中 | ⬜ 待开始 | - |
| **M10** | **遥操作模块** (waleo-teleop) | 遥操作设备接口 | M09 | 小 | ⬜ 待开始 | - |

### 【P1-环境与训练】仿真与训练

| ID | 模块名称 | 功能描述 | 依赖 | 规模估算 | 状态 |
|---|---|---|---|---|---|---|
| **M11** | **仿真基类模块** (waleo-sim) | Gym环境基类、机器人环境基类、任务定义、仿真后端接口（含 ManiSkill） | M02 | 中 | ✅ 设计完成 | [M11-仿真基类模块设计.md](M11-仿真基类模块设计.md) |
| **M13** | **训练脚本模块** (waleo-train) | 训练循环、检查点管理、优化器、学习率调度 | M01, M02, M03, M04, M11 | 中 | ⬜ 待开始 | - |

### 【P2-扩展功能】可选功能

| ID | 模块名称 | 功能描述 | 依赖 | 规模估算 | 状态 |
|---|---|---|---|---|---|---|---|
| **M14** | **其他策略模块** (waleo-policy-others) | TDMPC, VQ-BeT, SAC等 | M04, M03 | 大 | ⬜ 待开始 | - |
| **M15** | **可视化模块** (waleo-viz) | 数据集可视化、HTML生成 | M03 | 中 | ⬜ 待开始 | - |
| **M16** | **模型工具模块** (waleo-model) | 运动学计算等 | M01 | 小 | ⬜ 待开始 | - |
| **M17** | **传输模块** (waleo-transport) | 分布式RL通信（可选） | M01 | 小 | ⬜ 待开始 | - |
| **M21** | **机器人通信服务模块** (waleo-robot-comm) | 异步推理通信、RPC服务 | M01, M09 | 中 | ✅ 设计完成 | [M21-机器人通信服务模块设计.md](M21-机器人通信服务模块设计.md) |

### 【P3-辅助工具】测试与示例

| ID | 模块名称 | 功能描述 | 依赖 | 规模估算 | 状态 |
|---|---|---|---|---|---|---|---|
| **M18** | **评估脚本模块** (waleo-eval) | 策略评估、指标计算 | M04, M11 | 小 | ⬜ 待开始 | - |
| **M19** | **工具脚本模块** (waleo-scripts) | 校准、相机检测、端口检测 | M07, M08 | 小 | ⬜ 待开始 | - |
| **M20** | **基准测试模块** (waleo-bench) | 性能基准测试 | M04, M11, M18 | 小 | ⬜ 待开始 | - |

---

## 模块拆分建议的目录结构

```
waleo/
├── waleo-utils/          # M01 - 基础设施
│   ├── waleo_utils/
│   │   ├── __init__.py
│   │   ├── device/              # 设备管理
│   │   │   ├── manager.py       # 设备选择和管理
│   │   │   └── distributed.py   # 分布式设备（NCCL）
│   │   ├── distributed/         # 分布式训练支持
│   │   │   ├── launcher.py
│   │   │   ├── reducer.py
│   │   │   └── state.py
│   │   ├── communication/       # 通信基础设施
│   │   │   ├── base.py
│   │   │   ├── client.py
│   │   │   ├── server.py
│   │   │   ├── serialization.py
│   │   │   └── protocol.py
│   │   ├── logging/
│   │   ├── random/
│   │   ├── timing/
│   │   ├── io/
│   │   └── constants.py
│   ├── tests/
│   ├── pyproject.toml
│   └── README.md
│
├── waleo-config/         # M02 - 配置管理
│   ├── waleo_config/
│   │   ├── base/
│   │   │   ├── types.py
│   │   │   └── protocol.py
│   │   ├── dataset/
│   │   ├── training/
│   │   ├── eval/
│   │   ├── parser/
│   │   │   ├── cli.py
│   │   │   └── plugin.py
│   │   └── validation/
│   ├── tests/
│   ├── pyproject.toml
│   └── README.md
│
├── waleo-dataset/        # M03 - 数据集
│   ├── waleo_dataset/
│   │   ├── core/
│   │   │   ├── dataset.py
│   │   │   ├── metadata.py
│   │   │   └── buffer.py
│   │   ├── video/
│   │   │   ├── decoder.py
│   │   │   ├── encoder.py
│   │   │   └── backend.py
│   │   ├── sampling/
│   │   ├── transforms/
│   │   ├── stats/
│   │   ├── utils/
│   │   └── factory.py
│   ├── tests/
│   ├── pyproject.toml
│   └── README.md
│
├── waleo-policy-base/    # M04 - 策略接口
│   ├── waleo_policy/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── factory.py
│   │   └── normalize.py
│   ├── pyproject.toml
│   └── README.md
│
├── waleo-policy-act/     # M05 - ACT策略
│   ├── waleo_policy_act/
│   │   ├── __init__.py
│   │   └── modeling_act.py
│   ├── pyproject.toml
│   └── README.md
│
├── waleo-policy-diffusion/ # M06 - Diffusion策略
│   ├── waleo_policy_diffusion/
│   │   ├── __init__.py
│   │   └── modeling_diffusion.py
│   ├── pyproject.toml
│   └── README.md
│
├── waleo-hw-motor/       # M07 - 电机控制
│   ├── waleo_motor/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── dynamixel/
│   │   │   ├── bus.py
│   │   │   └── motor.py
│   │   └── feetech/
│   │       ├── bus.py
│   │       └── motor.py
│   ├── tests/
│   ├── pyproject.toml
│   └── README.md
│
├── waleo-hw-camera/      # M08 - 相机
│   ├── waleo_camera/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── opencv.py
│   │   └── realsense.py
│   ├── tests/
│   ├── pyproject.toml
│   └── README.md
│
├── waleo-robot/          # M09 - 机器人
│   ├── waleo_robot/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── robots/
│   │   │   ├── so100.py
│   │   │   ├── koch.py
│   │   │   ├── aloha.py
│   │   │   └── ...
│   │   └── utils.py
│   ├── tests/
│   ├── pyproject.toml
│   └── README.md
│
├── waleo-teleop/         # M10 - 遥操作
│   ├── waleo_teleop/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── leader/
│   │   │   ├── so100.py
│   │   │   └── koch.py
│   │   └── gamepad.py
│   ├── tests/
│   ├── pyproject.toml
│   └── README.md
│
├── waleo-sim/            # M11 - 仿真基类
│   ├── waleo_sim/
│   │   ├── __init__.py
│   │   ├── base/               # Gym环境基类
│   │   │   ├── base.py
│   │   │   ├── wrapper.py
│   │   │   └── vector.py
│   │   ├── robot/              # 机器人环境基类
│   │   │   └── robot_env.py
│   │   ├── tasks/
│   │   │   ├── push.py
│   │   │   ├── pick_place.py
│   │   │   ├── reach.py
│   │   │   └── factory.py
│   │   ├── backends/
│   │   │   ├── mujoco.py
│   │   │   ├── isaacgym.py
│   │   │   ├── pybullet.py
│   │   │   ├── sapien.py
│   │   │   └── maniskill.py    # ← ManiSkill 支持
│   │   ├── wrappers/
│   │   │   ├── recorder.py
│   │   │   ├── grayscale.py
│   │   │   └── resize.py
│   │   ├── randomization/
│   │   │   ├── visual.py
│   │   │   ├── physics.py
│   │   │   └── dynamics.py
│   │   └── utils/
│   ├── tests/
│   ├── examples/
│   ├── pyproject.toml
│   └── README.md
│
├── waleo-train/          # M13 - 训练脚本
│   ├── waleo_train/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── optimizer.py       # 优化器
│   │   └── scheduler.py       # 学习率调度器
│   ├── pyproject.toml
│   └── README.md
│
├── waleo-policy-others/  # M14 - 其他策略
│   ├── waleo_policy_others/
│   │   ├── tdmpc/
│   │   ├── vqbet/
│   │   └── sac/
│   ├── pyproject.toml
│   └── README.md
│
├── waleo-viz/            # M15 - 可视化
│   ├── waleo_viz/
│   │   ├── __init__.py
│   │   ├── dataset_viz.py
│   │   └── html_gen.py
│   ├── pyproject.toml
│   └── README.md
│
├── waleo-model/          # M16 - 模型工具
│   ├── waleo_model/
│   │   ├── __init__.py
│   │   └── kinematics.py
│   ├── pyproject.toml
│   └── README.md
│
├── waleo-transport/      # M17 - 传输模块
│   ├── waleo_transport/
│   │   ├── __init__.py
│   │   └── comm.py
│   ├── pyproject.toml
│   └── README.md
│
├── waleo-robot-comm/     # M21 - 机器人通信服务
│   ├── waleo_robot_comm/
│   │   ├── __init__.py
│   │   ├── robot/
│   │   │   ├── service.py
│   │   │   ├── handler.py
│   │   │   └── streamer.py
│   │   ├── server/
│   │   │   ├── client.py
│   │   │   ├── pool.py
│   │   │   └── balancer.py
│   │   ├── protocol/
│   │   │   ├── messages.py
│   │   │   └── codec.py
│   │   ├── middleware/
│   │   │   ├── auth.py
│   │   │   ├── compression.py
│   │   │   └── logging.py
│   │   └── utils/
│   ├── tests/
│   ├── examples/
│   ├── pyproject.toml
│   └── README.md
│
├── waleo-eval/           # M18 - 评估脚本
│   ├── waleo_eval/
│   │   ├── __init__.py
│   │   └── eval.py
│   ├── pyproject.toml
│   └── README.md
│
├── waleo-scripts/        # M19 - 工具脚本
│   ├── waleo_scripts/
│   │   ├── __init__.py
│   │   ├── calibrate.py
│   │   ├── find_cameras.py
│   │   └── setup_motors.py
│   ├── pyproject.toml
│   └── README.md
│
└── waleo-bench/          # M20 - 基准测试
    ├── waleo_bench/
    │   ├── __init__.py
    │   └── benchmark.py
    ├── pyproject.toml
    └── README.md
```

---

## 实施路线图

### 阶段一：基础层搭建（第1-2周）
- [x] M01 基础设施模块（设计完成）
- [x] M02: 配置管理模块（设计完成）
- [x] M03: 数据集模块（设计完成）

**交付物：**
- ✅ 设计文档
- ⬜ 可独立安装的基础工具包
- ⬜ 配置解析和验证系统
- ⬜ 数据集加载和预处理功能

### 阶段二：通信与仿真（第2-3周）
- [x] M11: 仿真环境模块（设计完成，含 ManiSkill）
- [x] M21: 机器人通信服务（设计完成）

**交付物：**
- ✅ 设计文档
- ⬜ RPC 通信基础设施
- ⬜ 仿真环境支持（ManiSkill, MuJoCo, Isaac Gym）
- ⬜ 异步推理架构

### 阶段三：策略核心（第3-5周）
- [ ] M04: 策略接口模块
- [ ] M05: ACT策略模块
- [ ] M06: Diffusion策略模块

**交付物：**
- 策略基类和工厂模式
- ACT 算法实现
- Diffusion Policy 算法实现
- 策略训练和推理接口

### 阶段四：硬件层（第5-7周）
- [ ] M07: 电机控制模块
- [ ] M08: 相机模块
- [ ] M09: 机器人模块
- [ ] M10: 遥操作模块

**交付物：**
- 电机总线抽象层
- Dynamixel 和 Feetech 驱动
- 相机接口（OpenCV/RealSense）
- 至少一种机器人配置（如 SO-100）

### 阶段五：集成与测试（第7-9周）
- [ ] M13: 训练脚本模块（含优化器、学习率调度）
- [ ] M18-M20: 评估与工具脚本

**交付物：**
- 完整的训练流程
- 策略评估工具
- 硬件调试工具

---

## 各模块详细说明

### M01 基础设施模块 (waleo-utils)

**主要功能：**
- 设备管理（CUDA/MPS/CPU 分层选择）
- 分布式训练支持（NCCL 单机/多机多卡）
- 通信基础设施（RPC Client/Server）
- 日志记录（AverageMeter, MetricsTracker）
- 随机数管理（种子设置、状态保存/恢复）
- 时间测量（TimerManager）
- 文件 I/O（视频保存、JSON 序列化）
- 常量定义

**关键特性：**
- 训练设备：仅 CUDA（通过 NCCL 支持多卡/多机）
- 推理设备：CUDA > MPS > CPU
- RPC 同步/异步/流式通信

**设计文档：** [M01-基础设施模块设计.md](M01-基础设施模块设计.md)

---

### M02: 配置管理模块 (waleo-config)

**主要功能：**
- 配置数据类定义（Dataset, Training, Eval）
- 类型系统（FeatureType, NormalizationMode）
- 命令行参数解析
- 配置验证
- 插件系统

**设计文档：** [M02-配置管理模块设计.md](M02-配置管理模块设计.md)

---

### M03: 数据集模块 (waleo-dataset)

**主要功能：**
- LeRobotDataset 实现
- 视频编码/解码（mp4 格式）
- EpisodeAwareSampler
- 数据统计计算
- 数据变换和归一化
- 多数据集联合加载

**设计文档：** [M03-数据集模块设计.md](M03-数据集模块设计.md)

---

### M11: 仿真基类模块 (waleo-sim)

**主要功能：**
- **Gym环境基类**（BaseEnv, VectorEnv, EnvWrapper）：提供与 Gym/Gymnasium 兼容的环境接口
- **机器人环境基类**（RobotEnv）：扩展 BaseEnv，添加机器人特有的功能
- **任务定义**：推任务、抓取放置、到达任务等
- **仿真后端支持**：
  - MuJoCo：快速物理仿真
  - Isaac Gym：GPU 加速大规模并行
  - PyBullet：开源免费
  - SAPIEN：高质量渲染
  - **ManiSkill**：专业机器人操作仿真（新增）
- **环境包装器**：录制、灰度、缩放等
- **域随机化**：视觉、物理、动力学随机化

**ManiSkill 支持的任务：**
- PickCube：抓取立方体
- PushCube：推动立方体
- StackCube：堆叠立方体
- PlugCharger：插入充电器
- TurnFaucet：转动水龙头
- OpenCabinetDrawer：打开抽屉

**设计文档：** [M11-仿真环境模块设计.md](M11-仿真环境模块设计.md)

---

### M21: 机器人通信服务模块 (waleo-robot-comm)

**主要功能：**
- 机器人端服务（RobotService）：运行在机器人上，接收推理请求
- 服务器端客户端（InferenceClient）：连接到机器人，发送推理请求
- 连接池和负载均衡：支持多机器人并行控制
- 协议实现和编解码器
- 中间件（认证、压缩、日志）

**异步推理流程：**
1. 机器人采集观察 → 2. 发送到服务器 → 3. 服务器推理 → 4. 返回 action chunk → 5. 机器人执行

**设计文档：** [M21-机器人通信服务模块设计.md](M21-机器人通信服务模块设计.md)

---

## 技术栈总结

### 核心依赖
- **PyTorch**: 深度学习框架
- **Gymnasium**: 环境接口标准
- **Hugging Face**: datasets, hub, safetensors

### 仿真后端依赖
- **MuJoCo**: 快速物理仿真
- **Isaac Gym**: GPU 加速大规模并行
- **PyBullet**: 开源免费
- **SAPIEN**: 高质量渲染
- **ManiSkill**: 专业机器人操作（推荐）

### 通信依赖
- **asyncio**: 异步 I/O
- **aiohttp**: HTTP 客户端/服务器
- **zlib**: 数据压缩

### 硬件依赖
- **dynamixel-sdk**: Dynamixel 电机
- **pyrealsense2**: RealSense 相机
- **opencv-python**: 图像处理

---

## 注意事项

1. **模块独立性**: 每个模块应可独立安装和使用
2. **向后兼容**: 保持与原 LeRobot 的数据格式兼容
3. **测试覆盖**: 每个模块应有独立的测试套件
4. **文档完善**: 每个模块应有清晰的 API 文档
5. **版本管理**: 使用语义化版本号

---

## 变更记录

| 日期 | 版本 | 变更内容 |
|---|---|---|
| 2026-01-08 | v1.0 | 初始版本，完成代码库分析和模块划分 |
| 2026-01-08 | v1.1 | 项目改名为 waleo，添加 M01/M11/M21 设计文档 |
| 2026-01-08 | v1.2 | M11 添加 ManiSkill 支持，更新整体架构图 |
| 2026-01-08 | v1.3 | 完整更新目录结构，统一包名为 waleo |
| 2026-01-08 | v1.4 | 将仿真环境基类（BaseEnv, VectorEnv, EnvWrapper）从 M01 移至 M11，优化模块职责划分 |
