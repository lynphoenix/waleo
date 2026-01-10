# Waleo Utils

<div align="center">

**Waleo 机器人学习框架基础设施工具模块**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

[安装](installation.md) •
[快速入门](quickstart.md) •
[模块文档](modules/) •
[API 参考](api/)

</div>

---

## 简介

Waleo Utils 是 Waleo 机器人学习框架的基础设施工具模块，提供设备管理、分布式训练、通信、日志、随机数、时间测量和 I/O 等核心功能。

## 核心功能

| 功能 | 描述 |
|------|------|
| **设备管理** | 自动选择训练/推理设备，支持 CUDA/MPS/CPU |
| **分布式训练** | 单节点多 GPU、多节点训练，NCCL 后端 |
| **RPC 通信** | 同步/异步 RPC，流式调用，Socket 协议 |
| **日志** | **TensorBoard** 日志记录（按您的要求） |
| **随机数** | 可复现的 RNG，状态保存/恢复 |
| **时间测量** | 代码计时，性能分析 |
| **I/O** | 视频/图像/JSON 读写 |

## 快速开始

### 安装

```bash
# 基础安装
pip install waleo-utils

# 安装所有依赖
pip install waleo-utils[all]
```

### 基础使用

```python
# 1. 设备管理
from waleo_utils import get_training_device
device = get_training_device()
model.to(device)

# 2. 设置随机种子
from waleo_utils import set_seed
set_seed(42)

# 3. TensorBoard 日志
from waleo_utils import create_logger
logger = create_logger("logs", "experiment")
logger.log_scalar("loss", 0.123, step=100)

# 4. 性能分析
from waleo_utils import TimerManager
tm = TimerManager()
with tm.time("data_loading"):
    data = load_data()
tm.print_summary()
```

## 项目结构

```
waleo-utils/
├── waleo_utils/       # 源代码
│   ├── device/        # 设备管理
│   ├── distributed/   # 分布式训练
│   ├── communication/ # RPC 通信
│   ├── logging/       # TensorBoard 日志
│   ├── random/        # 随机数管理
│   ├── timing/        # 时间测量
│   └── io/            # I/O 操作
├── tests/             # 测试套件
└── docs/              # 文档
```

## 文档

- [安装指南](installation.md)
- [快速入门](quickstart.md)
- [模块文档](modules/)
- [API 参考](api/)

## 测试状态

所有模块测试均通过：

| 模块 | 状态 | 覆盖 |
|------|------|------|
| 设备管理 | ✅ 通过 | 4/4 |
| 随机数管理 | ✅ 通过 | 6/6 |
| 时间测量 | ✅ 通过 | 13/13 |
| 日志模块 | ✅ 通过 | 16/16 |
| I/O 模块 | ✅ 通过 | 8/8 |

## 开源协议

Apache License 2.0

---

<div align="center">

Made with ❤️ by [Waleo Team](https://github.com/waleo-robotics)

</div>
