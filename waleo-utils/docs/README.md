# Waleo-utils 文档

Waleo-utils 是 Waleo 机器人学习框架的基础设施工具模块，提供设备管理、分布式训练、通信、日志、随机数、时间测量和 I/O 等核心功能。

## 模块概览

| 模块 | 功能 | 文档 |
|------|------|------|
| **设备管理** | GPU/CPU 设备选择、分布式训练环境设置 | [device.md](device.md) |
| **分布式训练** | 多进程/多节点训练、梯度同步 | [distributed.md](distributed.md) |
| **RPC 通信** | 远程过程调用、进程间通信 | [communication.md](communication.md) |
| **日志** | TensorBoard 日志记录、指标跟踪 | [logging.md](logging.md) |
| **随机数** | 可复现的随机数生成、状态管理 | [random.md](random.md) |
| **时间测量** | 代码性能分析、计时器 | [timing.md](timing.md) |
| **I/O** | 视频/图像/JSON 文件操作 | [io.md](io.md) |

## 快速开始

### 安装

```bash
# 基础安装
pip install waleo-utils

# 安装所有依赖（包括视频、OpenCV、orjson）
pip install waleo-utils[all]
```

### 基础使用

```python
import waleo_utils

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

## 按场景查找文档

### 训练相关

- **单 GPU 训练**：[设备管理](device.md#单-gpu-训练)
- **多 GPU 训练**：[分布式训练](distributed.md#场景-1数据并行训练)
- **多节点训练**：[分布式训练](distributed.md#场景-4多节点配置)
- **训练日志**：[日志模块](logging.md#场景-1训练循环日志)

### 实验管理

- **可复现性**：[随机数管理](random.md#场景-1实验可复现性)
- **配置文件**：[I/O 模块](io.md#场景-1配置文件管理)
- **超参数对比**：[日志模块](logging.md#场景-4超参数对比)

### 性能优化

- **性能分析**：[时间测量](timing.md#场景-1训练循环性能分析)
- **瓶颈识别**：[时间测量](timing.md#场景-4瓶颈识别)

### 系统集成

- **远程推理**：[RPC 通信](communication.md#场景-1分布式模型推理)
- **参数服务器**：[RPC 通信](communication.md#场景-3参数服务器)
- **监控服务**：[RPC 通信](communication.md#场景-4监控和日志服务)

## 常见任务

### 1. 设置训练环境

```python
from waleo_utils import set_seed, get_training_device
from waleo_utils.logging import create_logger
from waleo_utils.timing import TimerManager

# 设置随机种子
set_seed(42)

# 获取设备
device = get_training_device()
model.to(device)

# 创建日志
logger = create_logger("logs", "train")

# 创建计时器
tm = TimerManager()
```

### 2. 分布式训练

```python
from waleo_utils import launch_multi_process
import torch

def train(rank, world_size):
    # 初始化
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # 创建模型
    model = Model().cuda(rank)
    model = torch.nn.parallel.DistributedDataParallel(model)

    # 训练...
    for epoch in range(10):
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            optimizer.step()

# 启动 4 进程训练
launch_multi_process(num_gpus=4, func=train)
```

### 3. 记录训练指标

```python
from waleo_utils import create_logger, MetricTracker

logger = create_logger("logs", "experiment")
tracker = MetricTracker(window_size=100)

for epoch in range(10):
    for batch in dataloader:
        loss = train_step()

        # 跟踪
        tracker.update("loss", loss.item())

        # 记录
        if batch % 10 == 0:
            logger.log_scalar("train/loss", loss.item(), logger.step)
            logger.log_scalar("train/avg_loss", tracker.get_average("loss"), logger.step)
            logger.increment_step()
```

### 4. 性能分析

```python
from waleo_utils import TimerManager

tm = TimerManager()

for batch in dataloader:
    with tm.time("data"):
        data = load_data()

    with tm.time("forward"):
        output = model(data)

    with tm.time("backward"):
        loss.backward()

# 打印统计
tm.print_summary()
```

## API 快速参考

### 设备管理

```python
get_training_device()      # 获取训练设备 (CUDA)
get_inference_device()     # 获取推理设备 (CUDA/MPS/CPU)
setup_distributed()        # 初始化分布式
get_rank()                 # 获取进程 rank
is_main_process()          # 判断是否主进程
```

### 日志

```python
create_logger(path, name)  # 创建日志记录器
logger.log_scalar()        # 记录标量
logger.log_image()         # 记录图像
MetricTracker()            # 指标跟踪器
```

### 时间测量

```python
Timer                      # 基础计时器
TimerManager               # 计时器管理器
timer(name)                # 简单计时上下文
```

### 随机数

```python
set_seed(seed)             # 设置随机种子
RNGManager(seed)           # RNG 管理器
```

### I/O

```python
save_json(data, path)      # 保存 JSON
load_json(path)            # 加载 JSON
save_image(image, path)    # 保存图像
save_video(frames, path)   # 保存视频
```

## 依赖项

### 必需依赖

```
torch>=1.12.0
numpy>=1.20.0
tensorboard>=2.10.0
pillow>=9.0.0
```

### 可选依赖

```bash
# 视频支持
pip install imageio>=2.20.0

# OpenCV
pip install opencv-python>=4.5.0

# 更快的 JSON
pip install orjson>=3.8.0
```

## 常见问题

### Q: 如何选择设备？

A: 使用 `get_training_device()` 用于训练（仅 CUDA），`get_inference_device()` 用于推理（CUDA/MPS/CPU）。

### Q: 如何确保可复现性？

A: 使用 `set_seed(42)` 设置随机种子，并确保数据加载器的确定性。

### Q: 如何在 TensorBoard 中查看日志？

A: 运行 `tensorboard --logdir=logs`，在浏览器访问 `http://localhost:6006`。

### Q: 如何进行分布式训练？

A: 使用 `launch_multi_process(num_gpus, func)` 启动多进程训练，或参考 [分布式训练文档](distributed.md)。

### Q: 如何处理大文件？

A: 使用流式处理，避免一次性加载全部数据到内存。

## 更多资源

- [主 README](../README.md)
- [源代码](../waleo_utils/)
- [测试用例](../tests/)

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

Apache License 2.0
