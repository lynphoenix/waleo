# 快速入门

本指南将帮助你快速上手 Waleo Utils。

## 5 分钟入门

### 1. 设备管理

```python
from waleo_utils import get_training_device, get_inference_device

# 训练设备（仅 CUDA）
device = get_training_device()
print(f"训练设备: {device}")  # cuda:0

# 推理设备（CUDA > MPS > CPU）
device = get_inference_device()
print(f"推理设备: {device}")  # 自动选择最佳设备
```

### 2. 设置随机种子

```python
from waleo_utils import set_seed

# 确保可复现
set_seed(42)

# 现在所有随机操作都是可复现的
import random
import torch

print(random.random())  # 每次运行相同
print(torch.randn(1))   # 每次运行相同
```

### 3. TensorBoard 日志

```python
from waleo_utils import create_logger

# 创建日志记录器
logger = create_logger("logs", "my_experiment")

# 记录标量
logger.log_scalar("train/loss", 0.5, step=0)
logger.log_scalar("train/accuracy", 0.9, step=0)

# 记录图像
import torch
image = torch.randn(3, 64, 64)
logger.log_image("predictions/sample", image, step=0)

# 查看：tensorboard --logdir=logs
```

### 4. 性能分析

```python
from waleo_utils import TimerManager

tm = TimerManager()

# 计时代码块
with tm.time("data_loading"):
    data = load_data()

with tm.time("forward"):
    output = model(data)

# 打印统计
tm.print_summary()
# 输出：
# data_loading: 0.1234s
# forward: 0.0567s
```

## 典型训练流程

### 完整示例

```python
import torch
from waleo_utils import (
    set_seed,
    get_training_device,
    create_logger,
    MetricTracker,
    TimerManager
)

# 1. 设置环境
set_seed(42)
device = get_training_device()
logger = create_logger("logs", "train")
tracker = MetricTracker()
tm = TimerManager()

# 2. 创建模型
model = torch.nn.Linear(10, 5).to(device)
optimizer = torch.optim.Adam(model.parameters())

# 3. 训练循环
for epoch in range(10):
    for batch_idx, (data, target) in enumerate(dataloader):
        # 计时
        with tm.time("forward"):
            output = model(data.to(device))

        with tm.time("loss"):
            loss = torch.nn.functional.cross_entropy(output, target.to(device))

        with tm.time("backward"):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 跟踪指标
        tracker.update("loss", loss.item())

        # 记录日志
        if batch_idx % 10 == 0:
            logger.log_scalar("batch/loss", loss.item(), logger.step)
            logger.log_scalar("batch/avg_loss", tracker.get_average("loss"), logger.step)
            logger.increment_step()

    # Epoch 指标
    avg_loss = tracker.get_average("loss")
    logger.log_scalar("epoch/loss", avg_loss, epoch)
    tracker.reset("loss")

    # 打印进度
    if epoch % 2 == 0:
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

print("训练完成！")
```

## 常用场景

### 场景 1：单 GPU 训练

```python
from waleo_utils import get_training_device
import torch.nn as nn

# 获取设备
device = get_training_device()

# 创建模型
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
).to(device)

# 训练
for data, target in dataloader:
    data, target = data.to(device), target.to(device)
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

### 场景 2：多 GPU 分布式训练

```python
from waleo_utils import launch_multi_process
import torch

def train(rank, world_size):
    # 初始化分布式
    torch.distributed.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )

    # 创建模型
    model = nn.Linear(10, 5).cuda(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # 训练...
    for batch in dataloader:
        output = model(batch)
        loss.backward()
        optimizer.step()

# 启动 4 进程训练
launch_multi_process(num_gpus=4, func=train)
```

### 场景 3：记录训练指标

```python
from waleo_utils import MetricTracker, create_logger

tracker = MetricTracker(window_size=100)
logger = create_logger("logs", "metrics")

# 训练时跟踪
for batch in dataloader:
    loss = train_step(batch)
    tracker.update("loss", loss.item())

    # 定期记录
    if batch % 10 == 0:
        logger.log_scalar("train/loss", loss.item(), logger.step)
        logger.log_scalar("train/avg_loss", tracker.get_average("loss"), logger.step)
        logger.log_scalar("train/std_loss", tracker.get_std("loss"), logger.step)
        logger.increment_step()
```

### 场景 4：性能优化分析

```python
from waleo_utils import TimerManager

tm = TimerManager()

# 分析各阶段耗时
for epoch in range(num_epochs):
    with tm.time("data_loading"):
        data = next(dataloader)

    with tm.time("forward"):
        output = model(data)

    with tm.time("backward"):
        loss.backward()
        optimizer.step()

    # 每 10 个 epoch 打印统计
    if epoch % 10 == 0:
        print(f"\n=== Epoch {epoch} 性能 ===")
        tm.print_summary()
        tm.reset()
```

### 场景 5：配置文件管理

```python
from waleo_utils import save_json, load_json

# 保存配置
config = {
    "model": {
        "layers": [64, 128, 256],
        "activation": "relu"
    },
    "training": {
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001
    }
}
save_json(config, "config.json")

# 加载配置
config = load_json("config.json")
model = create_model(**config["model"])
```

## 下一步

- [模块文档](modules/) - 详细的功能说明
- [API 参考](api/) - 完整的 API 文档
- [测试用例](https://github.com/waleo-robotics/waleo-utils/tree/main/tests) - 示例代码

## 获取帮助

- GitHub Issues: [https://github.com/waleo-robotics/waleo-utils/issues](https://github.com/waleo-robotics/waleo-utils/issues)
- 文档: [https://github.com/waleo-robotics/waleo-utils/docs](https://github.com/waleo-robotics/waleo-utils/tree/main/docs)
