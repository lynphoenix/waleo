# 设备管理模块 (Device)

## 概述

设备管理模块提供设备选择、管理和分布式设备支持，自动根据使用场景分层选择设备。

## 核心功能

### 1. 训练设备选择

训练时必须使用 CUDA 以获得最佳性能。

```python
from waleo_utils import get_training_device

device = get_training_device()  # 返回 cuda:0
model.to(device)
```

**注意**：如果 CUDA 不可用，会抛出 `RuntimeError`。

### 2. 推理设备选择

推理时可以使用任何可用设备，优先级为：CUDA > MPS > CPU。

```python
from waleo_utils import get_inference_device

device = get_inference_device()  # 自动选择最佳设备
model.to(device)
```

### 3. 字符串设备获取

根据字符串获取设备对象。

```python
from waleo_utils import get_device

device = get_device("cuda:0")
device = get_device("mps")
device = get_device("cpu")
```

### 4. 分布式训练设置

初始化分布式训练环境。

```python
from waleo_utils import setup_distributed, get_rank, get_world_size

# 从环境变量初始化（RANK, WORLD_SIZE, LOCAL_RANK）
setup_distributed()

rank = get_rank()          # 当前进程的全局 rank
world_size = get_world_size()  # 总进程数
local_rank = get_local_rank()  # 当前节点的本地 rank
```

### 5. 主进程判断

判断当前进程是否为主进程（rank 0）。

```python
from waleo_utils import is_main_process

if is_main_process():
    print("This is the main process")
    # 仅在主进程执行的操作（保存模型、打印日志等）
```

### 6. 进程同步

在所有进程间设置同步屏障。

```python
from waleo_utils import barrier

# 确保所有进程到达此点后再继续
barrier()
```

## 环境变量

分布式训练需要设置以下环境变量：

| 变量 | 说明 | 示例 |
|------|------|------|
| `RANK` | 进程的全局 rank | `0`, `1`, `2` |
| `WORLD_SIZE` | 总进程数 | `4`, `8` |
| `LOCAL_RANK` | 当前节点的本地 rank | `0`, `1` |
| `MASTER_ADDR` | 主节点地址 | `localhost` |
| `MASTER_PORT` | 主节点端口 | `29500` |

## 使用示例

### 单 GPU 训练

```python
import torch
import torch.nn as nn
from waleo_utils import get_training_device

# 创建模型
model = nn.Linear(10, 5)

# 获取训练设备
device = get_training_device()
model.to(device)

# 训练循环
for data, target in dataloader:
    data, target = data.to(device), target.to(device)
    output = model(data)
    # ...
```

### 多 GPU 分布式训练

```python
import torch
import torch.nn as nn
from waleo_utils import setup_distributed, get_rank, is_main_process

# 初始化分布式环境
setup_distributed()

rank = get_rank()

# 创建模型并包装为 DDP
model = nn.Linear(10, 5).cuda()
model = nn.parallel.DistributedDataParallel(model)

# 仅在主进程保存模型
if is_main_process():
    torch.save(model.state_dict(), "model.pth")
```

### 推理

```python
from waleo_utils import get_inference_device

# 自动选择最佳可用设备
device = get_inference_device()
model.to(device)

# 执行推理
with torch.no_grad():
    output = model(input.to(device))
```

## API 参考

### 函数

| 函数 | 说明 |
|------|------|
| `get_training_device()` | 获取训练设备（仅 CUDA） |
| `get_inference_device()` | 获取推理设备（CUDA > MPS > CPU） |
| `get_device(device_str)` | 根据字符串获取设备 |
| `setup_distributed()` | 初始化分布式环境 |
| `get_world_size()` | 获取总进程数 |
| `get_rank()` | 获取当前进程 rank |
| `get_local_rank()` | 获取本地 rank |
| `is_main_process()` | 判断是否为主进程 |
| `barrier()` | 进程同步屏障 |

## 注意事项

1. **训练设备**：训练时必须使用 CUDA，如果 CUDA 不可用会抛出异常
2. **分布式初始化**：必须在创建模型之前调用 `setup_distributed()`
3. **进程同步**：使用 `barrier()` 确保所有进程同步，避免死锁
4. **主进程操作**：保存模型、打印日志等操作应在主进程执行
