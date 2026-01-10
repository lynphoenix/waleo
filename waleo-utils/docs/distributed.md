# 分布式训练模块 (Distributed)

## 概述

分布式训练模块提供多进程启动、梯度同步和状态管理功能，支持单节点多 GPU 和多节点分布式训练。

## 核心功能

### 1. 多进程训练启动

#### 单节点多 GPU

```python
from waleo_utils import launch_multi_process

def training_function(rank, world_size):
    """在每个进程中执行的训练函数"""
    # rank: 当前进程的编号 (0, 1, 2, ...)
    # world_size: 总进程数
    print(f"Rank {rank} of {world_size}")

    # 训练代码...
    train_model()

# 启动 4 个进程（使用 4 个 GPU）
launch_multi_process(num_gpus=4, func=training_function)
```

#### 多节点训练

```python
from waleo_utils import launch_multinode

# 节点配置
nodes_config = [
    {"hostname": "node1", "gpus": 4},
    {"hostname": "node2", "gpus": 4},
]

# 启动多节点训练
launch_multinode(
    nodes_config=nodes_config,
    func=training_function,
    master_port=29500
)
```

### 2. 梯度同步

#### All-Reduce

```python
from waleo_utils import all_reduce
import torch

# 在每个进程中计算梯度
local_gradient = compute_gradient()

# 聚合所有进程的梯度
global_gradient = all_reduce(local_gradient, op="avg")
# op: "avg", "sum", "max", "min"
```

#### Broadcast

```python
from waleo_utils import broadcast
import torch

# 仅在 rank 0 上初始化
if rank == 0:
    model_state = init_model()
else:
    model_state = None

# 广播到所有进程
model_state = broadcast(model_state, src=0)
```

#### All-Gather

```python
from waleo_utils import all_gather
import torch

# 每个进程有部分数据
local_data = torch.randn(10)

# 收集所有进程的数据
all_data = all_gather(local_data)
# 返回形状: (world_size * 10,)
```

### 3. 分布式状态管理

```python
from waleo_utils.distributed import DistributedState, get_state

# 获取当前分布式状态
state = get_state()

print(f"Rank: {state.rank}")
print(f"World size: {state.world_size}")
print(f"Local rank: {state.local_rank}")
print(f"Is main: {state.is_main}")
```

## 使用场景

### 场景 1：数据并行训练

```python
from waleo_utils import launch_multi_process
import torch
import torch.nn as nn
import torch.distributed as dist

def train(rank, world_size):
    # 初始化进程组
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )

    # 创建模型并移到本地 GPU
    model = nn.Linear(10, 5).cuda(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # 创建数据加载器（每个进程不同的数据）
    dataset = MyDataset()
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler
    )

    # 训练
    for epoch in range(10):
        for batch in dataloader:
            # 训练步骤...
            loss = model(batch)
            loss.backward()
            optimizer.step()

# 启动训练
launch_multi_process(num_gpus=4, func=train)
```

### 场景 2：梯度累积

```python
from waleo_utils import launch_multi_process
import torch

def train_with_accumulation(rank, world_size):
    # ... 初始化 ...

    accumulation_steps = 4

    for i, batch in enumerate(dataloader):
        output = model(batch)
        loss = criterion(output, target) / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            # 同步梯度
            for param in model.parameters():
                if param.grad is not None:
                    from waleo_utils import all_reduce
                    param.grad = all_reduce(param.grad, op="avg")

            optimizer.step()
            optimizer.zero_grad()

launch_multi_process(num_gpus=4, func=train_with_accumulation)
```

### 场景 3：分布式评估

```python
from waleo_utils import launch_multi_process, all_reduce, get_rank

def evaluate(rank, world_size):
    # ... 初始化 ...

    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            output = model(batch)
            loss = criterion(output, target)

            total_loss += loss.item() * len(batch)
            total_samples += len(batch)

    # 聚合所有进程的损失
    from waleo_utils import all_reduce
    total_loss_tensor = torch.tensor([total_loss])
    total_samples_tensor = torch.tensor([total_samples])

    global_loss = all_reduce(total_loss_tensor, op="sum")
    global_samples = all_reduce(total_samples_tensor, op="sum")

    # 仅主进程打印
    if get_rank() == 0:
        avg_loss = global_loss / global_samples
        print(f"Average loss: {avg_loss:.4f}")

launch_multi_process(num_gpus=4, func=evaluate)
```

### 场景 4：多节点配置

```python
from waleo_utils import launch_multinode

# 节点配置
nodes = [
    {"hostname": "worker-1", "gpus": 8},
    {"hostname": "worker-2", "gpus": 8},
    {"hostname": "worker-3", "gpus": 8},
    {"hostname": "worker-4", "gpus": 8},
]

def training_function(rank, world_size):
    # rank 从 0 到 31（4 节点 x 8 GPU）
    print(f"Rank {rank} starting training...")

    # 计算节点和设备
    node_id = rank // 8
    local_rank = rank % 8

    print(f"Node {node_id}, Local Rank {local_rank}")

    # ... 训练代码 ...

# 启动多节点训练
launch_multinode(
    nodes_config=nodes,
    func=training_function,
    master_port=29500
)
```

## 环境变量

分布式训练需要设置以下环境变量：

### 单节点多 GPU

`launch_multi_process` 会自动设置：
- `RANK`: 进程的全局 rank
- `WORLD_SIZE`: 总进程数
- `LOCAL_RANK`: 本地 GPU 编号
- `MASTER_ADDR`: 主节点地址（默认 localhost）
- `MASTER_PORT`: 主节点端口（默认 29500）

### 多节点

需要手动设置或通过 `launch_multinode` 自动设置：
- `MASTER_ADDR`: 主节点 hostname
- `MASTER_PORT`: 主节点端口（需确保防火墙允许）
- `NODE_RANK`: 当前节点的编号（0, 1, 2, ...）

## API 参考

### 函数

#### 多进程启动

| 函数 | 说明 |
|------|------|
| `launch_multi_process(num_gpus, func, args)` | 启动单节点多进程训练 |
| `launch_multinode(nodes_config, func, args)` | 启动多节点训练 |
| `run_main(func, args)` | 在主进程中执行函数 |

#### 梯度同步

| 函数 | 说明 |
|------|------|
| `all_reduce(tensor, op)` | All-Reduce 操作 |
| `broadcast(tensor, src)` | 从源进程广播 |
| `all_gather(tensor)` | 收集所有进程的数据 |
| `reduce_scalar(value, op)` | 标量 Reduce |

#### 状态管理

| 类/函数 | 说明 |
|---------|------|
| `DistributedState` | 分布式状态数据类 |
| `get_state()` | 获取当前分布式状态 |

## 注意事项

1. **NCCL 后端**：多 GPU 训练推荐使用 NCCL 后端
2. **端口选择**：确保主节点端口未被占用
3. **网络配置**：多节点训练需要节点间网络互通
4. **同步点**：注意避免死锁，确保所有进程到达同步点
5. **内存使用**：每个进程独立占用内存，注意 GPU 内存限制

## 最佳实践

### 1. 初始化模板

```python
def setup_distributed_training():
    from waleo_utils import get_rank, setup_distributed
    import torch

    # 初始化分布式
    setup_distributed()

    rank = get_rank()

    # 设置设备
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    return device, rank
```

### 2. 安全的训练循环

```python
def safe_training_loop(model, dataloader, optimizer):
    from waleo_utils import get_rank

    for epoch in range(num_epochs):
        # 确保所有进程同步
        if get_rank() == 0:
            dataloader.sampler.set_epoch(epoch)

        for batch in dataloader:
            # 训练步骤
            loss = model(batch)
            loss.backward()
            optimizer.step()

        # 同步点
        from waleo_utils import barrier
        barrier()
```

### 3. 仅主进程操作

```python
from waleo_utils import is_main_process

def save_checkpoint(model, path):
    if is_main_process():
        torch.save(model.state_dict(), path)

def log_metrics(metrics):
    if is_main_process():
        print(f"Metrics: {metrics}")
```

### 4. 错误处理

```python
def distributed_training():
    try:
        # 初始化
        setup_distributed()

        # 训练
        train_model()

    except Exception as e:
        # 确保所有进程都看到错误
        if is_main_process():
            print(f"Error in training: {e}")
        raise

    finally:
        # 清理
        cleanup_distributed()
```

## 故障排查

### 问题 1：NCCL 超时

```
NCCL error: unhandled system error
```

解决：
- 检查网络连接
- 增加 NCCL 超时时间：
```python
import os
os.environ["NCCL_BLOCKING_WAIT"] = "1"
```

### 问题 2：端口占用

```
Address already in use
```

解决：
- 更换端口：
```python
launch_multi_process(num_gpus=4, func=train, master_port=29501)
```

### 问题 3：死锁

症状：程序挂起，无输出

解决：
- 确保所有进程到达同步点
- 检查条件分支是否一致
- 使用 timeout 检测

### 问题 4：内存不足

解决：
- 减少 batch size
- 使用梯度累积
- 使用混合精度训练
