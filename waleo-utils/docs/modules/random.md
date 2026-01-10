# 随机数管理模块 (Random)

## 概述

随机数管理模块提供可复现的随机数生成和状态管理，确保实验的可复现性。

## 核心功能

### 1. 设置全局随机种子

设置所有随机数生成器（Python、NumPy、PyTorch）的种子。

```python
from waleo_utils import set_seed

# 设置随机种子
set_seed(42)

# 现在所有随机操作都是可复现的
import random
import numpy as np
import torch

print(random.random())  # 每次运行结果相同
print(np.random.randn())  # 每次运行结果相同
print(torch.randn(1))  # 每次运行结果相同
```

### 2. RNG 管理器

高级的随机数状态管理器，支持状态保存和恢复。

```python
from waleo_utils import RNGManager

# 创建 RNG 管理器
rng = RNGManager(seed=42)

# 生成一些随机数
import random
values = [random.random() for _ in range(5)]

# 保存当前状态
state = rng.get_state()

# 继续生成...
more_values = [random.random() for _ in range(5)]

# 恢复到之前的状态
rng.set_state(state)

# 现在会生成相同的序列
repeat_values = [random.random() for _ in range(5)]
assert repeat_values == more_values
```

### 3. 状态持久化

保存和加载 RNG 状态到文件。

```python
from waleo_utils import RNGManager

rng = RNGManager(seed=42)

# 执行一些随机操作
...

# 保存状态到文件
rng.save_state("rng_state.pkl")

# 之后可以恢复状态
rng.load_state("rng_state.pkl")
```

### 4. 分支 RNG

在临时上下文中使用不同的随机种子，不影响主序列。

```python
from waleo_utils import RNGManager
import random

rng = RNGManager(seed=42)

main_value = random.random()

# 在分支中使用不同种子
with rng.fork_rng(seed=123):
    forked_value = random.random()
    # 这里使用 seed=123

# 恢复后继续主序列
next_main_value = random.random()
# 这里继续使用 seed=42
```

### 5. DataLoader Worker 种子

为 DataLoader 的 worker 进程设置独立的随机种子。

```python
from waleo_utils import seed_worker
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    worker_init_fn=seed_worker,  # 每个 worker 有独立的种子
)
```

## 使用场景

### 场景 1：实验可复现性

```python
from waleo_utils import set_seed
import torch
import numpy as np

# 在训练脚本开始时设置种子
set_seed(42)

# 训练
model = MyModel()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(100):
    # 每次运行都会得到相同的结果
    loss = train_step(model, optimizer)
```

### 场景 2：调试随机问题

```python
from waleo_utils import RNGManager
import random

rng = RNGManager(seed=42)

# 在问题发生前保存状态
rng.save_state("before_issue.pkl")

# 执行可能有问题的代码
problematic_value = random.random()

# 如果发现问题，可以恢复到之前的状态重新调试
rng.load_state("before_issue.pkl")
```

### 场景 3：多进程训练

```python
from waleo_utils import set_seed, get_rank

# 每个进程使用不同的种子
rank = get_rank()
set_seed(42 + rank)

# 现在每个进程有独立但可复现的随机序列
```

### 场景 4：条件随机性

```python
from waleo_utils import RNGManager
import random

rng = RNGManager(seed=42)

# 主序列
main_values = []
for i in range(5):
    # 某些条件下需要特殊的随机性
    if i == 2:
        with rng.fork_rng(seed=100):
            special_value = random.random()
            # 使用特殊种子
    else:
        normal_value = random.random()
        # 使用主种子
    main_values.append(random.random())
```

## API 参考

### 函数

| 函数 | 说明 |
|------|------|
| `set_seed(seed)` | 设置全局随机种子 |
| `get_seed()` | 获取当前随机种子 |
| `seed_worker(worker_id)` | 为 DataLoader worker 设置种子 |

### 类

#### RNGManager

| 方法 | 说明 |
|------|------|
| `__init__(seed)` | 初始化 RNG 管理器 |
| `seed(seed)` | 设置新种子 |
| `get_state()` | 获取当前状态 |
| `set_state(state)` | 设置状态 |
| `save_state(path)` | 保存状态到文件 |
| `load_state(path)` | 从文件加载状态 |
| `fork_rng(seed)` | 创建分支 RNG |

#### ForkedRNG

上下文管理器，用于在临时上下文中使用不同的随机种子。

## 注意事项

1. **CUDA 随机性**：设置 CUDA 随机种子会增加一些性能开销，因为禁用了 cuDNN 的某些优化
2. **完全可复现性**：要实现完全可复现，还需要设置：
   ```python
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   ```
3. **多进程**：在分布式训练中，每个进程应使用不同的种子
4. **线程安全**：RNG 管理器不是线程安全的，多线程环境需要额外同步

## 最佳实践

### 1. 训练脚本模板

```python
from waleo_utils import set_seed

def main(seed=42):
    # 1. 设置随机种子
    set_seed(seed)

    # 2. 创建模型
    model = create_model()

    # 3. 训练
    train(model)

if __name__ == "__main__":
    main(seed=42)
```

### 2. 实验追踪

```python
from waleo_utils import RNGManager

rng = RNGManager(seed=42)

# 记录状态
states = []
for epoch in range(10):
    states.append(rng.get_state())
    train_one_epoch()

# 可以回到任意 epoch 的状态
rng.set_state(states[5])  # 回到 epoch 5
```

### 3. 调试随机 Bug

```python
from waleo_utils import RNGManager

def debug_step():
    rng = RNGManager(seed=42)

    for step in range(1000):
        rng.save_state(f"state_{step}.pkl")
        result = process_step(step)

        if result.is_invalid():
            # 恢复到问题发生前的状态
            rng.load_state(f"state_{step}.pkl")
            # 在调试器中检查
            breakpoint()
```
