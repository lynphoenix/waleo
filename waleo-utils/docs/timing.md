# 时间测量模块 (Timing)

## 概述

时间测量模块提供代码执行时间测量和性能分析功能，帮助识别性能瓶颈。

## 核心功能

### 1. 基础计时器 (Timer)

简单的计时器，支持累积时间。

```python
from waleo_utils import Timer

# 方式 1：手动控制
t = Timer()
t.start()
# ... 执行操作 ...
elapsed = t.stop()
print(f"操作耗时: {elapsed:.3f}秒")

# 方式 2：使用上下文管理器
with Timer() as t:
    # ... 执行操作 ...
    pass
print(f"操作耗时: {t.elapsed:.3f}秒")

# 时间累积
t = Timer()
t.start()
time.sleep(0.05)
t.stop()

t.start()
time.sleep(0.03)
t.stop()

# t.elapsed = 0.08 (累积时间)
```

### 2. 计时器管理器 (TimerManager)

管理多个命名计时器，支持层次化计时。

```python
from waleo_utils import TimerManager

tm = TimerManager()

# 方式 1：手动控制
tm.start("data_loading")
data = load_data()
tm.stop("data_loading")

tm.start("forward_pass")
output = model(input)
tm.stop("forward_pass")

# 方式 2：使用上下文管理器
with tm.time("preprocessing"):
    data = preprocess(data)

# 获取时间
data_time = tm.elapsed("data_loading")
forward_time = tm.elapsed("forward_pass")

# 打印摘要
tm.print_summary()
```

### 3. 代码计时器 (CodeTimer)

方便的计时工具，自动打印时间。

```python
from waleo_utils import CodeTimer

# 上下文管理器
with CodeTimer("数据加载"):
    data = load_data()
# 自动打印: "数据加载 took X.XX seconds"

# 装饰器
@CodeTimer("模型训练")
def train_model():
    # ... 训练代码 ...
    pass

train_model()
# 自动打印: "模型训练 took X.XX seconds"
```

### 4. time_function 装饰器

简单的函数计时装饰器。

```python
from waleo_utils import time_function

@time_function
def expensive_function():
    # ... 耗时操作 ...
    return result

result = expensive_function()
# 自动打印: "expensive_function took X.XX seconds"
```

### 5. timer 工具

简单的计时上下文管理器。

```python
from waleo_utils import timer

with timer("自定义操作"):
    # ... 操作 ...
    pass
# 自动打印: "自定义操作: X.XXs"
```

## 使用场景

### 场景 1：训练循环性能分析

```python
from waleo_utils import TimerManager

tm = TimerManager()

for epoch in range(num_epochs):
    with tm.time("data_loading"):
        data = next(dataloader)

    with tm.time("forward"):
        output = model(data)

    with tm.time("backward"):
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        tm.print_summary()
```

### 场景 2：函数级性能测量

```python
from waleo_utils import CodeTimer

@CodeTimer("数据预处理")
def preprocess(data):
    # ... 处理逻辑 ...
    return processed_data

@CodeTimer("特征提取")
def extract_features(data):
    # ... 特征提取 ...
    return features

# 自动显示每个函数的执行时间
processed = preprocess(raw_data)
features = extract_features(processed)
```

### 场景 3：层次化性能分析

```python
from waleo_utils import TimerManager

tm = TimerManager()

with tm.time("total"):
    with tm.time("load_data"):
        data = load_data()

    with tm.time("preprocess", parent="total"):
        with tm.time("normalize"):
            data = normalize(data)
        with tm.time("augment"):
            data = augment(data)

    with tm.time("training", parent="total"):
        # ... 训练 ...
        pass

summary = tm.get_summary()
# 可以查看每个阶段的耗时及其父子关系
```

### 场景 4：瓶颈识别

```python
from waleo_utils import TimerManager
import torch

tm = TimerManager()

# 测试不同操作的性能
with tm.time("cpu_computation"):
    result = torch.randn(1000, 1000).cpu()

with tm.time("gpu_transfer"):
    result = result.cuda()

with tm.time("gpu_computation"):
    result = result @ result.T

tm.print_summary()
# 识别最耗时的操作
```

## 高级用法

### 1. 自定义日志记录

```python
from waleo_utils import CodeTimer
import logging

logger = logging.getLogger(__name__)

with CodeTimer("操作", logger=logger):
    # ... 操作 ...
    pass
# 日志会记录时间而不是直接打印
```

### 2. 条件计时

```python
from waleo_utils import TimerManager

tm = TimerManager()

if debug_mode:
    with tm.time("debug_operation"):
        # ... 操作 ...
        pass

# 仅在 debug 模式下测量
if tm.has_timer("debug_operation"):
    print(f"调试操作耗时: {tm.elapsed('debug_operation'):.3f}s")
```

### 3. 性能对比

```python
from waleo_utils import Timer

# 测试方法 A
t = Timer()
for _ in range(100):
    t.start()
    method_a()
    t.stop()
time_a = t.elapsed

# 测试方法 B
t.reset()
for _ in range(100):
    t.start()
    method_b()
    t.stop()
time_b = t.elapsed

print(f"方法 A: {time_a:.3f}s")
print(f"方法 B: {time_b:.3f}s")
print(f"加速比: {time_a/time_b:.2f}x")
```

### 4. 实时监控

```python
from waleo_utils import TimerManager

tm = TimerManager()

for batch in dataloader:
    with tm.time("batch"):
        with tm.time("forward"):
            output = model(batch)

        with tm.time("loss"):
            loss = compute_loss(output)

        with tm.time("backward"):
            loss.backward()

    # 每隔一定时间输出
    if tm.get_count("batch") % 100 == 0:
        avg = tm.get_average("batch")
        print(f"平均批次时间: {avg:.3f}s")
```

## API 参考

### 类

#### Timer

| 方法/属性 | 说明 |
|-----------|------|
| `start()` | 开始计时 |
| `stop()` | 停止计时，返回经过的时间 |
| `reset()` | 重置计时器 |
| `elapsed` | 累积时间（秒） |

#### TimerManager

| 方法 | 说明 |
|------|------|
| `start(name, parent)` | 开始命名计时器 |
| `stop(name)` | 停止命名计时器 |
| `elapsed(name)` | 获取累积时间 |
| `reset(name)` | 重置计时器（name=None 重置全部） |
| `time(name, parent)` | 计时上下文管理器 |
| `has_timer(name)` | 检查计时器是否存在 |
| `list_timers()` | 列出所有计时器 |
| `get_summary()` | 获取摘要统计 |
| `print_summary()` | 打印摘要 |

#### CodeTimer

| 参数 | 说明 |
|------|------|
| `name` | 计时名称 |
| `logger` | 日志记录器（可选） |

### 函数

| 函数 | 说明 |
|------|------|
| `time_function(func)` | 函数计时装饰器 |
| `timer(name)` | 计时上下文管理器 |

## 输出示例

### TimerManager.print_summary() 输出

```
=== Timer Summary ===
data_loading (parent: None): 2.3456s
forward (parent: None): 1.8234s
backward (parent: None): 0.9123s
=====================
```

## 注意事项

1. **开销**：计时本身会有少量开销，不适合极短的操作（< 微秒级）
2. **线程安全**：TimerManager 不是线程安全的，多线程环境需要加锁
3. **累积时间**：Timer 会累积所有调用的时间，注意适时 reset
4. **精度**：使用 `time.perf_counter()` 提供最高可用精度

## 最佳实践

### 1. 训练脚本模板

```python
from waleo_utils import TimerManager

tm = TimerManager()

def train_epoch(model, dataloader, optimizer):
    tm.reset()

    for batch in dataloader:
        with tm.time("batch"):
            with tm.time("forward"):
                output = model(batch)

            with tm.time("backward"):
                loss.backward()
                optimizer.step()

    # 每个 epoch 结束打印统计
    tm.print_summary()
```

### 2. 性能优化工作流

```python
from waleo_utils import TimerManager

tm = TimerManager()

# 1. 基准测量
with tm.time("baseline"):
    result = original_method()

# 2. 优化后测量
tm.reset("baseline")
with tm.time("optimized"):
    result = optimized_method()

# 3. 对比
baseline = tm.elapsed("baseline")
optimized = tm.elapsed("optimized")
print(f"性能提升: {(baseline/optimized - 1)*100:.1f}%")
```

### 3. 生产环境监控

```python
from waleo_utils import TimerManager
import logging

tm = TimerManager()
logger = logging.getLogger(__name__)

def monitored_function():
    with tm.time("operation"):
        # ... 操作 ...
        pass

    # 定期记录到日志
    summary = tm.get_summary()
    logger.info(f"Performance: {summary}")
    tm.reset()
```
