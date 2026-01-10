# 日志模块 (Logging with TensorBoard)

## 概述

日志模块基于 TensorBoard 提供训练过程的日志记录和指标跟踪功能。

## 核心功能

### 1. TensorBoard 日志记录器 (TBLogger)

基于 TensorBoard 的日志记录器，支持多种数据类型。

```python
from waleo_utils import create_logger

# 创建日志记录器
logger = create_logger(log_dir="logs", name="experiment")

# 记录标量
logger.log_scalar("train/loss", 0.123, step=100)
logger.log_scalar("train/accuracy", 0.956, step=100)

# 记录多个标量
logger.log_scalars("metrics", {"loss": 0.123, "acc": 0.956}, step=100)

# 记录图像
logger.log_image("train/prediction", image_tensor, step=100)

# 记录直方图
logger.log_histogram("weights", model.weight, step=100)

# 增加步数
logger.increment_step()

# 关闭日志
logger.close()
```

### 2. 指标跟踪器 (MetricTracker)

自动跟踪和统计训练指标。

```python
from waleo_utils import MetricTracker

# 创建指标跟踪器
tracker = MetricTracker(window_size=100)

# 更新指标
tracker.update("loss", 0.5)
tracker.update("loss", 0.3)
tracker.update("loss", 0.2)

# 获取统计
avg = tracker.get_average("loss")  # 0.333
std = tracker.get_std("loss")      # 标准差
min_val = tracker.get_min("loss")  # 0.2
max_val = tracker.get_max("loss")  # 0.5

# 获取最新值
latest = tracker.get_latest("loss")  # 0.2

# 获取摘要
summary = tracker.get_summary()
```

### 3. 平均值计量器 (AverageMeter)

简单易用的平均值计量器。

```python
from waleo_utils import AverageMeter

# 创建计量器
loss_meter = AverageMeter("loss", ":.4f")

# 更新
loss_meter.update(0.5, n=1)
loss_meter.update(0.3, n=1)

# 获取平均值
print(loss_meter.avg)  # 0.4

# 字符串表示
print(loss_meter)  # "loss 0.3000 (0.4000)"
```

### 4. 进度计量器 (ProgressMeter)

显示训练进度。

```python
from waleo_utils import ProgressMeter, AverageMeter

# 创建计量器
loss_meter = AverageMeter("loss", ":.4f")
acc_meter = AverageMeter("acc", ":.2f")
progress = ProgressMeter(
    num_batches=100,
    meters=[loss_meter, acc_meter],
    prefix="Train: "
)

# 更新并显示
loss_meter.update(0.5)
acc_meter.update(0.95)
print(progress.display(10))
# 输出: "Train: [ 10/100]	loss 0.5000 (0.5000)	acc 0.95 (0.95)"
```

## 使用场景

### 场景 1：训练循环日志

```python
from waleo_utils import create_logger, MetricTracker

logger = create_logger("logs", "mnist")
tracker = MetricTracker(window_size=100)

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # 训练步骤
        loss = train_step(model, data, target)

        # 更新指标
        tracker.update("loss", loss.item())

        # 记录到 TensorBoard
        if batch_idx % 10 == 0:
            logger.log_scalar("train/loss", loss.item(), logger.step)
            logger.increment_step()

    # 每个 epoch 记录平均指标
    avg_loss = tracker.get_average("loss")
    logger.log_scalar("epoch/loss", avg_loss, epoch)
```

### 场景 2：可视化模型

```python
from waleo_utils import create_logger

logger = create_logger("logs", "model_vis")

# 记录模型计算图
sample_input = torch.randn(1, 3, 224, 224)
logger.log_graph(model, sample_input)

# 记录权重分布
for name, param in model.named_parameters():
    logger.log_histogram(f"weights/{name}", param, step=0)
```

### 场景 3：图像日志

```python
from waleo_utils import create_logger
import torch

logger = create_logger("logs", "images")

# 记录单张图像
image = torch.randn(3, 64, 64)
logger.log_image("predictions/sample", image, step=0)

# 记录图像批次
images = torch.randn(8, 3, 64, 64)
logger.log_images("predictions/batch", images, step=0)

# 记录视频（T x H x W x C）
video = torch.randn(10, 64, 64, 3)
logger.log_video("episode/video", video, step=0, fps=30)
```

### 场景 4：超参数对比

```python
from waleo_utils import create_logger

logger = create_logger("logs", "hparam")

# 定义超参数和指标
hparam_dict = {
    "lr": 0.001,
    "batch_size": 32,
    "optimizer": "adam"
}

metric_dict = {
    "accuracy": 0.95,
    "loss": 0.123
}

# 记录到 TensorBoard（可以在 HP 插件中查看）
logger.log_hparams(hparam_dict, metric_dict)
```

## TensorBoard 使用

### 启动 TensorBoard

```bash
# 基础启动
tensorboard --logdir=logs

# 指定端口
tensorboard --logdir=logs --port 6006

# 自动重载
tensorboard --logdir=logs --reload_interval=30
```

### 在浏览器中查看

访问 `http://localhost:6006` 查看：

- **Scalars**：损失、准确率等标量指标
- **Images**：预测结果、特征图等
- **Histograms**：权重分布、梯度分布
- **Graphs**：模型计算图
- **HPARAMS**：超参数对比

## 高级用法

### 1. 多实验对比

```python
from waleo_utils import create_logger

# 为每个实验创建独立的日志目录
for lr in [0.001, 0.01, 0.1]:
    logger = create_logger("logs", f"experiment_lr_{lr}")
    # 训练并记录
    train_with_lr(lr, logger)
```

### 2. 嵌套指标组织

```python
from waleo_utils import create_logger

logger = create_logger("logs", "nested")

# 使用 / 组织指标层级
logger.log_scalar("train/loss", 0.5, step=0)
logger.log_scalar("train/accuracy", 0.9, step=0)
logger.log_scalar("val/loss", 0.6, step=0)
logger.log_scalar("val/accuracy", 0.85, step=0)

# 在 TensorBoard 中会自动分组显示
```

### 3. 实时监控

```python
from waleo_utils import create_logger, MetricTracker

logger = create_logger("logs", "monitor")
tracker = MetricTracker(window_size=100)

for batch in dataloader:
    loss = train_step(batch)
    tracker.update("loss", loss.item())

    # 实时记录
    logger.log_scalar("实时/loss", loss.item(), logger.step)

    # 记录移动平均
    if tracker.get_count("loss") > 10:
        avg = tracker.get_average("loss")
        logger.log_scalar("实时/avg_loss", avg, logger.step)

    logger.increment_step()
```

### 4. 自定义布局

```python
from waleo_utils import create_logger

logger = create_logger("logs", "custom")

# 使用前缀组织不同类型的指标
logger.log_scalar("performance/accuracy", 0.95, step=0)
logger.log_scalar("performance/precision", 0.93, step=0)
logger.log_scalar("performance/recall", 0.91, step=0)

logger.log_scalar("training/loss", 0.123, step=0)
logger.log_scalar("training/lr", 0.001, step=0)

logger.log_scalar("data/queue_size", 100, step=0)
```

## API 参考

### 类

#### TBLogger

| 方法 | 说明 |
|------|------|
| `log_scalar(tag, value, step)` | 记录标量 |
| `log_scalars(main_tag, tag_scalar_dict, step)` | 记录多个标量 |
| `log_image(tag, image, step, dataformats)` | 记录图像 |
| `log_images(tag, images, step, dataformats)` | 记录图像批次 |
| `log_histogram(tag, values, step)` | 记录直方图 |
| `log_graph(model, input_to_model)` | 记录模型图 |
| `log_hparams(hparam_dict, metric_dict)` | 记录超参数 |
| `log_text(tag, text, step)` | 记录文本 |
| `log_video(tag, video, step, fps)` | 记录视频 |
| `log_audio(tag, audio, step, sample_rate)` | 记录音频 |
| `increment_step()` | 增加内部步数 |
| `flush()` | 刷新到磁盘 |
| `close()` | 关闭记录器 |

#### MetricTracker

| 方法 | 说明 |
|------|------|
| `update(name, value)` | 更新指标 |
| `get_average(name, window)` | 获取平均值 |
| `get_sum(name)` | 获取总和 |
| `get_count(name)` | 获取计数 |
| `get_std(name, window)` | 获取标准差 |
| `get_min(name)` | 获取最小值 |
| `get_max(name)` | 获取最大值 |
| `get_latest(name)` | 获取最新值 |
| `get_all(name)` | 获取所有历史值 |
| `has_metric(name)` | 检查指标是否存在 |
| `reset(name)` | 重置指标 |
| `list_metrics()` | 列出所有指标 |
| `get_summary()` | 获取摘要统计 |

#### AverageMeter

| 方法/属性 | 说明 |
|-----------|------|
| `update(val, n)` | 更新计量器 |
| `reset()` | 重置计量器 |
| `avg` | 平均值 |
| `val` | 当前值 |
| `sum` | 总和 |
| `count` | 计数 |

## 注意事项

1. **刷新频率**：定期调用 `flush()` 或设置合适的 `flush_secs` 确保数据写入磁盘
2. **磁盘空间**：长时间训练会产生大量日志，定期清理或使用子目录
3. **性能影响**：过于频繁的日志记录会影响训练速度
4. **数据类型**：确保图像数据在 [0, 255] 范围内或 [0, 1] 范围内
5. **关闭日志**：训练结束后调用 `close()` 确保所有数据写入

## 最佳实践

### 1. 训练脚本模板

```python
from waleo_utils import create_logger, MetricTracker

def train(model, dataloader, num_epochs, log_dir):
    logger = create_logger(log_dir, "train")
    tracker = MetricTracker(window_size=100)

    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(dataloader):
            # 训练
            loss = train_step(model, batch)

            # 跟踪指标
            tracker.update("loss", loss.item())

            # 定期记录
            if batch_idx % 10 == 0:
                # 当前指标
                logger.log_scalar("batch/loss", loss.item(), logger.step)
                logger.log_scalar("batch/avg_loss", tracker.get_average("loss"), logger.step)
                logger.increment_step()

        # Epoch 指标
        avg_loss = tracker.get_average("loss")
        logger.log_scalar("epoch/loss", avg_loss, epoch)
        tracker.reset("loss")

    logger.close()
```

### 2. 调试可视化

```python
from waleo_utils import create_logger

logger = create_logger("logs", "debug")

# 记录中间结果
logger.log_image("debug/input", input_tensor, step=0)
output = model(input_tensor)
logger.log_image("debug/output", output_tensor, step=0)

# 记录梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        logger.log_histogram(f"gradients/{name}", param.grad, step=0)
```

### 3. 实验管理

```python
from waleo_utils import create_logger
from datetime import datetime

# 使用时间戳创建唯一的实验目录
exp_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
logger = create_logger("logs", exp_name)

# 记录实验配置
config = {
    "model": "resnet18",
    "lr": 0.001,
    "batch_size": 32,
}
logger.log_text("config", str(config), step=0)

# 训练...
```
