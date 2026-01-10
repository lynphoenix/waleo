# Waleo Utils

Waleo 基础设施工具模块 - 提供设备管理、分布式训练、通信、日志、随机数、时间测量和 I/O 等基础设施功能。

## 功能特性

### 设备管理 (Device Management)
- 自动选择训练和推理设备（CUDA、MPS、CPU）
- 分布式训练环境初始化和管理
- 进程间同步

### 分布式训练 (Distributed Training)
- 单节点多 GPU 训练启动器
- 多节点训练启动器
- 梯度同步和聚合
- 分布式状态管理

### RPC 通信 (RPC Communication)
- 同步和异步 RPC 客户端
- RPC 服务端
- 流式 RPC 调用
- 基于 Socket 的通信协议

### 日志 (Logging with TensorBoard)
- TensorBoard 日志记录器
- 指标跟踪和统计
- 支持标量、图像、视频、直方图等多种数据类型

### 随机数管理 (Random Number Management)
- 可复现的随机数生成
- RNG 状态保存和恢复
- 分支 RNG 上下文管理器

### 时间测量 (Timing)
- 代码块计时
- 计时器管理器
- 性能分析工具

### I/O 工具 (I/O Utilities)
- 视频读写（支持 imageio 和 OpenCV）
- 图像读写
- JSON 读写（支持复杂数据类型）

## 安装

```bash
# 基础安装
pip install waleo-utils

# 安装所有可选依赖
pip install waleo-utils[all]

# 仅安装视频相关依赖
pip install waleo-utils[video]

# 仅安装 OpenCV
pip install waleo-utils[opencv]

# 开发安装
pip install waleo-utils[dev]
```

## 快速开始

### 设备管理

```python
import torch
from waleo_utils import get_training_device, get_inference_device

# 获取训练设备（仅 CUDA）
device = get_training_device()
model.to(device)

# 获取推理设备（CUDA > MPS > CPU）
device = get_inference_device()
```

### 分布式训练

```python
from waleo_utils import setup_distributed, get_rank, is_main_process

# 初始化分布式环境
setup_distributed()

# 获取当前进程的 rank
rank = get_rank()

# 仅在主进程执行
if is_main_process():
    print("Running on main process")
```

### TensorBoard 日志

```python
from waleo_utils import create_logger

# 创建日志记录器
logger = create_logger(log_dir="logs", name="experiment")

# 记录标量
logger.log_scalar("train/loss", 0.123, step=100)
logger.log_scalar("train/accuracy", 0.956, step=100)

# 记录图像
logger.log_image("train/image", image_tensor, step=100)

# 增加步数
logger.increment_step()
```

### 指标跟踪

```python
from waleo_utils import MetricTracker

# 创建指标跟踪器
tracker = MetricTracker(window_size=100)

# 更新指标
tracker.update("loss", 0.123)
tracker.update("accuracy", 0.956)

# 获取平均值
avg_loss = tracker.get_average("loss")
print(f"Average loss: {avg_loss}")
```

### RPC 通信

```python
from waleo_utils import RPCServer, RPCClient

# 服务端
def add(a, b):
    return a + b

server = RPCServer(host="0.0.0.0", port=5000)
server.register_method("add", add)
server.start()

# 客户端
client = RPCClient(host="localhost", port=5000)
result = client.call("add", 1, 2)
print(f"Result: {result}")
```

### 时间测量

```python
from waleo_utils import TimerManager, timer

# 使用计时器管理器
tm = TimerManager()

with tm.time("data_loading"):
    data = load_data()

with tm.time("forward_pass"):
    output = model(input)

tm.print_summary()

# 使用简单计时器
with timer("Training"):
    train_model()
```

### 随机数管理

```python
from waleo_utils import set_seed, RNGManager

# 设置全局随机种子
set_seed(42)

# 使用 RNG 管理器
rng = RNGManager(seed=42)
rng.save_state("rng_state.pkl")
rng.load_state("rng_state.pkl")
```

### I/O 操作

```python
from waleo_utils import save_video, load_video, save_json, load_json

# 保存视频
save_video(frames, "output.mp4", fps=30)

# 加载视频
frames = load_video("input.mp4")

# 保存 JSON
save_json({"loss": 0.123}, "metrics.json")

# 加载 JSON
data = load_json("config.json")
```

## 架构设计

Waleo Utils 遵循模块化设计，每个子模块都专注于特定功能：

```
waleo-utils/
├── waleo_utils/
│   ├── constants.py          # 常量定义
│   ├── device/               # 设备管理
│   ├── distributed/          # 分布式训练
│   ├── communication/        # RPC 通信
│   ├── logging/              # TensorBoard 日志
│   ├── random/               # 随机数管理
│   ├── timing/               # 时间测量
│   └── io/                   # I/O 工具
└── tests/                    # 单元测试
```

## 依赖项

### 必需依赖
- torch >= 1.12.0
- numpy >= 1.20.0
- tensorboard >= 2.10.0
- pillow >= 9.0.0

### 可选依赖
- imageio >= 2.20.0 (视频 I/O)
- opencv-python >= 4.5.0 (视频/图像 I/O)
- orjson >= 3.8.0 (快速 JSON)

## 许可证

Apache License 2.0

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

- GitHub: https://github.com/waleo-robotics/waleo-utils
- Issues: https://github.com/waleo-robotics/waleo-utils/issues
