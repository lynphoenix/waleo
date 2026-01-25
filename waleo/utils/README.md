# Waleo Utils

Waleo 基础设施工具模块，提供设备管理、分布式训练、通信、日志、随机数、时间测量和 I/O 等核心功能。

## 特性

- 🖥️ **设备管理**: 智能设备选择（训练/推理分离），分布式训练支持
- 🌐 **分布式训练**: NCCL多机多卡支持，梯度同步工具
- 📡 **RPC通信**: 异步/同步RPC框架，高效序列化
- 📝 **日志系统**: TensorBoard集成，指标追踪，进度显示
- 🎲 **随机数管理**: 可复现性保证，状态管理
- ⏱️ **性能测量**: 多种计时工具，性能分析
- 💾 **文件I/O**: 视频/图像/JSON读写

## 安装

`waleo.utils` 模块是 waleo 主包的一部分：

```bash
cd /path/to/waleo
pip install -e .
```

## 快速开始

### 设备管理

```python
from waleo.utils import get_training_device, get_inference_device

# 训练（仅CUDA）
device = get_training_device()

# 推理（CUDA > MPS > CPU）
device = get_inference_device()

# 查询设备能力
from waleo.utils import get_device_capabilities
caps = get_device_capabilities()
print(f"CUDA可用: {caps['cuda']['available']}")
```

### 分布式训练

```python
from waleo.utils import setup_distributed, get_rank, launch_multi_process

def train_worker(rank, world_size):
    setup_distributed()
    device = f"cuda:{rank}"
    # 训练代码...

# 启动4卡训练
launch_multi_process(num_gpus=4, func=train_worker, args=())
```

### 日志记录

```python
from waleo.utils import TBLogger, MetricTracker, AverageMeter

# TensorBoard日志
logger = TBLogger("runs/experiment")
logger.log_scalar("loss", 0.5, step=100)

# 指标追踪
tracker = MetricTracker()
tracker.update("loss", 0.5)
tracker.update("accuracy", 0.95)

# 平均值计算
meter = AverageMeter("loss")
for loss in [0.5, 0.3, 0.4]:
    meter.update(loss)
print(f"平均损失: {meter.avg}")  # 0.4
```

### 随机数管理

```python
from waleo.utils import RNGManager, set_seed

# 简单使用
set_seed(42)

# 高级使用
rng = RNGManager(seed=42)
state = rng.get_state()
rng.save_state("checkpoint/rng_state.pkl")

# 恢复状态
rng.load_state("checkpoint/rng_state.pkl")
```

### 时间测量

```python
from waleo.utils import Timer, CodeTimer, timer

# 方式1: 基础计时器
t = Timer()
t.start()
# ... your code ...
t.stop()
print(f"耗时: {t.elapsed:.4f}s")

# 方式2: 上下文管理器
with CodeTimer("forward_pass"):
    output = model(input)

# 方式3: 装饰器
@timer("training_step")
def train_step():
    # ... training code ...
    pass
```

### RPC 通信

```python
from waleo.utils import RPCServer, RPCClient

# 服务端
class MyService(RPCServer):
    def __init__(self):
        super().__init__(port=5000)
        self.register_method("predict", self.predict)

    def predict(self, data):
        return {"result": "success"}

service = MyService()
service.start()

# 客户端
client = RPCClient(host="localhost", port=5000)
client.connect()
result = client.call("predict", data={"input": [1, 2, 3]})
```

### 序列化

```python
from waleo.utils import serialize, deserialize, serialize_tensor

# 基本对象
data = {"key": "value", "numbers": [1, 2, 3]}
serialized = serialize(data)
recovered = deserialize(serialized)

# Tensor序列化
import torch
tensor = torch.randn(3, 3)
tensor_bytes = serialize_tensor(tensor)
recovered_tensor = deserialize_tensor(tensor_bytes)
```

### 文件 I/O

```python
from waleo.utils import save_json, load_json, save_video, save_image

# JSON
data = {"experiment": "test", "accuracy": 0.95}
save_json(data, "results.json")
loaded = load_json("results.json")

# 视频
import numpy as np
frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(30)]
save_video("output.mp4", frames, fps=30)

# 图像
image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
save_image("output.png", image)
```

## API 参考

### 常量 (6个)
- `WALES_HOME` - 缓存目录
- `OBS_STATE`, `OBS_IMAGE` - 观测键名
- `ACTION`, `REWARD` - 动作/奖励键名
- `DEFAULT_RPC_PORT` - 默认RPC端口

### 设备管理 (13个)
- `get_training_device()`, `get_inference_device()`, `get_device()`
- `is_device_available()`, `get_device_count()`, `get_device_capabilities()`
- `setup_distributed()`, `get_world_size()`, `get_rank()`, `get_local_rank()`
- `is_main_process()`, `barrier()`, `destroy_distributed()`

### 分布式训练 (6个)
- `launch_multi_process()`, `launch_multinode()`
- `all_reduce()`, `broadcast()`, `all_gather()`
- `DistributedState`

### RPC 通信 (7个)
- `RPCEndpoint`, `RPCMessage`, `MessageType`, `ErrorCode`
- `RPCClient`, `AsyncRPCClient`, `RPCServer`

### 序列化 (6个)
- `serialize()`, `deserialize()`
- `serialize_tensor()`, `deserialize_tensor()`
- `serialize_ndarray()`, `deserialize_ndarray()`

### 日志系统 (5个)
- `TBLogger` - TensorBoard日志
- `create_logger()` - 创建logger
- `MetricTracker` - 指标追踪
- `AverageMeter` - 平均值计算
- `ProgressMeter` - 进度显示

### 随机数管理 (5个)
- `RNGManager` - 随机数管理器
- `ForkedRNG` - Fork安全的RNG
- `set_seed()`, `get_seed()`, `seed_worker()`

### 时间测量 (5个)
- `Timer` - 基础计时器
- `TimerManager` - 计时器管理
- `CodeTimer` - 代码块计时
- `time_function()`, `timer()` - 计时装饰器

### 文件 I/O (9个)
- `save_video()`, `load_video()` - 视频I/O
- `save_image()`, `load_image()` - 图像I/O
- `save_json()`, `load_json()` - JSON I/O
- `save_json_custom()`, `update_json()`, `get_json_value()` - JSON工具

## 测试

```bash
# 运行完备测试
conda run -n waleo python tests/test_utils/test_comprehensive.py

# 运行单元测试
pytest tests/test_utils/
```

测试覆盖：**120/120** 测试通过 ✅

## 性能优化建议

1. **设备选择**: 训练用`get_training_device()`，推理用`get_inference_device()`
2. **计时分析**: 使用`@timer`装饰器找出性能瓶颈
3. **随机数管理**: 使用`RNGManager`统一管理，确保可复现
4. **日志记录**: 使用`TBLogger`实时监控训练过程

## 依赖

- Python >= 3.9
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- aiohttp >= 3.9.0 (RPC)
- imageio >= 2.31.0 (视频)

## 许可证

与 Waleo 项目相同

## 相关文档

- [设计文档](../../docs/design/M01-基础设施模块设计.md)
- [Waleo 主页](../../README.md)
