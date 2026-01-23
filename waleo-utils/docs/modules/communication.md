# RPC 通信模块 (Communication)

## 概述

RPC 通信模块提供远程过程调用（RPC）功能，支持同步和异步通信，用于进程间或机器间的方法调用。

## 核心功能

### 1. RPC 服务端

创建 RPC 服务端并注册可远程调用的方法。

```python
from waleo_utils import RPCServer

# 定义可远程调用的方法
def add(a, b):
    return a + b

def process_data(data):
    # 处理数据
    result = [x * 2 for x in data]
    return result

# 创建服务端
server = RPCServer(host="0.0.0.0", port=5000)

# 注册方法
server.register_method("add", add)
server.register_method("process_data", process_data)

# 启动服务（阻塞）
server.start(block=True)

# 或在后台运行
server.start(block=False)
# ... 其他操作 ...
server.stop()
```

### 2. 同步 RPC 客户端

```python
from waleo_utils import RPCClient

# 创建客户端
client = RPCClient(host="localhost", port=5000)

# 连接
client.connect()

# 调用远程方法
result = client.call("add", 1, 2)
print(f"Result: {result}")  # 3

# 使用关键字参数
result = client.call("add", a=5, b=3)
print(f"Result: {result}")  # 8

# 传递复杂数据
data = [1, 2, 3, 4, 5]
result = client.call("process_data", data)
print(f"Result: {result}")  # [2, 4, 6, 8, 10]

# 断开连接
client.disconnect()
```

### 3. 异步 RPC 客户端

```python
from waleo_utils.communication import AsyncRPCClient
import asyncio

async def async_example():
    # 创建异步客户端
    client = AsyncRPCClient(host="localhost", port=5000)

    # 使用上下文管理器
    async with client:
        # 异步调用
        result = await client.call("add", 10, 20)
        print(f"Result: {result}")  # 30

    # 自动断开连接

# 运行异步代码
asyncio.run(async_example_example())
```

### 4. 流式 RPC

服务端支持生成器的流式响应。

```python
# 服务端：定义生成器方法
def stream_data(num_items):
    for i in range(num_items):
        yield f"item_{i}"

server.register_method("stream_data", stream_data)

# 客户端：流式接收
client = RPCClient(host="localhost", port=5000)

for item in client.call_stream("stream_data", 5):
    print(item)
# 输出：
# item_0
# item_1
# item_2
# item_3
# item_4
```

## 使用场景

### 场景 1：分布式模型推理

```python
# 服务端：GPU 推理服务器
import torch
from waleo_utils import RPCServer

class ModelServer:
    def __init__(self):
        self.model = load_model()
        self.model.cuda()
        self.model.eval()

    def predict(self, input_data):
        with torch.no_grad():
            output = self.model(input_data.cuda())
        return output.cpu().numpy()

server = RPCServer(host="0.0.0.0", port=5000)
model_server = ModelServer()
server.register_method("predict", model_server.predict)
server.start()
```

```python
# 客户端：CPU 机器调用 GPU 服务器
from waleo_utils import RPCClient

client = RPCClient(host="gpu-server", port=5000)

# 准备输入
input_data = prepare_input()

# 调用远程推理
output = client.call("predict", input_data)

# 使用输出
process_output(output)
```

### 场景 2：数据预处理服务

```python
# 服务端：数据预处理
from waleo_utils import RPCServer

def preprocess_batch(images):
    """图像预处理"""
    # 归一化、调整大小等
    processed = []
    for img in images:
        processed.append(preprocess_single(img))
    return processed

def augment_data(data):
    """数据增强"""
    # 随机翻转、旋转等
    return augment(data)

server = RPCServer(host="0.0.0.0", port=5001)
server.register_method("preprocess_batch", preprocess_batch)
server.register_method("augment_data", augment_data)
server.start()
```

```python
# 客户端：训练进程
from waleo_utils import RPCClient

client = RPCClient(host="preprocess-server", port=5001)

# 批量预处理
images = load_images()
processed = client.call("preprocess_batch", images)

# 数据增强
augmented = client.call("augment_data", processed)
```

### 场景 3：参数服务器

```python
# 服务端：参数服务器
from waleo_utils import RPCServer
import numpy as np

class ParameterServer:
    def __init__(self):
        self.params = np.random.randn(1000)

    def get_params(self):
        return self.params.copy()

    def update_params(self, gradients):
        self.params += 0.01 * gradients
        return True

    def push_pull(self, local_params, local_gradients):
        # 推送梯度，拉取参数
        self.update_params(local_gradients)
        return self.get_params()

server = RPCServer(host="0.0.0.0", port=5002)
ps = ParameterServer()
server.register_method("get_params", ps.get_params)
server.register_method("update_params", ps.update_params)
server.register_method("push_pull", ps.push_pull)
server.start()
```

### 场景 4：监控和日志服务

```python
# 服务端：日志收集器
from waleo_utils import RPCServer
from collections import defaultdict

class Logger:
    def __init__(self):
        self.metrics = defaultdict(list)

    def log_metric(self, name, value, timestamp):
        self.metrics[name].append((timestamp, value))
        return True

    def get_metrics(self, name):
        return self.metrics.get(name, [])

    def get_latest(self, name):
        if name in self.metrics and self.metrics[name]:
            return self.metrics[name][-1]
        return None

server = RPCServer(host="0.0.0.0", port=5003)
logger = Logger()
server.register_method("log_metric", logger.log_metric)
server.register_method("get_metrics", logger.get_metrics)
server.register_method("get_latest", logger.get_latest)
server.start()
```

```python
# 客户端：训练进程
from waleo_utils import RPCClient
import time

client = RPCClient(host="log-server", port=5003)

# 记录指标
client.call("log_metric", "loss", 0.5, time.time())
client.call("log_metric", "accuracy", 0.9, time.time())

# 查询指标
metrics = client.call("get_metrics", "loss")
latest = client.call("get_latest", "accuracy")
```

## 高级用法

### 1. 超时控制

```python
from waleo_utils import RPCClient

# 设置超时时间（秒）
client = RPCClient(host="localhost", port=5000, timeout=10.0)

try:
    result = client.call("slow_operation", data)
except RuntimeError as e:
    if "timeout" in str(e).lower():
        print("Operation timed out")
```

### 2. 上下文管理器

```python
from waleo_utils import RPCClient

# 自动连接和断开
with RPCClient(host="localhost", port=5000) as client:
    result = client.call("method", arg1, arg2)
    # 自动处理连接和清理
```

### 3. 多服务端管理

```python
from waleo_utils import RPCClient

# 连接多个服务
servers = [
    RPCClient(host="server1", port=5000),
    RPCClient(host="server2", port=5000),
    RPCClient(host="server3", port=5000),
]

# 连接所有
for server in servers:
    server.connect()

# 并行调用
results = []
for server in servers:
    result = server.call("process", data)
    results.append(result)

# 清理
for server in servers:
    server.disconnect()
```

### 4. 错误处理

```python
from waleo_utils import RPCClient

client = RPCClient(host="localhost", port=5000)

try:
    result = client.call("method", args)
except RuntimeError as e:
    if "Connection refused" in str(e):
        print("服务端未运行")
    elif "timeout" in str(e).lower():
        print("请求超时")
    elif "RPC error" in str(e):
        print(f"RPC 错误: {e}")
    else:
        print(f"未知错误: {e}")
```

## 协议说明

### 消息类型

| 类型 | 说明 |
|------|------|
| `REQUEST` | 普通请求 |
| `RESPONSE` | 普通响应 |
| `STREAM_START` | 流式开始 |
| `STREAM_CHUNK` | 流式数据块 |
| `STREAM_END` | 流式结束 |
| `ERROR` | 错误消息 |

### 错误码

| 错误码 | 说明 |
|--------|------|
| `OK` | 成功 |
| `NOT_FOUND` | 方法不存在 |
| `INVALID_REQUEST` | 无效请求 |
| `INTERNAL_ERROR` | 内部错误 |
| `TIMEOUT` | 超时 |
| `SERIALIZATION_ERROR` | 序列化错误 |
| `DESERIALIZATION_ERROR` | 反序列化错误 |

### 序列化

使用 pickle 进行序列化，支持：
- 基本数据类型（int, float, str, list, dict, 等）
- NumPy 数组
- PyTorch 张量
- 自定义对象（需可 pickle）

## API 参考

### 类

#### RPCServer

| 方法 | 说明 |
|------|------|
| `__init__(host, port, max_workers)` | 创建服务端 |
| `register_method(name, func)` | 注册方法 |
| `unregister_method(name)` | 取消注册方法 |
| `list_methods()` | 列出所有方法 |
| `start(block)` | 启动服务 |
| `stop()` | 停止服务 |

#### RPCClient

| 方法 | 说明 |
|------|------|
| `__init__(host, port, timeout)` | 创建客户端 |
| `connect()` | 连接到服务端 |
| `disconnect()` | 断开连接 |
| `is_connected()` | 检查连接状态 |
| `call(method, *args, **kwargs)` | 同步调用 |
| `call_stream(method, *args, **kwargs)` | 流式调用 |
| `call_async(method, *args, **kwargs)` | 异步调用 |

#### AsyncRPCClient

| 方法 | 说明 |
|------|------|
| `__init__(host, port, timeout)` | 创建异步客户端 |
| `connect()` | 连接到服务端 |
| `disconnect()` | 断开连接 |
| `call(method, *args, **kwargs)` | 异步调用 |

## 注意事项

1. **序列化**：确保所有参数和返回值可序列化
2. **线程安全**：RPCServer 使用线程池，方法应线程安全
3. **错误处理**：服务端方法应处理异常并返回合适的错误信息
4. **超时设置**：合理设置超时时间避免长时间阻塞
5. **连接管理**：使用完毕后断开连接释放资源

## 最佳实践

### 1. 服务端模板

```python
from waleo_utils import RPCServer
import logging

class MyService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def method1(self, arg1, arg2):
        try:
            # 业务逻辑
            result = process(arg1, arg2)
            return result
        except Exception as e:
            self.logger.error(f"Error in method1: {e}")
            raise

# 创建服务端
server = RPCServer(host="0.0.0.0", port=5000, max_workers=4)
service = MyService()

# 注册方法
server.register_method("method1", service.method1)

# 启动
try:
    server.start()
except KeyboardInterrupt:
    print("Shutting down...")
finally:
    server.stop()
```

### 2. 客户端模板

```python
from waleo_utils import RPCClient
import time

class ServiceClient:
    def __init__(self, host, port, max_retries=3):
        self.host = host
        self.port = port
        self.max_retries = max_retries
        self.client = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def connect(self):
        for attempt in range(self.max_retries):
            try:
                self.client = RPCClient(self.host, self.port)
                self.client.connect()
                return
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    raise

    def disconnect(self):
        if self.client:
            self.client.disconnect()

    def call(self, method, *args, **kwargs):
        if not self.client or not self.client.is_connected():
            self.connect()
        return self.client.call(method, *args, **kwargs)

# 使用
with ServiceClient("localhost", 5000) as client:
    result = client.call("method1", arg1, arg2)
```

### 3. 方法注册装饰器

```python
from waleo_utils import RPCServer

class RPCHelper:
    def __init__(self, server):
        self.server = server

    def register(self, name):
        def decorator(func):
            self.server.register_method(name, func)
            return func
        return decorator

# 使用
server = RPCServer(host="0.0.0.0", port=5000)
rpc = RPCHelper(server)

@rpc.register("add")
def add(a, b):
    return a + b

@rpc.register("multiply")
def multiply(a, b):
    return a * b

server.start()
```
