# 设备管理 API

## 函数

### get_training_device

```python
def get_training_device() -> torch.device
```

获取训练设备（仅 CUDA）。

**返回**：`torch.device` - CUDA 设备

**异常**：
- `RuntimeError` - 如果 CUDA 不可用

**示例**：
```python
device = get_training_device()  # cuda:0
model.to(device)
```

---

### get_inference_device

```python
def get_inference_device() -> torch.device
```

获取推理设备（CUDA > MPS > CPU）。

**返回**：`torch.device` - 可用设备

**示例**：
```python
device = get_inference_device()  # 自动选择最佳设备
model.to(device)
```

---

### get_device

```python
def get_device(device_str: str) -> torch.device
```

根据字符串获取设备。

**参数**：
- `device_str` (str) - 设备字符串（"cuda", "cuda:0", "mps", "cpu"）

**返回**：`torch.device`

**示例**：
```python
device = get_device("cuda:0")
device = get_device("cpu")
```

---

### is_device_available

```python
def is_device_available(device_type: str) -> bool
```

检查设备是否可用。

**参数**：
- `device_type` (str) - 设备类型（"cuda", "mps"）

**返回**：`bool`

---

### get_device_count

```python
def get_device_count(device_type: str) -> int
```

获取设备数量。

**参数**：
- `device_type` (str) - 设备类型

**返回**：`int` - 设备数量

---

### get_device_capabilities

```python
def get_device_capabilities(device_type: str) -> dict
```

获取设备能力。

**参数**：
- `device_type` (str) - 设备类型

**返回**：`dict` - 设备能力信息

---

### setup_distributed

```python
def setup_distributed(backend: str = "nccl") -> None
```

初始化分布式训练环境。

**参数**：
- `backend` (str) - 后端类型（默认："nccl"）

**环境变量**：
- `RANK` - 进程 rank
- `WORLD_SIZE` - 总进程数
- `LOCAL_RANK` - 本地 rank
- `MASTER_ADDR` - 主节点地址
- `MASTER_PORT` - 主节点端口

---

### setup_distributed_multinode

```python
def setup_distributed_multinode(
    backend: str = "nccl",
    init_method: str = "tcp"
) -> None
```

初始化多节点分布式训练。

---

### get_world_size

```python
def get_world_size() -> int
```

获取总进程数。

**返回**：`int`

---

### get_rank

```python
def get_rank() -> int
```

获取当前进程 rank。

**返回**：`int`

---

### get_local_rank

```python
def get_local_rank() -> int
```

获取本地 rank。

**返回**：`int`

---

### is_main_process

```python
def is_main_process() -> bool
```

判断是否为主进程（rank 0）。

**返回**：`bool`

---

### barrier

```python
def barrier() -> None
```

进程同步屏障。

---

### destroy_distributed

```python
def destroy_distributed() -> None
```

清理分布式环境。
