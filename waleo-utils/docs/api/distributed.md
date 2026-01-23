# 分布式训练 API

## 函数

### launch_multi_process

```python
def launch_multi_process(
    num_gpus: int,
    func: Callable,
    args: Tuple = (),
) -> None
```

启动多进程训练（单节点）。

**参数**：
- `num_gpus` - GPU 数量
- `func` - 训练函数，签名为 `func(rank, world_size, *args)`
- `args` - 传递给函数的额外参数

**示例**：
```python
def train(rank, world_size):
    # 训练代码
    pass

launch_multi_process(num_gpus=4, func=train)
```

---

### launch_multinode

```python
def launch_multinode(
    nodes_config: List[Dict],
    func: Callable,
    args: Tuple = (),
) -> None
```

启动多节点训练。

**参数**：
- `nodes_config` - 节点配置列表
  ```python
  [
      {"hostname": "node1", "gpus": 8},
      {"hostname": "node2", "gpus": 8},
  ]
  ```
- `func` - 训练函数
- `args` - 额外参数

---

### all_reduce

```python
def all_reduce(
    tensor: torch.Tensor,
    op: str = "avg"
) -> torch.Tensor
```

All-Reduce 张量跨进程。

**参数**：
- `tensor` - 要聚合的张量
- `op` - 操作类型：`"avg"`, `"sum"`, `"max"`, `"min"`

**返回**：聚合后的张量

---

### broadcast

```python
def broadcast(
    tensor: torch.Tensor,
    src: int = 0
) -> torch.Tensor
```

从源进程广播张量。

---

### all_gather

```python
def all_gather(
    tensor: torch.Tensor
) -> torch.Tensor
```

收集所有进程的张量。

---

### reduce_scalar

```python
def reduce_scalar(
    value: float,
    op: str = "avg"
) -> float
```

标量 Reduce。

---

## 类

### DistributedState

```python
@dataclass
class DistributedState:
    rank: int
    world_size: int
    local_rank: int
    is_main: bool
```

分布式状态数据类。

#### 方法

##### get_state

```python
def get_state() -> DistributedState
```

获取当前分布式状态。
