# 随机数管理 API

## 函数

### set_seed

```python
def set_seed(seed: int) -> None
```

设置所有随机数生成器的种子。

**参数**：
- `seed` (int) - 随机种子

**影响的 RNG**：
- Python `random`
- NumPy `numpy.random`
- PyTorch `torch.manual_seed`

**示例**：
```python
set_seed(42)
# 现在所有随机操作都可复现
```

---

### get_seed

```python
def get_seed() -> Optional[int]
```

获取当前随机种子。

**返回**：`Optional[int]` - 当前种子值

---

### seed_worker

```python
def seed_worker(worker_id: int) -> None
```

为 DataLoader worker 设置种子。

**参数**：
- `worker_id` (int) - worker ID

**示例**：
```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    worker_init_fn=seed_worker
)
```

## 类

### RNGManager

```python
class RNGManager:
    def __init__(self, seed: Optional[int] = None)
```

随机数生成器管理器。

#### 方法

##### seed

```python
def seed(self, seed: int) -> None
```

设置新种子。

##### get_state

```python
def get_state(self) -> Dict[str, Any]
```

获取所有 RNG 的当前状态。

**返回**：`Dict[str, Any]` - 状态字典

##### set_state

```python
def set_state(self, state: Dict[str, Any]) -> None
```

设置所有 RNG 的状态。

##### save_state

```python
def save_state(self, path: Path) -> None
```

保存 RNG 状态到文件。

##### load_state

```python
def load_state(self, path: Path) -> None
```

从文件加载 RNG 状态。

##### fork_rng

```python
def fork_rng(self, seed: Optional[int] = None) -> ForkedRNG
```

创建分支 RNG。

**返回**：`ForkedRNG` - 分支 RNG 上下文管理器

---

### ForkedRNG

```python
class ForkedRNG:
    def __init__(self, seed: int)
```

分支 RNG 上下文管理器。

#### 示例

```python
with rng.fork_rng(seed=123):
    # 使用不同的种子
    value = random.random()

# 恢复到主序列
```

---

### seed_worker

```python
def seed_worker(worker_id: int) -> None
```

为 DataLoader worker 设置种子。
