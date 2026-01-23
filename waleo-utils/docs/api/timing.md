# 时间测量 API

## 类

### Timer

```python
class Timer:
    def __init__(self)
```

基础计时器。

#### 方法

##### start

```python
def start(self) -> None
```

开始计时。

##### stop

```python
def stop(self) -> float
```

停止计时并返回经过的时间。

**返回**：`float` - 经过的时间（秒）

##### reset

```python
def reset(self) -> None
```

重置计时器。

#### 属性

##### elapsed

```python
@property
def elapsed(self) -> float
```

获取累积时间（秒）。

#### 示例

```python
t = Timer()
t.start()
# ... 操作 ...
elapsed = t.stop()

# 或使用上下文管理器
with Timer() as t:
    # ... 操作 ...
print(t.elapsed)
```

---

### TimerManager

```python
class TimerManager:
    def __init__(self)
```

计时器管理器。

#### 方法

##### start

```python
def start(self, name: str, parent: Optional[str] = None) -> None
```

开始命名计时器。

**参数**：
- `name` (str) - 计时器名称
- `parent` (str) - 父计时器名称

##### stop

```python
def stop(self, name: str) -> float
```

停止命名计时器。

**返回**：`float` - 经过的时间

##### elapsed

```python
def elapsed(self, name: str) -> float
```

获取命名计时器的累积时间。

##### reset

```python
def reset(self, name: Optional[str] = None) -> None
```

重置计时器。

**参数**：
- `name` (str) - 计时器名称，None 表示重置所有

##### time

```python
@contextmanager
def time(self, name: str, parent: Optional[str] = None)
```

计时上下文管理器。

##### has_timer

```python
def has_timer(self, name: str) -> bool
```

检查计时器是否存在。

##### list_timers

```python
def list_timers(self) -> List[str]
```

列出所有计时器名称。

##### get_summary

```python
def get_summary(self) -> Dict[str, Dict[str, float]]
```

获取所有计时器的摘要。

##### print_summary

```python
def print_summary(self) -> None
```

打印计时器摘要。

#### 示例

```python
tm = TimerManager()

with tm.time("operation"):
    # ... 操作 ...
    pass

tm.print_summary()
```

---

### CodeTimer

```python
class CodeTimer:
    def __init__(self, name: str = "", fmt: str = ":f", logger=None)
```

代码计时器。

#### 示例

```python
# 上下文管理器
with CodeTimer("操作"):
    # ... 操作 ...
    pass

# 装饰器
@CodeTimer("函数")
def my_function():
    # ... 代码 ...
    pass
```

---

### AverageMeter

```python
class AverageMeter:
    def __init__(self, name: str = "", fmt: str = ":f")
```

平均值计量器。

#### 方法

##### update

```python
def update(self, val: float, n: int = 1) -> None
```

更新计量器。

##### reset

```python
def reset(self) -> None
```

重置计量器。

#### 属性

- `avg` - 平均值
- `val` - 当前值
- `sum` - 总和
- `count` - 计数

---

### ProgressMeter

```python
class ProgressMeter:
    def __init__(self, num_batches: int, meters: List[AverageMeter], prefix: str = "")
```

进度计量器。

#### 方法

##### display

```python
def display(self, batch: int) -> str
```

显示进度。

## 函数

### time_function

```python
def time_function(func)
```

函数计时装饰器。

#### 示例

```python
@time_function
def my_function():
    # ... 代码 ...
    pass
```

---

### timer

```python
@contextmanager
def timer(name: str = "Timer")
```

简单计时上下文管理器。

#### 示例

```python
with timer("操作"):
    # ... 代码 ...
    pass
```
