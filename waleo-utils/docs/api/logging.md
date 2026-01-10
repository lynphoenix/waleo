# 日志模块 API

## 函数

### create_logger

```python
def create_logger(
    log_dir: Union[str, Path],
    name: str = "waleo",
    comment: str = ""
) -> TBLogger
```

创建日志记录器。

**参数**：
- `log_dir` - 基础日志目录
- `name` - 日志名称
- `comment` - 日志目录后缀注释

**返回**：`TBLogger`

## 类

### TBLogger

```python
class TBLogger:
    def __init__(
        self,
        log_dir: Union[str, Path],
        comment: str = "",
        flush_secs: int = 120,
    )
```

TensorBoard 日志记录器。

#### 方法

##### log_scalar

```python
def log_scalar(
    self,
    tag: str,
    value: float,
    step: Optional[int] = None,
) -> None
```

记录标量。

##### log_scalars

```python
def log_scalars(
    self,
    main_tag: str,
    tag_scalar_dict: Dict[str, float],
    step: Optional[int] = None,
) -> None
```

记录多个标量。

##### log_image

```python
def log_image(
    self,
    tag: str,
    image: Union[np.ndarray, torch.Tensor],
    step: Optional[int] = None,
    dataformats: str = "CHW",
) -> None
```

记录图像。

##### log_images

```python
def log_images(
    self,
    tag: str,
    images: Union[np.ndarray, torch.Tensor],
    step: Optional[int] = None,
    dataformats: str = "NCHW",
) -> None
```

记录多张图像。

##### log_histogram

```python
def log_histogram(
    self,
    tag: str,
    values: Union[np.ndarray, torch.Tensor],
    step: Optional[int] = None,
) -> None
```

记录直方图。

##### log_graph

```python
def log_graph(
    self,
    model: torch.nn.Module,
    input_to_model: Union[torch.Tensor, tuple],
) -> None
```

记录模型计算图。

##### log_hparams

```python
def log_hparams(
    self,
    hparam_dict: Dict[str, Any],
    metric_dict: Dict[str, float],
) -> None
```

记录超参数和指标。

##### log_video

```python
def log_video(
    self,
    tag: str,
    video: Union[np.ndarray, torch.Tensor],
    step: Optional[int] = None,
    fps: int = 4,
) -> None
```

记录视频。

##### increment_step

```python
def increment_step(self) -> None
```

增加内部步数计数器。

##### flush

```python
def flush(self) -> None
```

刷新日志到磁盘。

##### close

```python
def close(self) -> None
```

关闭日志记录器。

---

### MetricTracker

```python
class MetricTracker:
    def __init__(
        self,
        logger: Optional[TBLogger] = None,
        window_size: int = 100,
    )
```

指标跟踪器。

#### 方法

##### update

```python
def update(self, name: str, value: Union[float, int, torch.Tensor, np.ndarray]) -> None
```

更新指标。

##### get_average

```python
def get_average(self, name: str, window: Optional[int] = None) -> float
```

获取平均值。

##### get_std

```python
def get_std(self, name: str, window: Optional[int] = None) -> float
```

获取标准差。

##### get_min

```python
def get_min(self, name: str) -> float
```

获取最小值。

##### get_max

```python
def get_max(self, name: str) -> float
```

获取最大值。

##### get_latest

```python
def get_latest(self, name: str) -> Optional[float]
```

获取最新值。

##### get_summary

```python
def get_summary(self) -> Dict[str, Dict[str, float]]
```

获取所有指标的摘要统计。

##### reset

```python
def reset(self, name: Optional[str] = None) -> None
```

重置指标。

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
- `sum` - 总和
- `count` - 计数

---

### ProgressMeter

```python
class ProgressMeter:
    def __init__(
        self,
        num_batches: int,
        meters: List[AverageMeter],
        prefix: str = "",
    )
```

进度计量器。

#### 方法

##### display

```python
def display(self, batch: int) -> str
```

显示进度。
