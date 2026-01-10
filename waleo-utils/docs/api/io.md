# I/O 操作 API

## 函数

### save_json

```python
def save_json(
    data: Any,
    path: Union[str, Path],
    indent: int = 2,
    use_orjson: bool = True,
) -> None
```

保存数据到 JSON 文件。

**参数**：
- `data` - 要保存的数据
- `path` - 保存路径
- `indent` - 缩进空格数
- `use_orjson` - 是否使用 orjson

---

### load_json

```python
def load_json(
    path: Union[str, Path],
    use_orjson: bool = True,
) -> Any
```

从 JSON 文件加载数据。

---

### save_json_custom

```python
def save_json_custom(
    data: Any,
    path: Union[str, Path],
    indent: int = 2,
) -> None
```

使用自定义编码器保存 JSON。

支持特殊类型：
- `Path` 对象
- `np.ndarray`
- 枚举类型
- 有 `to_dict()` 方法的对象

---

### update_json

```python
def update_json(
    path: Union[str, Path],
    updates: Dict[str, Any],
    create: bool = False,
) -> None
```

更新 JSON 文件中的字段。

---

### get_json_value

```python
def get_json_value(
    path: Union[str, Path],
    key: str,
    default: Any = None,
) -> Any
```

从 JSON 文件获取单个值。

支持嵌套键（用 `.` 分隔）：
```python
value = get_json_value("config.json", "model.learning_rate")
```

---

### save_image

```python
def save_image(
    image: Union[np.ndarray, torch.Tensor],
    path: Union[str, Path],
) -> None
```

保存单张图像。

**支持格式**：
- `Tensor` (H, W, C) 或 (H, W)
- `ndarray` (H, W, C) 或 (H, W)
- 数据类型：`uint8` [0, 255] 或 `float` [0, 1]

---

### load_image

```python
def load_image(
    path: Union[str, Path],
) -> np.ndarray
```

加载单张图像。

**返回**：`np.ndarray` (H, W, C)

---

### save_video

```python
def save_video(
    frames: Union[np.ndarray, torch.Tensor],
    path: Union[str, Path],
    fps: int = 30,
    codec: str = "mp4v",
    backend: str = "imageio",
) -> None
```

保存视频。

**参数**：
- `frames` - 视频帧 (T, H, W, C) 或 (T, H, W)
- `path` - 保存路径
- `fps` - 帧率
- `codec` - 编解码器（OpenCV）
- `backend` - 后端选择（"imageio" 或 "opencv"）

---

### load_video

```python
def load_video(
    path: Union[str, Path],
    backend: str = "imageio",
) -> np.ndarray
```

加载视频。

**返回**：`np.ndarray` (T, H, W, C)
