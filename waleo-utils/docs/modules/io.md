# I/O 模块 (Input/Output)

## 概述

I/O 模块提供文件输入输出功能，包括视频、图像和 JSON 文件的读写。

## 核心功能

### 1. JSON 文件操作

#### 基础 JSON 读写

```python
from waleo_utils import save_json, load_json

# 保存 JSON
data = {
    "model_config": {
        "layers": [64, 128, 256],
        "activation": "relu"
    },
    "training": {
        "epochs": 100,
        "batch_size": 32
    }
}
save_json(data, "config.json")

# 加载 JSON
loaded = load_json("config.json")
assert loaded == data
```

#### 自定义类型序列化

```python
from waleo_utils.io.json_io import save_json_custom
from pathlib import Path
import numpy as np

# 支持特殊类型
data = {
    "path": Path("/tmp/data"),  # Path 对象
    "array": np.array([1, 2, 3]),  # NumPy 数组
    "enum": SomeEnum.VALUE  # 枚举类型
}

save_json_custom(data, "custom.json")
```

#### JSON 更新

```python
from waleo_utils.io.json_io import update_json

# 更新现有文件中的字段
update_json("config.json", {"training.epochs": 200})

# 如果文件不存在会创建
update_json("new_config.json", {"key": "value"}, create=True)
```

#### 嵌套值获取

```python
from waleo_utils.io.json_io import get_json_value

# 使用 . 访问嵌套字段
epochs = get_json_value("config.json", "training.epochs")
batch_size = get_json_value("config.json", "training.batch_size")

# 提供默认值
unknown = get_json_value("config.json", "unknown.field", default=42)
```

### 2. 图像文件操作

```python
from waleo_utils import save_image, load_image
import torch
import numpy as np

# 保存图像（多种格式）
# PyTorch Tensor (H, W, C) 或 (H, W)
image_tensor = torch.randint(0, 255, (64, 64, 3), dtype=torch.uint8)
save_image(image_tensor, "output.png")

# NumPy 数组
image_np = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
save_image(image_np, "output.jpg")

# 自动归一化浮点图像
image_float = torch.rand(64, 64, 3)  # [0, 1] 范围
save_image(image_float, "output.png")  # 自动缩放到 [0, 255]

# 加载图像
loaded = load_image("input.png")
# 返回 NumPy 数组 (H, W, C)
```

### 3. 视频文件操作

#### 保存视频

```python
from waleo_utils import save_video
import numpy as np

# 生成视频帧 (T, H, W, C)
frames = np.random.randint(0, 255, (100, 64, 64, 3), dtype=np.uint8)

# 保存视频（需要 imageio 或 opencv-python）
save_video(frames, "output.mp4", fps=30)

# 指定后端
save_video(frames, "output.mp4", fps=30, backend="imageio")
save_video(frames, "output.mp4", fps=30, backend="opencv")
```

#### 加载视频

```python
from waleo_utils import load_video

# 加载视频
frames = load_video("input.mp4")
# 返回 NumPy 数组 (T, H, W, C)

print(f"视频帧数: {len(frames)}")
print(f"帧形状: {frames[0].shape}")
```

## 使用场景

### 场景 1：配置文件管理

```python
from waleo_utils import save_json, load_json

# 保存训练配置
config = {
    "model": {
        "type": "resnet18",
        "num_classes": 10
    },
    "training": {
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001
    },
    "data": {
        "dataset": "cifar10",
        "data_dir": "./data"
    }
}
save_json(config, "configs/train_config.json")

# 加载并使用
config = load_json("configs/train_config.json")
model = create_model(**config["model"])
train(model, **config["training"])
```

### 场景 2：训练指标保存

```python
from waleo_utils import save_json

# 保存训练历史
history = {
    "train_loss": [0.8, 0.6, 0.4, 0.3, 0.2],
    "train_acc": [0.7, 0.8, 0.85, 0.9, 0.92],
    "val_loss": [0.9, 0.7, 0.5, 0.4, 0.35],
    "val_acc": [0.68, 0.78, 0.82, 0.88, 0.90]
}
save_json(history, "outputs/training_history.json")
```

### 场景 3：图像数据增强

```python
from waleo_utils import save_image, load_image
import numpy as np

# 加载图像
image = load_image("input.jpg")

# 应用增强
augmented = augment_image(image)

# 保存结果
save_image(augmented, "augmented.jpg")
```

### 场景 4：视频处理

```python
from waleo_utils import save_video, load_video

# 加载视频
frames = load_video("input.mp4")

# 处理每一帧
processed_frames = []
for frame in frames:
    processed = process_frame(frame)
    processed_frames.append(processed)

# 保存处理后的视频
save_video(np.array(processed_frames), "output.mp4", fps=30)
```

### 场景 5：强化学习轨迹记录

```python
from waleo_utils import save_json

# 记录一个 episode
episode = {
    "observations": [obs.tolist() for obs in trajectory],
    "actions": [action.tolist() for action in actions],
    "rewards": rewards,
    "dones": dones,
    "info": {
        "episode_length": len(rewards),
        "total_reward": sum(rewards)
    }
}
save_json(episode, f"trajectories/ep_{episode_id}.json")
```

## 高级用法

### 1. 批量图像处理

```python
from waleo_utils import save_image, load_image
from pathlib import Path

# 批量处理图像
input_dir = Path("images/input")
output_dir = Path("images/output")
output_dir.mkdir(exist_ok=True)

for img_path in input_dir.glob("*.jpg"):
    # 加载
    img = load_image(img_path)

    # 处理
    processed = process_image(img)

    # 保存
    output_path = output_dir / img_path.name
    save_image(processed, output_path)
```

### 2. 配置文件合并

```python
from waleo_utils.io.json_io import merge_json_files

# 基础配置
base_config = {
    "model": {"layers": [64, 128]},
    "training": {"epochs": 100}
}

# 覆盖配置
override_config = {
    "training": {"learning_rate": 0.001}
}

# 合并
merge_json_files(
    "base.json",
    "override.json",
    "merged.json"
)
# 结果: {"model": {...}, "training": {"epochs": 100, "learning_rate": 0.001}}
```

### 3. 视频帧提取

```python
from waleo_utils import load_video, save_image
import numpy as np

# 加载视频并提取特定帧
frames = load_video("video.mp4")

# 提取关键帧（每隔 N 帧）
key_frames = frames[::10]  # 每 10 帧取 1 帧

# 保存为图像
for i, frame in enumerate(key_frames):
    save_image(frame, f"key_frames/frame_{i:04d}.png")
```

### 4. 数据集元数据

```python
from waleo_utils import save_json

# 创建数据集元数据
metadata = {
    "dataset_name": "custom_dataset",
    "version": "1.0",
    "split": {
        "train": {"size": 10000, "files": [...]},
        "val": {"size": 2000, "files": [...]},
        "test": {"size": 2000, "files": [...]}
    },
    "statistics": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    }
}
save_json(metadata, "dataset/metadata.json")
```

## API 参考

### 函数

#### JSON 操作

| 函数 | 说明 |
|------|------|
| `save_json(data, path, indent, use_orjson)` | 保存 JSON 文件 |
| `load_json(path, use_orjson)` | 加载 JSON 文件 |
| `save_json_custom(data, path, indent)` | 保存含特殊类型的 JSON |
| `update_json(path, updates, create)` | 更新 JSON 文件 |
| `get_json_value(path, key, default)` | 获取嵌套 JSON 值 |

#### 图像操作

| 函数 | 说明 |
|------|------|
| `save_image(image, path)` | 保存图像（PNG/JPG） |
| `load_image(path)` | 加载图像 |

#### 视频操作

| 函数 | 说明 |
|------|------|
| `save_video(frames, path, fps, backend)` | 保存视频（MP4/等） |
| `load_video(path, backend)` | 加载视频 |

## 依赖项

### 必需
- `pillow`：图像 I/O（基础）
- `numpy`：数组操作

### 可选

#### 视频/图像高级功能
- `imageio >= 2.20.0`：视频读写
- `opencv-python >= 4.5.0`：视频/图像读写

#### JSON 性能优化
- `orjson >= 3.8.0`：更快的 JSON 序列化

安装可选依赖：
```bash
pip install waleo-utils[all]  # 安装所有
pip install waleo-utils[video]  # 仅视频
pip install waleo-utils[opencv]  # 仅 OpenCV
pip install waleo-utils[json]  # 仅 orjson
```

## 数据格式

### 图像格式

支持的图像格式：
- 输入：`Tensor` (H, W, C) 或 `H, W`，或 `ndarray` (H, W, C) 或 (H, W)
- 数据类型：`uint8` [0, 255] 或 `float` [0, 1]
- 通道顺序：RGB
- 输出格式：根据扩展名自动选择（PNG, JPG, 等）

### 视频格式

支持的视频格式：
- 输入：`Tensor` (T, H, W, C) 或 `ndarray` (T, H, W, C)
- 数据类型：`uint8` [0, 255]
- 通道顺序：RGB
- 编码器：根据后端选择

### JSON 格式

- 标准 JSON 格式
- 支持嵌套结构
- 使用 `save_json_custom` 支持：
  - `Path` 对象 → 转换为字符串
  - `np.ndarray` → 转换为列表
  - 枚举类型 → 转换为值
  - 有 `to_dict()` 方法的对象 → 调用该方法

## 注意事项

1. **内存管理**：加载大视频可能消耗大量内存，考虑分帧处理
2. **图像归一化**：浮点图像应在 [0, 1] 范围，会自动缩放到 [0, 255]
3. **视频编码**：确保系统安装了相应的编解码器
4. **文件权限**：确保有写入权限
5. **路径处理**：自动创建父目录

## 最佳实践

### 1. 配置文件结构

```python
# 推荐的配置文件结构
configs/
├── base.json          # 基础配置
├── models/
│   ├── resnet.json
│   └── transformer.json
└── experiments/
    ├── exp1.json
    └── exp2.json
```

### 2. 输出目录组织

```python
from pathlib import Path
from waleo_utils import save_json, save_image

# 按实验组织输出
exp_dir = Path("outputs") / f"exp_{timestamp}"
exp_dir.mkdir(exist_ok=True)

(exp_dir / "checkpoints").mkdir(exist_ok=True)
(exp_dir / "logs").mkdir(exist_ok=True)
(exp_dir / "visualizations").mkdir(exist_ok=True)

# 保存到相应目录
save_json(config, exp_dir / "config.json")
save_image(image, exp_dir / "visualizations" / "sample.png")
```

### 3. 数据验证

```python
from waleo_utils import save_json, load_json

# 保存前验证
def save_config(config, path):
    # 验证配置
    required_keys = ["model", "training", "data"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key: {key}")

    # 保存
    save_json(config, path)

# 加载后验证
def load_config(path):
    config = load_json(path)

    # 验证版本
    if "version" not in config:
        raise ValueError("Config missing version")

    return config
```
