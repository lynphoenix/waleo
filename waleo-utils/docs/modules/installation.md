# 安装指南

## 环境要求

- Python 3.8 或更高版本
- PyTorch 1.12 或更高版本
- CUDA 11.0+（推荐，用于 GPU 加速）

## 使用 Conda（推荐）

### 创建新环境

```bash
# 创建 Python 3.10 环境
conda create -n waleo python=3.10 -y
conda activate waleo
```

### 安装依赖

```bash
# 安装 PyTorch（CUDA 版本）
pip install torch torchvision torchaudio

# 或安装 CPU 版本
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 使用 pip 安装

### 基础安装

```bash
pip install waleo-utils
```

### 完整安装（包含所有可选依赖）

```bash
pip install waleo-utils[all]
```

### 选择性安装

```bash
# 仅安装视频支持
pip install waleo-utils[video]

# 仅安装 OpenCV
pip install waleo-utils[opencv]

# 仅安装 orjson（更快的 JSON）
pip install waleo-utils[json]
```

## 从源码安装

```bash
# 克隆仓库
git clone https://github.com/waleo-robotics/waleo-utils.git
cd waleo-utils

# 安装
pip install -e .
```

## 依赖项

### 必需依赖

```
torch>=1.12.0
numpy>=1.20.0
tensorboard>=2.10.0
pillow>=9.0.0
```

### 可选依赖

| 功能 | 依赖 | 版本 |
|------|------|------|
| 视频 I/O | `imageio` | ≥2.20.0 |
| 图像/视频 | `opencv-python` | ≥4.5.0 |
| 快速 JSON | `orjson` | ≥3.8.0 |

## 验证安装

```python
# 测试导入
import waleo_utils
print(f"Waleo Utils 版本: {waleo_utils.__version__}")

# 测试设备管理
from waleo_utils import get_inference_device
device = get_inference_device()
print(f"推理设备: {device}")

# 测试日志
from waleo_utils import create_logger
logger = create_logger("test", "test")
print("日志模块正常")
```

## 常见问题

### Q: CUDA 不可用？

A: 检查 PyTorch CUDA 版本：
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Q: TensorBoard 无法启动？

A: 确保 TensorBoard 已安装：
```bash
pip install tensorboard
```

### Q: 视频保存失败？

A: 安装 imageio 或 opencv-python：
```bash
pip install imageio opencv-python
```

## 下一步

- [快速入门](quickstart.md)
- [模块文档](modules/)
- [API 参考](api/)
