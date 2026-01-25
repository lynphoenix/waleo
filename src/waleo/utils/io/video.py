"""
视频 I/O 工具

提供视频保存和加载功能
"""

from pathlib import Path
from typing import Optional, Union
import numpy as np
import torch

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


def save_video(
    frames: Union[np.ndarray, torch.Tensor],
    path: Union[str, Path],
    fps: int = 30,
    codec: str = "mp4v",
    backend: str = "imageio",
) -> None:
    """保存视频

    Args:
        frames: 视频帧数组，形状为 (T, H, W, C) 或 (T, H, W)
        path: 保存路径
        fps: 帧率
        codec: 编解码器（用于 OpenCV）
        backend: 后端选择，"imageio" 或 "opencv"

    Raises:
        ImportError: 如果选定的后端不可用
        ValueError: 如果帧数据格式无效
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 转换为 numpy array
    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().numpy()

    # 检查形状
    if frames.ndim not in [3, 4]:
        raise ValueError(f"Invalid frames shape: {frames.shape}. Expected (T, H, W) or (T, H, W, C)")

    # 确保是 (T, H, W, C) 格式
    if frames.ndim == 3:
        frames = frames[..., None]

    # 归一化到 uint8
    if frames.dtype != np.uint8:
        if frames.max() <= 1.0:
            frames = (frames * 255).astype(np.uint8)
        else:
            frames = frames.astype(np.uint8)

    # 选择后端
    if backend == "imageio":
        if not IMAGEIO_AVAILABLE:
            raise ImportError("imageio is not installed. Install it with: pip install imageio")
        _save_video_imageio(frames, path, fps)
    elif backend == "opencv":
        if not OPENCV_AVAILABLE:
            raise ImportError("opencv-python is not installed. Install it with: pip install opencv-python")
        _save_video_opencv(frames, path, fps, codec)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _save_video_imageio(frames: np.ndarray, path: Path, fps: int) -> None:
    """使用 imageio 保存视频"""
    with imageio.get_writer(path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)


def _save_video_opencv(frames: np.ndarray, path: Path, fps: int, codec: str) -> None:
    """使用 OpenCV 保存视频"""
    fourcc = cv2.VideoWriter_fourcc(*codec)
    height, width = frames.shape[1:3]
    out = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    for frame in frames:
        # OpenCV 使用 BGR 格式，如果是 RGB 则需要转换
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()


def load_video(
    path: Union[str, Path],
    backend: str = "imageio",
) -> np.ndarray:
    """加载视频

    Args:
        path: 视频路径
        backend: 后端选择，"imageio" 或 "opencv"

    Returns:
        np.ndarray: 视频帧数组，形状为 (T, H, W, C)

    Raises:
        ImportError: 如果选定的后端不可用
        FileNotFoundError: 如果文件不存在
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    if backend == "imageio":
        if not IMAGEIO_AVAILABLE:
            raise ImportError("imageio is not installed. Install it with: pip install imageio")
        return _load_video_imageio(path)
    elif backend == "opencv":
        if not OPENCV_AVAILABLE:
            raise ImportError("opencv-python is not installed. Install it with: pip install opencv-python")
        return _load_video_opencv(path)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _load_video_imageio(path: Path) -> np.ndarray:
    """使用 imageio 加载视频"""
    reader = imageio.get_reader(path)
    frames = []
    for frame in reader:
        frames.append(frame)
    return np.array(frames)


def _load_video_opencv(path: Path) -> np.ndarray:
    """使用 OpenCV 加载视频"""
    cap = cv2.VideoCapture(str(path))
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 转换 BGR 到 RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return np.array(frames)


def save_image(
    image: Union[np.ndarray, torch.Tensor],
    path: Union[str, Path],
) -> None:
    """保存单张图像

    Args:
        image: 图像数据，形状为 (H, W, C) 或 (H, W)
        path: 保存路径
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 转换为 numpy array
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    # 确保是 (H, W, C) 格式
    if image.ndim == 2:
        image = image[..., None]

    # 归一化到 uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    # 使用 imageio 或 OpenCV 保存
    if IMAGEIO_AVAILABLE:
        imageio.imwrite(path, image)
    elif OPENCV_AVAILABLE:
        # 转换 RGB 到 BGR for OpenCV
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), image)
    else:
        # 使用 PIL 作为后备
        from PIL import Image
        Image.fromarray(image).save(path)


def load_image(
    path: Union[str, Path],
) -> np.ndarray:
    """加载单张图像

    Args:
        path: 图像路径

    Returns:
        np.ndarray: 图像数组，形状为 (H, W, C)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    # 使用 imageio 或 OpenCV 加载
    if IMAGEIO_AVAILABLE:
        image = imageio.imread(path)
    elif OPENCV_AVAILABLE:
        image = cv2.imread(str(path))
        # 转换 BGR 到 RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        # 使用 PIL 作为后备
        from PIL import Image
        image = np.array(Image.open(path))

    return image
