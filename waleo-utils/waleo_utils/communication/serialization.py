"""
序列化工具

提供对象（包括 tensor、numpy 等）的序列化和反序列化功能
"""

import pickle
import io
import torch
import numpy as np
from typing import Any


def serialize(obj: Any) -> bytes:
    """序列化对象（支持 tensor、numpy 等）

    Args:
        obj: 要序列化的对象

    Returns:
        bytes: 序列化后的字节数据

    Raises:
        TypeError: 如果对象类型不支持
    """
    try:
        # 使用 pickle 序列化
        buffer = io.BytesIO()
        pickle.dump(obj, buffer)
        return buffer.getvalue()
    except Exception as e:
        raise TypeError(f"Failed to serialize object: {e}")


def deserialize(data: bytes) -> Any:
    """反序列化对象

    Args:
        data: 序列化后的字节数据

    Returns:
        Any: 反序列化后的对象

    Raises:
        TypeError: 如果反序列化失败
    """
    try:
        buffer = io.BytesIO(data)
        obj = pickle.load(buffer)
        return obj
    except Exception as e:
        raise TypeError(f"Failed to deserialize object: {e}")


def serialize_tensor(tensor: torch.Tensor) -> bytes:
    """序列化张量

    Args:
        tensor: PyTorch 张量

    Returns:
        bytes: 序列化后的字节数据
    """
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()


def deserialize_tensor(data: bytes) -> torch.Tensor:
    """反序列化张量

    Args:
        data: 序列化后的字节数据

    Returns:
        torch.Tensor: PyTorch 张量
    """
    buffer = io.BytesIO(data)
    tensor = torch.load(buffer)
    return tensor


def serialize_ndarray(array: np.ndarray) -> bytes:
    """序列化 numpy 数组

    Args:
        array: numpy 数组

    Returns:
        bytes: 序列化后的字节数据
    """
    return pickle.dumps(array)


def deserialize_ndarray(data: bytes) -> np.ndarray:
    """反序列化 numpy 数组

    Args:
        data: 序列化后的字节数据

    Returns:
        np.ndarray: numpy 数组
    """
    return pickle.loads(data)
