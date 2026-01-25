"""
设备管理模块

提供设备选择和管理功能，根据使用场景分层选择设备
"""

import torch
from typing import Optional


def get_training_device() -> torch.device:
    """获取训练设备（仅 CUDA）

    训练时必须使用 CUDA 以获得最佳性能

    Returns:
        torch.device: CUDA 设备

    Raises:
        RuntimeError: 如果 CUDA 不可用
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training but not available")

    return torch.device("cuda:0")


def get_inference_device() -> torch.device:
    """获取推理设备（CUDA > MPS > CPU）

    推理时可以使用任何可用设备

    Returns:
        torch.device: 可用设备（优先级：CUDA > MPS > CPU）
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps:0")
    else:
        return torch.device("cpu")


def get_device(device_str: str) -> torch.device:
    """根据字符串获取设备

    Args:
        device_str: 设备字符串 ("cuda", "cuda:0", "mps", "cpu")

    Returns:
        torch.device: 对应的设备

    Raises:
        ValueError: 如果设备字符串无效或设备不可用
    """
    device_str = device_str.lower().strip()

    # 解析设备类型和编号
    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(f"CUDA is not available")
        if device_str == "cuda":
            return torch.device("cuda:0")
        return torch.device(device_str)

    elif device_str.startswith("mps"):
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError(f"MPS is not available")
        if device_str == "mps":
            return torch.device("mps:0")
        return torch.device(device_str)

    elif device_str == "cpu":
        return torch.device("cpu")

    else:
        raise ValueError(f"Unknown device: {device_str}")


def is_device_available(device_str: str) -> bool:
    """检查设备是否可用

    Args:
        device_str: 设备字符串

    Returns:
        bool: 设备是否可用
    """
    try:
        get_device(device_str)
        return True
    except (ValueError, RuntimeError):
        return False


def get_device_count(device_type: str) -> int:
    """获取指定类型的设备数量

    Args:
        device_type: 设备类型 ("cuda", "mps", "cpu")

    Returns:
        int: 设备数量

    Raises:
        ValueError: 如果设备类型无效
    """
    device_type = device_type.lower().strip()

    if device_type == "cuda":
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    elif device_type == "mps":
        return 1 if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else 0
    elif device_type == "cpu":
        return 1
    else:
        raise ValueError(f"Unknown device type: {device_type}")


def get_device_capabilities() -> dict:
    """获取设备能力信息

    Returns:
        dict: 设备能力字典，包含：
            - cuda: available (bool), count (int), nccl (bool)
            - mps: available (bool), count (int)
            - cpu: available (bool)
    """
    capabilities = {}

    # CUDA
    cuda_available = torch.cuda.is_available()
    capabilities["cuda"] = {
        "available": cuda_available,
        "count": torch.cuda.device_count() if cuda_available else 0,
        "nccl": hasattr(torch.distributed, "nccl") if cuda_available else False,
    }

    # MPS
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    capabilities["mps"] = {
        "available": mps_available,
        "count": 1 if mps_available else 0,
        "nccl": False,
    }

    # CPU
    capabilities["cpu"] = {
        "available": True,
    }

    return capabilities
