"""
设备管理模块

提供设备选择、管理和分布式设备支持
"""

from waleo.utils.device.manager import (
    get_training_device,
    get_inference_device,
    get_device,
    is_device_available,
    get_device_count,
    get_device_capabilities,
)

from waleo.utils.device.distributed import (
    setup_distributed,
    setup_distributed_multinode,
    get_world_size,
    get_rank,
    get_local_rank,
    is_main_process,
    barrier,
    destroy_distributed,
)

__all__ = [
    # 基础设备管理
    "get_training_device",
    "get_inference_device",
    "get_device",
    "is_device_available",
    "get_device_count",
    "get_device_capabilities",
    # 分布式设备
    "setup_distributed",
    "setup_distributed_multinode",
    "get_world_size",
    "get_rank",
    "get_local_rank",
    "is_main_process",
    "barrier",
    "destroy_distributed",
]
