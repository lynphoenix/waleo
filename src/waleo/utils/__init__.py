"""
Waleo Utils - 基础设施工具模块

提供设备管理、分布式训练、通信、日志、随机数、时间测量和 I/O 等基础设施功能
"""

__version__ = "0.1.0"

# 常量
from waleo.utils.constants import (
    WALES_HOME,
    OBS_STATE,
    OBS_IMAGE,
    ACTION,
    REWARD,
    DEFAULT_RPC_PORT,
)

# 设备管理
from waleo.utils.device import (
    get_training_device,
    get_inference_device,
    get_device,
    setup_distributed,
    get_world_size,
    get_rank,
    get_local_rank,
    is_main_process,
    barrier,
)

# 设备能力查询（补充）
from waleo.utils.device.manager import (
    is_device_available,
    get_device_count,
    get_device_capabilities,
)

# 分布式环境管理（补充）
from waleo.utils.device.distributed import (
    destroy_distributed,
)

# 分布式训练
from waleo.utils.distributed import (
    launch_multi_process,
    launch_multinode,
    all_reduce,
    broadcast,
    all_gather,
    DistributedState,
)

# RPC 通信
from waleo.utils.communication import (
    RPCEndpoint,
    RPCMessage,
    MessageType,
    ErrorCode,
    RPCClient,
    AsyncRPCClient,
    RPCServer,
)

# 序列化工具
from waleo.utils.communication.serialization import (
    serialize,
    deserialize,
    serialize_tensor,
    deserialize_tensor,
    serialize_ndarray,
    deserialize_ndarray,
)

# 日志 (TensorBoard)
from waleo.utils.logging import (
    TBLogger,
    create_logger,
    MetricTracker,
    AverageMeter,
    ProgressMeter,
)

# 随机数管理
from waleo.utils.random import (
    RNGManager,
    ForkedRNG,
    set_seed,
    get_seed,
    seed_worker,
)

# 时间测量
from waleo.utils.timing import (
    Timer,
    TimerManager,
    CodeTimer,
    time_function,
    timer,
)

# I/O
from waleo.utils.io import (
    save_video,
    load_video,
    save_image,
    load_image,
    save_json,
    load_json,
    save_json_custom,
    update_json,
    get_json_value,
)

__all__ = [
    # 版本
    "__version__",

    # 常量
    "WALES_HOME",
    "OBS_STATE",
    "OBS_IMAGE",
    "ACTION",
    "REWARD",
    "DEFAULT_RPC_PORT",

    # 设备管理
    "get_training_device",
    "get_inference_device",
    "get_device",
    "is_device_available",
    "get_device_count",
    "get_device_capabilities",
    "setup_distributed",
    "get_world_size",
    "get_rank",
    "get_local_rank",
    "is_main_process",
    "barrier",
    "destroy_distributed",

    # 分布式训练
    "launch_multi_process",
    "launch_multinode",
    "all_reduce",
    "broadcast",
    "all_gather",
    "DistributedState",

    # RPC 通信
    "RPCEndpoint",
    "RPCMessage",
    "MessageType",
    "ErrorCode",
    "RPCClient",
    "AsyncRPCClient",
    "RPCServer",

    # 序列化
    "serialize",
    "deserialize",
    "serialize_tensor",
    "deserialize_tensor",
    "serialize_ndarray",
    "deserialize_ndarray",

    # 日志
    "TBLogger",
    "create_logger",
    "MetricTracker",
    "AverageMeter",
    "ProgressMeter",

    # 随机数管理
    "RNGManager",
    "ForkedRNG",
    "set_seed",
    "get_seed",
    "seed_worker",

    # 时间测量
    "Timer",
    "TimerManager",
    "CodeTimer",
    "time_function",
    "timer",

    # I/O
    "save_video",
    "load_video",
    "save_image",
    "load_image",
    "save_json",
    "load_json",
    "save_json_custom",
    "update_json",
    "get_json_value",
]
