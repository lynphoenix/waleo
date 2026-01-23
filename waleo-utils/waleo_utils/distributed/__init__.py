"""
分布式训练支持模块

提供多进程启动、梯度同步和状态管理
"""

from waleo_utils.distributed.launcher import (
    launch_multi_process,
    launch_multinode,
    run_main,
)

from waleo_utils.distributed.reducer import (
    all_reduce,
    broadcast,
    all_gather,
    reduce_scalar,
)

from waleo_utils.distributed.state import (
    DistributedState,
    get_state,
)

__all__ = [
    # 多进程启动
    "launch_multi_process",
    "launch_multinode",
    "run_main",
    # 梯度同步
    "all_reduce",
    "broadcast",
    "all_gather",
    "reduce_scalar",
    # 状态管理
    "DistributedState",
    "get_state",
]
