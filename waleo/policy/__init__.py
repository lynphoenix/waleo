"""Policy interface module.

This module provides a unified interface for robot learning policies, including:
- BasePolicy: Abstract base class for all policies
- PolicyConfig: Configuration base class
- Normalize/Unnormalize: Input/output normalization utilities
- PolicyFactory: Registration and creation of policy instances
- Evaluation metrics: Common metrics for policy evaluation
- Distributed training: DDP utilities for multi-GPU training
- Concrete policies: ACT and other implementations
"""

from .base import (
    BasePolicy,
    FeatureType,
    NormalizationMode,
    PolicyConfig,
    PolicyFeature,
)
from .distributed import (
    DDPCheckpointManager,
    cleanup_ddp,
    get_rank,
    get_world_size,
    is_main_process,
    main_process_first,
    print_rank_0,
    reduce_dict,
    reduce_tensor,
    setup_ddp,
    synchronize,
    wrap_policy_ddp,
)
from .factory import PolicyFactory
from .normalize import Normalize, Unnormalize
from .policies import ACTConfig, ACTPolicy
from .utils import (
    compute_average_return,
    compute_max_return,
    compute_metrics,
    compute_min_return,
    compute_std_return,
    compute_success_rate,
)

__all__ = [
    # Base
    "BasePolicy",
    "PolicyConfig",
    "FeatureType",
    "NormalizationMode",
    "PolicyFeature",
    # Normalize
    "Normalize",
    "Unnormalize",
    # Factory
    "PolicyFactory",
    # Policies
    "ACTPolicy",
    "ACTConfig",
    # Distributed
    "setup_ddp",
    "cleanup_ddp",
    "get_rank",
    "get_world_size",
    "is_main_process",
    "main_process_first",
    "wrap_policy_ddp",
    "reduce_tensor",
    "reduce_dict",
    "DDPCheckpointManager",
    "print_rank_0",
    "synchronize",
    # Utils
    "compute_success_rate",
    "compute_average_return",
    "compute_std_return",
    "compute_min_return",
    "compute_max_return",
    "compute_metrics",
]
