"""Policy interface module.

This module provides a unified interface for robot learning policies, including:
- BasePolicy: Abstract base class for all policies
- PolicyConfig: Configuration base class
- Normalize/Unnormalize: Input/output normalization utilities
- PolicyFactory: Registration and creation of policy instances
- Evaluation metrics: Common metrics for policy evaluation
"""

from .base import (
    BasePolicy,
    FeatureType,
    NormalizationMode,
    PolicyConfig,
    PolicyFeature,
)
from .factory import PolicyFactory
from .normalize import Normalize, Unnormalize
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
    # Utils
    "compute_success_rate",
    "compute_average_return",
    "compute_std_return",
    "compute_min_return",
    "compute_max_return",
    "compute_metrics",
]
