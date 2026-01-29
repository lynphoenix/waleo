"""Data types and enums for policy module."""

from dataclasses import dataclass
from enum import Enum


class FeatureType(Enum):
    """Feature type enumeration for policy inputs/outputs."""

    STATE = "state"  # Robot proprioceptive state
    VISUAL = "visual"  # Image observations
    ENV = "env_state"  # Environment state
    ACTION = "action"  # Action


class NormalizationMode(Enum):
    """Normalization mode for policy features."""

    MEAN_STD = "MEAN_STD"  # (x - mean) / std
    MIN_MAX = "MIN_MAX"  # Map to [-1, 1]
    IDENTITY = "IDENTITY"  # No normalization


@dataclass
class PolicyFeature:
    """Policy feature definition.

    Args:
        type: Feature type (STATE, VISUAL, ENV, ACTION)
        shape: Feature shape tuple
        name: Optional feature name
    """

    type: FeatureType
    shape: tuple[int, ...]
    name: str | None = None

    def __post_init__(self):
        """Validate feature definition."""
        if not isinstance(self.shape, tuple):
            raise TypeError(f"shape must be tuple, got {type(self.shape)}")
        if not all(isinstance(s, int) and s > 0 for s in self.shape):
            raise ValueError(f"shape must contain positive integers, got {self.shape}")
