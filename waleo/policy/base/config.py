"""Policy configuration base class."""

from dataclasses import dataclass, field
from typing import Any

from .types import NormalizationMode


@dataclass
class PolicyConfig:
    """Base configuration class for policies.

    Args:
        name: Policy name
        input_shapes: Dictionary mapping input names to their shapes
        output_shapes: Dictionary mapping output names to their shapes
        normalization_mapping: Dictionary mapping feature names to normalization modes
        device: Device to run policy on ("cpu" or "cuda")
        kwargs: Additional keyword arguments for policy-specific configuration
    """

    name: str = "base_policy"
    input_shapes: dict[str, tuple] = field(default_factory=dict)
    output_shapes: dict[str, tuple] = field(default_factory=dict)
    normalization_mapping: dict[str, NormalizationMode] = field(default_factory=dict)
    device: str = "cuda"
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration."""
        # Validate device
        if self.device not in ("cpu", "cuda"):
            raise ValueError(f"device must be 'cpu' or 'cuda', got '{self.device}'")

        # Validate shapes are tuples
        for key, shape in self.input_shapes.items():
            if not isinstance(shape, tuple):
                raise TypeError(f"input_shapes[{key}] must be tuple, got {type(shape)}")

        for key, shape in self.output_shapes.items():
            if not isinstance(shape, tuple):
                raise TypeError(f"output_shapes[{key}] must be tuple, got {type(shape)}")

        # Validate normalization_mapping values are NormalizationMode
        for key, mode in self.normalization_mapping.items():
            if not isinstance(mode, NormalizationMode):
                raise TypeError(
                    f"normalization_mapping[{key}] must be NormalizationMode, got {type(mode)}"
                )
