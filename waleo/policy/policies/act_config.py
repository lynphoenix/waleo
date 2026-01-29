"""Configuration for ACT (Action Chunking with Transformers) policy."""

from dataclasses import dataclass, field

from ..base.config import PolicyConfig
from ..base.types import NormalizationMode


@dataclass
class ACTConfig(PolicyConfig):
    """Configuration for ACT policy.

    Args:
        name: Policy name
        input_shapes: Dictionary mapping input names to their shapes
        output_shapes: Dictionary mapping output names to their shapes
        normalization_mapping: Dictionary mapping feature names to normalization modes
        device: Device to run policy on
        kwargs: Additional keyword arguments

        # ACT-specific parameters
        backbone: CNN backbone type ("resnet18", "resnet34", "resnet50")
        backbone_pretrained: Whether to use pretrained backbone weights
        hidden_dim: Hidden dimension of transformer
        nheads: Number of attention heads
        enc_layers: Number of transformer encoder layers
        dec_layers: Number of transformer decoder layers
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout rate
        chunk_size: Number of actions to predict (action chunking)
        camera_names: List of camera names to use
        state_dim: Dimension of robot state
        num_queries: Number of query embeddings (usually same as chunk_size)
    """

    # ACT-specific parameters
    backbone: str = "resnet18"
    backbone_pretrained: bool = True
    hidden_dim: int = 512
    nheads: int = 8
    enc_layers: int = 4
    dec_layers: int = 1
    dim_feedforward: int = 3200
    dropout: float = 0.1
    chunk_size: int = 100
    camera_names: list[str] = field(default_factory=lambda: ["top"])
    state_dim: int = 14  # Robot proprioceptive state dimension
    num_queries: int = 100  # Usually same as chunk_size

    def __post_init__(self):
        """Validate ACT configuration."""
        super().__post_init__()

        # Validate backbone
        valid_backbones = ["resnet18", "resnet34", "resnet50"]
        if self.backbone not in valid_backbones:
            raise ValueError(
                f"backbone must be one of {valid_backbones}, got '{self.backbone}'"
            )

        # Validate dimensions
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")

        if self.nheads <= 0:
            raise ValueError(f"nheads must be positive, got {self.nheads}")

        if self.hidden_dim % self.nheads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by nheads ({self.nheads})"
            )

        if self.enc_layers <= 0:
            raise ValueError(f"enc_layers must be positive, got {self.enc_layers}")

        if self.dec_layers <= 0:
            raise ValueError(f"dec_layers must be positive, got {self.dec_layers}")

        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")

        if self.num_queries <= 0:
            raise ValueError(f"num_queries must be positive, got {self.num_queries}")

        if len(self.camera_names) == 0:
            raise ValueError("camera_names must not be empty")
