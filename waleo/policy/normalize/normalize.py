"""Input normalization for policies."""

import numpy as np
import torch
import torch.nn as nn

from ..base.types import NormalizationMode


class Normalize(nn.Module):
    """Input normalization module.

    This module normalizes policy inputs using statistics computed from training data.
    It wraps the normalization functions from waleo.dataset.transforms.normalization
    and integrates them into the policy as a PyTorch module.

    Normalization statistics are registered as buffers to ensure they move with the
    model to the correct device.

    Args:
        input_shapes: Dictionary mapping input names to their shapes
        normalization_mapping: Dictionary mapping input names to normalization modes
        stats: Optional dictionary of normalization statistics. Format:
            {
                "observation.state": {
                    "mean": np.ndarray,
                    "std": np.ndarray,
                    "min": np.ndarray,
                    "max": np.ndarray,
                },
                ...
            }
    """

    def __init__(
        self,
        input_shapes: dict[str, tuple],
        normalization_mapping: dict[str, NormalizationMode],
        stats: dict[str, dict] | None = None,
    ):
        super().__init__()
        self.input_shapes = input_shapes
        self.normalization_mapping = normalization_mapping

        # Register statistics as buffers (ensures device movement)
        self.stats = {}
        if stats is not None:
            for key, stat in stats.items():
                self.stats[key] = {}
                for stat_name, stat_value in stat.items():
                    buffer_name = f"{key}_{stat_name}".replace(".", "_")
                    value_tensor = torch.from_numpy(np.array(stat_value)).float()
                    self.register_buffer(buffer_name, value_tensor)
                    self.stats[key][stat_name] = buffer_name

    def forward(self, batch: dict) -> dict:
        """Normalize batch inputs.

        Args:
            batch: Dictionary containing input tensors

        Returns:
            Dictionary with normalized tensors
        """
        from waleo.dataset.transforms.normalization import normalize_observation

        normalized_batch = {}

        for key, value in batch.items():
            if key in self.normalization_mapping:
                mode = self.normalization_mapping[key]
                mode_str = mode.value

                # Convert to numpy for normalization function
                value_np = value.cpu().numpy() if isinstance(value, torch.Tensor) else value

                # Get stats for this key
                key_stats = {}
                if key in self.stats:
                    for stat_name, buffer_name in self.stats[key].items():
                        buffer = getattr(self, buffer_name)
                        key_stats[stat_name] = buffer.cpu().numpy()

                # Normalize using M03 function
                stats_dict = {key: key_stats} if key_stats else {}
                normalized = normalize_observation(
                    {key: value_np}, stats_dict, normalization_mode=mode_str
                )[key]

                # Convert back to tensor on same device as input
                device = value.device if isinstance(value, torch.Tensor) else torch.device("cpu")
                normalized_batch[key] = torch.from_numpy(normalized).float().to(device)
            else:
                # No normalization for this key
                normalized_batch[key] = value

        return normalized_batch
