"""Output unnormalization for policies."""

import numpy as np
import torch
import torch.nn as nn

from ..base.types import NormalizationMode


class Unnormalize(nn.Module):
    """Output unnormalization module.

    This module unnormalizes policy outputs to convert them back to their original
    scale. It wraps the unnormalization functions from waleo.dataset.transforms.normalization
    and integrates them into the policy as a PyTorch module.

    Unnormalization statistics are registered as buffers to ensure they move with the
    model to the correct device.

    Args:
        output_shapes: Dictionary mapping output names to their shapes
        normalization_mapping: Dictionary mapping output names to normalization modes
        stats: Optional dictionary of normalization statistics. Format:
            {
                "action": {
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
        output_shapes: dict[str, tuple],
        normalization_mapping: dict[str, NormalizationMode],
        stats: dict[str, dict] | None = None,
    ):
        super().__init__()
        self.output_shapes = output_shapes
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
        """Unnormalize batch outputs.

        Args:
            batch: Dictionary containing output tensors

        Returns:
            Dictionary with unnormalized tensors
        """
        from waleo.dataset.transforms.normalization import unnormalize_action

        unnormalized_batch = {}

        for key, value in batch.items():
            if key in self.normalization_mapping:
                mode = self.normalization_mapping[key]
                mode_str = mode.value

                # Convert to numpy for unnormalization function
                value_np = value.cpu().numpy() if isinstance(value, torch.Tensor) else value

                # Get stats for this key
                key_stats = {}
                if key in self.stats:
                    for stat_name, buffer_name in self.stats[key].items():
                        buffer = getattr(self, buffer_name)
                        key_stats[stat_name] = buffer.cpu().numpy()

                # Unnormalize using M03 function
                # Note: unnormalize_action expects stats with "action" key
                stats_dict = {"action": key_stats} if key_stats else {}
                unnormalized = unnormalize_action(value_np, stats_dict, normalization_mode=mode_str)

                # Convert back to tensor on same device as input
                device = value.device if isinstance(value, torch.Tensor) else torch.device("cpu")
                unnormalized_batch[key] = torch.from_numpy(unnormalized).float().to(device)
            else:
                # No unnormalization for this key
                unnormalized_batch[key] = value

        return unnormalized_batch
