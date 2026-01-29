"""ACT (Action Chunking with Transformers) policy implementation.

Reference: https://arxiv.org/abs/2304.13705
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from ..base.policy import BasePolicy
from ..factory import PolicyFactory
from .act_config import ACTConfig


class ResNetBackbone(nn.Module):
    """ResNet backbone for visual feature extraction."""

    def __init__(self, backbone_name: str, pretrained: bool = True):
        super().__init__()

        # Load ResNet model
        if backbone_name == "resnet18":
            resnet = models.resnet18(pretrained=pretrained)
            self.feature_dim = 512
        elif backbone_name == "resnet34":
            resnet = models.resnet34(pretrained=pretrained)
            self.feature_dim = 512
        elif backbone_name == "resnet50":
            resnet = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        # Remove the final fc layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # 1x1 conv to reduce channel dimension
        self.conv = nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1)

    def forward(self, x):
        """Extract visual features.

        Args:
            x: Input images of shape (B, C, H, W)

        Returns:
            Features of shape (B, feature_dim, H', W')
        """
        x = self.backbone(x)
        x = self.conv(x)
        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder."""

    def __init__(
        self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src):
        """Encode input sequence.

        Args:
            src: Input of shape (B, N, D)

        Returns:
            Encoded features of shape (B, N, D)
        """
        return self.encoder(src)


class TransformerDecoder(nn.Module):
    """Transformer decoder."""

    def __init__(
        self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float
    ):
        super().__init__()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )

        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, tgt, memory):
        """Decode with cross-attention to encoder output.

        Args:
            tgt: Target queries of shape (B, M, D)
            memory: Encoder output of shape (B, N, D)

        Returns:
            Decoded features of shape (B, M, D)
        """
        return self.decoder(tgt, memory)


@PolicyFactory.register()
class ACTPolicy(BasePolicy):
    """ACT (Action Chunking with Transformers) policy.

    This policy predicts a sequence of actions (action chunk) using a
    Transformer-based architecture similar to DETR.

    Architecture:
        1. CNN backbone extracts visual features from images
        2. Linear projection embeds robot state
        3. Visual and state features are concatenated
        4. Transformer encoder processes the combined features
        5. Transformer decoder with learnable queries predicts action chunk
        6. Action head predicts final actions

    Reference: https://arxiv.org/abs/2304.13705
    """

    name = "act"
    config_class = ACTConfig

    def __init__(self, config: ACTConfig):
        super().__init__(config)

        self.config: ACTConfig = config
        self.chunk_size = config.chunk_size
        self.num_cameras = len(config.camera_names)

        # CNN backbone for each camera
        self.backbones = nn.ModuleList(
            [
                ResNetBackbone(config.backbone, config.backbone_pretrained)
                for _ in range(self.num_cameras)
            ]
        )

        # Get backbone feature dimension
        backbone_feature_dim = self.backbones[0].feature_dim

        # Input projections
        self.visual_proj = nn.Linear(backbone_feature_dim, config.hidden_dim)
        self.state_proj = nn.Linear(config.state_dim, config.hidden_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, config.hidden_dim))

        # Transformer encoder
        self.encoder = TransformerEncoder(
            d_model=config.hidden_dim,
            nhead=config.nheads,
            num_layers=config.enc_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
        )

        # Learnable query embeddings
        self.query_embed = nn.Parameter(torch.randn(1, config.num_queries, config.hidden_dim))

        # Transformer decoder
        self.decoder = TransformerDecoder(
            d_model=config.hidden_dim,
            nhead=config.nheads,
            num_layers=config.dec_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
        )

        # Action head
        # Get action dimension from output_shapes
        self.action_dim = config.output_shapes["action"][0]
        self.action_head = nn.Linear(config.hidden_dim, self.action_dim)

        # For action chunk prediction during inference
        self.predicted_actions = None
        self.action_index = 0

    def encode_observations(self, batch: dict):
        """Encode visual and state observations.

        Args:
            batch: Dictionary containing:
                - "observation.images.{camera}": Images (B, C, H, W)
                - "observation.state": Robot state (B, state_dim)

        Returns:
            Encoded features of shape (B, N, hidden_dim)
        """
        batch_size = batch["observation.state"].shape[0]

        # Extract visual features from each camera
        visual_features = []
        for i, camera_name in enumerate(self.config.camera_names):
            key = f"observation.images.{camera_name}"
            if key not in batch:
                raise ValueError(f"Missing camera observation: {key}")

            images = batch[key]  # (B, C, H, W)

            # Extract features
            features = self.backbones[i](images)  # (B, feature_dim, H', W')

            # Flatten spatial dimensions
            B, C, H, W = features.shape
            features = features.view(B, C, H * W).transpose(1, 2)  # (B, H'*W', C)

            # Project to hidden_dim
            features = self.visual_proj(features)  # (B, H'*W', hidden_dim)

            visual_features.append(features)

        # Concatenate features from all cameras
        visual_features = torch.cat(visual_features, dim=1)  # (B, N_visual, hidden_dim)

        # Encode robot state
        state = batch["observation.state"]  # (B, state_dim)
        state_features = self.state_proj(state).unsqueeze(1)  # (B, 1, hidden_dim)

        # Concatenate visual and state features
        features = torch.cat([state_features, visual_features], dim=1)  # (B, N, hidden_dim)

        # Add positional encoding
        seq_len = features.shape[1]
        features = features + self.pos_encoding[:, :seq_len, :]

        return features

    def forward(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """Training forward pass.

        Args:
            batch: Dictionary containing:
                - "observation.images.{camera}": Images
                - "observation.state": Robot state
                - "actions": Ground truth action chunk (B, chunk_size, action_dim)

        Returns:
            Tuple of (loss, logs)
        """
        # Encode observations
        encoded_features = self.encode_observations(batch)  # (B, N, hidden_dim)

        # Transformer encoder
        memory = self.encoder(encoded_features)  # (B, N, hidden_dim)

        # Transformer decoder with query embeddings
        batch_size = encoded_features.shape[0]
        queries = self.query_embed.expand(batch_size, -1, -1)  # (B, num_queries, hidden_dim)
        decoded = self.decoder(queries, memory)  # (B, num_queries, hidden_dim)

        # Predict actions
        predicted_actions = self.action_head(decoded)  # (B, num_queries, action_dim)

        # Take only chunk_size predictions
        predicted_actions = predicted_actions[:, : self.chunk_size, :]  # (B, chunk_size, action_dim)

        # Compute loss
        ground_truth_actions = batch["actions"]  # (B, chunk_size, action_dim)
        loss = F.mse_loss(predicted_actions, ground_truth_actions)

        # Compute metrics
        with torch.no_grad():
            mae = F.l1_loss(predicted_actions, ground_truth_actions)

        logs = {
            "loss": loss.item(),
            "mae": mae.item(),
        }

        return loss, logs

    def select_action(self, batch: dict) -> torch.Tensor:
        """Inference action selection with action chunking.

        During inference, we predict a chunk of actions once, then execute them
        sequentially. This reduces the frequency of policy queries and improves
        temporal consistency.

        Args:
            batch: Dictionary containing observations

        Returns:
            Action tensor of shape (B, action_dim)
        """
        batch_size = batch["observation.state"].shape[0]

        # Check if we need to predict new action chunk
        if self.predicted_actions is None or self.action_index >= self.chunk_size:
            # Encode observations
            encoded_features = self.encode_observations(batch)

            # Transformer encoder
            memory = self.encoder(encoded_features)

            # Transformer decoder
            queries = self.query_embed.expand(batch_size, -1, -1)
            decoded = self.decoder(queries, memory)

            # Predict action chunk
            predicted_actions = self.action_head(decoded)
            self.predicted_actions = predicted_actions[:, : self.chunk_size, :]

            # Reset action index
            self.action_index = 0

        # Get current action from chunk
        action = self.predicted_actions[:, self.action_index, :]
        self.action_index += 1

        return action

    def reset(self):
        """Reset policy state (clear predicted actions)."""
        self.predicted_actions = None
        self.action_index = 0

    def get_optim_params(self) -> list[dict]:
        """Return optimizer parameter groups.

        We use different learning rates for backbone and other components.
        """
        # Backbone parameters (lower learning rate if pretrained)
        backbone_params = []
        for backbone in self.backbones:
            backbone_params.extend(list(backbone.parameters()))

        # Other parameters
        other_params = []
        for name, module in self.named_children():
            if name != "backbones":
                other_params.extend(list(module.parameters()))

        if self.config.backbone_pretrained:
            # Lower learning rate for pretrained backbone
            return [
                {"params": backbone_params, "lr": 1e-5},
                {"params": other_params, "lr": 1e-4},
            ]
        else:
            # Same learning rate for all
            return [{"params": self.parameters(), "lr": 1e-4}]
