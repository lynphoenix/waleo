"""Example of creating a custom policy with waleo.policy.

This example demonstrates:
1. Creating a custom policy by inheriting from BasePolicy
2. Using PolicyFactory to register the policy
3. Implementing all required abstract methods
4. Using normalization utilities
5. Saving and loading policies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field

from waleo.policy import (
    BasePolicy,
    PolicyConfig,
    PolicyFactory,
    NormalizationMode,
    Normalize,
    Unnormalize,
)


# Step 1: Define a custom policy configuration
@dataclass
class MLPPolicyConfig(PolicyConfig):
    """Configuration for MLP policy."""

    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"


# Step 2: Implement custom policy
@PolicyFactory.register()
class MLPPolicy(BasePolicy):
    """Simple MLP policy for demonstration.

    This policy uses a multi-layer perceptron to map observations to actions.
    It includes normalization of inputs and unnormalization of outputs.
    """

    name = "mlp_policy"
    config_class = MLPPolicyConfig

    def __init__(self, config: MLPPolicyConfig):
        super().__init__(config)

        # Get dimensions
        self.obs_dim = config.input_shapes["observation.state"][0]
        self.action_dim = config.output_shapes["action"][0]

        # Build MLP layers
        layers = []
        prev_dim = self.obs_dim

        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation(config.activation))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, self.action_dim))

        self.network = nn.Sequential(*layers)

        # Normalization (if stats are provided)
        self.normalize = None
        self.unnormalize = None

        if "normalization_stats" in config.kwargs:
            stats = config.kwargs["normalization_stats"]
            self.normalize = Normalize(
                input_shapes=config.input_shapes,
                normalization_mapping=config.normalization_mapping,
                stats=stats,
            )
            self.unnormalize = Unnormalize(
                output_shapes=config.output_shapes,
                normalization_mapping={"action": NormalizationMode.MEAN_STD},
                stats=stats,
            )

    def _get_activation(self, name: str):
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
        }
        return activations[name]

    def get_optim_params(self) -> list[dict]:
        """Return optimizer parameters."""
        return [{"params": self.parameters(), "lr": 3e-4}]

    def reset(self):
        """Reset policy state (no state in MLP, so no-op)."""
        pass

    def forward(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """Training forward pass with MSE loss."""
        # Normalize inputs if available
        if self.normalize is not None:
            batch = self.normalize(batch)

        # Forward pass
        obs = batch["observation.state"]
        action_gt = batch["action"]

        action_pred = self.network(obs)

        # Compute loss
        loss = F.mse_loss(action_pred, action_gt)

        # Compute metrics
        with torch.no_grad():
            mae = F.l1_loss(action_pred, action_gt)

        logs = {
            "loss": loss.item(),
            "mae": mae.item(),
        }

        return loss, logs

    def select_action(self, batch: dict) -> torch.Tensor:
        """Inference action selection."""
        # Normalize inputs if available
        if self.normalize is not None:
            batch = self.normalize(batch)

        # Forward pass
        obs = batch["observation.state"]
        action = self.network(obs)

        # Unnormalize outputs if available
        if self.unnormalize is not None:
            action = self.unnormalize({"action": action})["action"]

        return action


def main():
    """Demonstrate custom policy usage."""

    print("=" * 80)
    print("Custom Policy Example")
    print("=" * 80)

    # Create policy configuration
    config = MLPPolicyConfig(
        name="mlp_policy",
        input_shapes={"observation.state": (10,)},
        output_shapes={"action": (6,)},
        normalization_mapping={
            "observation.state": NormalizationMode.MEAN_STD,
        },
        hidden_dims=[128, 128],
        activation="relu",
        device="cpu",
    )

    print(f"\n1. Creating policy with config:")
    print(f"   - Input shape: {config.input_shapes}")
    print(f"   - Output shape: {config.output_shapes}")
    print(f"   - Hidden dims: {config.hidden_dims}")

    # Create policy using factory
    policy = PolicyFactory.create("mlp_policy", config)
    print(f"\n2. Policy created: {policy.name}")
    print(f"   - Number of parameters: {sum(p.numel() for p in policy.parameters())}")

    # Create sample batch
    batch = {
        "observation.state": torch.randn(4, 10),
        "action": torch.randn(4, 6),
    }

    print(f"\n3. Training forward pass:")
    policy.train()
    loss, logs = policy.forward(batch)
    print(f"   - Loss: {logs['loss']:.4f}")
    print(f"   - MAE: {logs['mae']:.4f}")

    # Inference
    print(f"\n4. Inference:")
    policy.eval()
    with torch.no_grad():
        action = policy.select_action(batch)
    print(f"   - Action shape: {action.shape}")
    print(f"   - Action mean: {action.mean().item():.4f}")

    # Save policy
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "policy_checkpoint"

        print(f"\n5. Saving policy to: {save_path}")
        policy.save(save_path)

        # Load policy
        print(f"\n6. Loading policy from: {save_path}")
        loaded_policy = MLPPolicy.load(save_path)

        # Verify loaded policy produces same output
        with torch.no_grad():
            action_loaded = loaded_policy.select_action(batch)

        assert torch.allclose(action, action_loaded), "Loaded policy output differs!"
        print(f"   - Loaded policy matches original: ✓")

    # List all registered policies
    print(f"\n7. Registered policies:")
    for policy_name in PolicyFactory.list_policies():
        print(f"   - {policy_name}")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
