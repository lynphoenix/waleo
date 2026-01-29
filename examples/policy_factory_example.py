"""Example of using PolicyFactory for policy management.

This example demonstrates:
1. Registering multiple policies
2. Creating policies from factory
3. Listing available policies
4. Using custom names for registration
"""

import torch
import torch.nn as nn
from dataclasses import dataclass

from waleo.policy import (
    BasePolicy,
    PolicyConfig,
    PolicyFactory,
    NormalizationMode,
)


# Policy 1: Simple Linear Policy
@PolicyFactory.register()
class LinearPolicy(BasePolicy):
    """Simple linear policy."""

    name = "linear"
    config_class = PolicyConfig

    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        obs_dim = config.input_shapes["observation.state"][0]
        action_dim = config.output_shapes["action"][0]
        self.linear = nn.Linear(obs_dim, action_dim)

    def get_optim_params(self):
        return [{"params": self.parameters()}]

    def reset(self):
        pass

    def forward(self, batch):
        obs = batch["observation.state"]
        action_gt = batch["action"]
        action_pred = self.linear(obs)
        loss = nn.functional.mse_loss(action_pred, action_gt)
        return loss, {"loss": loss.item()}

    def select_action(self, batch):
        return self.linear(batch["observation.state"])


# Policy 2: Two-Layer MLP Policy
@dataclass
class TwoLayerConfig(PolicyConfig):
    """Config for two-layer policy."""

    hidden_dim: int = 128


@PolicyFactory.register()
class TwoLayerPolicy(BasePolicy):
    """Two-layer MLP policy."""

    name = "two_layer"
    config_class = TwoLayerConfig

    def __init__(self, config: TwoLayerConfig):
        super().__init__(config)
        obs_dim = config.input_shapes["observation.state"][0]
        action_dim = config.output_shapes["action"][0]

        self.network = nn.Sequential(
            nn.Linear(obs_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, action_dim),
        )

    def get_optim_params(self):
        return [{"params": self.parameters()}]

    def reset(self):
        pass

    def forward(self, batch):
        obs = batch["observation.state"]
        action_gt = batch["action"]
        action_pred = self.network(obs)
        loss = nn.functional.mse_loss(action_pred, action_gt)
        return loss, {"loss": loss.item()}

    def select_action(self, batch):
        return self.network(batch["observation.state"])


# Policy 3: Using custom registration name
@PolicyFactory.register("advanced_mlp")
class AdvancedMLPPolicy(BasePolicy):
    """Advanced MLP policy registered with custom name."""

    name = "advanced_mlp_internal"  # Internal name (not used for registration)
    config_class = PolicyConfig

    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        obs_dim = config.input_shapes["observation.state"][0]
        action_dim = config.output_shapes["action"][0]

        self.network = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def get_optim_params(self):
        return [{"params": self.parameters()}]

    def reset(self):
        pass

    def forward(self, batch):
        obs = batch["observation.state"]
        action_gt = batch["action"]
        action_pred = self.network(obs)
        loss = nn.functional.mse_loss(action_pred, action_gt)
        return loss, {"loss": loss.item()}

    def select_action(self, batch):
        return self.network(batch["observation.state"])


def main():
    """Demonstrate PolicyFactory usage."""

    print("=" * 80)
    print("PolicyFactory Example")
    print("=" * 80)

    # List all registered policies
    print("\n1. Available policies:")
    for policy_name in PolicyFactory.list_policies():
        policy_cls = PolicyFactory.get_policy_class(policy_name)
        print(f"   - {policy_name} ({policy_cls.__name__})")

    # Check if specific policies are registered
    print("\n2. Check registration:")
    print(f"   - 'linear' registered: {PolicyFactory.is_registered('linear')}")
    print(f"   - 'two_layer' registered: {PolicyFactory.is_registered('two_layer')}")
    print(
        f"   - 'advanced_mlp' registered: {PolicyFactory.is_registered('advanced_mlp')}"
    )
    print(
        f"   - 'nonexistent' registered: {PolicyFactory.is_registered('nonexistent')}"
    )

    # Create shared config
    base_config = PolicyConfig(
        input_shapes={"observation.state": (10,)},
        output_shapes={"action": (6,)},
        normalization_mapping={
            "observation.state": NormalizationMode.MEAN_STD,
        },
        device="cpu",
    )

    # Create different policies from factory
    print("\n3. Creating policies from factory:")

    # Linear policy
    linear_policy = PolicyFactory.create("linear", base_config)
    linear_params = sum(p.numel() for p in linear_policy.parameters())
    print(f"   - Linear policy: {linear_params} parameters")

    # Two-layer policy with custom config
    two_layer_config = TwoLayerConfig(
        input_shapes={"observation.state": (10,)},
        output_shapes={"action": (6,)},
        hidden_dim=256,
        device="cpu",
    )
    two_layer_policy = PolicyFactory.create("two_layer", two_layer_config)
    two_layer_params = sum(p.numel() for p in two_layer_policy.parameters())
    print(f"   - Two-layer policy: {two_layer_params} parameters")

    # Advanced MLP policy (using custom registration name)
    advanced_policy = PolicyFactory.create("advanced_mlp", base_config)
    advanced_params = sum(p.numel() for p in advanced_policy.parameters())
    print(f"   - Advanced MLP policy: {advanced_params} parameters")

    # Test all policies with sample data
    print("\n4. Testing policies with sample data:")
    batch = {
        "observation.state": torch.randn(4, 10),
        "action": torch.randn(4, 6),
    }

    policies = [
        ("linear", linear_policy),
        ("two_layer", two_layer_policy),
        ("advanced_mlp", advanced_policy),
    ]

    for name, policy in policies:
        policy.train()
        loss, logs = policy.forward(batch)

        policy.eval()
        with torch.no_grad():
            action = policy.select_action(batch)

        print(f"   - {name:15} | Loss: {logs['loss']:.4f} | Action: {action.shape}")

    # Demonstrate error handling
    print("\n5. Error handling:")
    try:
        PolicyFactory.create("nonexistent_policy", base_config)
    except ValueError as e:
        print(f"   - Creating unknown policy raises ValueError: ✓")
        print(f"   - Error message: {str(e)[:60]}...")

    # Get policy class
    print("\n6. Getting policy classes:")
    linear_cls = PolicyFactory.get_policy_class("linear")
    print(f"   - 'linear' class: {linear_cls.__name__}")
    print(f"   - Is BasePolicy subclass: {issubclass(linear_cls, BasePolicy)}")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
