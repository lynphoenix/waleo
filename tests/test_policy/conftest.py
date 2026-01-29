"""Test fixtures for policy module tests."""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from waleo.policy import BasePolicy, NormalizationMode, PolicyConfig


@pytest.fixture
def mock_policy_config():
    """Create a mock policy configuration."""
    return PolicyConfig(
        name="mock_policy",
        input_shapes={"observation.state": (10,)},
        output_shapes={"action": (6,)},
        normalization_mapping={
            "observation.state": NormalizationMode.MEAN_STD,
            "action": NormalizationMode.MEAN_STD,
        },
        device="cpu",
    )


@pytest.fixture
def mock_stats():
    """Create mock normalization statistics."""
    return {
        "observation.state": {
            "mean": np.array([0.5] * 10),
            "std": np.array([0.5] * 10),
            "min": np.array([-1.0] * 10),
            "max": np.array([1.0] * 10),
        },
        "action": {
            "mean": np.array([0.5] * 6),
            "std": np.array([0.5] * 6),
            "min": np.array([-1.0] * 6),
            "max": np.array([1.0] * 6),
        },
    }


class MockPolicy(BasePolicy):
    """Mock policy for testing."""

    name = "mock_policy"
    config_class = PolicyConfig

    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        input_dim = config.input_shapes["observation.state"][0]
        output_dim = config.output_shapes["action"][0]
        self.network = nn.Linear(input_dim, output_dim)

    def get_optim_params(self) -> list[dict]:
        """Return optimizer parameters."""
        return [{"params": self.parameters()}]

    def reset(self):
        """Reset policy state (no-op for this simple policy)."""
        pass

    def forward(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """Training forward pass."""
        obs = batch["observation.state"]
        action = batch["action"]
        pred = self.network(obs)
        loss = F.mse_loss(pred, action)
        return loss, {"loss": loss.item()}

    def select_action(self, batch: dict) -> torch.Tensor:
        """Inference action selection."""
        obs = batch["observation.state"]
        return self.network(obs)


@pytest.fixture
def mock_policy(mock_policy_config):
    """Create a mock policy instance."""
    return MockPolicy(mock_policy_config)


@pytest.fixture
def sample_batch():
    """Create a sample batch for testing."""
    return {
        "observation.state": torch.randn(4, 10),
        "action": torch.randn(4, 6),
    }
