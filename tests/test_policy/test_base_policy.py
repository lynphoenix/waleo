"""Tests for BasePolicy abstract class."""

import tempfile
from pathlib import Path

import pytest
import torch

from waleo.policy import BasePolicy, PolicyConfig


def test_mock_policy_creation(mock_policy):
    """Test that mock policy can be created."""
    assert isinstance(mock_policy, BasePolicy)
    assert mock_policy.name == "mock_policy"
    assert isinstance(mock_policy.config, PolicyConfig)


def test_policy_forward(mock_policy, sample_batch):
    """Test policy forward pass."""
    loss, logs = mock_policy.forward(sample_batch)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar loss
    assert isinstance(logs, dict)
    assert "loss" in logs


def test_policy_select_action(mock_policy, sample_batch):
    """Test policy action selection."""
    action = mock_policy.select_action(sample_batch)

    assert isinstance(action, torch.Tensor)
    assert action.shape == (4, 6)  # batch_size=4, action_dim=6


def test_policy_eval_action(mock_policy, sample_batch):
    """Test policy eval_action method."""
    result = mock_policy.eval_action(sample_batch)

    assert isinstance(result, dict)
    assert "action" in result
    assert "success" in result
    assert result["success"] is True
    assert result["action"].shape == (4, 6)


def test_policy_get_optim_params(mock_policy):
    """Test policy get_optim_params method."""
    params = mock_policy.get_optim_params()

    assert isinstance(params, list)
    assert len(params) > 0
    assert all(isinstance(p, dict) for p in params)
    assert all("params" in p for p in params)


def test_policy_reset(mock_policy):
    """Test policy reset method."""
    # Should not raise
    mock_policy.reset()


def test_policy_save_load(mock_policy, sample_batch):
    """Test policy save and load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "policy_checkpoint"

        # Get initial prediction
        with torch.no_grad():
            initial_action = mock_policy.select_action(sample_batch)

        # Save policy
        mock_policy.save(save_path)

        # Verify checkpoint exists
        assert (save_path / "policy.pth").exists()

        # Load policy
        loaded_policy = type(mock_policy).load(save_path)

        # Verify loaded policy produces same output
        with torch.no_grad():
            loaded_action = loaded_policy.select_action(sample_batch)

        assert torch.allclose(initial_action, loaded_action)


def test_policy_device_movement(mock_policy):
    """Test policy device movement."""
    # Move to CPU (already on CPU, but test the method)
    mock_policy.to("cpu")
    assert mock_policy.device == torch.device("cpu")
    assert mock_policy.config.device == "cpu"

    # Test that tensors are on the correct device
    sample_batch = {
        "observation.state": torch.randn(4, 10),
    }
    action = mock_policy.select_action(sample_batch)
    assert action.device == torch.device("cpu")


def test_policy_config_validation():
    """Test policy config validation."""
    # Valid config
    config = PolicyConfig(
        name="test",
        input_shapes={"obs": (10,)},
        output_shapes={"act": (6,)},
        device="cpu",
    )
    assert config.device == "cpu"

    # Invalid device
    with pytest.raises(ValueError, match="device must be"):
        PolicyConfig(device="invalid")

    # Invalid shape type
    with pytest.raises(TypeError, match="must be tuple"):
        PolicyConfig(input_shapes={"obs": [10]})  # type: ignore


def test_abstract_methods_must_be_implemented():
    """Test that abstract methods must be implemented."""

    class IncompletePolicy(BasePolicy):
        name = "incomplete"
        config_class = PolicyConfig

        def __init__(self, config):
            super().__init__(config)

    # Should not be able to instantiate
    config = PolicyConfig(device="cpu")
    with pytest.raises(TypeError, match="abstract methods"):
        IncompletePolicy(config)
