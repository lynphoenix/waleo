"""Tests for ACT policy."""

import pytest
import torch

from waleo.policy import ACTConfig, ACTPolicy, PolicyFactory


@pytest.fixture
def act_config():
    """Create ACT configuration for testing."""
    return ACTConfig(
        name="act",
        input_shapes={
            "observation.images.top": (3, 128, 128),
            "observation.state": (14,),
        },
        output_shapes={"action": (7,)},
        device="cpu",
        # ACT-specific
        backbone="resnet18",
        backbone_pretrained=False,  # Faster for testing
        hidden_dim=256,
        nheads=4,
        enc_layers=2,
        dec_layers=1,
        dim_feedforward=1024,
        dropout=0.1,
        chunk_size=10,
        camera_names=["top"],
        state_dim=14,
        num_queries=10,
    )


@pytest.fixture
def act_policy(act_config):
    """Create ACT policy for testing."""
    return ACTPolicy(act_config)


@pytest.fixture
def sample_batch():
    """Create sample batch for testing."""
    return {
        "observation.images.top": torch.randn(2, 3, 128, 128),
        "observation.state": torch.randn(2, 14),
        "actions": torch.randn(2, 10, 7),  # (batch, chunk_size, action_dim)
    }


def test_act_config_validation():
    """Test ACT config validation."""
    # Valid config
    config = ACTConfig(
        input_shapes={"observation.images.top": (3, 128, 128), "observation.state": (14,)},
        output_shapes={"action": (7,)},
        device="cpu",
        camera_names=["top"],
        state_dim=14,
    )
    assert config.device == "cpu"

    # Invalid backbone
    with pytest.raises(ValueError, match="backbone must be one of"):
        ACTConfig(
            input_shapes={},
            output_shapes={"action": (7,)},
            backbone="invalid",
            camera_names=["top"],
        )

    # Invalid hidden_dim/nheads
    with pytest.raises(ValueError, match="must be divisible by"):
        ACTConfig(
            input_shapes={},
            output_shapes={"action": (7,)},
            hidden_dim=100,
            nheads=7,
            camera_names=["top"],
        )

    # Empty camera_names
    with pytest.raises(ValueError, match="must not be empty"):
        ACTConfig(
            input_shapes={},
            output_shapes={"action": (7,)},
            camera_names=[],
        )


def test_act_policy_creation(act_policy):
    """Test ACT policy can be created."""
    assert isinstance(act_policy, ACTPolicy)
    assert act_policy.name == "act"
    assert act_policy.chunk_size == 10


def test_act_policy_registered():
    """Test ACT policy is registered in factory."""
    assert PolicyFactory.is_registered("act")
    assert "act" in PolicyFactory.list_policies()


def test_act_forward(act_policy, sample_batch):
    """Test ACT forward pass."""
    act_policy.train()
    loss, logs = act_policy.forward(sample_batch)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar
    assert isinstance(logs, dict)
    assert "loss" in logs
    assert "mae" in logs


def test_act_select_action(act_policy, sample_batch):
    """Test ACT action selection."""
    act_policy.eval()

    # Remove actions from batch (not needed for inference)
    inference_batch = {
        "observation.images.top": sample_batch["observation.images.top"],
        "observation.state": sample_batch["observation.state"],
    }

    with torch.no_grad():
        # First action selection (predicts full chunk)
        action1 = act_policy.select_action(inference_batch)
        assert action1.shape == (2, 7)  # (batch, action_dim)

        # Second action selection (uses cached chunk)
        action2 = act_policy.select_action(inference_batch)
        assert action2.shape == (2, 7)

        # Actions should be different (from different positions in chunk)
        assert not torch.equal(action1, action2)


def test_act_reset(act_policy, sample_batch):
    """Test ACT reset clears action chunk."""
    act_policy.eval()

    inference_batch = {
        "observation.images.top": sample_batch["observation.images.top"],
        "observation.state": sample_batch["observation.state"],
    }

    with torch.no_grad():
        # Select action (creates chunk)
        action1 = act_policy.select_action(inference_batch)

        # Reset
        act_policy.reset()

        # Select action again (should create new chunk)
        action2 = act_policy.select_action(inference_batch)

        # Actions might be same (both first in chunk), but chunks are different
        assert act_policy.action_index == 1  # Should be at index 1 after one selection


def test_act_get_optim_params(act_policy):
    """Test ACT optimizer parameters."""
    params = act_policy.get_optim_params()

    assert isinstance(params, list)
    assert len(params) > 0
    assert all(isinstance(p, dict) for p in params)


def test_act_encode_observations(act_policy, sample_batch):
    """Test observation encoding."""
    inference_batch = {
        "observation.images.top": sample_batch["observation.images.top"],
        "observation.state": sample_batch["observation.state"],
    }

    features = act_policy.encode_observations(inference_batch)

    assert isinstance(features, torch.Tensor)
    assert features.ndim == 3  # (batch, seq_len, hidden_dim)
    assert features.shape[0] == 2  # batch_size
    assert features.shape[2] == 256  # hidden_dim


def test_act_missing_camera_error(act_policy):
    """Test error when camera observation is missing."""
    bad_batch = {
        "observation.state": torch.randn(2, 14),
        # Missing observation.images.top
    }

    with pytest.raises(ValueError, match="Missing camera observation"):
        act_policy.encode_observations(bad_batch)


def test_act_multiple_cameras():
    """Test ACT with multiple cameras."""
    config = ACTConfig(
        input_shapes={
            "observation.images.top": (3, 128, 128),
            "observation.images.wrist": (3, 128, 128),
            "observation.state": (14,)},
        output_shapes={"action": (7,)},
        device="cpu",
        backbone="resnet18",
        backbone_pretrained=False,
        camera_names=["top", "wrist"],
        state_dim=14,
        chunk_size=10,
    )

    policy = ACTPolicy(config)

    batch = {
        "observation.images.top": torch.randn(2, 3, 128, 128),
        "observation.images.wrist": torch.randn(2, 3, 128, 128),
        "observation.state": torch.randn(2, 14),
    }

    features = policy.encode_observations(batch)
    assert features.shape[0] == 2  # batch_size


def test_act_action_chunking_cycle(act_policy, sample_batch):
    """Test that action chunking cycles through chunk correctly."""
    act_policy.eval()

    inference_batch = {
        "observation.images.top": sample_batch["observation.images.top"],
        "observation.state": sample_batch["observation.state"],
    }

    with torch.no_grad():
        actions = []
        # Select chunk_size actions (should use same chunk)
        for i in range(act_policy.chunk_size):
            action = act_policy.select_action(inference_batch)
            actions.append(action)
            assert act_policy.action_index == i + 1

        # Next action should trigger new chunk prediction
        action_new = act_policy.select_action(inference_batch)
        assert act_policy.action_index == 1  # Reset to 1 after predicting new chunk


def test_act_factory_create(act_config):
    """Test creating ACT policy from factory."""
    policy = PolicyFactory.create("act", act_config)

    assert isinstance(policy, ACTPolicy)
    assert policy.name == "act"
