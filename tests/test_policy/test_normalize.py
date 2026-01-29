"""Tests for normalization utilities."""

import numpy as np
import pytest
import torch

from waleo.policy import Normalize, NormalizationMode, Unnormalize


def test_normalize_creation(mock_policy_config, mock_stats):
    """Test Normalize module creation."""
    normalize = Normalize(
        input_shapes=mock_policy_config.input_shapes,
        normalization_mapping=mock_policy_config.normalization_mapping,
        stats=mock_stats,
    )

    assert isinstance(normalize, Normalize)
    assert normalize.input_shapes == mock_policy_config.input_shapes


def test_normalize_mean_std(mock_stats):
    """Test MEAN_STD normalization."""
    # Create stats with known values
    stats = {
        "observation.state": {
            "mean": np.array([5.0] * 10),
            "std": np.array([2.0] * 10),
        }
    }

    normalize = Normalize(
        input_shapes={"observation.state": (10,)},
        normalization_mapping={"observation.state": NormalizationMode.MEAN_STD},
        stats=stats,
    )

    # Test normalization
    batch = {"observation.state": torch.tensor([[5.0] * 10])}
    normalized = normalize(batch)

    # Should be close to zero (mean-centered)
    assert torch.allclose(normalized["observation.state"], torch.zeros(1, 10), atol=1e-5)


def test_normalize_identity():
    """Test IDENTITY normalization (no-op)."""
    normalize = Normalize(
        input_shapes={"observation.state": (10,)},
        normalization_mapping={"observation.state": NormalizationMode.IDENTITY},
        stats=None,
    )

    batch = {"observation.state": torch.randn(4, 10)}
    normalized = normalize(batch)

    # Should be unchanged
    assert torch.equal(batch["observation.state"], normalized["observation.state"])


def test_normalize_mixed_keys(mock_stats):
    """Test normalization with mixed normalized/unnormalized keys."""
    normalize = Normalize(
        input_shapes={"observation.state": (10,), "observation.image": (3, 64, 64)},
        normalization_mapping={"observation.state": NormalizationMode.MEAN_STD},
        stats=mock_stats,
    )

    batch = {
        "observation.state": torch.randn(4, 10),
        "observation.image": torch.randn(4, 3, 64, 64),
    }
    normalized = normalize(batch)

    # State should be normalized (different)
    assert not torch.equal(batch["observation.state"], normalized["observation.state"])

    # Image should be unchanged
    assert torch.equal(batch["observation.image"], normalized["observation.image"])


def test_unnormalize_creation(mock_policy_config, mock_stats):
    """Test Unnormalize module creation."""
    unnormalize = Unnormalize(
        output_shapes=mock_policy_config.output_shapes,
        normalization_mapping={"action": NormalizationMode.MEAN_STD},
        stats=mock_stats,
    )

    assert isinstance(unnormalize, Unnormalize)
    assert unnormalize.output_shapes == mock_policy_config.output_shapes


def test_unnormalize_mean_std():
    """Test MEAN_STD unnormalization."""
    # Create stats with known values
    stats = {
        "action": {
            "mean": np.array([5.0] * 6),
            "std": np.array([2.0] * 6),
        }
    }

    unnormalize = Unnormalize(
        output_shapes={"action": (6,)},
        normalization_mapping={"action": NormalizationMode.MEAN_STD},
        stats=stats,
    )

    # Test unnormalization of zero (should become mean)
    batch = {"action": torch.zeros(1, 6)}
    unnormalized = unnormalize(batch)

    # Should be close to mean
    expected = torch.tensor([[5.0] * 6])
    assert torch.allclose(unnormalized["action"], expected, atol=1e-5)


def test_normalize_unnormalize_reversibility(mock_stats):
    """Test that normalize and unnormalize are reversible."""
    normalize = Normalize(
        input_shapes={"observation.state": (10,)},
        normalization_mapping={"observation.state": NormalizationMode.MEAN_STD},
        stats=mock_stats,
    )

    unnormalize = Unnormalize(
        output_shapes={"observation.state": (10,)},
        normalization_mapping={"observation.state": NormalizationMode.MEAN_STD},
        stats=mock_stats,
    )

    # Original data
    original = {"observation.state": torch.randn(4, 10)}

    # Normalize then unnormalize
    normalized = normalize(original)
    restored = unnormalize(normalized)

    # Should be close to original
    assert torch.allclose(original["observation.state"], restored["observation.state"], atol=1e-5)


def test_normalize_device_movement(mock_stats):
    """Test that normalization works with device movement."""
    normalize = Normalize(
        input_shapes={"observation.state": (10,)},
        normalization_mapping={"observation.state": NormalizationMode.MEAN_STD},
        stats=mock_stats,
    )

    # Test on CPU
    batch = {"observation.state": torch.randn(4, 10)}
    normalized = normalize(batch)
    assert normalized["observation.state"].device == torch.device("cpu")


def test_normalize_without_stats():
    """Test normalization without stats (should handle gracefully)."""
    normalize = Normalize(
        input_shapes={"observation.state": (10,)},
        normalization_mapping={"observation.state": NormalizationMode.MEAN_STD},
        stats=None,
    )

    batch = {"observation.state": torch.randn(4, 10)}
    # Should not raise, but may not normalize properly without stats
    normalized = normalize(batch)
    assert "observation.state" in normalized
