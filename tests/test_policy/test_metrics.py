"""Tests for policy evaluation metrics."""

import numpy as np
import pytest

from waleo.policy.utils import (
    compute_average_return,
    compute_max_return,
    compute_metrics,
    compute_min_return,
    compute_std_return,
    compute_success_rate,
)


def test_compute_success_rate():
    """Test success rate computation."""
    episode_returns = [1.0, 2.0, -1.0, 3.0, 0.5]

    # All positive returns succeed (threshold=0)
    success_rate = compute_success_rate(episode_returns, success_threshold=0.0)
    assert success_rate == 0.8  # 4 out of 5

    # Higher threshold
    success_rate = compute_success_rate(episode_returns, success_threshold=1.0)
    assert success_rate == 0.4  # 2 out of 5 (2.0 and 3.0)


def test_compute_success_rate_numpy():
    """Test success rate with numpy array."""
    episode_returns = np.array([1.0, 2.0, -1.0, 3.0, 0.5])
    success_rate = compute_success_rate(episode_returns, success_threshold=0.0)
    assert success_rate == 0.8


def test_compute_average_return():
    """Test average return computation."""
    episode_returns = [1.0, 2.0, 3.0, 4.0, 5.0]
    avg_return = compute_average_return(episode_returns)
    assert avg_return == 3.0


def test_compute_average_return_numpy():
    """Test average return with numpy array."""
    episode_returns = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    avg_return = compute_average_return(episode_returns)
    assert avg_return == 3.0


def test_compute_std_return():
    """Test standard deviation computation."""
    episode_returns = [1.0, 2.0, 3.0, 4.0, 5.0]
    std_return = compute_std_return(episode_returns)
    expected_std = np.std([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(std_return - expected_std) < 1e-6


def test_compute_min_return():
    """Test minimum return computation."""
    episode_returns = [1.0, 2.0, -1.0, 3.0, 0.5]
    min_return = compute_min_return(episode_returns)
    assert min_return == -1.0


def test_compute_max_return():
    """Test maximum return computation."""
    episode_returns = [1.0, 2.0, -1.0, 3.0, 0.5]
    max_return = compute_max_return(episode_returns)
    assert max_return == 3.0


def test_compute_metrics():
    """Test comprehensive metrics computation."""
    episode_returns = [1.0, 2.0, -1.0, 3.0, 0.5]

    metrics = compute_metrics(episode_returns, success_threshold=0.0)

    assert "success_rate" in metrics
    assert "average_return" in metrics
    assert "std_return" in metrics
    assert "min_return" in metrics
    assert "max_return" in metrics

    assert metrics["success_rate"] == 0.8
    assert abs(metrics["average_return"] - 1.1) < 1e-6
    assert metrics["min_return"] == -1.0
    assert metrics["max_return"] == 3.0
    assert metrics["std_return"] > 0


def test_compute_metrics_all_success():
    """Test metrics when all episodes succeed."""
    episode_returns = [1.0, 2.0, 3.0, 4.0, 5.0]

    metrics = compute_metrics(episode_returns, success_threshold=0.0)

    assert metrics["success_rate"] == 1.0
    assert metrics["average_return"] == 3.0
    assert metrics["min_return"] == 1.0
    assert metrics["max_return"] == 5.0


def test_compute_metrics_all_fail():
    """Test metrics when all episodes fail."""
    episode_returns = [-1.0, -2.0, -3.0, -4.0, -5.0]

    metrics = compute_metrics(episode_returns, success_threshold=0.0)

    assert metrics["success_rate"] == 0.0
    assert metrics["average_return"] == -3.0
    assert metrics["min_return"] == -5.0
    assert metrics["max_return"] == -1.0


def test_compute_metrics_single_episode():
    """Test metrics with single episode."""
    episode_returns = [2.5]

    metrics = compute_metrics(episode_returns, success_threshold=0.0)

    assert metrics["success_rate"] == 1.0
    assert metrics["average_return"] == 2.5
    assert metrics["std_return"] == 0.0
    assert metrics["min_return"] == 2.5
    assert metrics["max_return"] == 2.5


def test_compute_metrics_types():
    """Test that all metrics return float type."""
    episode_returns = [1.0, 2.0, 3.0]

    metrics = compute_metrics(episode_returns)

    for key, value in metrics.items():
        assert isinstance(value, float), f"Metric {key} should be float, got {type(value)}"
