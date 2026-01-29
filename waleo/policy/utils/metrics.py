"""Policy evaluation metrics."""

import numpy as np


def compute_success_rate(
    episode_returns: list[float] | np.ndarray, success_threshold: float = 0.0
) -> float:
    """Compute success rate from episode returns.

    An episode is considered successful if its return exceeds the success threshold.

    Args:
        episode_returns: List or array of episode returns
        success_threshold: Threshold for considering an episode successful

    Returns:
        Success rate as a float in [0, 1]
    """
    episode_returns = np.array(episode_returns)
    successes = episode_returns > success_threshold
    return float(np.mean(successes))


def compute_average_return(episode_returns: list[float] | np.ndarray) -> float:
    """Compute average episode return.

    Args:
        episode_returns: List or array of episode returns

    Returns:
        Average return as a float
    """
    return float(np.mean(episode_returns))


def compute_std_return(episode_returns: list[float] | np.ndarray) -> float:
    """Compute standard deviation of episode returns.

    Args:
        episode_returns: List or array of episode returns

    Returns:
        Standard deviation of returns as a float
    """
    return float(np.std(episode_returns))


def compute_min_return(episode_returns: list[float] | np.ndarray) -> float:
    """Compute minimum episode return.

    Args:
        episode_returns: List or array of episode returns

    Returns:
        Minimum return as a float
    """
    return float(np.min(episode_returns))


def compute_max_return(episode_returns: list[float] | np.ndarray) -> float:
    """Compute maximum episode return.

    Args:
        episode_returns: List or array of episode returns

    Returns:
        Maximum return as a float
    """
    return float(np.max(episode_returns))


def compute_metrics(
    episode_returns: list[float] | np.ndarray, success_threshold: float = 0.0
) -> dict[str, float]:
    """Compute all evaluation metrics from episode returns.

    Args:
        episode_returns: List or array of episode returns
        success_threshold: Threshold for considering an episode successful

    Returns:
        Dictionary containing:
            - success_rate: Fraction of episodes exceeding success_threshold
            - average_return: Mean episode return
            - std_return: Standard deviation of episode returns
            - min_return: Minimum episode return
            - max_return: Maximum episode return
    """
    return {
        "success_rate": compute_success_rate(episode_returns, success_threshold),
        "average_return": compute_average_return(episode_returns),
        "std_return": compute_std_return(episode_returns),
        "min_return": compute_min_return(episode_returns),
        "max_return": compute_max_return(episode_returns),
    }
