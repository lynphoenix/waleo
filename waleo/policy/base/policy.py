"""Base policy abstract class."""

from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn

from .config import PolicyConfig


class BasePolicy(nn.Module, ABC):
    """Abstract base class for all policies.

    All policy implementations must inherit from this class and implement
    the abstract methods: get_optim_params, reset, forward, and select_action.

    Attributes:
        config_class: Policy-specific configuration class (must be set by subclass)
        name: Policy name (must be set by subclass)
        config: Policy configuration instance
        device: PyTorch device for computation
    """

    config_class: type[PolicyConfig]  # Must be set by subclass
    name: str  # Must be set by subclass

    def __init__(self, config: PolicyConfig):
        """Initialize base policy.

        Args:
            config: Policy configuration
        """
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

    @abstractmethod
    def get_optim_params(self) -> list[dict]:
        """Return optimizer parameter groups.

        This method should return a list of parameter dictionaries that can be
        passed to a PyTorch optimizer. This allows for different learning rates
        or weight decay values for different parts of the network.

        Returns:
            List of parameter group dictionaries

        Example:
            return [
                {"params": self.encoder.parameters(), "lr": 1e-4},
                {"params": self.decoder.parameters(), "lr": 3e-4},
            ]
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset policy state.

        This method should reset any internal state of the policy, such as
        hidden states in recurrent networks or action buffers in temporal
        policies. Called at the beginning of each episode during evaluation.
        """
        pass

    @abstractmethod
    def forward(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """Training forward pass.

        Args:
            batch: Dictionary containing batch data with keys like:
                - "observation.state": robot state tensor
                - "observation.image": image tensor (if applicable)
                - "action": ground truth action tensor
                - etc.

        Returns:
            Tuple of (loss, logs) where:
                - loss: Scalar loss tensor for backpropagation
                - logs: Dictionary of logging metrics (e.g., {"loss": float, "accuracy": float})
        """
        pass

    @abstractmethod
    def select_action(self, batch: dict) -> torch.Tensor:
        """Inference action selection.

        Args:
            batch: Dictionary containing batch data with keys like:
                - "observation.state": robot state tensor
                - "observation.image": image tensor (if applicable)

        Returns:
            Action tensor of shape (batch_size, action_dim)
        """
        pass

    def save(self, path: Path | str):
        """Save policy checkpoint.

        Args:
            path: Directory path to save checkpoint
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.config,
        }

        torch.save(checkpoint, path / "policy.pth")

    @classmethod
    def load(cls, path: Path | str):
        """Load policy from checkpoint.

        Args:
            path: Directory path containing checkpoint

        Returns:
            Loaded policy instance in eval mode
        """
        path = Path(path)
        checkpoint = torch.load(path / "policy.pth", map_location="cpu", weights_only=False)

        config = checkpoint["config"]
        policy = cls(config)
        policy.load_state_dict(checkpoint["model_state_dict"])
        policy.eval()

        return policy

    def eval_action(self, batch: dict) -> dict:
        """Evaluate action with metadata.

        This method wraps select_action and returns additional metadata
        useful for evaluation and debugging.

        Args:
            batch: Dictionary containing batch data

        Returns:
            Dictionary containing:
                - "action": Action as numpy array
                - "success": Whether action selection succeeded
                - Additional policy-specific metadata
        """
        with torch.no_grad():
            action = self.select_action(batch)

        return {
            "action": action.cpu().numpy(),
            "success": True,
        }

    def to(self, device: torch.device | str):
        """Move policy to device and update config.

        Args:
            device: Target device

        Returns:
            Self for chaining
        """
        super().to(device)
        self.device = torch.device(device)
        self.config.device = str(device)
        return self
