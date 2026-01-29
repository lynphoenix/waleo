"""Tests for PolicyFactory."""

import pytest
import torch.nn as nn

from waleo.policy import BasePolicy, PolicyConfig, PolicyFactory


def test_factory_register_decorator():
    """Test policy registration using decorator."""

    # Create a fresh registry for testing
    original_registry = PolicyFactory._registry.copy()
    PolicyFactory._registry.clear()

    try:

        @PolicyFactory.register()
        class TestPolicy(BasePolicy):
            name = "test_policy"
            config_class = PolicyConfig

            def __init__(self, config):
                super().__init__(config)
                self.net = nn.Linear(10, 6)

            def get_optim_params(self):
                return [{"params": self.parameters()}]

            def reset(self):
                pass

            def forward(self, batch):
                return torch.tensor(0.0), {}

            def select_action(self, batch):
                return self.net(batch["observation.state"])

        # Check registration
        assert PolicyFactory.is_registered("test_policy")
        assert "test_policy" in PolicyFactory.list_policies()

    finally:
        # Restore original registry
        PolicyFactory._registry = original_registry


def test_factory_register_custom_name():
    """Test policy registration with custom name."""

    original_registry = PolicyFactory._registry.copy()
    PolicyFactory._registry.clear()

    try:

        @PolicyFactory.register("custom_name")
        class TestPolicy(BasePolicy):
            name = "test_policy"
            config_class = PolicyConfig

            def __init__(self, config):
                super().__init__(config)
                self.net = nn.Linear(10, 6)

            def get_optim_params(self):
                return [{"params": self.parameters()}]

            def reset(self):
                pass

            def forward(self, batch):
                return torch.tensor(0.0), {}

            def select_action(self, batch):
                return self.net(batch["observation.state"])

        # Check registration with custom name
        assert PolicyFactory.is_registered("custom_name")
        assert "custom_name" in PolicyFactory.list_policies()

    finally:
        PolicyFactory._registry = original_registry


def test_factory_create_policy():
    """Test policy creation from factory."""

    original_registry = PolicyFactory._registry.copy()
    PolicyFactory._registry.clear()

    try:

        @PolicyFactory.register()
        class TestPolicy(BasePolicy):
            name = "test_policy"
            config_class = PolicyConfig

            def __init__(self, config):
                super().__init__(config)
                self.net = nn.Linear(10, 6)

            def get_optim_params(self):
                return [{"params": self.parameters()}]

            def reset(self):
                pass

            def forward(self, batch):
                return torch.tensor(0.0), {}

            def select_action(self, batch):
                return self.net(batch["observation.state"])

        # Create policy
        config = PolicyConfig(
            name="test_policy",
            input_shapes={"observation.state": (10,)},
            output_shapes={"action": (6,)},
            device="cpu",
        )
        policy = PolicyFactory.create("test_policy", config)

        assert isinstance(policy, TestPolicy)
        assert isinstance(policy, BasePolicy)

    finally:
        PolicyFactory._registry = original_registry


def test_factory_create_unknown_policy():
    """Test creating unknown policy raises error."""

    original_registry = PolicyFactory._registry.copy()
    PolicyFactory._registry.clear()

    try:
        config = PolicyConfig(device="cpu")

        with pytest.raises(ValueError, match="Unknown policy"):
            PolicyFactory.create("nonexistent_policy", config)

    finally:
        PolicyFactory._registry = original_registry


def test_factory_list_policies():
    """Test listing registered policies."""

    original_registry = PolicyFactory._registry.copy()
    PolicyFactory._registry.clear()

    try:

        @PolicyFactory.register()
        class Policy1(BasePolicy):
            name = "policy1"
            config_class = PolicyConfig

            def __init__(self, config):
                super().__init__(config)

            def get_optim_params(self):
                return []

            def reset(self):
                pass

            def forward(self, batch):
                return torch.tensor(0.0), {}

            def select_action(self, batch):
                return torch.zeros(1, 6)

        @PolicyFactory.register()
        class Policy2(BasePolicy):
            name = "policy2"
            config_class = PolicyConfig

            def __init__(self, config):
                super().__init__(config)

            def get_optim_params(self):
                return []

            def reset(self):
                pass

            def forward(self, batch):
                return torch.tensor(0.0), {}

            def select_action(self, batch):
                return torch.zeros(1, 6)

        policies = PolicyFactory.list_policies()
        assert len(policies) == 2
        assert "policy1" in policies
        assert "policy2" in policies

    finally:
        PolicyFactory._registry = original_registry


def test_factory_is_registered():
    """Test checking if policy is registered."""

    original_registry = PolicyFactory._registry.copy()
    PolicyFactory._registry.clear()

    try:

        @PolicyFactory.register()
        class TestPolicy(BasePolicy):
            name = "test_policy"
            config_class = PolicyConfig

            def __init__(self, config):
                super().__init__(config)

            def get_optim_params(self):
                return []

            def reset(self):
                pass

            def forward(self, batch):
                return torch.tensor(0.0), {}

            def select_action(self, batch):
                return torch.zeros(1, 6)

        assert PolicyFactory.is_registered("test_policy")
        assert not PolicyFactory.is_registered("nonexistent")

    finally:
        PolicyFactory._registry = original_registry


def test_factory_get_policy_class():
    """Test getting policy class from factory."""

    original_registry = PolicyFactory._registry.copy()
    PolicyFactory._registry.clear()

    try:

        @PolicyFactory.register()
        class TestPolicy(BasePolicy):
            name = "test_policy"
            config_class = PolicyConfig

            def __init__(self, config):
                super().__init__(config)

            def get_optim_params(self):
                return []

            def reset(self):
                pass

            def forward(self, batch):
                return torch.tensor(0.0), {}

            def select_action(self, batch):
                return torch.zeros(1, 6)

        policy_cls = PolicyFactory.get_policy_class("test_policy")
        assert policy_cls is TestPolicy

        with pytest.raises(ValueError, match="Unknown policy"):
            PolicyFactory.get_policy_class("nonexistent")

    finally:
        PolicyFactory._registry = original_registry


def test_factory_register_non_basepolicy():
    """Test that registering non-BasePolicy raises error."""

    original_registry = PolicyFactory._registry.copy()
    PolicyFactory._registry.clear()

    try:

        class NotAPolicy:
            pass

        with pytest.raises(TypeError, match="must be a subclass of BasePolicy"):

            @PolicyFactory.register()
            class BadPolicy(NotAPolicy):  # type: ignore
                pass

    finally:
        PolicyFactory._registry = original_registry


def test_factory_duplicate_registration():
    """Test that duplicate registration raises error."""

    original_registry = PolicyFactory._registry.copy()
    PolicyFactory._registry.clear()

    try:

        @PolicyFactory.register()
        class TestPolicy1(BasePolicy):
            name = "test_policy"
            config_class = PolicyConfig

            def __init__(self, config):
                super().__init__(config)

            def get_optim_params(self):
                return []

            def reset(self):
                pass

            def forward(self, batch):
                return torch.tensor(0.0), {}

            def select_action(self, batch):
                return torch.zeros(1, 6)

        # Try to register another policy with same name
        with pytest.raises(ValueError, match="already registered"):

            @PolicyFactory.register()
            class TestPolicy2(BasePolicy):
                name = "test_policy"
                config_class = PolicyConfig

                def __init__(self, config):
                    super().__init__(config)

                def get_optim_params(self):
                    return []

                def reset(self):
                    pass

                def forward(self, batch):
                    return torch.tensor(0.0), {}

                def select_action(self, batch):
                    return torch.zeros(1, 6)

    finally:
        PolicyFactory._registry = original_registry
