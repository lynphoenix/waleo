"""Policy factory for registration and creation."""

from ..base.config import PolicyConfig
from ..base.policy import BasePolicy


class PolicyFactory:
    """Factory for registering and creating policy instances.

    This factory provides a centralized registry for policy types and
    a decorator-based registration mechanism for easy policy registration.

    Example:
        @PolicyFactory.register()
        class MyPolicy(BasePolicy):
            name = "my_policy"
            config_class = MyPolicyConfig
            ...

        # Or with custom name
        @PolicyFactory.register("custom_name")
        class MyPolicy(BasePolicy):
            ...

        # Create policy
        config = MyPolicyConfig(...)
        policy = PolicyFactory.create("my_policy", config)
    """

    # Registry: policy_name -> policy_class
    _registry: dict[str, type[BasePolicy]] = {}

    @classmethod
    def register(cls, name: str | None = None):
        """Register a policy class (decorator).

        Args:
            name: Optional policy name. If not provided, uses the class's `name` attribute.

        Returns:
            Decorator function

        Raises:
            TypeError: If the class is not a subclass of BasePolicy
            ValueError: If the policy name is already registered

        Example:
            @PolicyFactory.register()
            class MyPolicy(BasePolicy):
                name = "my_policy"
                ...

            @PolicyFactory.register("custom_name")
            class AnotherPolicy(BasePolicy):
                ...
        """

        def decorator(policy_cls: type[BasePolicy]):
            # Validate that it's a BasePolicy subclass
            if not issubclass(policy_cls, BasePolicy):
                raise TypeError(
                    f"{policy_cls.__name__} must be a subclass of BasePolicy, "
                    f"got {policy_cls.__bases__}"
                )

            # Get policy name
            policy_name = name if name is not None else policy_cls.name

            # Check if already registered
            if policy_name in cls._registry:
                raise ValueError(
                    f"Policy '{policy_name}' is already registered. "
                    f"Existing class: {cls._registry[policy_name].__name__}"
                )

            # Register
            cls._registry[policy_name] = policy_cls

            return policy_cls

        return decorator

    @classmethod
    def create(cls, name: str, config: PolicyConfig, **kwargs) -> BasePolicy:
        """Create a policy instance.

        Args:
            name: Policy name (must be registered)
            config: Policy configuration
            **kwargs: Additional keyword arguments passed to policy constructor

        Returns:
            Initialized policy instance

        Raises:
            ValueError: If the policy name is not registered
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown policy: '{name}'. Available policies: {available}")

        policy_cls = cls._registry[name]
        return policy_cls(config, **kwargs)

    @classmethod
    def list_policies(cls) -> list[str]:
        """List all registered policy names.

        Returns:
            List of registered policy names
        """
        return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a policy is registered.

        Args:
            name: Policy name

        Returns:
            True if the policy is registered, False otherwise
        """
        return name in cls._registry

    @classmethod
    def get_policy_class(cls, name: str) -> type[BasePolicy]:
        """Get the policy class for a given name.

        Args:
            name: Policy name

        Returns:
            Policy class

        Raises:
            ValueError: If the policy name is not registered
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown policy: '{name}'. Available policies: {available}")

        return cls._registry[name]
