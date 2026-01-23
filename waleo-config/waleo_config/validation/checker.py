"""
配置验证模块

提供配置对象验证和一致性检查功能
"""

from typing import Any, Dict, List


def validate_config(config: Any) -> bool:
    """验证单个配置对象

    Args:
        config: 配置对象

    Returns:
        是否验证通过

    Examples:
        >>> config = TrainingConfig(batch_size=32)
        >>> validate_config(config)
        True
        >>> config = TrainingConfig(batch_size=-1)
        >>> validate_config(config)
        False
    """
    # 如果配置类有 validate 方法，使用它
    if hasattr(config, "validate"):
        try:
            return config.validate()
        except Exception:
            return False

    # 否则，检查是否有 __post_init__ 方法
    if hasattr(config, "__post_init__"):
        try:
            # 重新触发验证
            config.__post_init__()
            return True
        except Exception:
            return False

    # 没有验证方法，假设配置有效
    return True


def check_config_consistency(configs: Dict[str, Any]) -> List[str]:
    """检查多个配置之间的一致性

    Args:
        configs: 配置名字典

    Returns:
        问题列表，如果没有问题则返回空列表

    Examples:
        >>> train_cfg = TrainingConfig(batch_size=32)
        >>> eval_cfg = EvalConfig(n_episodes=50)
        >>> issues = check_config_consistency({
        ...     "training": train_cfg,
        ...     "eval": eval_cfg
        ... })
        >>> issues
        []
    """
    issues = []

    # 检查训练和评估配置的兼容性
    if "training" in configs and "eval" in configs:
        train_cfg = configs["training"]
        eval_cfg = configs["eval"]

        # 检查 eval_freq 和 n_episodes 的关系
        if hasattr(train_cfg, "eval_freq") and hasattr(eval_cfg, "n_episodes"):
            if train_cfg.eval_freq > 0 and eval_cfg.n_episodes == 0:
                issues.append(
                    "Training eval_freq is set but eval n_episodes is 0. " +
                    "Evaluation will not run."
                )

    # 检查数据集和训练配置的兼容性
    if "dataset" in configs and "training" in configs:
        dataset_cfg = configs["dataset"]
        train_cfg = configs["training"]

        # 检查 batch_size 和 num_workers 的关系
        if hasattr(train_cfg, "batch_size") and hasattr(train_cfg, "num_workers"):
            if train_cfg.num_workers > train_cfg.batch_size:
                issues.append(
                    f"Training num_workers ({train_cfg.num_workers}) > " +
                    f"batch_size ({train_cfg.batch_size}). Some workers will be unused."
                )

    # 检查设备和 batch_size 的关系
    if "training" in configs:
        train_cfg = configs["training"]
        if hasattr(train_cfg, "device") and hasattr(train_cfg, "batch_size"):
            if train_cfg.device == "cpu" and train_cfg.batch_size > 64:
                issues.append(
                    f"Large batch_size ({train_cfg.batch_size}) on CPU may be slow. " +
                    "Consider reducing batch_size or using GPU."
                )

    return issues


def validate_all(configs: Dict[str, Any]) -> bool:
    """验证所有配置对象及其一致性

    Args:
        configs: 配置名字典

    Returns:
        是否全部验证通过
    """
    # 验证单个配置
    for name, config in configs.items():
        if not validate_config(config):
            return False

    # 检查一致性
    issues = check_config_consistency(configs)
    return len(issues) == 0


def get_validation_errors(config: Any) -> List[str]:
    """获取配置对象的验证错误信息

    Args:
        config: 配置对象

    Returns:
        错误信息列表
    """
    errors = []

    if hasattr(config, "validate"):
        if not config.validate():
            # 尝试调用 __post_init__ 获取详细错误
            try:
                config.__post_init__()
            except Exception as e:
                errors.append(str(e))
            else:
                errors.append(f"{config.__class__.__name__} validation failed")
    elif hasattr(config, "__post_init__"):
        try:
            config.__post_init__()
        except Exception as e:
            errors.append(str(e))

    return errors
