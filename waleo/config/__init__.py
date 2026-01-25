"""Waleo 配置管理基础设施模块

此模块只提供配置管理的基础设施和工具，不包含具体的领域配置。
各模块的具体配置类应在各自模块中定义。

基础设施包括：
- 配置协议和类型定义
- 配置解析和合并工具
- 配置验证框架
- 通用配置（设备、日志、路径等）

领域配置位置：
- waleo.sim: EnvConfig, CameraConfig
- waleo.training: TrainingConfig (待创建)
- waleo.data: DatasetConfig (待创建)
"""

__version__ = "0.1.0"

# 基础协议和类型
from waleo.config.base import (
    ConfigProtocol,
    FileConfigProtocol,
    FeatureType,
    NormalizationMode,
    PolicyFeature,
    DictLike,
)

# 解析和合并工具
from waleo.config.parser import (
    parse_arg,
    parse_arg_value,
    parse_config,
    parse_arg_list,
    get_nested_attr,
    set_nested_attr,
    get_cli_overrides,
    filter_args,
    merge_configs,
    # 插件系统
    register_config,
    get_registered_configs,
    unregister_config,
    load_plugin,
    load_plugins_from_package,
    parse_plugin_args,
    create_config,
    list_plugins,
)

# 验证工具
from waleo.config.validation import (
    validate_config,
    check_config_consistency,
    validate_all,
    get_validation_errors,
)

# 通用配置（跨模块共享）
from waleo.config.common import (
    DeviceConfig,
    LogConfig,
    PathConfig,
    SeedConfig,
)

__all__ = [
    # 版本
    "__version__",

    # ===== 基础协议和类型 =====
    "ConfigProtocol",
    "FileConfigProtocol",
    "FeatureType",
    "NormalizationMode",
    "PolicyFeature",
    "DictLike",

    # ===== 解析和合并工具 =====
    # CLI 解析
    "parse_arg",
    "parse_arg_value",
    "parse_config",
    "parse_arg_list",
    "get_nested_attr",
    "set_nested_attr",
    "get_cli_overrides",
    "filter_args",
    "merge_configs",

    # 插件系统
    "register_config",
    "get_registered_configs",
    "unregister_config",
    "load_plugin",
    "load_plugins_from_package",
    "parse_plugin_args",
    "create_config",
    "list_plugins",

    # ===== 验证工具 =====
    "validate_config",
    "check_config_consistency",
    "validate_all",
    "get_validation_errors",

    # ===== 通用配置（跨模块共享）=====
    "DeviceConfig",
    "LogConfig",
    "PathConfig",
    "SeedConfig",
]
