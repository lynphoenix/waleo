"""
解析器模块

提供命令行解析和插件系统功能
"""

from waleo_config.parser.cli import (
    parse_arg,
    parse_arg_value,
    parse_config,
    parse_arg_list,
    get_nested_attr,
    set_nested_attr,
    get_cli_overrides,
    filter_args,
    merge_configs,
)

from waleo_config.parser.plugin import (
    register_config,
    get_registered_configs,
    unregister_config,
    load_plugin,
    load_plugins_from_package,
    parse_plugin_args,
    create_config,
    list_plugins,
)

__all__ = [
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
]
