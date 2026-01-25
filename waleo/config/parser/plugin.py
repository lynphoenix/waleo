"""
插件系统

提供动态加载插件和解析插件参数的功能
"""

import importlib
import pkgutil
from typing import Any, Dict, List, Optional


# 全局插件注册表
_config_registry: Dict[str, type] = {}


def register_config(cls: type) -> type:
    """注册配置类的装饰器

    Args:
        cls: 配置类

    Returns:
        配置类本身

    Examples:
        >>> @register_config
        ... class MyPolicyConfig:
        ...     pass
    """
    _config_registry[cls.__name__] = cls
    return cls


def get_registered_configs() -> Dict[str, type]:
    """获取所有已注册的配置类

    Returns:
        配置类名字典
    """
    return _config_registry.copy()


def unregister_config(cls: type) -> None:
    """注销配置类

    Args:
        cls: 配置类
    """
    if cls.__name__ in _config_registry:
        del _config_registry[cls.__name__]


def load_plugin(plugin_path: str) -> None:
    """加载插件包

    Args:
        plugin_path: 插件模块路径（如 "my_package.custom_plugins"）

    Raises:
        ImportError: 如果插件路径不存在

    Examples:
        >>> load_plugin("my_company.custom_plugins")
    """
    try:
        importlib.import_module(plugin_path)
    except ImportError as e:
        raise ImportError(
            f"Failed to load plugin '{plugin_path}': {e}. " +
            "Make sure the package is installed and the path is correct."
        )


def load_plugins_from_package(package_path: str) -> List[str]:
    """从包中加载所有插件

    Args:
        package_path: 包路径

    Returns:
        加载的插件模块名列表
    """
    try:
        package = importlib.import_module(package_path)
    except ImportError as e:
        raise ImportError(f"Failed to import package '{package_path}': {e}")

    loaded = []

    for _, name, _ in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
        try:
            importlib.import_module(name)
            loaded.append(name)
        except ImportError:
            # 跳过无法导入的模块
            continue

    return loaded


def parse_plugin_args(suffix: str, args: List[str]) -> Dict[str, Any]:
    """解析插件相关参数

    Args:
        suffix: 插件后缀（如 "plugin"）
        args: 命令行参数列表

    Returns:
        插件参数字典

    Examples:
        >>> args = ["--myplugin.option1=value1", "--myplugin.option2=value2"]
        >>> parse_plugin_args("myplugin", args)
        {"option1": "value1", "option2": "value2"}
    """
    from waleo.config.parser.cli import parse_arg_list

    all_args = parse_arg_list(args)
    result = {}

    prefix = f"{suffix}."

    for key, value in all_args.items():
        if key.startswith(prefix):
            # 移除前缀
            new_key = key[len(prefix):]
            result[new_key] = value

    return result


def create_config(class_name: str, **kwargs) -> Any:
    """创建已注册的配置类实例

    Args:
        class_name: 配置类名
        **kwargs: 配置参数

    Returns:
        配置对象实例

    Raises:
        ValueError: 如果配置类未注册
    """
    if class_name not in _config_registry:
        raise ValueError(
            f"Config class '{class_name}' is not registered. " +
            f"Available classes: {list(_config_registry.keys())}"
        )

    cls = _config_registry[class_name]
    return cls(**kwargs)


def list_plugins() -> List[str]:
    """列出所有已注册的配置类

    Returns:
        配置类名列表
    """
    return list(_config_registry.keys())
