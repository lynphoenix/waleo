"""
命令行参数解析器

提供从命令行解析配置参数的功能
"""

import ast
import re
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, List, Optional, Type, TypeVar


T = TypeVar("T", bound=Any)


def parse_arg(arg_name: str, args: Optional[List[str]] = None) -> Optional[str]:
    """解析单个命令行参数

    Args:
        arg_name: 参数名（不含前缀 --）
        args: 命令行参数列表，默认为 sys.argv[1:]

    Returns:
        参数值字符串，如果未找到则返回 None

    Examples:
        >>> parse_arg("batch_size", ["--batch_size=128"])
        "128"
    """
    if args is None:
        args = sys.argv[1:]

    prefix = f"--{arg_name}="

    for arg in args:
        if arg.startswith(prefix):
            return arg[len(prefix):]

    return None


def parse_arg_value(value_str: str) -> Any:
    """解析参数值字符串

    Args:
        value_str: 值字符串

    Returns:
        解析后的值

    Examples:
        >>> parse_arg_value("128")
        128
        >>> parse_arg_value("[1,2,3]")
        [1, 2, 3]
        >>> parse_arg_value("true")
        True
    """
    # 处理布尔值的特殊情况（JSON 格式）
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False
    if value_str.lower() == "null":
        return None
    
    # 尝试解析为 Python 字面量
    try:
        return ast.literal_eval(value_str)
    except (ValueError, SyntaxError):
        # 如果失败，返回字符串
        return value_str


def get_nested_attr(obj: Any, attr_path: str) -> Any:
    """获取嵌套属性

    Args:
        obj: 对象
        attr_path: 属性路径，使用点号分隔（如 "optimizer.lr"）

    Returns:
        属性值

    Examples:
        >>> config = TrainingConfig(optimizer=OptimizerConfig(lr=0.001))
        >>> get_nested_attr(config, "optimizer.lr")
        0.001
    """
    parts = attr_path.split(".")
    current = obj

    for part in parts:
        if hasattr(current, part):
            current = getattr(current, part)
        else:
            raise AttributeError(f"'{obj}' has no attribute '{attr_path}'")

    return current


def set_nested_attr(obj: Any, attr_path: str, value: Any) -> None:
    """设置嵌套属性

    Args:
        obj: 对象
        attr_path: 属性路径，使用点号分隔（如 "optimizer.lr"）
        value: 要设置的值
    """
    parts = attr_path.split(".")
    current = obj

    # 导航到最后一个属性之前
    for part in parts[:-1]:
        current = getattr(current, part)

    # 设置最后一个属性
    setattr(current, parts[-1], value)


def get_cli_overrides(field_name: str, args: Optional[List[str]] = None) -> List[str]:
    """获取指定字段的所有命令行覆盖参数

    Args:
        field_name: 字段名
        args: 命令行参数列表

    Returns:
        匹配的参数值列表

    Examples:
        >>> args = ["--optimizer.lr=0.001", "--optimizer.type=adamw"]
        >>> get_cli_overrides("optimizer", args)
        ["--optimizer.lr=0.001", "--optimizer.type=adamw"]
    """
    if args is None:
        args = sys.argv[1:]

    prefix = f"--{field_name}"
    result = []

    for arg in args:
        if arg.startswith(prefix):
            # 必须是 --field_name 或 --field_name.xxx 的形式
            if len(arg) == len(prefix) or arg[len(prefix)] == ".":
                result.append(arg)

    return result


def filter_args(fields_to_filter: str | List[str], args: Optional[List[str]] = None) -> List[str]:
    """过滤掉指定字段的参数

    Args:
        fields_to_filter: 要过滤的字段名或字段名列表
        args: 命令行参数列表

    Returns:
        过滤后的参数列表
    """
    if isinstance(fields_to_filter, str):
        fields_to_filter = [fields_to_filter]

    if args is None:
        args = sys.argv[1:]

    result = []
    for arg in args:
        should_filter = False
        for field in fields_to_filter:
            prefix = f"--{field}"
            if arg.startswith(prefix):
                if len(arg) == len(prefix) or arg[len(prefix)] == ".":
                    should_filter = True
                    break

        if not should_filter:
            result.append(arg)

    return result


def parse_config(
    config_cls: Type[T],
    args: Optional[List[str]] = None,
    config_path: Optional[Path | str] = None,
) -> T:
    """解析命令行参数并创建配置对象

    Args:
        config_cls: 配置类（必须是 dataclass）
        args: 命令行参数列表，默认为 sys.argv[1:]
        config_path: 配置文件路径（YAML 或 JSON）

    Returns:
        配置对象实例

    Raises:
        ValueError: 如果参数解析失败

    Examples:
        >>> # 假设命令行: --batch_size=128 --optimizer.lr=0.001
        >>> config = parse_config(TrainingConfig)
        >>> config.batch_size
        128
        >>> config.optimizer.lr
        0.001
    """
    if args is None:
        args = sys.argv[1:]

    # 1. 从文件加载基础配置（如果指定）
    if config_path is not None:
        config_path = Path(config_path)
        if config_path.suffix == ".yaml" or config_path.suffix == ".yml":
            if hasattr(config_cls, "from_yaml"):
                config = config_cls.from_yaml(config_path)
            else:
                raise ValueError(f"{config_cls.__name__} does not support YAML loading")
        elif config_path.suffix == ".json":
            if hasattr(config_cls, "from_json"):
                config = config_cls.from_json(config_path)
            else:
                raise ValueError(f"{config_cls.__name__} does not support JSON loading")
        else:
            # 尝试 from_dict
            import json
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            config = config_cls.from_dict(config_dict)
    else:
        # 使用默认值创建
        config = config_cls()

    # 2. 解析命令行覆盖参数
    for arg in args:
        if not arg.startswith("--"):
            continue

        # 移除 -- 前缀
        arg = arg[2:]

        # 分割参数名和值
        if "=" not in arg:
            continue

        param_name, value_str = arg.split("=", 1)

        try:
            # 解析值
            value = parse_arg_value(value_str)

            # 设置嵌套属性
            set_nested_attr(config, param_name, value)

        except (AttributeError, ValueError) as e:
            raise ValueError(f"Failed to parse parameter '{param_name}={value_str}': {e}")

    # 3. 验证最终配置
    if hasattr(config, "validate"):
        if not config.validate():
            raise ValueError(f"Invalid configuration after applying CLI overrides")

    return config


def parse_arg_list(args: Optional[List[str]] = None) -> dict[str, Any]:
    """将命令行参数解析为字典

    Args:
        args: 命令行参数列表

    Returns:
        参数字典

    Examples:
        >>> parse_arg_list(["--batch_size=128", "--optimizer.lr=0.001"])
        {"batch_size": 128, "optimizer.lr": 0.001}
    """
    if args is None:
        args = sys.argv[1:]

    result = {}

    for arg in args:
        if not arg.startswith("--"):
            continue

        arg = arg[2:]

        if "=" not in arg:
            continue

        param_name, value_str = arg.split("=", 1)
        result[param_name] = parse_arg_value(value_str)

    return result


def merge_configs(base: Any, overrides: dict[str, Any]) -> Any:
    """合并配置

    Args:
        base: 基础配置对象
        overrides: 覆盖值字典

    Returns:
        合并后的配置对象（修改 base 并返回）
    """
    for key, value in overrides.items():
        if "." in key:
            set_nested_attr(base, key, value)
        else:
            setattr(base, key, value)

    return base
