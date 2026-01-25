"""
JSON I/O 工具

提供 JSON 文件读写功能，支持复杂数据类型
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Union

try:
    import orjson
    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False


def save_json(
    data: Any,
    path: Union[str, Path],
    indent: int = 2,
    use_orjson: bool = True,
) -> None:
    """保存数据到 JSON 文件

    Args:
        data: 要保存的数据（dict, list 等）
        path: 保存路径
        indent: 缩进空格数
        use_orjson: 是否使用 orjson（更快）

    Raises:
        TypeError: 如果数据类型无法序列化
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if use_orjson and ORJSON_AVAILABLE:
        # orjson 默认不保留格式，使用紧凑模式
        options = orjson.OPT_INDENT_2 if indent > 0 else 0
        with open(path, "wb") as f:
            f.write(orjson.dumps(data, option=options))
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(
    path: Union[str, Path],
    use_orjson: bool = True,
) -> Any:
    """从 JSON 文件加载数据

    Args:
        path: 文件路径
        use_orjson: 是否使用 orjson（更快）

    Returns:
        Any: 加载的数据

    Raises:
        FileNotFoundError: 如果文件不存在
        json.JSONDecodeError: 如果 JSON 格式无效
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    if use_orjson and ORJSON_AVAILABLE:
        with open(path, "rb") as f:
            return orjson.loads(f.read())
    else:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


class JSONEncoder(json.JSONEncoder):
    """自定义 JSON 编码器

    支持更多数据类型的序列化
    """

    def default(self, obj: Any) -> Any:
        """处理默认无法序列化的对象

        Args:
            obj: 要序列化的对象

        Returns:
            Any: 可序列化的表示
        """
        # 处理 Path 对象
        if isinstance(obj, Path):
            return str(obj)

        # 处理 numpy 数组
        if hasattr(obj, "tolist"):
            return obj.tolist()

        # 处理枚举类型
        if hasattr(obj, "value"):
            return obj.value

        # 处理带有 to_dict 方法的对象
        if hasattr(obj, "to_dict"):
            return obj.to_dict()

        # 尝试转换为字符串
        try:
            return str(obj)
        except Exception:
            return super().default(obj)


def save_json_custom(
    data: Any,
    path: Union[str, Path],
    indent: int = 2,
) -> None:
    """使用自定义编码器保存 JSON

    Args:
        data: 要保存的数据
        path: 保存路径
        indent: 缩进空格数
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, cls=JSONEncoder, ensure_ascii=False)


def merge_json_files(
    base_path: Union[str, Path],
    override_path: Union[str, Path],
    output_path: Union[str, Path],
) -> None:
    """合并两个 JSON 文件

    Args:
        base_path: 基础 JSON 文件路径
        override_path: 覆盖 JSON 文件路径
        output_path: 输出文件路径
    """
    base = load_json(base_path)
    override = load_json(override_path)

    if isinstance(base, dict) and isinstance(override, dict):
        merged = {**base, **override}
    elif isinstance(base, list) and isinstance(override, list):
        merged = base + override
    else:
        merged = override

    save_json(merged, output_path)


def update_json(
    path: Union[str, Path],
    updates: Dict[str, Any],
    create: bool = False,
) -> None:
    """更新 JSON 文件中的字段

    Args:
        path: JSON 文件路径
        updates: 要更新的字段字典
        create: 如果文件不存在是否创建

    Raises:
        FileNotFoundError: 如果文件不存在且 create=False
    """
    path = Path(path)

    if path.exists():
        data = load_json(path)
        if not isinstance(data, dict):
            raise ValueError(f"JSON file does not contain a dict: {path}")
        data.update(updates)
    else:
        if not create:
            raise FileNotFoundError(f"JSON file not found: {path}")
        data = updates

    save_json(data, path)


def get_json_value(
    path: Union[str, Path],
    key: str,
    default: Any = None,
) -> Any:
    """从 JSON 文件获取单个值

    Args:
        path: JSON 文件路径
        key: 键名（支持嵌套，用 . 分隔）
        default: 默认值

    Returns:
        Any: 获取的值

    Example:
        value = get_json_value("config.json", "model.learning_rate")
    """
    data = load_json(path)

    # 处理嵌套键
    keys = key.split(".")
    value = data

    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default

    return value
