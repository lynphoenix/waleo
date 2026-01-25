# Waleo Tools

Waleo 项目的开发工具集合。

## 目录结构

```
tools/
└── mesh_simplification/    # Mesh 简化工具
    ├── __init__.py         # 包初始化
    ├── simplify_meshes.py  # Mesh 简化核心逻辑
    ├── urdf_updater.py     # URDF 文件更新工具
    ├── example_rj2506.py   # RJ2506 优化示例
    └── README.md           # 详细文档
```

## 工具列表

### Mesh Simplification（网格简化）

通用的机器人模型网格简化工具，用于优化仿真训练性能。

**功能**：
- 支持多种 mesh 格式（STL, OBJ, GLB等）
- 多种简化算法（体素聚类、二次误差度量）
- 批量处理整个目录
- 自动更新 URDF 配置文件

**快速使用**：
```python
from tools.mesh_simplification import simplify_directory

simplify_directory(
    source_dir="assets/robots/MyRobot/meshes",
    target_dir="assets/robots/MyRobot_simplified/meshes",
    target_reduction=0.95,  # 减少95%面数
)
```

**文档**：[tools/mesh_simplification/README.md](./mesh_simplification/README.md)

## 添加新工具

如果你开发了新的工具，请：

1. 在 `tools/` 下创建新目录（如 `tools/my_tool/`）
2. 添加 `__init__.py` 和核心代码
3. 创建 `README.md` 说明用途和用法
4. 添加使用示例
5. 更新本文档

## 工具开发指南

### 设计原则

1. **通用性**：工具应该通用，不要硬编码特定项目路径
2. **配置化**：关键参数应该可配置
3. **模块化**：清晰的 API 和职责分离
4. **文档化**：完整的文档和示例
5. **健壮性**：错误处理和输入验证

### 代码规范

- 使用 Type Hints
- 添加 Docstring（Google 风格）
- 提供命令行接口（可选）
- 包含单元测试（推荐）

### 示例模板

```python
#!/usr/bin/env python3
"""工具简短描述

详细说明工具的用途和功能。
"""

from pathlib import Path
from typing import Union, Optional

def my_tool_function(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    option: str = "default",
) -> dict:
    """函数简短描述

    Args:
        input_path: 输入路径
        output_path: 输出路径
        option: 选项说明

    Returns:
        结果字典

    Examples:
        >>> result = my_tool_function("input.txt", "output.txt")
        >>> print(result['status'])
        'success'
    """
    # 实现代码
    pass

if __name__ == "__main__":
    # 命令行接口
    import sys
    if len(sys.argv) < 3:
        print("Usage: python my_tool.py <input> <output>")
        sys.exit(1)

    result = my_tool_function(sys.argv[1], sys.argv[2])
    print(f"完成: {result}")
```

## 许可证

与 Waleo 项目相同
