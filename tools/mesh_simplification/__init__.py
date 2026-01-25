"""Mesh Simplification Tools for Robot Models

通用的机器人模型 mesh 简化工具，支持：
- STL/OBJ/GLB 格式的 mesh 简化
- 批量处理整个机器人模型目录
- URDF 文件的自动更新
- 可配置的简化参数

Examples:
    >>> from tools.mesh_simplification import simplify_robot_meshes
    >>> simplify_robot_meshes(
    ...     robot_name="RJ2506",
    ...     source_dir="path/to/meshes",
    ...     target_reduction=0.95,  # 减少95%的面数
    ... )
"""

from .simplify_meshes import (
    simplify_mesh,
    simplify_directory,
    estimate_voxel_size,
)

from .urdf_updater import (
    create_simplified_urdf,
    patch_urdf_mesh_paths,
)

__all__ = [
    "simplify_mesh",
    "simplify_directory",
    "estimate_voxel_size",
    "create_simplified_urdf",
    "patch_urdf_mesh_paths",
]
