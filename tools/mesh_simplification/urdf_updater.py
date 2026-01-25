#!/usr/bin/env python3
"""URDF 文件更新工具

自动更新 URDF 文件中的 mesh 引用路径。
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_simplified_urdf(
    source_urdf: Union[str, Path],
    output_urdf: Union[str, Path],
    robot_name: str,
    simplified_name: Optional[str] = None,
    visual_mesh_dir: str = "meshes",
    collision_mesh_dir: str = "meshes_collision",
    mesh_format: str = ".glb",
    original_format: str = ".STL",
    verify: bool = True,
) -> Dict[str, int]:
    """创建简化版 URDF 文件

    Args:
        source_urdf: 源 URDF 文件路径
        output_urdf: 输出 URDF 文件路径
        robot_name: 原始机器人名称
        simplified_name: 简化版机器人名称（默认为 {robot_name}_simplified）
        visual_mesh_dir: visual mesh 目录名
        collision_mesh_dir: collision mesh 目录名
        mesh_format: 目标 mesh 格式
        original_format: 原始 mesh 格式
        verify: 是否验证替换结果

    Returns:
        替换统计信息
    """
    source_urdf = Path(source_urdf)
    output_urdf = Path(output_urdf)

    if not source_urdf.exists():
        raise FileNotFoundError(f"Source URDF not found: {source_urdf}")

    if simplified_name is None:
        simplified_name = f"{robot_name}_simplified"

    # 读取源文件
    with open(source_urdf, 'r') as f:
        content = f.read()

    original_content = content

    # 替换函数
    def replace_visual(match):
        block = match.group(0)
        # 替换包路径
        block = block.replace(
            f'package://{robot_name}/meshes/',
            f'package://{simplified_name}/{visual_mesh_dir}/'
        )
        # 替换文件扩展名
        block = block.replace(f'{original_format}" />', f'{mesh_format}" />')
        return block

    def replace_collision(match):
        block = match.group(0)
        # 替换包路径
        block = block.replace(
            f'package://{robot_name}/meshes/',
            f'package://{simplified_name}/{collision_mesh_dir}/'
        )
        # 替换文件扩展名
        block = block.replace(f'{original_format}" />', f'{mesh_format}" />')
        return block

    # 执行替换
    content = re.sub(r'<visual>.*?</visual>', replace_visual, content, flags=re.DOTALL)
    content = re.sub(r'<collision>.*?</collision>', replace_collision, content, flags=re.DOTALL)

    # 创建输出目录
    output_urdf.parent.mkdir(parents=True, exist_ok=True)

    # 写入文件
    with open(output_urdf, 'w') as f:
        f.write(content)

    # 验证
    stats = {
        'visual_refs': content.count(f'{simplified_name}/{visual_mesh_dir}/'),
        'collision_refs': content.count(f'{simplified_name}/{collision_mesh_dir}/'),
        'total_changes': 0,
    }

    if verify:
        original_visual = original_content.count(f'{robot_name}/meshes/')
        current_visual = stats['visual_refs']
        current_collision = stats['collision_refs']

        if original_visual != (current_visual + current_collision):
            logger.warning(
                f"Verification mismatch: original {original_visual} refs, "
                f"now {current_visual} visual + {current_collision} collision"
            )

        stats['total_changes'] = current_visual + current_collision

        logger.info(f"Created simplified URDF: {output_urdf}")
        logger.info(f"Visual mesh refs: {current_visual}")
        logger.info(f"Collision mesh refs: {current_collision}")

    return stats


def patch_urdf_mesh_paths(
    urdf_path: Union[str, Path],
    replacements: Dict[str, str],
    backup: bool = True,
) -> Dict[str, int]:
    """直接修补 URDF 文件中的 mesh 路径

    警告：此函数会直接修改原文件！建议先备份。

    Args:
        urdf_path: URDF 文件路径
        replacements: 替换规则字典 {旧路径: 新路径}
        backup: 是否创建备份

    Returns:
        替换统计信息
    """
    urdf_path = Path(urdf_path)

    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    # 备份
    if backup:
        backup_path = urdf_path.with_suffix(urdf_path.suffix + '.bak')
        import shutil
        shutil.copy2(urdf_path, backup_path)
        logger.info(f"Backup created: {backup_path}")

    # 读取
    with open(urdf_path, 'r') as f:
        content = f.read()

    # 执行替换
    stats = {}
    for old_path, new_path in replacements.items():
        count = content.count(old_path)
        content = content.replace(old_path, new_path)
        stats[old_path] = count

    # 写回
    with open(urdf_path, 'w') as f:
        f.write(content)

    total = sum(stats.values())
    logger.info(f"Patched {total} mesh references in {urdf_path}")

    return stats


def update_urdf_mesh_format(
    urdf_path: Union[str, Path],
    old_format: str = ".STL",
    new_format: str = ".glb",
    backup: bool = True,
) -> int:
    """更新 URDF 中的 mesh 文件格式

    Args:
        urdf_path: URDF 文件路径
        old_format: 旧格式（包含点）
        new_format: 新格式（包含点）
        backup: 是否备份

    Returns:
        替换数量
    """
    replacements = {
        f'{old_format}" />': f'{new_format}" />'
    }
    stats = patch_urdf_mesh_paths(urdf_path, replacements, backup)
    return stats.get(f'{old_format}" />', 0)


def list_urdf_meshes(urdf_path: Union[str, Path]) -> Dict[str, List[str]]:
    """列出 URDF 中引用的所有 mesh 文件

    Args:
        urdf_path: URDF 文件路径

    Returns:
        {'visual': [...], 'collision': [...]}
    """
    urdf_path = Path(urdf_path)

    with open(urdf_path, 'r') as f:
        content = f.read()

    # 提取 mesh 文件名
    visual_meshes = re.findall(
        r'<visual>.*?<mesh\s+filename="([^"]+)".*?</visual>',
        content,
        re.DOTALL
    )

    collision_meshes = re.findall(
        r'<collision>.*?<mesh\s+filename="([^"]+)".*?</collision>',
        content,
        re.DOTALL
    )

    return {
        'visual': visual_meshes,
        'collision': collision_meshes,
    }


if __name__ == "__main__":
    # 简单的命令行接口
    import sys

    if len(sys.argv) < 4:
        print("Usage:")
        print("  Create simplified URDF:")
        print("    python urdf_updater.py create <source.urdf> <output.urdf> <robot_name>")
        print("  List meshes:")
        print("    python urdf_updater.py list <urdf_file>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "create":
        source = sys.argv[2]
        output = sys.argv[3]
        robot_name = sys.argv[4]
        stats = create_simplified_urdf(source, output, robot_name)
        print(f"Created: {output}")
        print(f"Visual refs: {stats['visual_refs']}")
        print(f"Collision refs: {stats['collision_refs']}")

    elif command == "list":
        urdf_path = sys.argv[2]
        meshes = list_urdf_meshes(urdf_path)
        print(f"Visual meshes ({len(meshes['visual'])}):")
        for m in meshes['visual'][:5]:
            print(f"  - {m}")
        if len(meshes['visual']) > 5:
            print(f"  ... and {len(meshes['visual']) - 5} more")

        print(f"\nCollision meshes ({len(meshes['collision'])}):")
        for m in meshes['collision'][:5]:
            print(f"  - {m}")
        if len(meshes['collision']) > 5:
            print(f"  ... and {len(meshes['collision']) - 5} more")