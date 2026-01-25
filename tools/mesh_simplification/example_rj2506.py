#!/usr/bin/env python3
"""
RJ2506 Mesh 优化示例

演示如何使用 mesh_simplification 工具优化 RJ2506 机器人模型。
这是之前临时脚本的重构版本。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.mesh_simplification import (
    simplify_directory,
    create_simplified_urdf,
)


def optimize_rj2506(
    source_robot_dir: str,
    target_robot_dir: str,
    visual_reduction: float = 0.95,
    collision_reduction: float = 0.93,
):
    """优化 RJ2506 机器人模型

    Args:
        source_robot_dir: 源机器人目录（如 RJ2506/）
        target_robot_dir: 目标目录（如 RJ2506_simplified/）
        visual_reduction: visual mesh 减少比例
        collision_reduction: collision mesh 减少比例
    """
    source_dir = Path(source_robot_dir)
    target_dir = Path(target_robot_dir)

    print("=" * 70)
    print("RJ2506 机器人模型优化")
    print("=" * 70)
    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_dir}")
    print(f"Visual 减少: {visual_reduction*100:.0f}%")
    print(f"Collision 减少: {collision_reduction*100:.0f}%")
    print("")

    # 1. 简化 visual meshes
    print("步骤 1/3: 简化 visual meshes...")
    print("-" * 70)
    visual_stats = simplify_directory(
        source_dir=source_dir / "meshes",
        target_dir=target_dir / "meshes",
        pattern="*.STL",
        output_format=".glb",
        target_reduction=visual_reduction,
        method="vertex_clustering",
    )

    # 2. 简化 collision meshes
    print("\n步骤 2/3: 简化 collision meshes...")
    print("-" * 70)
    collision_stats = simplify_directory(
        source_dir=source_dir / "meshes",
        target_dir=target_dir / "meshes_collision",
        pattern="*.STL",
        output_format=".glb",
        target_reduction=collision_reduction,
        method="vertex_clustering",
    )

    # 3. 创建简化版 URDF
    print("\n步骤 3/3: 创建简化版 URDF...")
    print("-" * 70)

    # 查找所有 URDF 文件
    urdf_files = list((source_dir / "urdf").glob("*.urdf"))
    if not urdf_files:
        print("⚠️  未找到 URDF 文件，跳过...")
    else:
        for urdf_file in urdf_files:
            output_urdf = target_dir / "urdf" / urdf_file.name
            urdf_stats = create_simplified_urdf(
                source_urdf=urdf_file,
                output_urdf=output_urdf,
                robot_name=source_dir.name,
                simplified_name=target_dir.name,
            )
            print(f"  创建: {output_urdf.name}")
            print(f"    - Visual refs: {urdf_stats['visual_refs']}")
            print(f"    - Collision refs: {urdf_stats['collision_refs']}")

    # 总结
    print("\n" + "=" * 70)
    print("✅ 优化完成！")
    print("=" * 70)

    total_visual_before = sum(s['original_faces'] for s in visual_stats.values() if 'original_faces' in s)
    total_visual_after = sum(s['simplified_faces'] for s in visual_stats.values() if 'simplified_faces' in s)
    total_collision_before = sum(s['original_faces'] for s in collision_stats.values() if 'original_faces' in s)
    total_collision_after = sum(s['simplified_faces'] for s in collision_stats.values() if 'simplified_faces' in s)

    print(f"\nVisual meshes:")
    print(f"  面数: {total_visual_before:,} -> {total_visual_after:,}")
    if total_visual_before > 0:
        print(f"  减少: {(1 - total_visual_after/total_visual_before)*100:.1f}%")

    print(f"\nCollision meshes:")
    print(f"  面数: {total_collision_before:,} -> {total_collision_after:,}")
    if total_collision_before > 0:
        print(f"  减少: {(1 - total_collision_after/total_collision_before)*100:.1f}%")

    print(f"\n📁 输出目录: {target_dir}")
    print("")


if __name__ == "__main__":
    # 示例：优化 ManiSkill 中的 RJ2506
    import os

    # 检测 ManiSkill 安装路径
    try:
        import mani_skill
        maniskill_path = Path(mani_skill.__file__).parent
        robot_assets_dir = maniskill_path / "assets" / "robots"

        if (robot_assets_dir / "RJ2506").exists():
            optimize_rj2506(
                source_robot_dir=str(robot_assets_dir / "RJ2506"),
                target_robot_dir=str(robot_assets_dir / "RJ2506_simplified"),
                visual_reduction=0.95,
                collision_reduction=0.93,
            )
        else:
            print("未找到 RJ2506 机器人，请检查 ManiSkill 安装")

    except ImportError:
        print("请先安装 ManiSkill: pip install mani-skill")
        sys.exit(1)
