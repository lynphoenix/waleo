#!/usr/bin/env python3
"""通用的 Mesh 简化工具

支持多种简化算法和输入/输出格式。
"""

import os
from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple
import time
import logging

try:
    import trimesh
    import open3d as o3d
    import numpy as np
except ImportError as e:
    raise ImportError(
        f"Missing required package: {e.name}. "
        "Install with: pip install trimesh open3d"
    ) from e

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class MeshSimplifier:
    """Mesh 简化器

    支持多种简化算法：
    - vertex_clustering: 体素聚类（快速，适合大规模简化）
    - quadric_decimation: 二次误差度量（质量高，速度慢）
    """

    SUPPORTED_INPUT = {'.stl', '.obj', '.ply', '.glb', '.gltf'}
    SUPPORTED_OUTPUT = {'.stl', '.obj', '.ply', '.glb'}

    def __init__(
        self,
        method: str = "vertex_clustering",
        voxel_size: Optional[float] = None,
        target_faces: Optional[int] = None,
        target_reduction: Optional[float] = None,
    ):
        """初始化简化器

        Args:
            method: 简化方法 ("vertex_clustering" 或 "quadric_decimation")
            voxel_size: 体素大小（用于 vertex_clustering）
            target_faces: 目标面数
            target_reduction: 目标减少比例（0-1，如0.9表示减少90%）
        """
        self.method = method
        self.voxel_size = voxel_size
        self.target_faces = target_faces
        self.target_reduction = target_reduction

        if method not in ["vertex_clustering", "quadric_decimation"]:
            raise ValueError(f"Unsupported method: {method}")

    def simplify(
        self,
        mesh: Union[str, Path, trimesh.Trimesh],
        output_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[trimesh.Trimesh, Dict]:
        """简化单个 mesh

        Args:
            mesh: 输入 mesh（文件路径或 trimesh 对象）
            output_path: 输出路径（可选）

        Returns:
            (simplified_mesh, stats): 简化后的 mesh 和统计信息
        """
        # 加载 mesh
        if isinstance(mesh, (str, Path)):
            mesh_path = Path(mesh)
            if mesh_path.suffix.lower() not in self.SUPPORTED_INPUT:
                raise ValueError(f"Unsupported input format: {mesh_path.suffix}")
            mesh = trimesh.load(mesh_path)

        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Expected Trimesh, got {type(mesh)}")

        # 记录原始信息
        original_faces = len(mesh.faces)
        original_vertices = len(mesh.vertices)

        # 执行简化
        start_time = time.time()

        if self.method == "vertex_clustering":
            simplified = self._vertex_clustering(mesh)
        elif self.method == "quadric_decimation":
            simplified = self._quadric_decimation(mesh)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        elapsed_time = time.time() - start_time

        # 统计信息
        stats = {
            'original_faces': original_faces,
            'original_vertices': original_vertices,
            'simplified_faces': len(simplified.faces),
            'simplified_vertices': len(simplified.vertices),
            'reduction_ratio': 1 - len(simplified.faces) / original_faces,
            'elapsed_time': elapsed_time,
            'method': self.method,
        }

        # 保存
        if output_path:
            output_path = Path(output_path)
            if output_path.suffix.lower() not in self.SUPPORTED_OUTPUT:
                raise ValueError(f"Unsupported output format: {output_path.suffix}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            simplified.export(output_path)
            stats['output_path'] = str(output_path)

        return simplified, stats

    def _vertex_clustering(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """使用体素聚类简化"""
        # 转换为 Open3D mesh
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
        mesh_o3d.compute_vertex_normals()

        # 计算 voxel size
        if self.voxel_size is None:
            voxel_size = estimate_voxel_size(
                mesh,
                target_faces=self.target_faces,
                target_reduction=self.target_reduction,
            )
        else:
            voxel_size = self.voxel_size

        # 简化
        simplified_o3d = mesh_o3d.simplify_vertex_clustering(
            voxel_size=voxel_size,
            contraction=o3d.geometry.SimplificationContraction.Average
        )

        # 转回 trimesh
        vertices = np.asarray(simplified_o3d.vertices)
        faces = np.asarray(simplified_o3d.triangles)
        return trimesh.Trimesh(vertices=vertices, faces=faces)

    def _quadric_decimation(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """使用二次误差度量简化"""
        # 转换为 Open3D mesh
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
        mesh_o3d.compute_vertex_normals()

        # 计算目标三角形数
        if self.target_faces is None:
            if self.target_reduction is not None:
                target_faces = int(len(mesh.faces) * (1 - self.target_reduction))
            else:
                target_faces = len(mesh.faces) // 2  # 默认减少50%
        else:
            target_faces = self.target_faces

        # 简化
        simplified_o3d = mesh_o3d.simplify_quadric_decimation(
            target_number_of_triangles=target_faces
        )

        # 转回 trimesh
        vertices = np.asarray(simplified_o3d.vertices)
        faces = np.asarray(simplified_o3d.triangles)
        return trimesh.Trimesh(vertices=vertices, faces=faces)


def simplify_mesh(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    method: str = "vertex_clustering",
    target_reduction: float = 0.95,
    voxel_size: Optional[float] = None,
    **kwargs
) -> Dict:
    """简化单个 mesh 文件（便捷函数）

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        method: 简化方法
        target_reduction: 目标减少比例（0-1）
        voxel_size: 体素大小（可选，会自动估算）

    Returns:
        统计信息字典
    """
    simplifier = MeshSimplifier(
        method=method,
        voxel_size=voxel_size,
        target_reduction=target_reduction,
        **kwargs
    )
    _, stats = simplifier.simplify(input_path, output_path)
    return stats


def simplify_directory(
    source_dir: Union[str, Path],
    target_dir: Union[str, Path],
    pattern: str = "*.STL",
    output_format: str = ".glb",
    method: str = "vertex_clustering",
    target_reduction: float = 0.95,
    voxel_size: Optional[float] = None,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """批量简化目录中的 mesh 文件

    Args:
        source_dir: 源目录
        target_dir: 目标目录
        pattern: 文件匹配模式（支持通配符）
        output_format: 输出格式（.stl, .obj, .glb等）
        method: 简化方法
        target_reduction: 目标减少比例
        voxel_size: 体素大小（可选）
        verbose: 是否显示详细信息

    Returns:
        每个文件的统计信息字典
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # 查找文件
    files = sorted(source_dir.glob(pattern))
    if not files:
        logger.warning(f"No files matching '{pattern}' found in {source_dir}")
        return {}

    if verbose:
        logger.info(f"Found {len(files)} files matching '{pattern}'")
        logger.info(f"Method: {method}, Target reduction: {target_reduction*100:.1f}%")
        logger.info("=" * 70)

    # 创建简化器
    simplifier = MeshSimplifier(
        method=method,
        voxel_size=voxel_size,
        target_reduction=target_reduction,
    )

    # 批量处理
    all_stats = {}
    total_before = 0
    total_after = 0
    success_count = 0

    for i, input_path in enumerate(files, 1):
        output_path = target_dir / (input_path.stem + output_format)

        try:
            _, stats = simplifier.simplify(input_path, output_path)
            all_stats[input_path.name] = stats

            total_before += stats['original_faces']
            total_after += stats['simplified_faces']
            success_count += 1

            if verbose:
                ratio = stats['reduction_ratio'] * 100
                logger.info(
                    f"[{i:2d}/{len(files)}] {input_path.name:30s} "
                    f"{stats['original_faces']:8,} -> {stats['simplified_faces']:8,} "
                    f"({ratio:5.1f}% reduction)"
                )

        except Exception as e:
            logger.error(f"[{i:2d}/{len(files)}] {input_path.name}: ERROR - {str(e)[:60]}")
            all_stats[input_path.name] = {'error': str(e)}

    # 总结
    if verbose:
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"Completed: {success_count}/{len(files)} files")
        logger.info(f"Total faces: {total_before:,} -> {total_after:,}")
        if total_before > 0:
            overall_reduction = (1 - total_after / total_before) * 100
            logger.info(f"Overall reduction: {overall_reduction:.1f}%")
        logger.info("=" * 70)

    return all_stats


def estimate_voxel_size(
    mesh: Union[trimesh.Trimesh, str, Path],
    target_faces: Optional[int] = None,
    target_reduction: Optional[float] = None,
) -> float:
    """估算合适的 voxel size

    根据目标面数或减少比例，估算体素聚类所需的 voxel size。

    Args:
        mesh: Mesh 对象或文件路径
        target_faces: 目标面数
        target_reduction: 目标减少比例（0-1）

    Returns:
        估算的 voxel size
    """
    if isinstance(mesh, (str, Path)):
        mesh = trimesh.load(mesh)

    # 获取 mesh 尺寸
    bbox = mesh.bounds
    bbox_size = bbox[1] - bbox[0]
    max_dimension = max(bbox_size)

    original_faces = len(mesh.faces)

    # 计算目标面数
    if target_faces is None:
        if target_reduction is not None:
            target_faces = int(original_faces * (1 - target_reduction))
        else:
            target_faces = original_faces // 10  # 默认减少90%

    # 启发式估算
    # voxel size 与面数的关系大致为: faces ∝ (size / voxel_size)^3
    ratio = (original_faces / max(target_faces, 1)) ** (1/3)
    voxel_size = max_dimension / (100 * ratio)

    # 限制范围
    voxel_size = max(0.001, min(voxel_size, max_dimension / 10))

    return voxel_size


if __name__ == "__main__":
    # 简单的命令行接口
    import sys

    if len(sys.argv) < 3:
        print("Usage: python simplify_meshes.py <input_file> <output_file> [reduction_ratio]")
        print("   or: python simplify_meshes.py <source_dir> <target_dir> --batch [reduction_ratio]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if "--batch" in sys.argv:
        reduction = float(sys.argv[4]) if len(sys.argv) > 4 else 0.95
        simplify_directory(input_path, output_path, target_reduction=reduction)
    else:
        reduction = float(sys.argv[3]) if len(sys.argv) > 3 else 0.95
        stats = simplify_mesh(input_path, output_path, target_reduction=reduction)
        print(f"Simplified: {stats['original_faces']:,} -> {stats['simplified_faces']:,} faces")
