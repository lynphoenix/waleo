#!/usr/bin/env python3
"""
简化RJ2506的collision mesh
目标：从78万面简化到约5万面（与Panda相当）
"""

import trimesh
import open3d as o3d
import os
import numpy as np
import time

def main():
    print('=' * 70)
    print('RJ2506 Collision Mesh 简化')
    print('=' * 70)
    print(f'开始时间: {time.strftime("%Y-%m-%d %H:%M:%S")}')

    # 配置
    source_dir = '/root/miniconda3/envs/waleo/lib/python3.10/site-packages/mani_skill/assets/robots/RJ2506/meshes'
    target_dir = '/root/miniconda3/envs/waleo/lib/python3.10/site-packages/mani_skill/assets/robots/RJ2506_simplified/meshes_collision'
    VOXEL_SIZE = 0.008  # 调整此值控制简化程度

    os.makedirs(target_dir, exist_ok=True)

    stl_files = sorted([f for f in os.listdir(source_dir) if f.endswith('.STL')])
    print(f'找到 {len(stl_files)} 个STL文件')
    print(f'Voxel size: {VOXEL_SIZE}')
    print('=' * 70)

    total_before = 0
    total_after = 0
    success_count = 0

    for i, stl_file in enumerate(stl_files, 1):
        source_path = os.path.join(source_dir, stl_file)
        glb_file = stl_file.replace('.STL', '.glb')
        target_path = os.path.join(target_dir, glb_file)

        try:
            # 加载STL
            mesh = trimesh.load(source_path)
            vertices = mesh.vertices
            faces = mesh.faces

            before = len(faces)
            total_before += before

            # 使用Open3D简化
            mesh_o3d = o3d.geometry.TriangleMesh()
            mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
            mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
            mesh_o3d.compute_vertex_normals()

            simplified_o3d = mesh_o3d.simplify_vertex_clustering(
                voxel_size=VOXEL_SIZE,
                contraction=o3d.geometry.SimplificationContraction.Average
            )

            simp_verts = np.asarray(simplified_o3d.vertices)
            simp_faces = np.asarray(simplified_o3d.triangles)
            after = len(simp_faces)
            total_after += after

            # 保存为GLB
            simp_mesh = trimesh.Trimesh(vertices=simp_verts, faces=simp_faces)
            simp_mesh.export(target_path)

            success_count += 1

            if 'arm' in stl_file or 'hand' in stl_file:
                ratio = after / before * 100
                print(f'[{i:2d}/{len(stl_files)}] {stl_file:30s} {before:8,} -> {after:8,} faces ({ratio:5.1f}%)')

        except Exception as e:
            print(f'[{i:2d}/{len(stl_files)}] {stl_file}: ERROR - {str(e)[:60]}')

    print('')
    print('=' * 70)
    print(f'完成！成功: {success_count}/{len(stl_files)}')
    print(f'Collision总面数: {total_before:,} -> {total_after:,}')
    print(f'简化比例: {total_after / max(total_before, 1) * 100:.1f}%')
    print(f'结束时间: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 70)

if __name__ == '__main__':
    main()
