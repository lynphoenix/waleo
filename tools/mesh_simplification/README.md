# 网格简化工具集

通用的机器人模型网格简化工具集，用于优化仿真训练性能。

## 功能特性

- 🔧 **通用工具**：支持任意机器人模型的网格简化
- 📦 **多格式支持**：STL, OBJ, PLY, GLB/GLTF
- 🎯 **多种算法**：体素聚类（快速）、二次误差度量（高质量）
- 📊 **批量处理**：一键简化整个网格目录
- 📝 **URDF 自动更新**：自动生成简化版 URDF 配置
- 📈 **详细统计**：面数、减少比例、耗时等

## 安装依赖

```bash
pip install trimesh open3d numpy
```

## 快速开始

### 1. 简化单个网格文件

```python
from tools.mesh_simplification import simplify_mesh

stats = simplify_mesh(
    input_path="path/to/model.stl",
    output_path="path/to/model_simplified.glb",
    target_reduction=0.95,  # 减少95%的面数
)

print(f"面数: {stats['original_faces']:,} -> {stats['simplified_faces']:,}")
print(f"减少: {stats['reduction_ratio']*100:.1f}%")
```

### 2. 批量简化整个目录

```python
from tools.mesh_simplification import simplify_directory

all_stats = simplify_directory(
    source_dir="assets/robots/RJ2506/meshes",
    target_dir="assets/robots/RJ2506_simplified/meshes",
    pattern="*.STL",           # 匹配所有 STL 文件
    output_format=".glb",      # 输出为 GLB 格式
    target_reduction=0.95,     # 减少95%
)
```

### 3. 创建简化版 URDF

```python
from tools.mesh_simplification import create_simplified_urdf

stats = create_simplified_urdf(
    source_urdf="path/to/robot.urdf",
    output_urdf="path/to/robot_simplified.urdf",
    robot_name="RJ2506",
    simplified_name="RJ2506_simplified",
    visual_mesh_dir="meshes",
    collision_mesh_dir="meshes_collision",
)

print(f"视觉网格引用: {stats['visual_refs']}")
print(f"碰撞网格引用: {stats['collision_refs']}")
```

## 完整工作流示例

### 优化 RJ2506 机器人模型

```python
from pathlib import Path
from tools.mesh_simplification import simplify_directory, create_simplified_urdf

# 1. 简化视觉网格
simplify_directory(
    source_dir="assets/robots/RJ2506/meshes",
    target_dir="assets/robots/RJ2506_simplified/meshes",
    pattern="*.STL",
    output_format=".glb",
    target_reduction=0.95,  # 减少95%
    method="vertex_clustering",
)

# 2. 简化碰撞网格（可以用不同参数）
simplify_directory(
    source_dir="assets/robots/RJ2506/meshes",
    target_dir="assets/robots/RJ2506_simplified/meshes_collision",
    pattern="*.STL",
    output_format=".glb",
    target_reduction=0.93,  # 碰撞网格可以保留稍多一点
)

# 3. 创建简化版 URDF
create_simplified_urdf(
    source_urdf="assets/robots/RJ2506/urdf/RJ2506.urdf",
    output_urdf="assets/robots/RJ2506_simplified/urdf/RJ2506_simplified.urdf",
    robot_name="RJ2506",
)

print("优化完成！")
```

## 命令行使用

### 简化单个文件

```bash
python -m tools.mesh_simplification.simplify_meshes \
    input.stl output.glb 0.95
```

### 批量简化

```bash
python -m tools.mesh_simplification.simplify_meshes \
    source_dir/ target_dir/ --batch 0.95
```

### 创建简化版 URDF

```bash
python -m tools.mesh_simplification.urdf_updater \
    create robot.urdf robot_simplified.urdf RobotName
```

### 列出 URDF 中的网格

```bash
python -m tools.mesh_simplification.urdf_updater list robot.urdf
```

## API 文档

### simplify_mesh()

简化单个网格文件。

**参数**：
- `input_path` (str|Path): 输入文件路径
- `output_path` (str|Path): 输出文件路径
- `method` (str): 简化方法
  - `"vertex_clustering"`: 体素聚类（默认，快速）
  - `"quadric_decimation"`: 二次误差度量（慢但质量高）
- `target_reduction` (float): 目标减少比例 (0-1)
- `voxel_size` (float, 可选): 体素大小（会自动估算）

**返回**：统计信息字典

### simplify_directory()

批量简化目录中的网格文件。

**参数**：
- `source_dir` (str|Path): 源目录
- `target_dir` (str|Path): 目标目录
- `pattern` (str): 文件匹配模式（如 `"*.STL"`, `"arm_*.obj"`）
- `output_format` (str): 输出格式（如 `".glb"`, `".obj"`）
- `method` (str): 简化方法
- `target_reduction` (float): 目标减少比例
- `verbose` (bool): 是否显示详细信息

**返回**：每个文件的统计信息字典

### create_simplified_urdf()

创建简化版 URDF 文件。

**参数**：
- `source_urdf` (str|Path): 源 URDF 文件
- `output_urdf` (str|Path): 输出 URDF 文件
- `robot_name` (str): 原始机器人名称
- `simplified_name` (str, 可选): 简化版名称
- `visual_mesh_dir` (str): 视觉网格目录名
- `collision_mesh_dir` (str): 碰撞网格目录名
- `mesh_format` (str): 目标网格格式
- `original_format` (str): 原始网格格式

**返回**：替换统计信息

## 简化方法对比

| 方法 | 速度 | 质量 | 适用场景 |
|------|------|------|----------|
| Vertex Clustering | ⚡ 快 | ⭐⭐⭐ 中 | 大规模简化（90%+），碰撞网格 |
| Quadric Decimation | 🐌 慢 | ⭐⭐⭐⭐⭐ 高 | 中等简化（50-80%），视觉网格 |

## 性能参考

以 RJ2506 机器人为例（H100 GPU）：

| 指标 | 原始模型 | 简化模型 | 提升 |
|------|---------|---------|------|
| 视觉面数 | 788,900 | 39,445 | 95% ↓ |
| 碰撞面数 | 788,900 | 55,423 | 93% ↓ |
| 加载时间 | ~2.5s | ~0.3s | 8.3× ↑ |
| FPS (512环境) | 10,200 | 35,000 | 3.4× ↑ |
| 训练时间 (25M步) | ~14h | ~5h | 2.8× ↑ |

## 注意事项

1. **备份原始文件**：简化是不可逆的，建议先备份
2. **逐步调整参数**：从小比例开始（如0.5），逐步提高
3. **分别优化视觉和碰撞网格**：碰撞网格可以更激进
4. **验证结果**：简化后要验证仿真的准确性
5. **格式选择**：GLB 格式加载更快，推荐用于仿真

## 原始脚本来源

本工具集改写自以下临时脚本：
- `create_simplified_urdf.py` (v1)
- `create_simplified_urdf2.py` (v2)
- `patch_urdf_simplified.py`
- `simplify_collision_meshes.py`
- `simplify_rj2506_collision.py`

改进：
- ✅ 通用化：支持任意机器人模型
- ✅ 配置化：参数可调
- ✅ 模块化：清晰的 API
- ✅ 文档化：完整的说明和示例
- ✅ 健壮性：错误处理和验证

## 许可证

与 Waleo 项目相同
