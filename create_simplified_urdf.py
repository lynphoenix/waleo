import re

urdf_path = '/root/miniconda3/envs/waleo/lib/python3.10/site-packages/mani_skill/assets/robots/RJ2506/urdf/RJ2506_leftarm_only_noagv.urdf'
with open(urdf_path, 'r') as f:
    urdf_content = f.read()

# 修改visual mesh路径
urdf_content = urdf_content.replace('package://RJ2506/meshes/', 'package://RJ2506_simplified/meshes/')
urdf_content = urdf_content.replace('.STL" />', '.glb" />')

# 保存
output_path = '/root/miniconda3/envs/waleo/lib/python3.10/site-packages/mani_skill/assets/robots/RJ2506_simplified/urdf/RJ2506_full_simplified.urdf'
with open(output_path, 'w') as f:
    f.write(urdf_content)

print('Created full simplified URDF')
print('All mesh references now use simplified GLB files')
