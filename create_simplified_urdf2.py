import re

urdf_path = '/root/miniconda3/envs/waleo/lib/python3.10/site-packages/mani_skill/assets/robots/RJ2506/urdf/RJ2506_leftarm_only_noagv.urdf'
with open(urdf_path, 'r') as f:
    urdf_content = f.read()

# 按visual块替换
def replace_visual(match):
    block = match.group(0)
    return block.replace('package://RJ2506/meshes/', 'package://RJ2506_simplified/meshes/').replace('.STL" />', '.glb" />')

def replace_collision(match):
    block = match.group(0)
    return block.replace('package://RJ2506/meshes/', 'package://RJ2506_simplified/meshes_collision/').replace('.STL" />', '.glb" />')

# 替换visual部分
urdf_content = re.sub(r'<visual>.*?</visual>', replace_visual, urdf_content, flags=re.DOTALL)

# 替换collision部分  
urdf_content = re.sub(r'<collision>.*?</collision>', replace_collision, urdf_content, flags=re.DOTALL)

output_path = '/root/miniconda3/envs/waleo/lib/python3.10/site-packages/mani_skill/assets/robots/RJ2506_simplified/urdf/RJ2506_full_simplified.urdf'
with open(output_path, 'w') as f:
    f.write(urdf_content)

# 验证
visual_count = urdf_content.count('RJ2506_simplified/meshes/')
collision_count = urdf_content.count('RJ2506_simplified/meshes_collision/')
print('Created full simplified URDF')
print('Visual mesh refs (meshes/):', visual_count)
print('Collision mesh refs (meshes_collision/):', collision_count)
