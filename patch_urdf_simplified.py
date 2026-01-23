import re

urdf_path = '/root/miniconda3/envs/waleo/lib/python3.10/site-packages/mani_skill/assets/robots/RJ2506/urdf/RJ2506_leftarm_only_noagv.urdf'

with open(urdf_path, 'r') as f:
    content = f.read()

# 替换visual mesh引用
def replace_visual(match):
    block = match.group(0)
    return block.replace('package://RJ2506/meshes/', 'package://RJ2506_simplified/meshes/').replace('.STL" />', '.glb" />')

def replace_collision(match):
    block = match.group(0)
    return block.replace('package://RJ2506/meshes/', 'package://RJ2506_simplified/meshes_collision/').replace('.STL" />', '.glb" />')

content = re.sub(r'<visual>.*?</visual>', replace_visual, content, flags=re.DOTALL)
content = re.sub(r'<collision>.*?</collision>', replace_collision, content, flags=re.DOTALL)

with open(urdf_path, 'w') as f:
    f.write(content)

visual_count = content.count('RJ2506_simplified/meshes/')
collision_count = content.count('RJ2506_simplified/meshes_collision/')
print(f'Patched URDF with simplified meshes')
print(f'Visual refs: {visual_count}, Collision refs: {collision_count}')
