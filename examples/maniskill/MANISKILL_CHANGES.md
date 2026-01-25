# ManiSkill库修改总结

## 修改概述

为了支持RJ2506机器人在PickCube任务中的使用，对ManiSkill库进行了以下修改。

## 修改文件清单

### 文件1: `mani_skill/envs/tasks/tabletop/pick_cube_cfgs.py`

**修改位置**: `/home/smai/dc_dir/yes/envs/waleo/lib/python3.10/site-packages/mani_skill/envs/tasks/tabletop/pick_cube_cfgs.py`

**修改内容**: 添加rj2506的配置 (第70-80行)

```python
"rj2506": {
    "cube_half_size": 0.02,
    "goal_thresh": 0.025,
    "cube_spawn_half_size": 0.05,
    "cube_spawn_center": (-0.62, 0.29),  # Near TCP initial position for better reachability
    "max_goal_height": 0.3,
    "sensor_cam_eye_pos": [-0.5, 0.2, 0.6],
    "sensor_cam_target_pos": [-0.62, 0.29, 0.1],
    "human_cam_eye_pos": [-0.1, 0.7, 0.6],
    "human_cam_target_pos": [-0.62, 0.0, 0.35],
},
```

**说明**: 这是旧配置，使用了`cube_spawn_center: (-0.62, 0.29)`。最终改进为`(-0.35, 0.10)`（机器人正前方）。

### 文件2: `mani_skill/envs/tasks/tabletop/pick_cube.py`

**修改位置**: `/home/smai/dc_dir/yes/envs/waleo/lib/python3.10/site-packages/mani_skill/envs/tasks/tabletop/pick_cube.py`

#### 修改1: 添加RJ2506导入 (第9行)

```python
from mani_skill.agents.robots.rj2506 import RJ2506
```

#### 修改2: 添加rj2506到支持的机器人列表 (第44行)

```python
SUPPORTED_ROBOTS = [
    "panda",
    "fetch",
    "xarm6_robotiq",
    "so100",
    "widowxai",
    "rj2506",  # 新增
]
```

#### 修改3: 更新类型注解 (第46行)

```python
agent: Union[Panda, Fetch, XArm6Robotiq, SO100, WidowXAI, RJ2506]
```

#### 修改4: 添加rj2506加载位置配置 (第82-88行)

```python
def _load_agent(self, options: dict):
    # Robot-specific loading positions to ensure proper workspace configuration
    robot_load_positions = {
        "rj2506": [-0.95, 0, -0.35],  # Further left to avoid table collision
    }
    load_pos = robot_load_positions.get(self.robot_uids, [-0.615, 0, 0])
    super()._load_agent(options, sapien.Pose(p=load_pos))
```

## 修改原因

RJ2506是一个双臂机器人，身体结构较大。使用默认的加载位置 `[-0.615, 0, 0]` 会导致机器人与桌子发生碰撞。

通过将机器人向左移动0.335m（X从-0.615到-0.95），可以避免碰撞，让机器人正确放置在桌子旁边。

## 配置参数说明

| 参数 | 默认值 | RJ2506值 | 说明 |
|------|--------|----------|------|
| load_pos.x | -0.615 | -0.95 | 向左移动，避免碰撞 |
| load_pos.y | 0 | 0 | 保持不变 |
| load_pos.z | 0 | -0.35 | 稍微降低，确保稳定性 |

## 影响范围

- ✅ **向后兼容**: 其他机器人使用默认值，不受影响
- ✅ **作用域限定**: 只影响rj2506机器人的加载位置
- ✅ **无需修改现有代码**: 使用PickCube-v1环境时自动应用

## 测试验证

修改后已验证：
1. RJ2506机器人可以正确加载到PickCube环境
2. 机器人与桌子无碰撞
3. 训练可以正常运行
4. 其他机器人（Panda等）不受影响

## 对应的Pull Request建议

建议将此修改提交到ManiSkill upstream，标题为：
```
Add RJ2506 robot support for PickCube environment
```

包含以下内容：
- 支持RJ2506机器人在PickCube任务中使用
- 添加机器人特定的加载位置配置
- 避免机器人与桌子的碰撞问题
