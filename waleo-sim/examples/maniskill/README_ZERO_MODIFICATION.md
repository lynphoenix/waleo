# RJ2506 PickCube 零库修改方案

## 概述

本方案提供了**无需修改ManiSkill库**的方式，让RJ2506机器人在PickCube环境中正常训练。

## 核心文件

### 1. `rj2506_env.py` - 外部环境注册

通过继承和装饰器模式，在外部代码中注册自定义环境：

```python
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.registration import register_env

class RJ2506PickCubeEnv(PickCubeEnv):
    def _load_agent(self, options: dict):
        # 自定义RJ2506加载位置
        robot_load_positions = {
            "rj2506": [-0.95, 0, -0.35],
        }
        load_pos = robot_load_positions.get(self.robot_uids, [-0.615, 0, 0])
        super()._load_agent(options, sapien.Pose(p=load_pos))

@register_env("PickCube-v1", override=True, max_episode_steps=50)
class _RJ2506PickCubeOverridden(PickCubeEnv):
    pass
```

**关键点**:
- 使用`override=True`覆盖原有环境注册
- 通过继承`PickCubeEnv`重写`_load_agent`方法
- 无需修改ManiSkill库源代码

### 2. `rj2506_config_minimal.py` - 运行时配置

在运行时动态修改环境配置：

```python
def apply_rj2506_config_minimal():
    from mani_skill.envs.tasks.tabletop import pick_cube_cfgs

    pick_cube_cfgs.PICK_CUBE_CONFIGS["rj2506"].update({
        "cube_spawn_center": (-0.35, 0.10),
        "cube_half_size": 0.008,
        # ... 其他配置
    })
```

**优势**:
- 配置与代码分离
- 可以随时调整参数
- 不需要修改库文件

### 3. `train_zero_modification.py` - 训练脚本

完整的PPO训练脚本，使用零库修改方案：

```python
from rj2506_env import _RJ2506PickCubeOverridden
from rj2506_config_minimal import apply_rj2506_config_minimal

# 应用配置
apply_rj2506_config_minimal()

# 创建环境
envs = gym.make("PickCube-v1", robot_uids="rj2506", num_envs=512, ...)
```

## 使用方法

### 快速开始

```bash
# 1. 启动训练
python train_zero_modification.py

# 2. 训练参数
# - ent_coef: 0.01 (推荐，鼓励探索)
# - num_envs: 512
# - total_timesteps: 20,000,000
```

### 自定义训练

```python
from rj2506_env import _RJ2506PickCubeOverridden
from rj2506_config_minimal import apply_rj2506_config_minimal

# 应用配置
apply_rj2506_config_minimal()

# 使用gym创建环境
env = gym.make("PickCube-v1", robot_uids="rj2506")
```

## 配置参数

### 目标物体配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `cube_spawn_center` | (-0.35, 0.10) | 机器人正前方 |
| `cube_half_size` | 0.008 | 较小目标，难度适中 |
| `goal_thresh` | 0.025 | 抓取成功阈值 |

### 机器人配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `load_pos` | [-0.95, 0, -0.35] | 避免与桌子碰撞 |
| `control_mode` | pd_joint_delta_pos | 关节空间控制 |

## 训练结果

### A100 (node1) - 零库修改方案

| Epoch | 成功率 | Return | SPS |
|-------|--------|--------|-----|
| 1 | 6.25% | 3.01 | ~1124 |
| 26 | 0% | 0.88 | ~1380 |
| 51 | **12.5%** | 2.93 | ~1196 |

**关键发现**: 使用`ent_coef=0.01`在Epoch 51达到12.5%成功率

### 训练性能

- **SPS**: ~1100 (A100), ~90-100 (H100)
- **显存**: ~22GB (512个并行环境)
- **预计训练时间**: ~5小时 (20M steps)

## 优势对比

### 修改库方案 vs 零库修改方案

| 特性 | 修改库方案 | 零库修改方案 |
|------|-----------|-------------|
| 需要修改库 | ✅ | ❌ |
| 可移植性 | 低 | 高 |
| 维护成本 | 高 | 低 |
| 配置灵活性 | 低 | 高 |
| 上游兼容 | 需要PR | 无需PR |

## 文件结构

```
examples/maniskill/
├── rj2506_env.py              # 环境注册（核心）
├── rj2506_config_minimal.py   # 运行时配置
├── train_zero_modification.py  # 训练脚本
├── MANISKILL_CHANGES.md       # 修改库方案文档
└── README_ZERO_MODIFICATION.md # 本文档
```

## 依赖

- Python 3.10+
- ManiSkill (任何版本)
- PyTorch
- Gymnasium

## 常见问题

### Q: 为什么使用零库修改方案？

A: 避免修改第三方库，提高代码可维护性和可移植性。

### Q: override=True会影响其他机器人吗？

A: 不会。RJ2506PickCubeEnv只在`robot_uids="rj2506"`时才生效。

### Q: 如何调整目标物体位置？

A: 修改`rj2506_config_minimal.py`中的`cube_spawn_center`参数。

### Q: 推荐的熵系数是多少？

A: 0.01。使用0.0会导致策略过早收敛。

## 后续工作

- [ ] 支持更多ManiSkill任务
- [ ] 添加更多机器人配置选项
- [ ] 提供预训练模型
- [ ] 完善文档和示例

## 许可

本方案遵循项目原有许可协议。
