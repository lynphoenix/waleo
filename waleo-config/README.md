# Waleo Config

Waleo 配置管理模块，提供统一的配置数据类定义。

## 功能

- 环境配置 (`EnvConfig`)
- 相机配置 (`CameraConfig`)

## 使用示例

```python
from waleo_config import EnvConfig, CameraConfig

# 创建环境配置
env_config = EnvConfig(
    task="push",
    robot_type="so100",
    simulation_backend="mujoco",
    render_mode="rgb_array"
)

# 从字典创建
config_dict = {
    "task": "pick_place",
    "robot_type": "aloha",
    "num_envs": 8
}
env_config = EnvConfig.from_dict(config_dict)
```

## 配置参数

### EnvConfig

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `task` | str | *必需* | 任务类型 |
| `robot_type` | str | *必需* | 机器人类型 |
| `simulation_backend` | str | "mujoco" | 仿真后端 |
| `render_mode` | str | None | 渲染模式 |
| `headless` | bool | False | 无头模式 |
| `num_envs` | int | 1 | 并行环境数量 |
| `seed` | int | None | 随机种子 |
| `episode_length` | int | 1000 | 回合最大步数 |
| `reward_scale` | float | 1.0 | 奖励缩放 |

### CameraConfig

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `width` | int | 640 | 图像宽度 |
| `height` | int | 480 | 图像高度 |
| `fov` | float | 60.0 | 视场角（度） |
