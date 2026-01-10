# Waleo Sim

Waleo 仿真基类模块，提供统一的仿真环境接口，支持多种仿真后端。

## 特性

- ✅ **Gym/Gymnasium 兼容**：标准的环境接口
- ✅ **多后端支持**：MuJoCo, PyBullet, ManiSkill, Isaac Gym
- ✅ **向量化环境**：支持并行仿真，提高训练效率
- ✅ **机器人环境基类**：扩展 BaseEnv，添加机器人特有功能

## 安装

```bash
# 基础安装
pip install waleo-sim

# 安装特定后端
pip install waleo-sim[mujoco]      # MuJoCo
pip install waleo-sim[pybullet]    # PyBullet
pip install waleo-sim[maniskill]   # ManiSkill
pip install waleo-sim[isaacgym]    # Isaac Gym
```

## 快速开始

### 使用环境基类

```python
from waleo_sim import BaseEnv
import numpy as np

class MyEnv(BaseEnv):
    @property
    def observation_space(self):
        # 定义观察空间
        pass

    @property
    def action_space(self):
        # 定义动作空间
        pass

    def reset(self, seed=None, options=None):
        # 重置环境
        return observation, info

    def step(self, action):
        # 执行一步
        return observation, reward, terminated, truncated, info
```

### 使用仿真后端

```python
from waleo_config import EnvConfig
from waleo_sim.backends import ManiSkillBackend

# 创建配置
config = EnvConfig(
    task="pick_place",
    robot_type="so100",
    simulation_backend="maniskill",
    maniskill_task="PickCube",
    num_envs=8
)

# 创建后端
backend = ManiSkillBackend(config)
backend.initialize()

# 运行环境
obs, info = backend.reset()
for _ in range(1000):
    action = backend.action_space.sample()
    obs, reward, terminated, truncated, info = backend.step(action)

    if terminated:
        obs, info = backend.reset()

backend.close()
```

### 使用环境包装器

```python
from waleo_sim import EnvWrapper, BaseEnv

class MyWrapper(EnvWrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # 修改观察
        obs = self.process(obs)
        return obs, reward, terminated, truncated, info

# 使用包装器
env = MyEnv()
wrapped_env = MyWrapper(env)
```

## 支持的仿真后端

| 后端 | 特点 | 安装 |
|------|------|------|
| **MuJoCo** | 快速物理仿真 | `pip install mujoco` |
| **PyBullet** | 开源免费，易于使用 | `pip install pybullet` |
| **ManiSkill** | 机器人操作仿真 | `pip install maniskill2-api sapien` |
| **Isaac Gym** | GPU 加速大规模并行 | （需要 NVIDIA GPU）|

## 模块结构

```
waleo-sim/
├── waleo_sim/
│   ├── __init__.py         # 模块入口
│   ├── base/               # 环境基类
│   │   ├── __init__.py
│   │   └── base.py         # BaseEnv, EnvWrapper, VectorEnv, RobotEnv
│   ├── backends/           # 仿真后端
│   │   ├── __init__.py
│   │   ├── backend.py      # 后端抽象接口
│   │   ├── mujoco.py       # MuJoCo 后端
│   │   ├── pybullet.py     # PyBullet 后端
│   │   └── maniskill.py    # ManiSkill 后端
│   └── config.py           # 环境配置
├── tests/                  # 测试文件
├── examples/               # 使用示例
└── README.md
```

## API 文档

### BaseEnv

环境基类，与 Gym/Gymnasium API 兼容。

```python
class BaseEnv:
    @property
    @abstractmethod
    def observation_space(self):
        """观察空间"""

    @property
    @abstractmethod
    def action_space(self):
        """动作空间"""

    @abstractmethod
    def reset(self, seed=None, options=None):
        """重置环境"""

    @abstractmethod
    def step(self, action):
        """执行一步"""

    def render(self, mode="human"):
        """渲染环境"""

    def close(self):
        """关闭环境"""
```

### RobotEnv

机器人环境基类，扩展 BaseEnv。

```python
class RobotEnv(BaseEnv):
    def __init__(self, task, robot_type="so100", simulation_backend="mujoco", ...):
        # 初始化环境

    @abstractmethod
    def _load_robot_model(self):
        """加载机器人模型"""

    @abstractmethod
    def _setup_cameras(self):
        """设置相机"""

    @abstractmethod
    def _compute_reward(self, achieved_goal, desired_goal):
        """计算奖励"""

    @property
    @abstractmethod
    def robot_state(self):
        """获取机器人状态"""
```

### SimulationBackend

仿真后端抽象接口。

```python
class SimulationBackend:
    def __init__(self, config):
        """初始化后端"""

    @abstractmethod
    def initialize(self):
        """初始化仿真环境"""

    @abstractmethod
    def reset(self, seed=None):
        """重置环境"""

    @abstractmethod
    def step(self, action):
        """执行仿真步"""

    @abstractmethod
    def render(self, mode="rgb_array"):
        """渲染环境"""

    @abstractmethod
    def close(self):
        """关闭仿真环境"""
```

## ManiSkill 后端使用示例

```python
from waleo_config import EnvConfig
from waleo_sim.backends import ManiSkillBackend

# 创建配置
config = EnvConfig(
    task="push",
    robot_type="so100",
    simulation_backend="maniskill",
    maniskill_task="PushCube",  # ManiSkill 任务
    num_envs=8,                # 并行环境
    enable_visual_obs=True      # 启用视觉观察
)

# 创建后端
backend = ManiSkillBackend(config)
backend.initialize()

# 运行
obs, info = backend.reset()
for step in range(1000):
    action = policy.predict(obs)
    obs, reward, terminated, truncated, info = backend.step(action)

    if terminated:
        print("Task completed!")
        obs, info = backend.reset()

backend.close()
```

## ManiSkill 支持的任务

| 任务名称 | 描述 | 难度 |
|---------|------|------|
| `PickCube` | 从桌子上抓取立方体 | ⭐ |
| `PushCube` | 推动立方体到目标位置 | ⭐ |
| `StackCube` | 将立方体堆叠起来 | ⭐⭐ |
| `PlugCharger` | 将充电器插入插座 | ⭐⭐⭐ |
| `TurnFaucet` | 转动水龙头 | ⭐⭐ |
| `OpenCabinetDrawer` | 打开抽屉 | ⭐⭐ |

## 许可证

Apache License 2.0
