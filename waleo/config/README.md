# Waleo Config - 配置管理基础设施

Waleo Config 模块提供配置管理的基础设施和工具，采用**分布式配置架构**。

## 设计哲学

### 基础设施 vs 领域配置

**Config 模块职责**：
- ✅ 提供配置协议和类型定义
- ✅ 提供配置解析、合并、验证工具
- ✅ 提供跨模块共享的通用配置
- ❌ 不包含具体领域的配置类

**各模块职责**：
- ✅ 定义自己领域的配置类
- ✅ 继承 ConfigProtocol 或使用 dataclass
- ✅ 实现领域特定的验证逻辑

## 配置位置

| 配置类型 | 位置 | 说明 |
|---------|------|------|
| **基础设施** | `waleo.config` | 协议、工具、验证框架 |
| **通用配置** | `waleo.config` | DeviceConfig, LogConfig, PathConfig |
| **仿真配置** | `waleo.sim` | EnvConfig, CameraConfig |
| **训练配置** | (待创建) | TrainingConfig, PPOConfig |
| **数据配置** | (待创建) | DatasetConfig |

## 快速开始

### 1. 使用基础设施

```python
from waleo.config import (
    ConfigProtocol,       # 配置协议
    merge_configs,        # 配置合并
    validate_config,      # 配置验证
    DeviceConfig,         # 设备配置
    LogConfig,            # 日志配置
)

# 通用配置
device_config = DeviceConfig(device="cuda", device_id=0)
log_config = LogConfig(log_dir="logs", log_level="INFO")
```

### 2. 使用领域配置

```python
# 从各自模块导入领域配置
from waleo.sim import EnvConfig, CameraConfig

# 创建配置
env_config = EnvConfig(
    task="pick_place",
    robot_type="panda",
    simulation_backend="maniskill",
    num_envs=512,
)

camera_config = CameraConfig(
    width=128,
    height=128,
    fov=60.0,
)
```

### 3. 配置文件加载

```python
from waleo.sim import EnvConfig

# 从 YAML 加载
env_config = EnvConfig.from_yaml("configs/env.yaml")

# 从 JSON 加载
env_config = EnvConfig.from_json("configs/env.json")

# 保存配置
env_config.to_yaml("output/env.yaml")
env_config.to_json("output/env.json")
```

### 4. 配置合并

```python
from waleo.config import merge_configs
from waleo.sim import EnvConfig

# 基础配置
base_config = EnvConfig.from_yaml("configs/base.yaml")

# 合并覆盖
config = merge_configs(
    base_config.to_dict(),
    {"num_envs": 1024, "seed": 42}
)

# 从字典创建
final_config = EnvConfig.from_dict(config)
```

## API 参考

### 基础协议

#### ConfigProtocol
```python
@runtime_checkable
class ConfigProtocol(Protocol):
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, config_dict: dict) -> "ConfigProtocol": ...
    def validate(self) -> bool: ...
```

#### FileConfigProtocol
```python
@runtime_checkable
class FileConfigProtocol(Protocol):
    @classmethod
    def from_yaml(cls, path: Path | str) -> "FileConfigProtocol": ...
    @classmethod
    def from_json(cls, path: Path | str) -> "FileConfigProtocol": ...
    def to_yaml(self, path: Path | str) -> None: ...
    def to_json(self, path: Path | str) -> None: ...
```

### 解析工具

- `parse_config(config_dict, overrides)` - 解析配置字典
- `merge_configs(base, override)` - 合并两个配置
- `get_nested_attr(obj, path)` - 获取嵌套属性
- `set_nested_attr(obj, path, value)` - 设置嵌套属性

### 验证工具

- `validate_config(config)` - 验证单个配置
- `validate_all(configs)` - 验证多个配置
- `check_config_consistency(configs)` - 检查配置一致性
- `get_validation_errors(config)` - 获取验证错误

### 通用配置

#### DeviceConfig
```python
@dataclass
class DeviceConfig:
    device: str = "cuda"        # 设备类型
    device_id: int = 0          # 设备ID
    num_threads: int = 4        # CPU线程数
```

#### LogConfig
```python
@dataclass
class LogConfig:
    log_dir: str = "logs"       # 日志目录
    log_level: str = "INFO"     # 日志级别
    log_freq: int = 100         # 日志频率
```

#### PathConfig
```python
@dataclass
class PathConfig:
    data_dir: str = "data"      # 数据目录
    output_dir: str = "output"  # 输出目录
    cache_dir: str = ".cache"   # 缓存目录
```

#### SeedConfig
```python
@dataclass
class SeedConfig:
    seed: int = 42              # 随机种子
    deterministic: bool = False # 确定性模式
```

## 创建新的配置类

### 示例：创建训练配置

```python
# src/waleo/training/config.py
from dataclasses import dataclass, field
from waleo.config import ConfigProtocol

@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 64
    learning_rate: float = 1e-4
    num_epochs: int = 100

    def __post_init__(self):
        """配置验证"""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "TrainingConfig":
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        import yaml
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def validate(self) -> bool:
        try:
            self.__post_init__()
            return True
        except ValueError:
            return False
```

```python
# src/waleo/training/__init__.py
from waleo.training.config import TrainingConfig

__all__ = ["TrainingConfig"]
```

## 配置文件示例

### YAML 格式

```yaml
# configs/env.yaml
task: pick_place
robot_type: panda
simulation_backend: maniskill
num_envs: 512
seed: 42
episode_length: 200
observation_keys:
  - observation.state
  - observation.image
```

### JSON 格式

```json
{
  "task": "pick_place",
  "robot_type": "panda",
  "simulation_backend": "maniskill",
  "num_envs": 512,
  "seed": 42,
  "episode_length": 200,
  "observation_keys": [
    "observation.state",
    "observation.image"
  ]
}
```

## 设计原则

### 1. 职责分离
- Config 提供工具，不定义领域配置
- 各模块管理自己的配置

### 2. 低耦合
- Config 不依赖其他模块
- 模块之间通过协议解耦

### 3. 易扩展
- 新模块只需实现 ConfigProtocol
- 不需要修改 config 模块

### 4. 类型安全
- 使用 dataclass 和类型提示
- 配置验证在 __post_init__ 中

### 5. 统一接口
- 所有配置都支持 to_dict/from_dict
- 所有配置都支持 YAML/JSON 序列化

## 对比：集中式 vs 分布式

| 方面 | 集中式 | 分布式（当前） |
|-----|--------|---------------|
| 配置位置 | 全在 config | 分散在各模块 |
| 耦合度 | 高 | 低 |
| 扩展性 | 差 | 好 |
| 查找性 | 集中 | 需要知道位置 |
| 维护性 | 困难 | 简单 |

## 相关文档

- [整体架构设计](../../docs/Waleo整体架构设计.md)
- [M02 配置管理模块设计](../../docs/design/M02-配置管理模块设计.md)
- [Sim 模块配置](../sim/README.md)

## 许可证

与 Waleo 项目相同
