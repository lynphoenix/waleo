# Config 模块重构迁移指南

## 重构日期
2026-01-25

## 重构目标

将 Config 模块从**集中式配置**重构为**分布式配置架构**，提高模块独立性和可维护性。

---

## 变更概览

### 架构变化

**重构前（集中式）**：
```python
# 所有配置都从 config 导入
from waleo.config import EnvConfig, CameraConfig, TrainingConfig, DatasetConfig
```

**重构后（分布式）**：
```python
# 基础设施从 config 导入
from waleo.config import merge_configs, validate_config, DeviceConfig

# 领域配置从各自模块导入
from waleo.sim import EnvConfig, CameraConfig
from waleo.training import TrainingConfig  # (待创建)
from waleo.data import DatasetConfig       # (待创建)
```

### 核心理念

**Config 模块职责**：
- ✅ 提供配置协议和基础类型
- ✅ 提供配置工具（解析、合并、验证）
- ✅ 提供跨模块共享的通用配置
- ❌ 不再包含领域特定配置

**各模块职责**：
- ✅ 管理自己的配置类
- ✅ 实现领域特定验证
- ✅ 提供领域相关工具

---

## 破坏性变更

### 1. EnvConfig 和 CameraConfig 移动

**旧代码** ❌:
```python
from waleo.config import EnvConfig, CameraConfig
```

**新代码** ✅:
```python
from waleo.sim import EnvConfig, CameraConfig
```

### 2. Config 模块导出内容变化

#### 不再导出（已移除）

- ❌ `EnvConfig` → 移至 `waleo.sim`
- ❌ `CameraConfig` → 移至 `waleo.sim`
- ❌ `TrainingConfig` → 将移至 `waleo.training`（未来）
- ❌ `DatasetConfig` → 将移至 `waleo.data`（未来）
- ❌ `EvalConfig` → 将移至 `waleo.eval` 或 `waleo.training`（未来）

#### 新增导出（基础设施）

✅ **基础协议**：
- `ConfigProtocol`
- `FileConfigProtocol`
- `FeatureType`, `NormalizationMode`, `PolicyFeature`, `DictLike`

✅ **解析和合并工具**：
- `parse_config`, `merge_configs`
- `get_nested_attr`, `set_nested_attr`
- `parse_arg`, `parse_arg_value`
- 插件系统：`register_config`, `load_plugin`, `create_config`

✅ **验证工具**：
- `validate_config`, `validate_all`
- `check_config_consistency`
- `get_validation_errors`

✅ **通用配置**（跨模块共享）：
- `DeviceConfig` - 设备配置
- `LogConfig` - 日志配置
- `PathConfig` - 路径配置
- `SeedConfig` - 随机种子配置

---

## 迁移步骤

### 第1步：更新导入

使用查找替换工具批量更新：

```bash
# 查找所有使用旧导入的文件
grep -r "from waleo.config import.*EnvConfig" .

# 替换为新导入
sed -i 's/from waleo.config import EnvConfig/from waleo.sim import EnvConfig/g' *.py
sed -i 's/from waleo.config import CameraConfig/from waleo.sim import CameraConfig/g' *.py
```

### 第2步：验证代码

```python
# 运行测试确保无导入错误
pytest tests/

# 或手动测试导入
python -c "from waleo.sim import EnvConfig, CameraConfig; print('✅ Import OK')"
```

### 第3步：更新配置文件路径（如有）

如果有配置文件引用了模块路径，需要更新：

```yaml
# 旧配置 ❌
_target_: waleo.config.EnvConfig

# 新配置 ✅
_target_: waleo.sim.EnvConfig
```

---

## 示例代码迁移

### 示例 1：基础使用

**重构前**:
```python
from waleo.config import EnvConfig, CameraConfig

env_config = EnvConfig(task="pick_place", robot_type="panda")
camera_config = CameraConfig(width=128, height=128)
```

**重构后**:
```python
from waleo.sim import EnvConfig, CameraConfig

env_config = EnvConfig(task="pick_place", robot_type="panda")
camera_config = CameraConfig(width=128, height=128)
```

### 示例 2：配置合并

**重构前**:
```python
from waleo.config import EnvConfig

base = EnvConfig(task="pick_place", robot_type="panda")
# 手动合并或使用内部工具
```

**重构后**:
```python
from waleo.config import merge_configs
from waleo.sim import EnvConfig

base = EnvConfig.from_yaml("configs/base.yaml")
config_dict = merge_configs(base.to_dict(), {"num_envs": 1024})
env_config = EnvConfig.from_dict(config_dict)
```

### 示例 3：创建新配置类

**新增功能** - 使用基础设施创建自定义配置：

```python
from dataclasses import dataclass
from waleo.config import ConfigProtocol

@dataclass
class MyConfig:
    """自定义配置"""
    param1: int = 10
    param2: str = "default"

    def __post_init__(self):
        if self.param1 <= 0:
            raise ValueError("param1 must be positive")

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "MyConfig":
        return cls(**config_dict)

    def validate(self) -> bool:
        try:
            self.__post_init__()
            return True
        except ValueError:
            return False
```

---

## 配置位置索引

| 配置类 | 旧位置 | 新位置 | 状态 |
|--------|-------|--------|------|
| `EnvConfig` | `waleo.config` | `waleo.sim` | ✅ 已迁移 |
| `CameraConfig` | `waleo.config` | `waleo.sim` | ✅ 已迁移 |
| `DeviceConfig` | - | `waleo.config` | ✅ 通用配置 |
| `LogConfig` | - | `waleo.config` | ✅ 通用配置 |
| `PathConfig` | - | `waleo.config` | ✅ 通用配置 |
| `SeedConfig` | - | `waleo.config` | ✅ 通用配置 |
| `TrainingConfig` | `waleo.config.shared.training` | 待迁移到 `waleo.training` | ⏳ 计划中 |
| `DatasetConfig` | `waleo.config.shared.dataset` | 待迁移到 `waleo.data` | ⏳ 计划中 |
| `EvalConfig` | `waleo.config.shared.eval` | 待迁移到 `waleo.eval` | ⏳ 计划中 |

---

## 兼容性说明

### 不提供兼容层

为了保持代码简洁和避免混淆，此次重构**不提供向后兼容层**。所有使用旧导入路径的代码必须更新。

### 迁移难度

⭐ **极低**：主要是导入路径变更，功能完全相同

```python
# 只需改变导入路径
- from waleo.config import EnvConfig
+ from waleo.sim import EnvConfig

# API 完全相同
env_config = EnvConfig(task="pick_place", robot_type="panda")  # 不变
```

---

## 优势总结

### 1. 模块独立性 ↑

- Sim 模块现在包含自己的配置，可以独立使用
- Config 模块不依赖具体领域知识

### 2. 可扩展性 ↑

- 添加新模块不需要修改 config
- 用户可以创建自己的配置类

### 3. 代码清晰度 ↑

- 配置和功能在同一模块，更易理解
- Config 模块职责更清晰

### 4. 维护性 ↑

- 领域配置变更不影响 config 模块
- 减少跨模块依赖

---

## 常见问题

### Q: 为什么要这样重构？

**A**: 集中式配置导致：
- Config 模块臃肿，包含所有领域知识
- 高耦合，config 依赖 sim, training 等模块
- 扩展困难，添加新功能需修改 config

分布式配置解决了这些问题，符合软件工程最佳实践。

### Q: 如何知道配置在哪个模块？

**A**: 遵循原则：**配置跟随功能**
- 仿真相关 → `waleo.sim`
- 训练相关 → `waleo.training`
- 数据相关 → `waleo.data`
- 跨模块通用 → `waleo.config`

### Q: Config 模块现在做什么？

**A**: Config 提供**基础设施**：
- 配置协议定义（ConfigProtocol）
- 配置工具（merge, validate）
- 通用配置（device, log, path, seed）

### Q: 旧的 config.shared 目录怎么办？

**A**: 计划逐步迁移：
- `config.shared.training` → `waleo.training`
- `config.shared.dataset` → `waleo.data`
- `config.shared.eval` → `waleo.eval` 或 `waleo.training`

目前仍保留，避免一次性变更过大。

---

## 后续工作

### 已完成 ✅

1. ✅ 将 EnvConfig, CameraConfig 移至 waleo.sim
2. ✅ 重构 waleo.config 为基础设施
3. ✅ 创建迁移文档和使用指南
4. ✅ 更新主 README

### 计划中 ⏳

1. ⏳ 创建 waleo.training 模块，迁移 TrainingConfig
2. ⏳ 创建 waleo.data 模块，迁移 DatasetConfig
3. ⏳ 更新所有示例代码使用新导入
4. ⏳ 更新设计文档反映新架构

---

## 相关文档

- [配置管理 README](../config/README.md)
- [整体架构设计](../../docs/Waleo整体架构设计.md)
- [M02 配置管理设计](../../docs/design/M02-配置管理模块设计.md)

---

## 反馈

如有问题或建议，请提交 Issue 或联系维护者。
