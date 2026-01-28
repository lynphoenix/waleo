# waleo.sim 重构设计方案

> **日期**: 2026-01-26
> **状态**: 设计中
> **目标**: 清晰的多后端集成架构

---

## 1. 当前问题分析

### 1.1 目录结构混乱

```
当前 waleo/sim/
├── base/        ⚠️ 抽象基类（定义了但未充分使用）
├── backends/    ⚠️ 旧后端抽象层（ManiSkill3 不走这里）
├── tasks/       ⚠️ 任务封装（部分使用）
├── registry/    ✅ 资源管理（当前核心）
├── wrappers/    ✅ 配置注入（当前核心）
└── factory.py   ✅ 工厂函数（当前入口）
```

**问题**：
- 两套设计混在一起
- 职责不清晰
- 代码路径混乱

### 1.2 实际使用情况

**ManiSkill3**（主要使用）：
```python
make_env("PickCube-v1", backend="maniskill")
  → factory._make_maniskill_env()
  → gym.make()  # 直接调用，不走 backends/
```

**MuJoCo / PyBullet**：
```python
make_env("xxx", backend="mujoco")
  → factory._make_mujoco_env()
  → backends.MuJoCoBackend  # 走旧的后端抽象层
```

**不一致！** ManiSkill3 直接用 gym.make()，其他后端用抽象层。

---

## 2. 重构设计目标

### 2.1 核心原则

1. **统一接口**：所有后端通过相同的路径调用
2. **清晰职责**：每个目录/模块有明确使命
3. **无侵入**：保持对上游库的无侵入式集成
4. **可扩展**：易于添加新后端

### 2.2 设计理念

```
用户: make_env(task, backend, robot_id)
    ↓
factory: 解析参数，选择后端
    ↓
backends: 统一的后端抽象层
    ↓
registry: 资源管理（robot.yaml, URDF, mesh）
    ↓
wrappers: 配置注入（robot_pose, keyframes, camera）
    ↓
返回配置好的环境
```

---

## 3. 新目录结构

```
waleo/sim/
├── __init__.py              # 公开 API
├── config.py                # 配置类（EnvConfig, CameraConfig）
│
├── factory.py               # 🌟 工厂函数（用户入口）
│   └── make_env()           # 统一入口
│
├── backends/                # 🌟 后端实现（统一抽象）
│   ├── __init__.py
│   ├── base.py              # 后端抽象基类
│   ├── maniskill.py         # ManiSkill3 后端
│   ├── mujoco.py            # MuJoCo 后端
│   ├── pybullet.py          # PyBullet 后端
│   └── isaac.py             # Isaac Gym 后端（可选）
│
├── registry/                # 🌟 资源管理
│   ├── __init__.py
│   ├── resolver.py          # AssetResolver（路径解析）
│   └── robot.py             # RobotRegistry（YAML 加载）
│
├── wrappers/                # 🌟 配置注入
│   ├── __init__.py
│   └── robot.py             # RobotWrapper, CameraWrapper, TaskWrapper
│
└── utils/                   # 工具函数
    ├── __init__.py
    └── common.py            # 通用工具函数
```

**删除的目录**：
- `base/` - 抽象基类移到 `backends/base.py`
- `tasks/` - 任务封装逻辑集成到各后端

---

## 4. 核心模块设计

### 4.1 factory.py（统一入口）

```python
def make_env(
    task: str,
    backend: str = "maniskill",
    robot_id: str | None = None,
    **kwargs
) -> gym.Env:
    """创建仿真环境

    所有后端统一入口，内部路由到具体后端实现。

    Args:
        task: 任务名称（如 "PickCube-v1"）
        backend: 后端类型 ("maniskill", "mujoco", "pybullet", "isaac")
        robot_id: 自定义机器人 ID
        **kwargs: 后端特定参数

    Returns:
        gym.Env: 配置好的环境
    """
    # 1. 选择后端
    backend_cls = BackendFactory.get_backend(backend)

    # 2. 创建环境
    env = backend_cls.create(task, **kwargs)

    # 3. 应用自定义机器人配置
    if robot_id:
        env = apply_robot_config(env, robot_id, task)

    return env
```

### 4.2 backends/base.py（后端抽象）

```python
class SimulationBackend(ABC):
    """仿真后端抽象基类

    所有后端必须实现此接口，确保统一的调用方式。
    """

    @classmethod
    @abstractmethod
    def create(cls, task: str, **kwargs) -> gym.Env:
        """创建环境

        Args:
            task: 任务名称
            **kwargs: 后端特定参数

        Returns:
            gym.Env: 环境实例
        """
        pass

    @classmethod
    def is_available(cls) -> bool:
        """检查后端是否可用"""
        try:
            import_module(cls.import_name)
            return True
        except ImportError:
            return False

    @property
    @abstractmethod
    def import_name(self) -> str:
        """导入名称"""
        pass
```

### 4.3 backends/maniskill.py（ManiSkill3）

```python
class ManiSkillBackend(SimulationBackend):
    """ManiSkill3 后端

    直接使用 gym.make() 创建环境，符合 Gymnasium 标准。
    """

    import_name = "mani_skill"

    @classmethod
    def create(cls, task: str, **kwargs) -> gym.Env:
        """创建 ManiSkill3 环境

        Args:
            task: 如 "PickCube-v1"
            **kwargs: num_envs, obs_mode, control_mode 等

        Returns:
            gym.Env: ManiSkill3 环境
        """
        import mani_skill.envs

        # ManiSkill3 环境命名规范
        if not task.startswith("ManiSkill"):
            task = f"ManiSkill{task}"

        return gym.make(task, **kwargs)
```

### 4.4 backends/mujoco.py（MuJoCo）

```python
class MuJoCoBackend(SimulationBackend):
    """MuJoCo 后端

    通过 mujoco-python 包创建环境。
    """

    import_name = "mujoco"

    @classmethod
    def create(cls, task: str, **kwargs) -> gym.Env:
        """创建 MuJoCo 环境

        MuJoCo 任务需要 XML 文件，通过 task 参数指定路径。
        """
        import mujoco

        # 加载 XML 模型
        if task.endswith(".xml"):
            model = mujoco.MjModel.from_xml_path(task)
        else:
            # 从注册表查找
            model_path = resolve_model_path(task)
            model = mujoco.MjModel.from_xml_path(model_path)

        # 包装为 Gym 环境
        from waleo.sim.backends.mujoco_wrapper import MuJoCoWrapper
        return MuJoCoWrapper(model, **kwargs)
```

---

## 5. 职责划分

| 模块 | 职责 | 接口 |
|------|------|------|
| `factory.py` | 参数解析、后端路由 | `make_env()` |
| `backends/` | 后端实现、环境创建 | `SimulationBackend.create()` |
| `registry/` | 资源管理、YAML 加载 | `AssetResolver`, `RobotRegistry` |
| `wrappers/` | 配置注入、行为修改 | `RobotWrapper` 等 |
| `utils/` | 通用工具函数 | 辅助函数 |

---

## 6. 调用流程

```
用户代码
    ↓
make_env("PickCube-v1", backend="maniskill", robot_id="RJ2506")
    ↓
┌─────────────────────────────────────────────┐
│ factory.py                                   │
│  1. 解析参数                                 │
│  2. 选择后端: ManiSkillBackend              │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ backends/maniskill.py                        │
│  1. import mani_skill.envs                  │
│  2. env = gym.make("ManiSkillPickCube-v1")  │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ registry/robot.py                            │
│  1. RobotRegistry.get("RJ2506")             │
│  2. 加载 robot.yaml                          │
│  3. 获取 task_config                         │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ wrappers/robot.py                            │
│  1. CustomRobotWrapper(env, config)         │
│  2. 应用 robot_pose                          │
│  3. 应用 keyframes                           │
│  4. 应用 camera_config                       │
└─────────────────────────────────────────────┘
    ↓
返回配置好的环境
```

---

## 7. 后端对比

| 后端 | 状态 | 创建方式 | 任务指定 |
|------|------|---------|---------|
| **ManiSkill3** | ✅ 主要 | `gym.make()` | 字符串 ID |
| **MuJoCo** | ⚠️ 待实现 | `mujoco.MjModel` | XML 路径 |
| **PyBullet** | ⚠️ 待实现 | `pybullet.connect()` | URDF 路径 |
| **Isaac Gym** | ❌ 未实现 | `isaacgym.gymapi.acquire_gym()` | 待定 |

---

## 8. 迁移计划

### Phase 1: 重构 ManiSkill3（必须）
- [ ] 实现 `backends/base.py`
- [ ] 重构 `backends/maniskill.py`
- [ ] 更新 `factory.py` 使用新接口
- [ ] 删除旧的 `tasks/` 目录

### Phase 2: 实现其他后端（推荐）
- [ ] 实现 `backends/mujoco.py`
- [ ] 实现 `backends/pybullet.py`
- [ ] 添加测试

### Phase 3: 清理遗留代码（可选）
- [ ] 删除 `base/` 目录
- [ ] 更新文档

---

## 9. API 示例

### 9.1 统一接口

```python
from waleo.sim import make_env

# ManiSkill3
env = make_env("PickCube-v1", backend="maniskill")

# MuJoCo（未来）
env = make_env("my_robot.xml", backend="mujoco")

# PyBullet（未来）
env = make_env("kuka_iiwa", backend="pybullet")
```

### 9.2 自定义机器人

```python
# 所有后端统一方式
env = make_env(
    task="PickCube-v1",
    backend="maniskill",
    robot_id="RJ2506"  # 自动加载配置并注入
)
```

---

## 10. 文件清单

### 需要修改的文件
- `waleo/sim/__init__.py` - 更新导出
- `waleo/sim/factory.py` - 重构路由逻辑
- `waleo/sim/backends/__init__.py` - 新增后端注册

### 需要新建的文件
- `waleo/sim/backends/base.py` - 后端抽象基类
- `waleo/sim/backends/maniskill.py` - ManiSkill3 后端（重构）

### 需要删除的文件
- `waleo/sim/base/` - 整个目录
- `waleo/sim/tasks/` - 整个目录

### 需要更新的文件
- `waleo/sim/wrappers/__init__.py` - 更新导入
- `waleo/sim/registry/__init__.py` - 保持不变
- `tests/test_sim/` - 更新测试

---

## 11. 风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 破坏现有代码 | 高 | 保持向后兼容，保留旧 API |
| 后端差异过大 | 中 | 抽象层设计合理，允许后端特定参数 |
| 测试覆盖不足 | 中 | 逐步迁移，保持测试 |

---

## 12. 总结

**新设计优势**：
1. ✅ **职责清晰**：每个模块使命明确
2. ✅ **接口统一**：所有后端相同调用方式
3. ✅ **易于扩展**：添加新后端只需实现 `SimulationBackend`
4. ✅ **无侵入**：保持对上游库的无侵入式集成

**下一步**：确认方案后，开始 Phase 1 实施
