# Migration Documents（迁移文档）

记录 Waleo 项目重大架构变更的迁移指南。

## 文档列表

### Config模块重构迁移指南.md

**日期**：2026-01-25
**变更**：Config 模块从集中式架构改为分布式架构

**关键变化**：
- EnvConfig 和 CameraConfig 从 `waleo.config` 迁移到 `waleo.sim`
- waleo.config 现在只提供配置管理基础设施（协议、工具、通用配置）
- 领域配置由各自模块管理

**影响**：
- 需要更新导入语句
- 示例代码和测试需要修改

---

### 项目结构迁移-src-to-flat.md

**日期**：2026-01-25
**变更**：从 src-layout 迁移到 flat-layout

**关键变化**：
- 目录结构：`waleo/src/waleo/` → `waleo/waleo/`
- pyproject.toml 配置更新
- 所有路径引用更新

**影响**：
- 包仍然是 `waleo`，导入路径不变
- 开发需要 `pip install -e .`
- 更符合主流项目规范（NumPy, Pandas等）

---

### 包结构重构报告.md

**日期**：2026-01-25
**变更**：从 3 个独立包合并为单一包

**关键变化**：
- waleo-utils, waleo-config, waleo-sim → 单一 waleo 包
- 统一的 pyproject.toml 和版本管理
- 子模块作为包的一部分

**影响**：
- 安装更简单（`pip install waleo` 即可）
- 版本管理更容易
- 依赖关系更清晰

---

## 迁移时间线

```
2026-01-25
├── 上午：3个包合并为1个（包结构重构）
├── 中午：迁移到 flat-layout（项目结构迁移）
└── 下午：Config 模块分布式架构（Config重构）
```

## 所有迁移的共同原则

1. **向后兼容**：尽量保持 API 不变
2. **文档优先**：先写迁移指南再执行
3. **测试验证**：迁移后运行完整测试套件
4. **分步进行**：一次一个大变更
5. **记录原因**：说明为什么这样改

## 参考资料

- Python Packaging User Guide: https://packaging.python.org/
- NumPy 项目结构: https://github.com/numpy/numpy
- PEP 660 - Editable Installs: https://peps.python.org/pep-0660/
