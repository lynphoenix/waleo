# 项目结构迁移：从 src-layout 到 flat-layout

## 迁移日期
2026-01-25

## 迁移原因

1. **简洁性**：减少目录层次，更直观
2. **主流做法**：NumPy, Pandas, Requests 等知名项目都采用 flat-layout
3. **开发便利**：更容易导航和定位文件

## 结构变化

### 迁移前（src-layout）

```
waleo/
├── src/
│   └── waleo/          ← 包目录在 src/ 下
│       ├── utils/
│       ├── config/
│       └── sim/
├── tests/
├── examples/
└── pyproject.toml
```

### 迁移后（flat-layout）

```
waleo/
├── waleo/              ← 包目录直接在根目录
│   ├── utils/
│   ├── config/
│   └── sim/
├── tests/
├── examples/
└── pyproject.toml
```

## 变更详情

### 1. 目录结构

```bash
# 移动包目录
mv src/waleo ./waleo

# 删除空的 src 目录
rm -rf src/
```

### 2. pyproject.toml 配置

**修改前**：
```toml
[tool.setuptools.packages.find]
where = ["src"]
include = ["*"]

[tool.setuptools.package-dir]
"" = "src"
```

**修改后**：
```toml
[tool.setuptools.packages.find]
include = ["waleo*"]
exclude = ["tests*"]
```

### 3. 工具配置更新

**isort**:
```toml
# 修改前
src_paths = ["src", "tests"]

# 修改后
src_paths = ["waleo", "tests"]
```

**mypy**:
```toml
# 修改前
mypy_path = "src"

# 修改后
mypy_path = "waleo"
```

**pytest**:
```toml
# 修改前
addopts = "-v --cov=src --cov-report=term-missing"

# 修改后
addopts = "-v --cov=waleo --cov-report=term-missing"
```

### 4. 文档路径更新

更新了以下文档中的路径引用：
- `README.md` - 项目结构和快速链接
- `waleo/utils/README.md` - 相关文档链接
- `waleo/config/README.md` - 相关文档链接

## 对用户的影响

### 导入路径：无变化 ✅

```python
# 导入路径完全不变
from waleo.utils import get_training_device
from waleo.config import DeviceConfig
from waleo.sim import EnvConfig

# 功能完全相同
```

### 开发流程：无变化 ✅

```bash
# 安装方式不变
pip install -e .

# 测试方式不变
pytest tests/

# 使用方式不变
python examples/maniskill/train_ppo.py
```

### 文件路径：有变化 ⚠️

```
# 修改前
waleo/src/waleo/utils/device.py

# 修改后
waleo/waleo/utils/device.py
```

## 验证结果

### 1. 安装验证 ✅

```bash
$ pip install -e .
Successfully installed waleo-0.1.0

$ pip list | grep waleo
waleo    0.1.0    /root/data2/lyn/waleo
```

### 2. 导入验证 ✅

```python
>>> import waleo
>>> waleo.__version__
'0.1.0'

>>> from waleo import utils, config, sim
>>> # 所有子模块正常导入
```

### 3. 测试验证 ✅

```bash
$ python tests/test_utils/test_comprehensive.py
总计: 120/120 测试通过
成功率: 100.0%
```

## Flat-layout 的注意事项

### 1. 开发规范

**必须先安装包**：
```bash
# 在开发前必须先安装（editable mode）
pip install -e .

# 修改代码后无需重新安装（editable mode 自动生效）
```

**避免在项目根目录直接运行 Python**：
```bash
# ❌ 不推荐（可能导入本地文件而非安装的包）
cd /path/to/waleo
python

# ✅ 推荐（先安装，然后在任何目录运行）
pip install -e .
cd /tmp
python
```

### 2. CI/CD 配置

确保 CI 流程先安装包：

```yaml
# .github/workflows/test.yml
steps:
  - name: Install package
    run: pip install -e .

  - name: Run tests
    run: pytest tests/
```

### 3. IDE 配置

确保 IDE 识别正确的包路径：
- PyCharm: Mark Directory as > Sources Root（选择 waleo 目录）
- VSCode: 设置 `python.analysis.extraPaths` 为项目根目录

## 与 src-layout 的对比

| 特性 | src-layout | flat-layout（当前）|
|-----|-----------|------------------|
| 目录层次 | 多一层 src/ | 少一层，更简洁 |
| 误导入风险 | 低（强制安装）| 需要注意开发规范 |
| 主流程度 | 现代推荐 | 传统主流 |
| 知名项目 | Flask, FastAPI | NumPy, Pandas, Requests |
| 配置复杂度 | 略复杂 | 简单 |

## 回滚方案（如需）

如果需要回滚到 src-layout：

```bash
# 1. 创建 src 目录并移动包
mkdir src
mv waleo src/

# 2. 恢复 pyproject.toml 配置
[tool.setuptools.packages.find]
where = ["src"]
include = ["*"]

[tool.setuptools.package-dir]
"" = "src"

# 3. 重新安装
pip uninstall -y waleo
pip install -e .
```

## 相关 PR 和 Commit

- Commit: 即将创建
- 相关讨论: flat-layout vs src-layout 的架构选择

## 参考资料

### Flat-layout 示例项目
- [NumPy](https://github.com/numpy/numpy)
- [Pandas](https://github.com/pandas-dev/pandas)
- [Requests](https://github.com/psf/requests)
- [Django](https://github.com/django/django)

### Src-layout 示例项目
- [Flask](https://github.com/pallets/flask)
- [FastAPI](https://github.com/tiangolo/fastapi)
- [pydantic](https://github.com/pydantic/pydantic)

### 相关文档
- [Python Packaging User Guide](https://packaging.python.org/)
- [PEP 517 - Build System Interface](https://peps.python.org/pep-0517/)
- [PEP 660 - Editable Installs](https://peps.python.org/pep-0660/)

## 总结

✅ **迁移成功完成**

- 包结构从 src-layout 改为 flat-layout
- 所有测试通过（120/120）
- 导入路径无变化
- 用户使用无影响
- 文档已同步更新

**优势**：
- 目录结构更简洁
- 与主流项目一致
- 更容易导航

**注意事项**：
- 遵循开发规范（先安装）
- CI/CD 确保先安装再测试
