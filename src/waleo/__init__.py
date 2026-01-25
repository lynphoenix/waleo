"""
Waleo - Robotics Learning Framework

一个完整的机器人学习框架，提供：
- 基础设施工具 (waleo.utils)
- 配置管理系统 (waleo.config)
- 仿真环境 (waleo.sim)
"""

__version__ = "0.1.0"

# 导入子模块，使用户可以直接访问
from . import utils
from . import config
from . import sim

__all__ = [
    "utils",
    "config",
    "sim",
    "__version__",
]
