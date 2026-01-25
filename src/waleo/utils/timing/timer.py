"""
时间测量工具

提供代码执行时间测量和性能分析功能
"""

import time
from typing import Dict, Optional, List
from contextlib import contextmanager
from collections import defaultdict


class Timer:
    """计时器

    用于测量代码块的执行时间
    """

    def __init__(self):
        """初始化计时器"""
        self._start_time: Optional[float] = None
        self._elapsed: float = 0.0
        self._running: bool = False

    def start(self) -> None:
        """开始计时"""
        if not self._running:
            self._start_time = time.perf_counter()
            self._running = True

    def stop(self) -> float:
        """停止计时并返回经过的时间

        Returns:
            float: 经过的时间（秒）
        """
        if self._running and self._start_time is not None:
            elapsed = time.perf_counter() - self._start_time
            self._elapsed += elapsed
            self._running = False
            self._start_time = None
            return elapsed
        return 0.0

    def reset(self) -> None:
        """重置计时器"""
        self._start_time = None
        self._elapsed = 0.0
        self._running = False

    @property
    def elapsed(self) -> float:
        """获取累积时间（秒）"""
        total = self._elapsed
        if self._running and self._start_time is not None:
            total += time.perf_counter() - self._start_time
        return total

    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()


class TimerManager:
    """计时器管理器

    管理多个命名计时器，用于性能分析
    """

    def __init__(self):
        """初始化计时器管理器"""
        self._timers: Dict[str, Timer] = {}
        self._hierarchy: Dict[str, Optional[str]] = {}

    def start(self, name: str, parent: Optional[str] = None) -> None:
        """开始命名计时器

        Args:
            name: 计时器名称
            parent: 父计时器名称（用于层次化计时）
        """
        if name not in self._timers:
            self._timers[name] = Timer()
        self._hierarchy[name] = parent
        self._timers[name].start()

    def stop(self, name: str) -> float:
        """停止命名计时器

        Args:
            name: 计时器名称

        Returns:
            float: 经过的时间（秒）
        """
        if name in self._timers:
            return self._timers[name].stop()
        return 0.0

    def elapsed(self, name: str) -> float:
        """获取命名计时器的累积时间

        Args:
            name: 计时器名称

        Returns:
            float: 累积时间（秒）
        """
        if name in self._timers:
            return self._timers[name].elapsed
        return 0.0

    def reset(self, name: Optional[str] = None) -> None:
        """重置计时器

        Args:
            name: 计时器名称，None 表示重置所有
        """
        if name is None:
            for timer in self._timers.values():
                timer.reset()
        elif name in self._timers:
            self._timers[name].reset()

    def has_timer(self, name: str) -> bool:
        """检查计时器是否存在

        Args:
            name: 计时器名称

        Returns:
            bool: 是否存在
        """
        return name in self._timers

    def list_timers(self) -> List[str]:
        """列出所有计时器名称

        Returns:
            List[str]: 计时器名称列表
        """
        return list(self._timers.keys())

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """获取所有计时器的摘要

        Returns:
            Dict: 摘要字典，包含每个计时器的统计信息
        """
        summary = {}
        for name, timer in self._timers.items():
            summary[name] = {
                "elapsed": timer.elapsed,
                "parent": self._hierarchy.get(name),
            }
        return summary

    def print_summary(self) -> None:
        """打印计时器摘要"""
        summary = self.get_summary()
        print("\n=== Timer Summary ===")
        for name, stats in summary.items():
            parent_str = f" (parent: {stats['parent']})" if stats['parent'] else ""
            print(f"{name}{parent_str}: {stats['elapsed']:.4f}s")
        print("=====================\n")

    @contextmanager
    def time(self, name: str, parent: Optional[str] = None):
        """计时上下文管理器

        Args:
            name: 计时器名称
            parent: 父计时器名称

        Yields:
            Timer: 计时器实例
        """
        self.start(name, parent)
        try:
            yield self._timers[name]
        finally:
            self.stop(name)


class CodeTimer:
    """代码块计时器

    用于装饰器或上下文管理器
    """

    def __init__(self, name: str = "Code", logger=None):
        """初始化代码计时器

        Args:
            name: 计时名称
            logger: 日志记录器
        """
        self.name = name
        self.logger = logger
        self._start: Optional[float] = None

    def __enter__(self):
        """上下文管理器入口"""
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        if self._start is not None:
            elapsed = time.perf_counter() - self._start
            message = f"{self.name} took {elapsed:.4f} seconds"
            if self.logger:
                self.logger.info(message)
            else:
                print(message)

    def __call__(self, func):
        """装饰器模式"""
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


def time_function(func):
    """函数计时装饰器

    Args:
        func: 要计时的函数

    Returns:
        包装后的函数
    """
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f} seconds")
        return result
    return wrapper


@contextmanager
def timer(name: str = "Timer"):
    """简单计时上下文管理器

    Args:
        name: 计时器名称

    Yields:
        Timer: 计时器实例
    """
    t = Timer()
    t.start()
    try:
        yield t
    finally:
        elapsed = t.stop()
        print(f"{name}: {elapsed:.4f}s")
