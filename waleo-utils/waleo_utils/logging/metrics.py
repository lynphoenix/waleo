"""
指标跟踪工具

提供训练指标跟踪和统计功能
"""

from typing import Dict, List, Optional, Union
import numpy as np
import torch
from pathlib import Path
from waleo_utils.logging.formatter import TBLogger


class MetricTracker:
    """指标跟踪器

    跟踪和统计训练过程中的各项指标
    """

    def __init__(
        self,
        logger: Optional[TBLogger] = None,
        window_size: int = 100,
    ):
        """初始化指标跟踪器

        Args:
            logger: TensorBoard 日志记录器
            window_size: 滑动窗口大小
        """
        self.logger = logger
        self.window_size = window_size
        self._metrics: Dict[str, List[float]] = {}
        self._counts: Dict[str, int] = {}

    def update(
        self,
        name: str,
        value: Union[float, int, torch.Tensor, np.ndarray],
    ) -> None:
        """更新指标

        Args:
            name: 指标名称
            value: 指标值
        """
        # 转换为 float
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        elif isinstance(value, np.ndarray):
            value = float(value)
        else:
            value = float(value)

        # 添加到历史记录
        if name not in self._metrics:
            self._metrics[name] = []
            self._counts[name] = 0

        self._metrics[name].append(value)
        self._counts[name] += 1

        # 保持窗口大小
        if len(self._metrics[name]) > self.window_size:
            self._metrics[name].pop(0)

        # 记录到 TensorBoard
        if self.logger is not None:
            self.logger.log_scalar(f"metrics/{name}", value, self.logger.step)

    def get_average(self, name: str, window: Optional[int] = None) -> float:
        """获取平均值

        Args:
            name: 指标名称
            window: 窗口大小，None 表示全部历史

        Returns:
            float: 平均值

        Raises:
            ValueError: 如果指标不存在
        """
        if name not in self._metrics:
            raise ValueError(f"Metric '{name}' not found")

        values = self._metrics[name]
        if window is not None and len(values) > window:
            values = values[-window:]

        return float(np.mean(values))

    def get_sum(self, name: str) -> float:
        """获取总和

        Args:
            name: 指标名称

        Returns:
            float: 总和
        """
        if name not in self._metrics:
            raise ValueError(f"Metric '{name}' not found")

        return float(np.sum(self._metrics[name]))

    def get_count(self, name: str) -> int:
        """获取计数

        Args:
            name: 指标名称

        Returns:
            int: 计数
        """
        if name not in self._metrics:
            raise ValueError(f"Metric '{name}' not found")

        return self._counts[name]

    def get_std(self, name: str, window: Optional[int] = None) -> float:
        """获取标准差

        Args:
            name: 指标名称
            window: 窗口大小

        Returns:
            float: 标准差
        """
        if name not in self._metrics:
            raise ValueError(f"Metric '{name}' not found")

        values = self._metrics[name]
        if window is not None and len(values) > window:
            values = values[-window:]

        return float(np.std(values))

    def get_min(self, name: str) -> float:
        """获取最小值

        Args:
            name: 指标名称

        Returns:
            float: 最小值
        """
        if name not in self._metrics:
            raise ValueError(f"Metric '{name}' not found")

        return float(np.min(self._metrics[name]))

    def get_max(self, name: str) -> float:
        """获取最大值

        Args:
            name: 指标名称

        Returns:
            float: 最大值
        """
        if name not in self._metrics:
            raise ValueError(f"Metric '{name}' not found")

        return float(np.max(self._metrics[name]))

    def get_latest(self, name: str) -> Optional[float]:
        """获取最新值

        Args:
            name: 指标名称

        Returns:
            float: 最新值，如果不存在返回 None
        """
        if name not in self._metrics or len(self._metrics[name]) == 0:
            return None
        return self._metrics[name][-1]

    def get_all(self, name: str) -> List[float]:
        """获取所有历史值

        Args:
            name: 指标名称

        Returns:
            List[float]: 历史值列表
        """
        if name not in self._metrics:
            raise ValueError(f"Metric '{name}' not found")

        return self._metrics[name].copy()

    def has_metric(self, name: str) -> bool:
        """检查指标是否存在

        Args:
            name: 指标名称

        Returns:
            bool: 是否存在
        """
        return name in self._metrics

    def reset(self, name: Optional[str] = None) -> None:
        """重置指标

        Args:
            name: 指标名称，None 表示重置所有
        """
        if name is None:
            self._metrics.clear()
            self._counts.clear()
        elif name in self._metrics:
            self._metrics[name].clear()
            self._counts[name] = 0

    def list_metrics(self) -> List[str]:
        """列出所有指标名称

        Returns:
            List[str]: 指标名称列表
        """
        return list(self._metrics.keys())

    def log_summary(self, step: Optional[int] = None) -> None:
        """记录摘要到 TensorBoard

        Args:
            step: 步数
        """
        if self.logger is None:
            return

        for name in self._metrics:
            if len(self._metrics[name]) > 0:
                avg = self.get_average(name)
                self.logger.log_scalar(f"summary/{name}_avg", avg, step)

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """获取所有指标的摘要统计

        Returns:
            Dict: 摘要统计字典
        """
        summary = {}
        for name in self._metrics:
            if len(self._metrics[name]) > 0:
                summary[name] = {
                    "avg": self.get_average(name),
                    "std": self.get_std(name),
                    "min": self.get_min(name),
                    "max": self.get_max(name),
                    "count": self.get_count(name),
                    "latest": self.get_latest(name),
                }
        return summary


class AverageMeter:
    """平均值计量器

    用于计算和存储平均值和当前值
    """

    def __init__(self, name: str = "", fmt: str = ":f"):
        """初始化计量器

        Args:
            name: 名称
            fmt: 格式化字符串
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        """重置计量器"""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """更新计量器

        Args:
            val: 值
            n: 数量
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __str__(self) -> str:
        """字符串表示"""
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """进度计量器

    显示训练进度和指标
    """

    def __init__(
        self,
        num_batches: int,
        meters: List[AverageMeter],
        prefix: str = "",
    ):
        """初始化进度计量器

        Args:
            num_batches: 总批次数量
            meters: 计量器列表
            prefix: 前缀字符串
        """
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int) -> str:
        """显示进度

        Args:
            batch: 当前批次

        Returns:
            str: 格式化的进度字符串
        """
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return "\t".join(entries)

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        """获取批次格式化字符串"""
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
