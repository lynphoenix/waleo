"""
日志格式化工具

基于 TensorBoard 的日志格式化和输出
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch


class TBLogger:
    """TensorBoard 日志记录器

    提供基于 TensorBoard 的日志记录功能，支持标量、图像、直方图等
    """

    def __init__(
        self,
        log_dir: Union[str, Path],
        comment: str = "",
        flush_secs: int = 120,
    ):
        """初始化 TensorBoard 日志记录器

        Args:
            log_dir: 日志目录
            comment: 日志目录后缀注释
            flush_secs: 刷新间隔（秒）
        """
        self.log_dir = Path(log_dir)
        self.writer = SummaryWriter(
            log_dir=str(self.log_dir),
            comment=comment,
            flush_secs=flush_secs,
        )
        self._step = 0

    def log_scalar(
        self,
        tag: str,
        value: float,
        step: Optional[int] = None,
    ) -> None:
        """记录标量

        Args:
            tag: 标签名称
            value: 标量值
            step: 步数，如果为 None 则使用内部计数器
        """
        if step is None:
            step = self._step
        self.writer.add_scalar(tag, value, step)

    def log_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """记录多个标量

        Args:
            main_tag: 主标签名称
            tag_scalar_dict: 标签-值字典
            step: 步数
        """
        if step is None:
            step = self._step
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_image(
        self,
        tag: str,
        image: Union[np.ndarray, torch.Tensor],
        step: Optional[int] = None,
        dataformats: str = "CHW",
    ) -> None:
        """记录图像

        Args:
            tag: 标签名称
            image: 图像数据 (numpy array or torch tensor)
            step: 步数
            dataformats: 图像数据格式，默认为 CHW
        """
        if step is None:
            step = self._step

        if isinstance(image, torch.Tensor):
            image = image.detach().cpu()

        self.writer.add_image(tag, image, step, dataformats=dataformats)

    def log_images(
        self,
        tag: str,
        images: Union[np.ndarray, torch.Tensor],
        step: Optional[int] = None,
        dataformats: str = "NCHW",
    ) -> None:
        """记录多张图像

        Args:
            tag: 标签名称
            images: 图像批次数据
            step: 步数
            dataformats: 图像数据格式，默认为 NCHW
        """
        if step is None:
            step = self._step

        if isinstance(images, torch.Tensor):
            images = images.detach().cpu()

        self.writer.add_images(tag, images, step, dataformats=dataformats)

    def log_histogram(
        self,
        tag: str,
        values: Union[np.ndarray, torch.Tensor],
        step: Optional[int] = None,
    ) -> None:
        """记录直方图

        Args:
            tag: 标签名称
            values: 值数组
            step: 步数
        """
        if step is None:
            step = self._step

        if isinstance(values, torch.Tensor):
            values = values.detach().cpu()

        self.writer.add_histogram(tag, values, step)

    def log_histogram_raw(
        self,
        tag: str,
        min: float,
        max: float,
        num: int,
        sum: float,
        sum_squares: float,
        bucket_limits: np.ndarray,
        bucket_counts: np.ndarray,
        step: Optional[int] = None,
    ) -> None:
        """记录原始直方图数据

        Args:
            tag: 标签名称
            min: 最小值
            max: 最大值
            num: 数量
            sum: 总和
            sum_squares: 平方和
            bucket_limits: 分桶边界
            bucket_counts: 分桶计数
            step: 步数
        """
        if step is None:
            step = self._step
        self.writer.add_histogram_raw(
            tag, min, max, num, sum, sum_squares,
            bucket_limits, bucket_counts, step
        )

    def log_graph(
        self,
        model: torch.nn.Module,
        input_to_model: Union[torch.Tensor, tuple],
    ) -> None:
        """记录模型计算图

        Args:
            model: PyTorch 模型
            input_to_model: 模型输入
        """
        self.writer.add_graph(model, input_to_model)

    def log_hparams(
        self,
        hparam_dict: Dict[str, Any],
        metric_dict: Dict[str, float],
    ) -> None:
        """记录超参数和指标

        Args:
            hparam_dict: 超参数字典
            metric_dict: 指标字典
        """
        self.writer.add_hparams(hparam_dict, metric_dict)

    def log_text(
        self,
        tag: str,
        text: str,
        step: Optional[int] = None,
    ) -> None:
        """记录文本

        Args:
            tag: 标签名称
            text: 文本内容
            step: 步数
        """
        if step is None:
            step = self._step
        self.writer.add_text(tag, text, step)

    def log_video(
        self,
        tag: str,
        video: Union[np.ndarray, torch.Tensor],
        step: Optional[int] = None,
        fps: int = 4,
    ) -> None:
        """记录视频

        Args:
            tag: 标签名称
            video: 视频数据 (T x H x W x C) or (B x T x H x W x C)
            step: 步数
            fps: 帧率
        """
        if step is None:
            step = self._step

        if isinstance(video, torch.Tensor):
            video = video.detach().cpu()

        self.writer.add_video(tag, video, step, fps=fps)

    def log_audio(
        self,
        tag: str,
        audio: Union[np.ndarray, torch.Tensor],
        step: Optional[int] = None,
        sample_rate: int = 44100,
    ) -> None:
        """记录音频

        Args:
            tag: 标签名称
            audio: 音频数据
            step: 步数
            sample_rate: 采样率
        """
        if step is None:
            step = self._step

        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu()

        self.writer.add_audio(tag, audio, step, sample_rate=sample_rate)

    def log_embedding(
        self,
        mat: Union[np.ndarray, torch.Tensor],
        metadata: Optional[list] = None,
        label_img: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> None:
        """记录嵌入向量

        Args:
            mat: 嵌入矩阵
            metadata: 元数据标签列表
            label_img: 标签图像
        """
        if isinstance(mat, torch.Tensor):
            mat = mat.detach().cpu()

        if label_img is not None and isinstance(label_img, torch.Tensor):
            label_img = label_img.detach().cpu()

        self.writer.add_embedding(mat, metadata, label_img)

    def increment_step(self) -> None:
        """增加内部步数计数器"""
        self._step += 1

    @property
    def step(self) -> int:
        """获取当前步数"""
        return self._step

    @step.setter
    def step(self, value: int) -> None:
        """设置步数"""
        self._step = value

    def flush(self) -> None:
        """刷新日志到磁盘"""
        self.writer.flush()

    def close(self) -> None:
        """关闭日志记录器"""
        self.writer.close()

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()


def create_logger(
    log_dir: Union[str, Path],
    name: str = "waleo",
    comment: str = "",
) -> TBLogger:
    """创建日志记录器

    Args:
        log_dir: 基础日志目录
        name: 日志名称
        comment: 日志目录后缀注释

    Returns:
        TBLogger: 日志记录器实例
    """
    log_path = Path(log_dir) / name
    return TBLogger(log_path, comment)
