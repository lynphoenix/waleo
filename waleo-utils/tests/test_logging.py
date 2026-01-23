"""
日志模块测试（TensorBoard）
"""

import tempfile
import shutil
from pathlib import Path
import numpy as np
import torch
from waleo_utils.logging import (
    TBLogger,
    create_logger,
    MetricTracker,
    AverageMeter,
    ProgressMeter,
)


class TestTBLogger:
    """TensorBoard 日志记录器测试"""

    def __init__(self):
        self.temp_dir = None
        self.logger = None

    def setup(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = TBLogger(self.temp_dir)

    def teardown(self):
        """清理测试环境"""
        if self.logger:
            self.logger.close()
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_create_logger(self):
        """测试创建日志记录器"""
        self.setup()

        assert self.logger is not None
        assert self.logger.log_dir.exists()
        assert self.logger.step == 0
        print(f"✓ 创建日志记录器: {self.logger.log_dir}")

        self.teardown()

    def test_log_scalar(self):
        """测试记录标量"""
        self.setup()

        self.logger.log_scalar("test/loss", 0.123, step=0)
        self.logger.log_scalar("test/accuracy", 0.956, step=0)
        self.logger.increment_step()

        assert self.logger.step == 1
        print("✓ 记录标量")

        self.teardown()

    def test_log_scalars(self):
        """测试记录多个标量"""
        self.setup()

        self.logger.log_scalars(
            "metrics",
            {"loss": 0.123, "accuracy": 0.956},
            step=0
        )
        print("✓ 记录多个标量")

        self.teardown()

    def test_log_image(self):
        """测试记录图像"""
        self.setup()

        # 创建随机图像
        image = torch.randn(3, 64, 64)
        self.logger.log_image("test/image", image, step=0)
        print("✓ 记录图像")

        self.teardown()

    def test_log_images(self):
        """测试记录多张图像"""
        self.setup()

        # 创建图像批次
        images = torch.randn(4, 3, 64, 64)
        self.logger.log_images("test/images", images, step=0)
        print("✓ 记录图像批次")

        self.teardown()

    def test_log_histogram(self):
        """测试记录直方图"""
        self.setup()

        # 创建随机数据
        data = torch.randn(1000)
        self.logger.log_histogram("test/distribution", data, step=0)
        print("✓ 记录直方图")

        self.teardown()

    def test_logger_flush(self):
        """测试日志刷新"""
        self.setup()

        self.logger.log_scalar("test/value", 1.0, step=0)
        self.logger.flush()
        print("✓ 日志刷新")

        self.teardown()


class TestMetricTracker:
    """指标跟踪器测试"""

    def test_tracker_update(self):
        """测试指标更新"""
        tracker = MetricTracker()

        tracker.update("loss", 0.5)
        tracker.update("loss", 0.3)
        tracker.update("loss", 0.2)

        assert tracker.get_count("loss") == 3
        print("✓ 指标更新")

    def test_tracker_average(self):
        """测试平均值计算"""
        tracker = MetricTracker()

        tracker.update("loss", 0.5)
        tracker.update("loss", 0.3)
        tracker.update("loss", 0.2)

        avg = tracker.get_average("loss")
        assert abs(avg - 0.333) < 0.01
        print(f"✓ 平均值计算: {avg:.3f}")

    def test_tracker_statistics(self):
        """测试统计指标"""
        tracker = MetricTracker()

        values = [0.5, 0.3, 0.2, 0.4, 0.6]
        for v in values:
            tracker.update("metric", v)

        minimum = tracker.get_min("metric")
        maximum = tracker.get_max("metric")
        std = tracker.get_std("metric")

        assert abs(minimum - 0.2) < 0.01
        assert abs(maximum - 0.6) < 0.01
        print(f"✓ 统计指标: min={minimum:.2f}, max={maximum:.2f}, std={std:.3f}")

    def test_tracker_latest(self):
        """测试获取最新值"""
        tracker = MetricTracker()

        tracker.update("value", 1.0)
        tracker.update("value", 2.0)
        tracker.update("value", 3.0)

        latest = tracker.get_latest("value")
        assert latest == 3.0
        print(f"✓ 最新值: {latest}")

    def test_tracker_reset(self):
        """测试重置跟踪器"""
        tracker = MetricTracker()

        tracker.update("loss", 0.5)
        tracker.reset("loss")

        assert tracker.get_count("loss") == 0
        print("✓ 重置跟踪器")

    def test_tracker_summary(self):
        """测试摘要"""
        tracker = MetricTracker()

        for i in range(10):
            tracker.update("metric", float(i))

        summary = tracker.get_summary()
        assert "metric" in summary
        assert "avg" in summary["metric"]
        print("✓ 获取摘要")


class TestAverageMeter:
    """平均值计量器测试"""

    def test_average_meter(self):
        """测试平均值计量器"""
        meter = AverageMeter()

        meter.update(0.5, n=1)
        meter.update(0.3, n=1)
        meter.update(0.2, n=1)

        assert abs(meter.avg - 0.333) < 0.01
        assert meter.count == 3
        print(f"✓ 平均值计量器: avg={meter.avg:.3f}, count={meter.count}")

    def test_meter_batch(self):
        """测试批量更新"""
        meter = AverageMeter()

        meter.update(1.0, n=4)
        assert meter.avg == 1.0
        assert meter.count == 4
        print("✓ 批量更新")


class TestProgressMeter:
    """进度计量器测试"""

    def test_progress_meter(self):
        """测试进度计量器"""
        meters = [AverageMeter("loss", ":.4f"), AverageMeter("acc", ":.2f")]
        progress = ProgressMeter(num_batches=100, meters=meters, prefix="Train: ")

        meters[0].update(0.5)
        meters[1].update(0.95)

        display_str = progress.display(10)
        assert "Train:" in display_str
        print(f"✓ 进度显示: {display_str}")


def run_tests():
    """运行所有测试"""
    print("\n" + "="*50)
    print("日志模块测试（TensorBoard）")
    print("="*50 + "\n")

    passed = 0
    failed = 0

    # TBLogger 测试
    print("=== TBLogger 测试 ===")
    test_logger = TestTBLogger()
    logger_tests = [
        ("创建日志记录器", test_logger.test_create_logger),
        ("记录标量", test_logger.test_log_scalar),
        ("记录多个标量", test_logger.test_log_scalars),
        ("记录图像", test_logger.test_log_image),
        ("记录图像批次", test_logger.test_log_images),
        ("记录直方图", test_logger.test_log_histogram),
        ("日志刷新", test_logger.test_logger_flush),
    ]

    for name, test_func in logger_tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed += 1

    # MetricTracker 测试
    print("\n=== MetricTracker 测试 ===")
    tracker_tests = [
        ("指标更新", TestMetricTracker().test_tracker_update),
        ("平均值计算", TestMetricTracker().test_tracker_average),
        ("统计指标", TestMetricTracker().test_tracker_statistics),
        ("最新值", TestMetricTracker().test_tracker_latest),
        ("重置跟踪器", TestMetricTracker().test_tracker_reset),
        ("获取摘要", TestMetricTracker().test_tracker_summary),
    ]

    for name, test_func in tracker_tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed += 1

    # AverageMeter 测试
    print("\n=== AverageMeter 测试 ===")
    meter_tests = [
        ("平均值计量器", TestAverageMeter().test_average_meter),
        ("批量更新", TestAverageMeter().test_meter_batch),
    ]

    for name, test_func in meter_tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed += 1

    # ProgressMeter 测试
    print("\n=== ProgressMeter 测试 ===")
    progress_tests = [
        ("进度显示", TestProgressMeter().test_progress_meter),
    ]

    for name, test_func in progress_tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed += 1

    print("\n" + "-"*50)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("-"*50 + "\n")

    return failed == 0


if __name__ == "__main__":
    run_tests()
