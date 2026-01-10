"""
时间测量模块测试
"""

import time
import torch
from waleo_utils.timing import (
    Timer,
    TimerManager,
    CodeTimer,
    time_function,
    timer,
)


class TestTimer:
    """计时器测试"""

    def test_basic_timer(self):
        """测试基础计时器"""
        t = Timer()
        t.start()
        time.sleep(0.1)
        elapsed = t.stop()

        assert elapsed >= 0.1
        assert 0.5 > elapsed  # 应该在合理范围内
        print(f"✓ 基础计时器: {elapsed:.3f}s")

    def test_timer_context_manager(self):
        """测试计时器上下文管理器"""
        with Timer() as t:
            time.sleep(0.05)

        assert t.elapsed >= 0.05
        print(f"✓ 计时器上下文管理器: {t.elapsed:.3f}s")

    def test_timer_accumulation(self):
        """测试时间累积"""
        t = Timer()

        t.start()
        time.sleep(0.05)
        t.stop()

        t.start()
        time.sleep(0.05)
        t.stop()

        assert t.elapsed >= 0.1
        print(f"✓ 时间累积: {t.elapsed:.3f}s")

    def test_timer_reset(self):
        """测试计时器重置"""
        t = Timer()
        t.start()
        time.sleep(0.05)
        t.stop()
        t.reset()

        assert t.elapsed == 0.0
        print("✓ 计时器重置")


class TestTimerManager:
    """计时器管理器测试"""

    def test_named_timers(self):
        """测试命名计时器"""
        tm = TimerManager()

        tm.start("data_loading")
        time.sleep(0.02)
        tm.stop("data_loading")

        tm.start("forward")
        time.sleep(0.03)
        tm.stop("forward")

        assert tm.elapsed("data_loading") >= 0.02
        assert tm.elapsed("forward") >= 0.03
        print(f"✓ 命名计时器: data_loading={tm.elapsed('data_loading'):.3f}s, forward={tm.elapsed('forward'):.3f}s")

    def test_timer_context(self):
        """测试计时器上下文管理器"""
        tm = TimerManager()

        with tm.time("operation"):
            time.sleep(0.02)

        assert tm.elapsed("operation") >= 0.02
        print(f"✓ 计时器上下文: {tm.elapsed('operation'):.3f}s")

    def test_timer_hierarchy(self):
        """测试计时器层次结构"""
        tm = TimerManager()

        tm.start("total")
        with tm.time("inner", parent="total"):
            time.sleep(0.02)
        tm.stop("total")

        summary = tm.get_summary()
        assert "total" in summary
        assert "inner" in summary
        assert summary["inner"]["parent"] == "total"
        print("✓ 计时器层次结构")

    def test_timer_reset(self):
        """测试计时器重置"""
        tm = TimerManager()

        tm.start("timer1")
        time.sleep(0.02)
        tm.stop("timer1")

        tm.reset("timer1")
        assert tm.elapsed("timer1") == 0.0
        print("✓ 单个计时器重置")

    def test_reset_all(self):
        """测试重置所有计时器"""
        tm = TimerManager()

        tm.start("timer1")
        time.sleep(0.01)
        tm.stop("timer1")

        tm.start("timer2")
        time.sleep(0.01)
        tm.stop("timer2")

        tm.reset()

        assert tm.elapsed("timer1") == 0.0
        assert tm.elapsed("timer2") == 0.0
        print("✓ 重置所有计时器")


class TestCodeTimer:
    """代码计时器测试"""

    def test_code_timer_context(self):
        """测试代码计时器上下文"""
        with CodeTimer("Test Operation"):
            time.sleep(0.02)
        print("✓ 代码计时器上下文（应显示时间）")

    def test_code_timer_decorator(self):
        """测试代码计时器装饰器"""
        @CodeTimer("slow_function")
        def slow_function():
            time.sleep(0.02)
            return 42

        result = slow_function()
        assert result == 42
        print("✓ 代码计时器装饰器")


class TestTimeFunction:
    """time_function 装饰器测试"""

    def test_time_function_decorator(self):
        """测试 time_function 装饰器"""
        @time_function
        def example_function():
            time.sleep(0.02)
            return 42

        result = example_function()
        assert result == 42
        print("✓ time_function 装饰器（应显示时间）")


class TestTimerUtility:
    """计时器工具测试"""

    def test_timer_utility(self):
        """测试 timer 工具"""
        with timer("Custom Timer"):
            time.sleep(0.02)
        print("✓ timer 工具（应显示时间）")


def run_tests():
    """运行所有测试"""
    print("\n" + "="*50)
    print("时间测量模块测试")
    print("="*50 + "\n")

    # Timer 测试
    print("=== Timer 测试 ===")
    test_timer = TestTimer()
    timer_tests = [
        ("基础计时器", test_timer.test_basic_timer),
        ("上下文管理器", test_timer.test_timer_context_manager),
        ("时间累积", test_timer.test_timer_accumulation),
        ("计时器重置", test_timer.test_timer_reset),
    ]

    passed = 0
    failed = 0

    for name, test_func in timer_tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed += 1

    # TimerManager 测试
    print("\n=== TimerManager 测试 ===")
    test_tm = TestTimerManager()
    tm_tests = [
        ("命名计时器", test_tm.test_named_timers),
        ("计时器上下文", test_tm.test_timer_context),
        ("层次结构", test_tm.test_timer_hierarchy),
        ("重置单个", test_tm.test_timer_reset),
        ("重置全部", test_tm.test_reset_all),
    ]

    for name, test_func in tm_tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed += 1

    # 其他测试
    print("\n=== 其他测试 ===")
    test_ct = TestCodeTimer()
    test_tf = TestTimeFunction()
    test_util = TestTimerUtility()

    other_tests = [
        ("代码计时器上下文", test_ct.test_code_timer_context),
        ("代码计时器装饰器", test_ct.test_code_timer_decorator),
        ("time_function 装饰器", test_tf.test_time_function_decorator),
        ("timer 工具", test_util.test_timer_utility),
    ]

    for name, test_func in other_tests:
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
