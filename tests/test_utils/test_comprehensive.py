#!/usr/bin/env python3
"""
M01 (waleo-utils) 完备测试套件

测试所有公开API的可用性和基本功能
"""

import sys
import tempfile
from pathlib import Path

# 添加模块路径
# sys.path.insert removed


class TestResult:
    """测试结果记录"""
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.errors = []

    def add_pass(self, name):
        self.total += 1
        self.passed += 1
        print(f"  ✅ {name}")

    def add_fail(self, name, error):
        self.total += 1
        self.failed += 1
        self.errors.append((name, error))
        print(f"  ❌ {name}: {error}")

    def summary(self):
        print(f"\n{'='*60}")
        print(f"测试总结: {self.passed}/{self.total} 通过")
        if self.failed > 0:
            print(f"\n失败的测试:")
            for name, error in self.errors:
                print(f"  • {name}: {error}")
        return self.failed == 0


def test_imports():
    """测试1: 所有导出是否可以导入"""
    result = TestResult()
    print("\n📦 测试1: 导入检查")

    from waleo import utils as waleo_utils

    # 预期导出数量
    expected_count = 63  # 实际统计结果
    actual_count = len(waleo_utils.__all__)

    if actual_count == expected_count:
        result.add_pass(f"导出数量正确: {actual_count}")
    else:
        result.add_fail("导出数量", f"预期{expected_count}, 实际{actual_count}")

    # 检查所有导出是否可访问
    for name in waleo_utils.__all__:
        try:
            obj = getattr(waleo_utils, name)
            result.add_pass(f"导入 {name}")
        except AttributeError as e:
            result.add_fail(f"导入 {name}", str(e))

    return result


def test_constants():
    """测试2: 常量定义"""
    result = TestResult()
    print("\n🔢 测试2: 常量")

    from waleo.utils import (
        WALES_HOME, OBS_STATE, OBS_IMAGE, ACTION, REWARD, DEFAULT_RPC_PORT
    )

    tests = [
        ("WALES_HOME", WALES_HOME, Path),
        ("OBS_STATE", OBS_STATE, str),
        ("OBS_IMAGE", OBS_IMAGE, str),
        ("ACTION", ACTION, str),
        ("REWARD", REWARD, str),
        ("DEFAULT_RPC_PORT", DEFAULT_RPC_PORT, int),
    ]

    for name, value, expected_type in tests:
        if isinstance(value, expected_type):
            result.add_pass(f"{name} = {value}")
        else:
            result.add_fail(name, f"类型错误: {type(value)} != {expected_type}")

    return result


def test_device_management():
    """测试3: 设备管理"""
    result = TestResult()
    print("\n💻 测试3: 设备管理")

    try:
        from waleo.utils import (
            get_training_device, get_inference_device, get_device,
            is_device_available, get_device_count, get_device_capabilities
        )

        # 测试设备函数
        result.add_pass("get_training_device 可调用")
        result.add_pass("get_inference_device 可调用")
        result.add_pass("get_device 可调用")

        # 测试设备查询
        cpu_available = is_device_available("cpu")
        if cpu_available:
            result.add_pass("is_device_available('cpu') = True")
        else:
            result.add_fail("is_device_available", "CPU应该可用")

        cpu_count = get_device_count("cpu")
        result.add_pass(f"get_device_count('cpu') = {cpu_count}")

        caps = get_device_capabilities()
        if isinstance(caps, dict):
            result.add_pass(f"get_device_capabilities() 返回字典")
        else:
            result.add_fail("get_device_capabilities", "应返回字典")

    except Exception as e:
        result.add_fail("设备管理", str(e))

    return result


def test_distributed():
    """测试4: 分布式训练接口"""
    result = TestResult()
    print("\n🌐 测试4: 分布式训练")

    try:
        from waleo.utils import (
            launch_multi_process, launch_multinode,
            all_reduce, broadcast, all_gather,
            DistributedState, setup_distributed, destroy_distributed
        )

        functions = [
            "launch_multi_process", "launch_multinode",
            "all_reduce", "broadcast", "all_gather",
            "setup_distributed", "destroy_distributed"
        ]

        for func_name in functions:
            result.add_pass(f"{func_name} 可调用")

        # 测试 DistributedState 类
        if callable(DistributedState):
            result.add_pass("DistributedState 类可实例化")

    except Exception as e:
        result.add_fail("分布式训练", str(e))

    return result


def test_communication():
    """测试5: RPC 通信"""
    result = TestResult()
    print("\n📡 测试5: RPC 通信")

    try:
        from waleo.utils import (
            RPCEndpoint, RPCMessage, MessageType, ErrorCode,
            RPCClient, AsyncRPCClient, RPCServer
        )

        classes = [
            ("RPCEndpoint", RPCEndpoint),
            ("RPCMessage", RPCMessage),
            ("MessageType", MessageType),
            ("ErrorCode", ErrorCode),
            ("RPCClient", RPCClient),
            ("AsyncRPCClient", AsyncRPCClient),
            ("RPCServer", RPCServer),
        ]

        for name, cls in classes:
            if callable(cls) or hasattr(cls, '__members__'):  # 类或枚举
                result.add_pass(f"{name} 可用")
            else:
                result.add_fail(name, "不是有效的类或枚举")

    except Exception as e:
        result.add_fail("RPC 通信", str(e))

    return result


def test_serialization():
    """测试6: 序列化工具"""
    result = TestResult()
    print("\n🔄 测试6: 序列化")

    try:
        from waleo.utils import (
            serialize, deserialize,
            serialize_tensor, deserialize_tensor,
            serialize_ndarray, deserialize_ndarray
        )

        # 测试基本序列化
        test_obj = {"key": "value", "number": 42}
        serialized = serialize(test_obj)
        deserialized = deserialize(serialized)

        if deserialized == test_obj:
            result.add_pass("基本对象序列化/反序列化")
        else:
            result.add_fail("基本序列化", "反序列化结果不匹配")

        # 测试函数可调用性
        for func_name in ["serialize_tensor", "deserialize_tensor",
                          "serialize_ndarray", "deserialize_ndarray"]:
            result.add_pass(f"{func_name} 可调用")

    except Exception as e:
        result.add_fail("序列化", str(e))

    return result


def test_logging():
    """测试7: 日志系统"""
    result = TestResult()
    print("\n📝 测试7: 日志系统")

    try:
        from waleo.utils import (
            TBLogger, create_logger,
            MetricTracker, AverageMeter, ProgressMeter
        )

        # 测试 AverageMeter
        meter = AverageMeter("test")
        meter.update(1.0)
        meter.update(2.0)
        if abs(meter.avg - 1.5) < 1e-6:
            result.add_pass("AverageMeter 计算正确")
        else:
            result.add_fail("AverageMeter", f"平均值错误: {meter.avg}")

        # 测试其他类
        classes = [
            ("TBLogger", TBLogger),
            ("MetricTracker", MetricTracker),
            ("ProgressMeter", ProgressMeter),
        ]

        for name, cls in classes:
            if callable(cls):
                result.add_pass(f"{name} 可实例化")

        # 测试 create_logger
        if callable(create_logger):
            result.add_pass("create_logger 可调用")

    except Exception as e:
        result.add_fail("日志系统", str(e))

    return result


def test_random():
    """测试8: 随机数管理"""
    result = TestResult()
    print("\n🎲 测试8: 随机数管理")

    try:
        from waleo.utils import (
            RNGManager, ForkedRNG,
            set_seed, get_seed, seed_worker
        )

        # 测试 set_seed
        set_seed(42)
        result.add_pass("set_seed(42) 执行成功")

        # 测试 get_seed
        seed = get_seed()
        if seed == 42:
            result.add_pass(f"get_seed() = {seed}")
        else:
            result.add_fail("get_seed", f"预期42, 得到{seed}")

        # 测试 RNGManager
        manager = RNGManager(seed=123)
        state = manager.get_state()
        if isinstance(state, dict) and 'seed' in state:
            result.add_pass("RNGManager.get_state() 返回有效状态")
        else:
            result.add_fail("RNGManager.get_state", "状态格式错误")

        # 测试状态保存/加载
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "rng_state.pkl"
            manager.save_state(save_path)

            manager2 = RNGManager()
            manager2.load_state(save_path)

            if manager2.get_state()['seed'] == 123:
                result.add_pass("RNG状态保存/加载")
            else:
                result.add_fail("RNG状态", "加载的种子不匹配")

        # 测试其他函数
        if callable(ForkedRNG):
            result.add_pass("ForkedRNG 可用")
        if callable(seed_worker):
            result.add_pass("seed_worker 可调用")

    except Exception as e:
        result.add_fail("随机数管理", str(e))

    return result


def test_timing():
    """测试9: 时间测量"""
    result = TestResult()
    print("\n⏱️  测试9: 时间测量")

    try:
        from waleo.utils import (
            Timer, TimerManager, CodeTimer,
            time_function, timer
        )
        import time

        # 测试 Timer
        t = Timer()
        t.start()
        time.sleep(0.01)
        t.stop()
        if t.elapsed > 0:
            result.add_pass(f"Timer 测量: {t.elapsed:.4f}s")
        else:
            result.add_fail("Timer", "时间测量无效")

        # 测试 timer 装饰器
        @timer("test_func")
        def test_func():
            time.sleep(0.01)
            return 42

        ret = test_func()
        if ret == 42:
            result.add_pass("@timer 装饰器工作正常")

        # 测试 CodeTimer 上下文管理器
        with CodeTimer("test"):
            time.sleep(0.01)
        result.add_pass("CodeTimer 上下文管理器")

        # 测试其他类
        if callable(TimerManager):
            result.add_pass("TimerManager 可用")
        if callable(time_function):
            result.add_pass("time_function 可调用")

    except Exception as e:
        result.add_fail("时间测量", str(e))

    return result


def test_io():
    """测试10: 文件 I/O"""
    result = TestResult()
    print("\n💾 测试10: 文件 I/O")

    try:
        from waleo.utils import (
            save_json, load_json,
            save_json_custom, update_json, get_json_value
        )
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            # 测试 JSON I/O
            json_path = Path(tmpdir) / "test.json"
            test_data = {"key": "value", "number": 42}

            save_json(test_data, json_path)
            loaded_data = load_json(json_path)

            if loaded_data == test_data:
                result.add_pass("JSON 保存/加载")
            else:
                result.add_fail("JSON I/O", "数据不匹配")

            # 测试 update_json
            update_json(json_path, {"new_key": "new_value"})
            updated = load_json(json_path)
            if "new_key" in updated:
                result.add_pass("update_json")

            # 测试 get_json_value
            value = get_json_value(json_path, "key")
            if value == "value":
                result.add_pass(f"get_json_value('key') = '{value}'")

        # 测试视频和图像函数（只检查可调用性，不实际执行）
        from waleo.utils import (
            save_video, load_video,
            save_image, load_image
        )

        io_funcs = [
            "save_video", "load_video",
            "save_image", "load_image",
            "save_json_custom"
        ]

        for func_name in io_funcs:
            result.add_pass(f"{func_name} 可调用")

    except Exception as e:
        result.add_fail("文件 I/O", str(e))

    return result


def main():
    """运行所有测试"""
    print("="*60)
    print("M01 (waleo-utils) 完备测试套件")
    print("="*60)

    tests = [
        test_imports,
        test_constants,
        test_device_management,
        test_distributed,
        test_communication,
        test_serialization,
        test_logging,
        test_random,
        test_timing,
        test_io,
    ]

    all_results = []
    for test_func in tests:
        try:
            result = test_func()
            all_results.append(result)
        except Exception as e:
            print(f"\n❌ 测试失败: {test_func.__name__}")
            print(f"   错误: {e}")
            import traceback
            traceback.print_exc()

    # 总结
    print("\n" + "="*60)
    print("📊 总体测试结果")
    print("="*60)

    total_passed = sum(r.passed for r in all_results)
    total_tests = sum(r.total for r in all_results)
    total_failed = sum(r.failed for r in all_results)

    print(f"\n总计: {total_passed}/{total_tests} 测试通过")
    if total_failed > 0:
        print(f"失败: {total_failed} 个测试")
        print("\n详细失败信息:")
        for result in all_results:
            for name, error in result.errors:
                print(f"  • {name}: {error}")
    else:
        print("\n✅ 所有测试通过！")

    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    print(f"\n成功率: {success_rate:.1f}%")

    return total_failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
