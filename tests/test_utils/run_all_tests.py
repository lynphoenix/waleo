"""
Waleo-utils 全模块测试运行器
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.test_device import run_tests as test_device
from tests.test_random import run_tests as test_random
from tests.test_timing import run_tests as test_timing
from tests.test_logging import run_tests as test_logging
from tests.test_io import run_tests as test_io


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("Waleo-utils 全模块测试")
    print("="*60)

    all_passed = True

    # 运行各模块测试
    modules = [
        ("设备管理", test_device),
        ("随机数管理", test_random),
        ("时间测量", test_timing),
        ("日志模块", test_logging),
        ("I/O 模块", test_io),
    ]

    results = {}
    for name, test_func in modules:
        try:
            passed = test_func()
            results[name] = passed
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"✗ {name} 测试失败: {e}")
            results[name] = False
            all_passed = False

    # 打印总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)

    for name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name}: {status}")

    print("="*60)

    if all_passed:
        print("\n✓ 所有测试通过！")
        return 0
    else:
        print("\n✗ 部分测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
