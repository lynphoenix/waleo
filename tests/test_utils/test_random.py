"""
随机数管理模块测试
"""

import numpy as np
import torch
from waleo.utils.random import (
    RNGManager,
    ForkedRNG,
    set_seed,
    get_seed,
    seed_worker,
)


class TestRNGManager:
    """RNG 管理器测试"""

    def test_seed_reproducibility(self):
        """测试种子的可复现性"""
        # 使用相同种子生成两个序列
        rng1 = RNGManager(seed=42)
        values1 = [random := __import__("random").random() for _ in range(5)]

        rng2 = RNGManager(seed=42)
        __import__("random").seed(42)
        values2 = [__import__("random").random() for _ in range(5)]

        assert values1 == values2
        print("✓ 相同种子产生相同的随机序列")

    def test_rng_state_save_load(self):
        """测试 RNG 状态保存和加载"""
        import random

        rng = RNGManager(seed=42)

        # 生成一些随机数
        _ = random.random()
        _ = random.random()

        # 保存状态
        state = rng.get_state()

        # 继续生成
        value1 = random.random()

        # 恢复状态
        rng.set_state(state)

        # 应该生成相同的值
        value2 = random.random()
        assert value1 == value2
        print("✓ RNG 状态保存和加载正常工作")

    def test_forked_rng(self):
        """测试分支 RNG"""
        import random

        rng = RNGManager(seed=42)
        initial_value = random.random()

        # 在分支中使用不同种子
        with rng.fork_rng(seed=123):
            forked_value = random.random()

        # 恢复后应该继续原序列
        main_value = random.random()

        # 分支值应该与主序列不同
        assert forked_value != initial_value
        assert forked_value != main_value
        print("✓ Forked RNG 正确隔离")

    def test_torch_rng_seed(self):
        """测试 PyTorch RNG 种子"""
        rng1 = RNGManager(seed=42)
        tensor1 = torch.randn(5)

        rng2 = RNGManager(seed=42)
        tensor2 = torch.randn(5)

        assert torch.allclose(tensor1, tensor2)
        print("✓ PyTorch RNG 种子可复现")

    def test_numpy_rng_seed(self):
        """测试 NumPy RNG 种子"""
        rng1 = RNGManager(seed=42)
        arr1 = np.random.randn(5)

        rng2 = RNGManager(seed=42)
        arr2 = np.random.randn(5)

        assert np.allclose(arr1, arr2)
        print("✓ NumPy RNG 种子可复现")

    def test_set_seed_function(self):
        """测试 set_seed 函数"""
        from waleo.utils.random import set_seed

        set_seed(42)
        values1 = [np.random.randn() for _ in range(3)]

        set_seed(42)
        values2 = [np.random.randn() for _ in range(3)]

        assert np.allclose(values1, values2)
        print("✓ set_seed 函数正常工作")


def run_tests():
    """运行所有测试"""
    print("\n" + "="*50)
    print("随机数管理模块测试")
    print("="*50 + "\n")

    test = TestRNGManager()

    tests = [
        ("种子可复现性", test.test_seed_reproducibility),
        ("状态保存加载", test.test_rng_state_save_load),
        ("分支 RNG", test.test_forked_rng),
        ("PyTorch RNG", test.test_torch_rng_seed),
        ("NumPy RNG", test.test_numpy_rng_seed),
        ("set_seed 函数", test.test_set_seed_function),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
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
