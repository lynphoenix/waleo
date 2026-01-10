"""
设备管理模块测试
"""

import torch
from waleo_utils.device import (
    get_training_device,
    get_inference_device,
    get_device,
)


class TestDeviceManager:
    """设备管理器测试"""

    def test_get_training_device_with_cuda(self):
        """测试获取训练设备（有 CUDA）"""
        try:
            device = get_training_device()
            assert device.type == "cuda"
            print(f"✓ Training device: {device}")
        except RuntimeError as e:
            if "CUDA is required" in str(e):
                print("⊘ CUDA not available, skipping training device test")
            else:
                raise

    def test_get_inference_device(self):
        """测试获取推理设备"""
        device = get_inference_device()
        assert device.type in ["cuda", "mps", "cpu"]
        print(f"✓ Inference device: {device}")

    def test_get_device_training(self):
        """测试获取训练设备"""
        try:
            device = get_device(training=True)
            assert device.type == "cuda"
            print(f"✓ Training device (via get_device): {device}")
        except RuntimeError as e:
            if "CUDA is required" in str(e):
                print("⊘ CUDA not available, skipping training device test")
            else:
                raise

    def test_get_device_inference(self):
        """测试获取推理设备"""
        device = get_device(training=False)
        assert device.type in ["cuda", "mps", "cpu"]
        print(f"✓ Inference device (via get_device): {device}")

    def test_device_tensor_operations(self):
        """测试设备上的张量操作"""
        device = get_inference_device()

        # 创建张量并移动到设备
        tensor = torch.randn(3, 4)
        tensor = tensor.to(device)

        assert tensor.device.type == device.type
        print(f"✓ Tensor operations on {device}")

    def test_device_with_model(self):
        """测试模型在设备上的运行"""
        device = get_inference_device()

        # 创建简单模型
        model = torch.nn.Linear(10, 5)
        model = model.to(device)

        # 创建输入并运行
        input_tensor = torch.randn(2, 10).to(device)
        output = model(input_tensor)

        assert output.device.type == device.type
        assert output.shape == (2, 5)
        print(f"✓ Model inference on {device}")


def run_tests():
    """运行所有测试"""
    print("\n" + "="*50)
    print("设备管理模块测试")
    print("="*50 + "\n")

    test = TestDeviceManager()

    tests = [
        ("训练设备", test.test_get_training_device_with_cuda),
        ("推理设备", test.test_get_inference_device),
        ("张量操作", test.test_device_tensor_operations),
        ("模型推理", test.test_device_with_model),
    ]

    passed = 0
    failed = 0
    skipped = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except RuntimeError as e:
            if "CUDA is required" in str(e) or "not available" in str(e):
                skipped += 1
            else:
                print(f"✗ {name}: {e}")
                failed += 1
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed += 1

    print("\n" + "-"*50)
    print(f"测试结果: {passed} 通过, {failed} 失败, {skipped} 跳过")
    print("-"*50 + "\n")

    return failed == 0


if __name__ == "__main__":
    run_tests()
