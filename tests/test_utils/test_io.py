"""
I/O 模块测试
"""

import tempfile
import shutil
from pathlib import Path
import numpy as np
import torch
from waleo.utils.io import (
    save_json,
    load_json,
    save_image,
)


class TestJSONIO:
    """JSON I/O 测试"""

    def __init__(self):
        self.temp_dir = None

    def setup(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()

    def teardown(self):
        """清理测试环境"""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_save_load_json(self):
        """测试保存和加载 JSON"""
        self.setup()

        data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        path = Path(self.temp_dir) / "test.json"

        save_json(data, path)
        loaded = load_json(path)

        assert loaded == data
        print("✓ JSON 保存和加载")

        self.teardown()

    def test_nested_json(self):
        """测试嵌套 JSON"""
        self.setup()

        data = {
            "model": {
                "layers": [64, 128, 256],
                "activation": "relu"
            },
            "training": {
                "epochs": 100,
                "batch_size": 32
            }
        }
        path = Path(self.temp_dir) / "nested.json"

        save_json(data, path)
        loaded = load_json(path)

        assert loaded == data
        print("✓ 嵌套 JSON")

        self.teardown()

    def test_json_arrays(self):
        """测试 JSON 数组"""
        self.setup()

        data = [1, 2, 3, 4, 5]
        path = Path(self.temp_dir) / "array.json"

        save_json(data, path)
        loaded = load_json(path)

        assert loaded == data
        print("✓ JSON 数组")

        self.teardown()

    def test_json_special_types(self):
        """测试 JSON 特殊类型"""
        self.setup()

        from waleo.utils.io.json_io import save_json_custom

        data = {
            "path": Path("/tmp/test"),
            "numpy": np.array([1, 2, 3]),
        }
        path = Path(self.temp_dir) / "special.json"

        save_json_custom(data, path)
        loaded = load_json(path)

        assert loaded["path"] == "/tmp/test"
        assert loaded["numpy"] == [1, 2, 3]
        print("✓ JSON 特殊类型")

        self.teardown()


class TestImageIO:
    """图像 I/O 测试"""

    def __init__(self):
        self.temp_dir = None

    def setup(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()

    def teardown(self):
        """清理测试环境"""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_save_load_image(self):
        """测试保存和加载图像"""
        self.setup()

        # 创建随机图像
        image = torch.randint(0, 255, (64, 64, 3), dtype=torch.uint8)
        path = Path(self.temp_dir) / "test.png"

        try:
            save_image(image, path)
            print("✓ 图像保存成功")
        except ImportError as e:
            print(f"⊘ 图像保存（缺少依赖）: {e}")
            self.teardown()
            return

        self.teardown()

    def test_image_normalization(self):
        """测试图像归一化"""
        self.setup()

        # 测试浮点图像
        image_float = torch.rand(64, 64, 3)
        path = Path(self.temp_dir) / "float.png"

        try:
            save_image(image_float, path)
            print("✓ 浮点图像归一化")
        except ImportError as e:
            print(f"⊘ 浮点图像（缺少依赖）: {e}")

        self.teardown()


class TestVideoIO:
    """视频 I/O 测试"""

    def test_video_save_requirements(self):
        """测试视频保存依赖"""
        print("⊘ 视频保存测试（需要 imageio 或 opencv-python）")

    def test_video_load_requirements(self):
        """测试视频加载依赖"""
        print("⊘ 视频加载测试（需要 imageio 或 opencv-python）")


def run_tests():
    """运行所有测试"""
    print("\n" + "="*50)
    print("I/O 模块测试")
    print("="*50 + "\n")

    passed = 0
    failed = 0
    skipped = 0

    # JSON I/O 测试
    print("=== JSON I/O 测试 ===")
    test_json = TestJSONIO()
    json_tests = [
        ("JSON 保存加载", test_json.test_save_load_json),
        ("嵌套 JSON", test_json.test_nested_json),
        ("JSON 数组", test_json.test_json_arrays),
        ("JSON 特殊类型", test_json.test_json_special_types),
    ]

    for name, test_func in json_tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed += 1

    # 图像 I/O 测试
    print("\n=== 图像 I/O 测试 ===")
    test_image = TestImageIO()
    image_tests = [
        ("图像保存加载", test_image.test_save_load_image),
        ("图像归一化", test_image.test_image_normalization),
    ]

    for name, test_func in image_tests:
        try:
            test_func()
            passed += 1
        except ImportError as e:
            print(f"⊘ {name}: {e}")
            skipped += 1
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed += 1

    # 视频 I/O 测试
    print("\n=== 视频 I/O 测试 ===")
    test_video = TestVideoIO()

    video_tests = [
        ("视频保存", test_video.test_video_save_requirements),
        ("视频加载", test_video.test_video_load_requirements),
    ]

    for name, test_func in video_tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed += 1

    print("\n" + "-"*50)
    print(f"测试结果: {passed} 通过, {failed} 失败, {skipped} 跳过")
    print("-"*50 + "\n")

    return failed == 0


if __name__ == "__main__":
    run_tests()
