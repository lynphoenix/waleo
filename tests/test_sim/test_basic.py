"""测试 waleo-sim 模块基础功能"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_factory():
    """测试工厂函数"""
    print("测试工厂函数...")
    from waleo.sim import (
        make_env,
        list_available_backends,
        list_available_robots,
        list_available_tasks,
        get_backend,
    )

    # 测试后端列表
    backends = list_available_backends()
    print(f"✓ 可用后端: {backends}")
    assert "maniskill" in backends

    # 测试 get_backend
    backend_cls = get_backend("maniskill")
    print(f"✓ get_backend('maniskill') 返回: {backend_cls}")

    # 测试机器人列表
    robots = list_available_robots()
    print(f"✓ 可用机器人: {robots}")

    # 测试任务列表
    tasks = list_available_tasks("maniskill")
    print(f"✓ ManiSkill 任务数量: {len(tasks)}")
    if tasks:
        print(f"  示例任务: {tasks[:3]}")

    print("✓ 工厂函数测试通过\n")


def test_backends():
    """测试后端模块"""
    print("测试后端模块...")
    from waleo.sim.backends import (
        SimulationBackend,
        ManiSkillBackend,
        BackendError,
        BackendUnavailableError,
        BackendCreateError,
    )

    # 测试类导入
    print(f"✓ SimulationBackend: {SimulationBackend}")
    print(f"✓ ManiSkillBackend: {ManiSkillBackend}")
    print(f"✓ BackendError: {BackendError}")

    # 测试后端属性（name 是 classmethod，需要调用）
    print(f"✓ ManiSkillBackend.name(): {ManiSkillBackend.name()}")
    print(f"✓ ManiSkillBackend.import_name: {ManiSkillBackend.import_name}")

    # 测试可用性检查（可能失败，如果没有安装 mani_skill）
    try:
        available = ManiSkillBackend.is_available()
        print(f"✓ ManiSkillBackend.is_available(): {available}")
    except Exception as e:
        print(f"⚠ ManiSkillBackend.is_available() 检查失败（预期）: {e}")

    # 测试默认参数
    defaults = ManiSkillBackend.get_default_kwargs()
    print(f"✓ 默认参数: {defaults}")

    print("✓ 后端模块测试通过\n")


def test_registry():
    """测试注册表"""
    print("测试注册表...")
    from waleo.sim import get_robot_registry, get_asset_resolver

    # 测试机器人注册表
    registry = get_robot_registry()
    print(f"✓ get_robot_registry(): {registry}")

    robots = registry.list_robots()
    print(f"✓ 已注册机器人: {robots}")

    # 测试资源解析器
    resolver = get_asset_resolver()
    print(f"✓ get_asset_resolver(): {resolver}")

    print("✓ 注册表测试通过\n")


def test_wrappers():
    """测试包装器"""
    print("测试包装器...")
    from waleo.sim import (
        CustomRobotWrapper,
        TaskConfigWrapper,
        CameraConfigWrapper,
        create_wrapped_env,
    )

    print(f"✓ CustomRobotWrapper: {CustomRobotWrapper}")
    print(f"✓ TaskConfigWrapper: {TaskConfigWrapper}")
    print(f"✓ CameraConfigWrapper: {CameraConfigWrapper}")
    print(f"✓ create_wrapped_env: {create_wrapped_env}")

    print("✓ 包装器测试通过\n")


def test_imports():
    """测试所有公开 API 导入"""
    print("测试公开 API...")
    from waleo.sim import (
        # 版本
        __version__,
        # 工厂函数
        make_env,
        make,
        list_available_backends,
        list_available_robots,
        list_available_tasks,
        # 后端
        SimulationBackend,
        ManiSkillBackend,
        BackendError,
        # 注册表
        get_robot_registry,
        get_asset_resolver,
        # 包装器
        CustomRobotWrapper,
        TaskConfigWrapper,
        CameraConfigWrapper,
    )

    print(f"✓ 版本: {__version__}")
    print("✓ 所有公开 API 导入成功")
    print("✓ API 导入测试通过\n")


def test_backend_consistency():
    """测试后端一致性"""
    print("测试后端一致性...")
    from waleo.sim.backends import ManiSkillBackend
    from waleo.sim.backends.base import SimulationBackend

    # 验证继承关系
    assert issubclass(ManiSkillBackend, SimulationBackend)
    print("✓ ManiSkillBackend 继承自 SimulationBackend")

    # 验证必需方法
    assert hasattr(ManiSkillBackend, "create")
    assert hasattr(ManiSkillBackend, "is_available")
    assert hasattr(ManiSkillBackend, "get_default_kwargs")
    print("✓ ManiSkillBackend 实现了所有必需方法")

    # 验证类属性（name 是 classmethod，需要调用）
    assert ManiSkillBackend.name() == "maniskill"
    assert ManiSkillBackend.import_name == "mani_skill"
    print("✓ ManiSkillBackend 类属性正确")

    print("✓ 后端一致性测试通过\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Waleo-Sim 模块测试（重构后架构）")
    print("=" * 60)
    print()

    try:
        test_imports()
        test_factory()
        test_backends()
        test_registry()
        test_wrappers()
        test_backend_consistency()

        print("=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
