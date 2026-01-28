"""测试 RobotRegistry 和 RobotSpec"""

import pytest
import tempfile
from pathlib import Path

from waleo.sim.registry import get_robot_registry, RobotSpec


class TestRobotSpec:
    """测试 RobotSpec 数据类"""

    def test_robot_spec_creation(self):
        """测试创建 RobotSpec"""
        spec = RobotSpec(
            name="TestRobot",
            urdf_path=Path("/tmp/test.urdf"),
            robot_dir=Path("/tmp/robot"),
            dof=6,
        )
        assert spec.name == "TestRobot"
        assert spec.dof == 6
        assert spec.default_kwargs == {}

    def test_robot_spec_with_default_kwargs(self):
        """测试 RobotSpec 的 default_kwargs"""
        spec = RobotSpec(
            name="TestRobot",
            urdf_path=Path("/tmp/test.urdf"),
            robot_dir=Path("/tmp/robot"),
            dof=6,
            default_kwargs={"control_mode": "pd_joint_pos"},
        )
        assert spec.default_kwargs == {"control_mode": "pd_joint_pos"}

    def test_robot_spec_task_config(self):
        """测试任务配置"""
        spec = RobotSpec(
            name="TestRobot",
            urdf_path=Path("/tmp/test.urdf"),
            robot_dir=Path("/tmp/robot"),
            dof=6,
            task_configs={
                "PickCube-v1": {"robot_pose": {"offset": [0, 0, 0]}}
            }
        )
        config = spec.get_task_config("PickCube-v1")
        assert config is not None
        assert "robot_pose" in config

    def test_robot_spec_get_task_config_not_found(self):
        """测试获取不存在的任务配置"""
        spec = RobotSpec(
            name="TestRobot",
            urdf_path=Path("/tmp/test.urdf"),
            robot_dir=Path("/tmp/robot"),
            dof=6,
            task_configs={}
        )
        config = spec.get_task_config("NonExistent-v1")
        assert config is None


class TestRobotRegistry:
    """测试 RobotRegistry"""

    def test_registry_singleton(self):
        """测试注册表是单例"""
        from waleo.sim.registry.robot import get_robot_registry
        registry1 = get_robot_registry()
        registry2 = get_robot_registry()
        assert registry1 is registry2

    def test_list_robots(self):
        """测试列出所有机器人"""
        registry = get_robot_registry()
        robots = registry.list_robots()
        assert isinstance(robots, list)
        # 应该至少有 RJ2506
        assert "RJ2506" in robots

    def test_get_robot(self):
        """测试获取机器人规格"""
        registry = get_robot_registry()
        spec = registry.get("RJ2506")
        assert spec is not None
        assert spec.name == "RJ2506"

    def test_get_robot_not_found(self):
        """测试获取不存在的机器人"""
        registry = get_robot_registry()
        spec = registry.get("NonExistentRobot")
        assert spec is None

    def test_is_registered(self):
        """测试检查机器人是否注册"""
        registry = get_robot_registry()
        assert registry.is_registered("RJ2506")
        assert not registry.is_registered("NonExistentRobot")

    def test_register_robot(self):
        """测试手动注册机器人"""
        registry = get_robot_registry()

        # 创建临时机器人
        spec = RobotSpec(
            name="TempRobot",
            urdf_path=Path("/tmp/temp.urdf"),
            robot_dir=Path("/tmp/temp"),
            dof=3,
        )

        # 注册
        registry.register(spec)
        assert registry.is_registered("TempRobot")

        # 注销
        success = registry.unregister("TempRobot")
        assert success
        assert not registry.is_registered("TempRobot")

    def test_unregister_not_found(self):
        """测试注销不存在的机器人"""
        registry = get_robot_registry()
        success = registry.unregister("NonExistent")
        assert not success


class TestRobotYAML:
    """测试 YAML 配置加载"""

    def test_load_rj2506_yaml(self):
        """测试加载 RJ2506 YAML 配置"""
        registry = get_robot_registry()
        spec = registry.get("RJ2506")

        assert spec is not None
        assert spec.name == "RJ2506"
        assert spec.dof == 10
        # 检查 default_kwargs 包含 control_mode
        assert "control_mode" in spec.default_kwargs

    def test_rj2506_task_configs(self):
        """测试 RJ2506 任务配置"""
        registry = get_robot_registry()
        spec = registry.get("RJ2506")

        task_configs = spec.task_configs
        assert isinstance(task_configs, dict)
        # 应该有 PickCube-v1 配置
        assert "PickCube-v1" in task_configs

    def test_rj2506_default_kwargs(self):
        """测试 RJ2506 default_kwargs"""
        registry = get_robot_registry()
        spec = registry.get("RJ2506")

        # 检查 default_kwargs
        assert isinstance(spec.default_kwargs, dict)
        assert "control_mode" in spec.default_kwargs
        assert spec.default_kwargs["control_mode"] == "pd_joint_delta_pos"

    def test_backward_compatible_control_mode(self):
        """测试向后兼容：control_mode 自动迁移到 default_kwargs"""
        # 使用真实的机器人配置目录和文件来测试
        import tempfile
        import yaml
        import shutil

        from waleo.sim.registry import get_asset_resolver
        resolver = get_asset_resolver()

        # 获取 RJ2506 的真实路径
        rj2506_dir = resolver.resolve_robot_dir("RJ2506")
        if not rj2506_dir:
            pytest.skip("需要 RJ2506 机器人来测试")

        # 创建临时测试目录，复制真实的 URDF
        temp_dir = Path(tempfile.mkdtemp())
        try:
            test_urdf = rj2506_dir / "urdf" / "RJ2506.urdf"
            if not test_urdf.exists():
                pytest.skip("需要 RJ2506.urdf 来测试")

            temp_robot_dir = temp_dir / "TestRobot"
            temp_robot_dir.mkdir()
            (temp_robot_dir / "urdf").mkdir()

            # 复制 URDF
            shutil.copy(test_urdf, temp_robot_dir / "urdf" / "TestRobot.urdf")

            # 创建 YAML
            yaml_content = """
name: TestRobot
urdf: urdf/TestRobot.urdf
dof: 6
control_mode: pd_joint_pos
task_configs: {}
"""
            yaml_path = temp_robot_dir / "robot.yaml"
            yaml_path.write_text(yaml_content)

            spec = RobotSpec.from_yaml(yaml_path)
            # control_mode 应该自动迁移到 default_kwargs
            assert "control_mode" in spec.default_kwargs
            assert spec.default_kwargs["control_mode"] == "pd_joint_pos"
        finally:
            shutil.rmtree(temp_dir)

    def test_new_default_kwargs_format(self):
        """测试新的 default_kwargs 格式"""
        import tempfile
        import yaml
        import shutil

        from waleo.sim.registry import get_asset_resolver
        resolver = get_asset_resolver()

        # 获取 RJ2506 的真实路径
        rj2506_dir = resolver.resolve_robot_dir("RJ2506")
        if not rj2506_dir:
            pytest.skip("需要 RJ2506 机器人来测试")

        # 创建临时测试目录
        temp_dir = Path(tempfile.mkdtemp())
        try:
            test_urdf = rj2506_dir / "urdf" / "RJ2506.urdf"
            if not test_urdf.exists():
                pytest.skip("需要 RJ2506.urdf 来测试")

            temp_robot_dir = temp_dir / "TestRobot"
            temp_robot_dir.mkdir()
            (temp_robot_dir / "urdf").mkdir()

            # 复制 URDF
            shutil.copy(test_urdf, temp_robot_dir / "urdf" / "TestRobot.urdf")

            # 创建 YAML
            yaml_content = """
name: TestRobot
urdf: urdf/TestRobot.urdf
dof: 6
default_kwargs:
  control_mode: pd_joint_pos
  frame_skip: 5
task_configs: {}
"""
            yaml_path = temp_robot_dir / "robot.yaml"
            yaml_path.write_text(yaml_content)

            spec = RobotSpec.from_yaml(yaml_path)
            assert spec.default_kwargs["control_mode"] == "pd_joint_pos"
            assert spec.default_kwargs["frame_skip"] == 5
        finally:
            shutil.rmtree(temp_dir)


class TestRobotRegistryIntegration:
    """测试 RobotRegistry 集成"""

    def test_auto_discovery(self):
        """测试自动发现机器人"""
        registry = get_robot_registry()
        robots = registry.list_robots()
        # 应该自动发现 assets/robots 下的机器人
        assert len(robots) > 0

    def test_get_all_specs(self):
        """测试获取所有机器人规格"""
        registry = get_robot_registry()
        robots = registry.list_robots()

        for robot_name in robots:
            spec = registry.get(robot_name)
            assert spec is not None
            assert spec.name == robot_name
            assert spec.urdf_path.exists()
