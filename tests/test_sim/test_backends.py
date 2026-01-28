"""测试所有仿真后端"""

import pytest

from waleo.sim.backends import (
    ManiSkillBackend,
    MuJoCoBackend,
    PyBulletBackend,
    IsaacSimBackend,
    SimulationBackend,
    BackendUnavailableError,
    BackendCreateError,
)
from waleo.sim.factory import get_backend


class TestBackendInterface:
    """测试后端接口一致性"""

    def test_all_backends_have_import_name(self):
        """测试所有后端都有 import_name"""
        backends = [ManiSkillBackend, MuJoCoBackend, PyBulletBackend, IsaacSimBackend]
        for backend in backends:
            assert hasattr(backend, 'import_name')
            assert isinstance(backend.import_name, str)
            assert len(backend.import_name) > 0

    def test_all_backends_have_create_method(self):
        """测试所有后端都有 create 方法"""
        backends = [ManiSkillBackend, MuJoCoBackend, PyBulletBackend, IsaacSimBackend]
        for backend in backends:
            assert hasattr(backend, 'create')
            assert callable(backend.create)

    def test_all_backends_have_is_available_method(self):
        """测试所有后端都有 is_available 方法"""
        backends = [ManiSkillBackend, MuJoCoBackend, PyBulletBackend, IsaacSimBackend]
        for backend in backends:
            assert hasattr(backend, 'is_available')
            assert callable(backend.is_available)

    def test_all_backends_have_get_default_kwargs_method(self):
        """测试所有后端都有 get_default_kwargs 方法"""
        backends = [ManiSkillBackend, MuJoCoBackend, PyBulletBackend, IsaacSimBackend]
        for backend in backends:
            assert hasattr(backend, 'get_default_kwargs')
            assert callable(backend.get_default_kwargs)

    def test_all_backends_have_list_available_tasks_method(self):
        """测试所有后端都有 list_available_tasks 方法"""
        backends = [ManiSkillBackend, MuJoCoBackend, PyBulletBackend, IsaacSimBackend]
        for backend in backends:
            assert hasattr(backend, 'list_available_tasks')
            assert callable(backend.list_available_tasks)

    def test_get_default_kwargs_returns_dict(self):
        """测试 get_default_kwargs 返回字典"""
        backends = [ManiSkillBackend, MuJoCoBackend, PyBulletBackend, IsaacSimBackend]
        for backend in backends:
            kwargs = backend.get_default_kwargs()
            assert isinstance(kwargs, dict)

    def test_list_available_tasks_returns_list(self):
        """测试 list_available_tasks 返回列表"""
        backends = [ManiSkillBackend, MuJoCoBackend, PyBulletBackend, IsaacSimBackend]
        for backend in backends:
            tasks = backend.list_available_tasks()
            assert isinstance(tasks, list)
            assert len(tasks) > 0


class TestManiSkillBackend:
    """测试 ManiSkill 后端"""

    def test_import_name(self):
        """测试 import_name"""
        assert ManiSkillBackend.import_name == "mani_skill"

    def test_is_available(self):
        """测试后端可用性"""
        # 在测试环境中 ManiSkill 应该可用
        assert ManiSkillBackend.is_available()

    def test_get_default_kwargs(self):
        """测试默认参数"""
        kwargs = ManiSkillBackend.get_default_kwargs()
        assert "num_envs" in kwargs
        assert "obs_mode" in kwargs
        assert kwargs["obs_mode"] == "state"  # 扁平观察
        assert "control_mode" in kwargs
        assert "robot_uids" in kwargs

    def test_list_available_tasks(self):
        """测试任务列表"""
        tasks = ManiSkillBackend.list_available_tasks()
        assert "PickCube-v1" in tasks

    def test_create_environment(self):
        """测试创建环境"""
        env = ManiSkillBackend.create("PickCube-v1", num_envs=1)
        assert env is not None
        env.close()

    def test_create_with_kwargs(self):
        """测试使用参数创建环境"""
        # 使用 num_envs=1 避免 PhysX 重复初始化
        env = ManiSkillBackend.create(
            "PickCube-v1",
            num_envs=1,
            obs_mode="state"
        )
        assert env is not None
        obs, info = env.reset()
        assert obs is not None
        env.close()


class TestMuJoCoBackend:
    """测试 MuJoCo 后端"""

    def test_import_name(self):
        """测试 import_name"""
        assert MuJoCoBackend.import_name == "mujoco"

    def test_get_default_kwargs(self):
        """测试默认参数"""
        kwargs = MuJoCoBackend.get_default_kwargs()
        assert "frame_skip" in kwargs

    def test_list_available_tasks(self):
        """测试任务列表"""
        tasks = MuJoCoBackend.list_available_tasks()
        assert "Ant-v4" in tasks
        assert "Hopper-v4" in tasks

    def test_create_without_installation(self):
        """测试未安装时的行为"""
        # MuJoCo 可能未安装
        if not MuJoCoBackend.is_available():
            with pytest.raises(Exception):
                MuJoCoBackend.create("Ant-v4")


class TestPyBulletBackend:
    """测试 PyBullet 后端"""

    def test_import_name(self):
        """测试 import_name"""
        assert PyBulletBackend.import_name == "pybullet"

    def test_list_available_tasks(self):
        """测试任务列表"""
        tasks = PyBulletBackend.list_available_tasks()
        assert "KukaBulletEnv-v0" in tasks


class TestIsaacSimBackend:
    """测试 IsaacSim 后端"""

    def test_import_name(self):
        """测试 import_name"""
        assert IsaacSimBackend.import_name == "isaacsim"

    def test_get_default_kwargs(self):
        """测试默认参数"""
        kwargs = IsaacSimBackend.get_default_kwargs()
        assert "headless" in kwargs
        assert "device" in kwargs

    def test_list_available_tasks(self):
        """测试任务列表"""
        tasks = IsaacSimBackend.list_available_tasks()
        assert "Isaac-Lift-Cube-Franka-v0" in tasks


class TestBackendFactoryIntegration:
    """测试后端与工厂的集成"""

    def test_get_maniskill_backend(self):
        """测试通过工厂获取 ManiSkill 后端"""
        backend = get_backend("maniskill")
        assert backend is ManiSkillBackend

    def test_get_mujoco_backend(self):
        """测试通过工厂获取 MuJoCo 后端"""
        backend = get_backend("mujoco")
        assert backend is MuJoCoBackend

    def test_get_pybullet_backend(self):
        """测试通过工厂获取 PyBullet 后端"""
        backend = get_backend("pybullet")
        assert backend is PyBulletBackend

    def test_get_isaacsim_backend(self):
        """测试通过工厂获取 IsaacSim 后端"""
        backend = get_backend("isaacsim")
        assert backend is IsaacSimBackend

    def test_all_backends_registered(self):
        """测试所有后端都已注册"""
        from waleo.sim.factory import _BACKENDS
        expected = {"maniskill", "mujoco", "pybullet", "isaacsim"}
        assert set(_BACKENDS.keys()) == expected


class TestBackendErrorHandling:
    """测试后端错误处理"""

    @pytest.mark.skip(reason="需要未安装的后端进行测试，当前环境所有后端可能已安装")
    def test_unavailable_backend_error(self):
        """测试后端不可用时的错误

        注意：此测试需要一个真正未安装的后端。
        可以使用 mock 来模拟 ImportError。
        """
        # 使用 mock 模拟未安装的后端
        from unittest.mock import patch
        with patch.object(ManiSkillBackend, '_check_dependencies', side_effect=ImportError):
            assert not ManiSkillBackend.is_available()

    def test_invalid_task_error(self):
        """测试无效任务时的错误"""
        with pytest.raises(Exception):
            ManiSkillBackend.create("InvalidTask-v999")

    @pytest.mark.skip(reason="需要 mock gym.make 失败场景")
    def test_create_fallback_to_direct(self):
        """测试 gym.make 失败时的备用方法

        ManiSkill 后端有 _create_direct 备用方法。
        当 gym.make() 失败时应该回退到直接创建。
        """
        # TODO: 使用 mock 模拟 gym.make 失败
        pass
