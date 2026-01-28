"""测试 factory.py 的 make_env() 函数"""

import pytest
from pathlib import Path

from waleo.sim import make_env
from waleo.sim.factory import get_backend, list_available_backends, list_available_robots
from waleo.sim.backends.base import SimulationBackend
from waleo.sim.backends.maniskill import ManiSkillBackend


class TestMakeEnv:
    """测试 make_env() 工厂函数"""

    def test_get_backend(self):
        """测试获取后端"""
        # 获取已存在的后端
        backend = get_backend("maniskill")
        assert backend is ManiSkillBackend

        # 获取不存在的后端
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("nonexistent")

    def test_get_backend_case_insensitive(self):
        """测试后端名称不区分大小写"""
        backend1 = get_backend("ManiSkill")
        backend2 = get_backend("MANISKILL")
        backend3 = get_backend("maniskill")
        assert backend1 is backend2 is backend3 is ManiSkillBackend

    def test_list_available_backends(self):
        """测试列出可用后端"""
        backends = list_available_backends()
        assert isinstance(backends, list)
        # ManiSkill 应该总是可用（测试环境中有）
        assert "maniskill" in backends

    def test_list_available_robots(self):
        """测试列出可用机器人"""
        robots = list_available_robots()
        assert isinstance(robots, list)
        # 应该至少有 RJ2506
        assert "RJ2506" in robots

    def test_make_env_maniskill_default(self):
        """测试创建 ManiSkill 环境（默认参数）"""
        env = make_env("PickCube-v1")
        assert env is not None
        # 清理
        env.close()

    def test_make_env_with_obs_mode(self):
        """测试指定观察模式"""
        env = make_env("PickCube-v1", obs_mode="state")
        obs, info = env.reset()
        # state 模式应该返回扁平观察
        # 注意：具体形状取决于环境
        assert obs is not None
        env.close()

    def test_make_env_with_num_envs(self):
        """测试并行环境创建"""
        # 注意：GPU PhysX 只能初始化一次，跳过此测试
        # 实际使用中会在单独进程中测试
        pytest.skip("GPU PhysX 只能初始化一次")

    def test_make_env_backend_parameter(self):
        """测试 backend 参数"""
        # 显式指定后端
        env = make_env("PickCube-v1", backend="maniskill")
        assert env is not None
        env.close()

    def test_make_env_invalid_backend(self):
        """测试无效后端"""
        with pytest.raises(ValueError, match="Unknown backend"):
            make_env("PickCube-v1", backend="invalid")

    def test_make_env_backend_case_insensitive(self):
        """测试后端名称不区分大小写"""
        env1 = make_env("PickCube-v1", backend="ManiSkill")
        env1.close()

        env2 = make_env("PickCube-v1", backend="MANISKILL")
        env2.close()

        env3 = make_env("PickCube-v1", backend="maniskill")
        env3.close()

    def test_make_env_with_control_mode(self):
        """测试指定控制模式"""
        env = make_env("PickCube-v1", control_mode="pd_joint_pos")
        obs, info = env.reset()
        assert obs is not None
        env.close()

    def test_make_env_multiple_kwargs(self):
        """测试多个后端特定参数"""
        # 使用 num_envs=1 避免 PhysX 重复初始化
        env = make_env(
            "PickCube-v1",
            obs_mode="state",
            control_mode="pd_joint_pos",
            num_envs=1
        )
        obs, info = env.reset()
        assert obs is not None
        env.close()

    def test_make_env_robot_id_priority(self):
        """测试 robot_id 优先级"""
        # 如果 robot_id 存在，应该优先使用 robot_id
        # 这里只测试不会报错，实际功能需要注册的机器人
        env = make_env("PickCube-v1", robot_id="panda")
        env.close()


class TestMakeEnvErrors:
    """测试 make_env() 错误处理"""

    def test_invalid_task(self):
        """测试无效任务"""
        with pytest.raises(Exception):  # 具体异常类型取决于后端
            make_env("InvalidTask-v999")

    @pytest.mark.skip(reason="需要 mock 模拟后端不可用场景")
    def test_backend_unavailable(self):
        """测试后端不可用

        使用 mock 模拟未安装的后端依赖。
        """
        from unittest.mock import patch
        with patch.object(ManiSkillBackend, '_check_dependencies', side_effect=ImportError):
            with pytest.raises(Exception):  # BackendUnavailableError
                make_env("PickCube-v1")


class TestMakeEnvIntegration:
    """测试 make_env() 集成场景"""

    def test_reset_step_close(self):
        """测试完整的 reset-step-close 循环"""
        env = make_env("PickCube-v1", obs_mode="state")

        # reset
        obs, info = env.reset()
        assert obs is not None

        # step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs is not None

        # close
        env.close()
        assert True  # 如果没有异常就算通过

    def test_multiple_episodes(self):
        """测试多个 episodes"""
        env = make_env("PickCube-v1", obs_mode="state")

        for episode in range(3):
            obs, info = env.reset(seed=episode)
            done = False
            steps = 0
            while not done and steps < 10:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1

        env.close()

    def test_different_tasks(self):
        """测试不同任务"""
        tasks = ["PickCube-v1"]  # 可以添加更多
        for task in tasks:
            env = make_env(task)
            obs, info = env.reset()
            assert obs is not None
            env.close()
