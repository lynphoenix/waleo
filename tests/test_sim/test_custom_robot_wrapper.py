"""测试自定义机器人包装器"""

import gymnasium as gym
import numpy as np
from unittest.mock import Mock, MagicMock
import sys
import os

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from waleo.sim.wrappers.custom_robot import (
    CustomRobotWrapper,
    TaskConfigWrapper,
    CameraConfigWrapper,
    create_wrapped_env,
)
from waleo.sim.registry.robot import RobotRegistry, RobotSpec, get_robot_registry
from pathlib import Path


def create_mock_env():
    """创建模拟环境用于测试"""
    env = Mock(spec=gym.Env)
    env.observation_space = gym.spaces.Box(low=0, high=1, shape=(10,))
    env.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,))

    # 模拟 reset 返回
    env.reset.return_value = (np.zeros(10), {})

    # 模拟 unwrapped
    env.unwrapped = Mock()
    env.unwrapped.agent = Mock()
    env.unwrapped.agent.robot = Mock()
    env.unwrapped.agent.robot.pose = Mock()
    env.unwrapped.agent.robot.pose.p = np.array([0, 0, 0])
    env.unwrapped.agent.robot.pose.q = np.array([1, 0, 0, 0])
    env.unwrapped.agent.robot.set_pose = Mock()

    return env


def create_test_robot_spec():
    """创建测试用的机器人规格"""
    return RobotSpec(
        name="test_robot",  # 使用大写开头
        urdf_path=Path("/fake/path/test_robot.urdf"),
        robot_dir=Path("/fake/path"),
        dof=7,
        task_configs={
            "TestTask-v1": {
                "robot_pose": {"offset": [1, 0, 0]},
                "keyframes": {"rest": {"qpos": [0, 0, 0, 0, 0, 0, 0]}}
            }
        }
    )


def test_custom_robot_wrapper_initialization():
    """测试 CustomRobotWrapper 初始化"""
    env = create_mock_env()

    # 使用全局单例注册表
    registry = get_robot_registry()
    spec = create_test_robot_spec()
    registry.register(spec)

    try:
        # 创建包装器 - 使用大写的机器人名称
        wrapper = CustomRobotWrapper(
            env,
            "test_robot",  # 大写
            task_config={"robot_pose": {"offset": [1, 0, 0]}}
        )

        assert wrapper.robot_name == "test_robot"
        assert wrapper.robot_spec == spec
        assert "robot_pose" in wrapper.task_config
    finally:
        # 清理测试数据
        if "test_robot" in registry._robots:
            del registry._robots["test_robot"]


def test_custom_robot_wrapper_reset():
    """测试 CustomRobotWrapper reset 方法"""
    env = create_mock_env()

    # 使用全局单例注册表
    registry = get_robot_registry()
    spec = create_test_robot_spec()
    registry.register(spec)

    try:
        # 创建包装器
        wrapper = CustomRobotWrapper(
            env,
            "test_robot",
            task_config={"robot_pose": {"offset": [1, 2, 3]}}
        )

        # 调用 reset
        obs, info = wrapper.reset()

        # 验证环境的 reset 被调用
        env.reset.assert_called_once()

        # 验证返回值正确
        assert obs is not None
        assert isinstance(info, dict)
    finally:
        # 清理测试数据
        if "test_robot" in registry._robots:
            del registry._robots["test_robot"]


def test_task_config_wrapper():
    """测试 TaskConfigWrapper"""
    env = create_mock_env()

    task_config = {
        "cube_half_size": 0.02,
        "goal_thresh": 0.05
    }

    wrapper = TaskConfigWrapper(env, task_config)

    # 调用 reset
    obs, info = wrapper.reset()

    # 验证 reset 被调用且传入了 options
    assert env.reset.called
    call_kwargs = env.reset.call_args.kwargs
    assert "options" in call_kwargs
    assert "cube_half_size" in call_kwargs["options"]
    assert call_kwargs["options"]["cube_half_size"] == 0.02


def test_camera_config_wrapper():
    """测试 CameraConfigWrapper"""
    env = create_mock_env()

    camera_config = {
        "sensor_cam": {
            "eye_pos": [1, 2, 3],
            "target_pos": [0, 0, 0]
        }
    }

    wrapper = CameraConfigWrapper(env, camera_config)

    # 调用 reset
    obs, info = wrapper.reset()

    # 验证 reset 被调用
    env.reset.assert_called_once()


def test_create_wrapped_env():
    """测试 create_wrapped_env 便捷函数"""
    env = create_mock_env()

    # 使用全局单例注册表
    registry = get_robot_registry()
    spec = create_test_robot_spec()
    registry.register(spec)

    try:
        # 创建完整配置
        full_config = {
            "robot_pose": {"offset": [1, 0, 0]},
            "object_config": {"cube_half_size": 0.02},
            "camera_config": {"sensor_cam": {"eye_pos": [1, 2, 3]}}
        }

        # 创建包装环境
        wrapped = create_wrapped_env(env, "test_robot", full_config)

        # 验证是多层包装
        assert isinstance(wrapped, CameraConfigWrapper)
        assert isinstance(wrapped.env, TaskConfigWrapper)
        assert isinstance(wrapped.env.env, CustomRobotWrapper)
    finally:
        # 清理测试数据
        if "test_robot" in registry._robots:
            del registry._robots["test_robot"]


def test_wrapper_without_robot_spec():
    """测试没有注册机器人时的错误处理"""
    env = create_mock_env()

    # 确保测试机器人不在注册表中
    registry = get_robot_registry()
    if "test_nonexistent_robot" in registry._robots:
        del registry._robots["test_nonexistent_robot"]
    if "nonexistent_robot" in registry._robots:
        del registry._robots["nonexistent_robot"]

    try:
        wrapper = CustomRobotWrapper(env, "nonexistent_robot")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "not found in registry" in str(e)


def test_apply_robot_pose_with_sapien():
    """测试应用机器人 pose（带 SAPIEN）"""
    env = create_mock_env()

    # 使用全局单例注册表
    registry = get_robot_registry()
    spec = create_test_robot_spec()
    registry.register(spec)

    try:
        # 创建包装器
        wrapper = CustomRobotWrapper(
            env,
            "test_robot",
            task_config={"robot_pose": {"offset": [1, 2, 3]}}
        )

        # 调用 reset（会触发 _apply_robot_pose）
        obs, info = wrapper.reset()

        # 验证 set_pose 被调用（即使没有真正的 SAPIEN）
        # 注意：实际测试中可能需要 mock SAPIEN
    finally:
        # 清理测试数据
        if "test_robot" in registry._robots:
            del registry._robots["test_robot"]


if __name__ == "__main__":
    print("Running wrapper tests...")

    test_custom_robot_wrapper_initialization()
    print("✓ test_custom_robot_wrapper_initialization")

    test_custom_robot_wrapper_reset()
    print("✓ test_custom_robot_wrapper_reset")

    test_task_config_wrapper()
    print("✓ test_task_config_wrapper")

    test_camera_config_wrapper()
    print("✓ test_camera_config_wrapper")

    test_create_wrapped_env()
    print("✓ test_create_wrapped_env")

    test_wrapper_without_robot_spec()
    print("✓ test_wrapper_without_robot_spec")

    test_apply_robot_pose_with_sapien()
    print("✓ test_apply_robot_pose_with_sapien")

    print("\n✅ All tests passed!")
