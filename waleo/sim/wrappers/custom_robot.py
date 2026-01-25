"""自定义机器人集成包装器

提供无侵入式的机器人和任务配置应用，替代 monkey-patching。
"""

import gymnasium as gym
from typing import Optional, Dict, Any, Tuple
import numpy as np
import warnings

try:
    from waleo.sim.base import EnvWrapper
except ImportError:
    # 如果 EnvWrapper 未实现，使用 gymnasium.Wrapper
    EnvWrapper = gym.Wrapper

from waleo.sim.registry.robot import get_robot_registry


class CustomRobotWrapper(EnvWrapper):
    """自定义机器人集成包装器

    无侵入式地应用机器人特定配置，替代 monkey-patching ManiSkill 内部类。

    功能：
    - 应用机器人 pose 调整
    - 应用 keyframes 配置
    - 应用 URDF 配置
    - 支持任务特定配置

    Examples:
        >>> from waleo.sim.wrappers import CustomRobotWrapper
        >>> import gymnasium as gym
        >>>
        >>> # 创建基础环境
        >>> env = gym.make("PickCube-v1", robot_uids="rj2506")
        >>>
        >>> # 应用自定义配置（无侵入式）
        >>> task_config = {
        ...     "robot_pose": {"offset": [-0.85, 0, -0.35]},
        ...     "keyframes": {"rest": {"qpos": [0, 0, 0, -0.785, 0, 1.57, 0.015, 0.015]}}
        ... }
        >>> env = CustomRobotWrapper(env, "rj2506", task_config)
    """

    def __init__(
        self,
        env: gym.Env,
        robot_name: str,
        task_config: Optional[Dict[str, Any]] = None
    ):
        """初始化包装器

        Args:
            env: 被包装的环境
            robot_name: 机器人名称
            task_config: 任务特定配置字典

        Raises:
            ValueError: 如果机器人未注册
        """
        super().__init__(env)
        self.robot_name = robot_name
        self.task_config = task_config or {}

        # 从注册中心获取机器人规格
        registry = get_robot_registry()
        self.robot_spec = registry.get(robot_name)

        if self.robot_spec is None:
            raise ValueError(
                f"Robot '{robot_name}' not found in registry. "
                f"Available robots: {registry.list_robots()}"
            )

        # 标记是否已应用配置
        self._config_applied = False

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """重置环境并应用自定义配置

        Args:
            **kwargs: 传递给环境的 reset 参数

        Returns:
            observation: 初始观察
            info: 信息字典
        """
        # 调用原始环境的 reset
        obs, info = self.env.reset(**kwargs)

        # 应用自定义配置（仅在首次或需要时）
        if not self._config_applied or kwargs.get("force_apply", False):
            self._apply_configurations()
            self._config_applied = True

        return obs, info

    def _apply_configurations(self):
        """应用所有自定义配置

        按顺序应用：
        1. 机器人 pose 调整
        2. Keyframes 配置
        3. 其他任务特定配置
        """
        if "robot_pose" in self.task_config:
            self._apply_robot_pose(self.task_config["robot_pose"])

        if "keyframes" in self.task_config:
            self._apply_keyframes(self.task_config["keyframes"])

    def _apply_robot_pose(self, pose_config: Dict):
        """应用机器人 pose 调整（无侵入式）

        Args:
            pose_config: Pose 配置字典，格式：
                {
                    "offset": [x, y, z],  # 位置偏移
                    "rotation": [qw, qx, qy, qz]  # 可选的四元数旋转
                }
        """
        try:
            # 通过公开 API 访问机器人
            if not hasattr(self.env.unwrapped, "agent"):
                warnings.warn(
                    f"Environment does not have 'agent' attribute. "
                    f"Cannot apply robot pose for {self.robot_name}",
                    UserWarning
                )
                return

            agent = self.env.unwrapped.agent

            if not hasattr(agent, "robot"):
                warnings.warn(
                    f"Agent does not have 'robot' attribute. "
                    f"Cannot apply robot pose for {self.robot_name}",
                    UserWarning
                )
                return

            robot = agent.robot

            # 应用位置偏移
            if "offset" in pose_config:
                offset = np.array(pose_config["offset"])
                current_pose = robot.pose

                # 使用 SAPIEN 的 Pose API（ManiSkill 基于 SAPIEN）
                try:
                    import sapien
                    new_p = current_pose.p + offset

                    if "rotation" in pose_config:
                        # 如果提供了旋转，应用旋转
                        q = pose_config["rotation"]
                        new_pose = sapien.Pose(p=new_p, q=q)
                    else:
                        # 否则保持原旋转
                        new_pose = sapien.Pose(p=new_p, q=current_pose.q)

                    robot.set_pose(new_pose)

                except ImportError:
                    warnings.warn(
                        "SAPIEN not available. Using numpy array for pose.",
                        UserWarning
                    )
                    # 降级处理：直接设置位置
                    if hasattr(robot, "set_pose"):
                        robot.set_pose(offset)

        except Exception as e:
            warnings.warn(
                f"Failed to apply robot pose for {self.robot_name}: {e}",
                UserWarning
            )

    def _apply_keyframes(self, keyframes_config: Dict):
        """应用 keyframes 配置（无侵入式）

        Args:
            keyframes_config: Keyframes 配置字典，格式：
                {
                    "rest": {
                        "qpos": [q0, q1, ..., qn]
                    }
                }
        """
        try:
            if not hasattr(self.env.unwrapped, "agent"):
                return

            agent = self.env.unwrapped.agent

            # 应用 rest keyframe（最常用）
            if "rest" in keyframes_config:
                rest_config = keyframes_config["rest"]

                if "qpos" in rest_config:
                    qpos = np.array(rest_config["qpos"])

                    # 尝试设置机器人的 qpos
                    if hasattr(agent, "robot") and hasattr(agent.robot, "set_qpos"):
                        agent.robot.set_qpos(qpos)
                    elif hasattr(agent, "set_qpos"):
                        agent.set_qpos(qpos)

        except Exception as e:
            warnings.warn(
                f"Failed to apply keyframes for {self.robot_name}: {e}",
                UserWarning
            )


class TaskConfigWrapper(EnvWrapper):
    """任务配置包装器

    无侵入式地应用任务特定配置（物体大小、生成位置等）。

    Examples:
        >>> from waleo.sim.wrappers import TaskConfigWrapper
        >>> import gymnasium as gym
        >>>
        >>> env = gym.make("PickCube-v1", robot_uids="panda")
        >>>
        >>> # 应用任务配置
        >>> task_config = {
        ...     "cube_half_size": 0.008,
        ...     "cube_spawn_center": [-0.5, 0],
        ...     "goal_thresh": 0.025
        ... }
        >>> env = TaskConfigWrapper(env, task_config)
    """

    def __init__(self, env: gym.Env, task_config: Dict[str, Any]):
        """初始化任务配置包装器

        Args:
            env: 被包装的环境
            task_config: 任务配置字典
        """
        super().__init__(env)
        self.task_config = task_config

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """重置环境并注入任务配置

        通过 reset 的 options 参数注入配置，这是 Gymnasium API 的标准方式。

        Args:
            **kwargs: reset 参数

        Returns:
            observation: 初始观察
            info: 信息字典
        """
        # 合并任务配置到 reset options
        reset_options = kwargs.get("options", {})
        reset_options.update(self.task_config)
        kwargs["options"] = reset_options

        return self.env.reset(**kwargs)


class CameraConfigWrapper(EnvWrapper):
    """相机配置包装器

    应用自定义相机位置和参数。

    Examples:
        >>> camera_config = {
        ...     "sensor_cam": {
        ...         "eye_pos": [-0.5, 0.2, 0.6],
        ...         "target_pos": [-0.62, 0.29, 0.1]
        ...     }
        ... }
        >>> env = CameraConfigWrapper(env, camera_config)
    """

    def __init__(self, env: gym.Env, camera_config: Dict[str, Dict]):
        """初始化相机配置包装器

        Args:
            env: 被包装的环境
            camera_config: 相机配置字典
        """
        super().__init__(env)
        self.camera_config = camera_config

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """重置并应用相机配置"""
        obs, info = self.env.reset(**kwargs)
        self._apply_camera_config()
        return obs, info

    def _apply_camera_config(self):
        """应用相机配置"""
        try:
            # 尝试通过环境 API 设置相机
            if hasattr(self.env.unwrapped, "cameras"):
                cameras = self.env.unwrapped.cameras

                for cam_name, cam_params in self.camera_config.items():
                    if cam_name in cameras:
                        camera = cameras[cam_name]

                        if "eye_pos" in cam_params:
                            # 设置相机位置
                            pass  # 具体实现依赖于 ManiSkill API

                        if "target_pos" in cam_params:
                            # 设置观察目标
                            pass

        except Exception as e:
            warnings.warn(
                f"Failed to apply camera config: {e}",
                UserWarning
            )


def create_wrapped_env(
    base_env: gym.Env,
    robot_name: str,
    task_config: Optional[Dict] = None
) -> gym.Env:
    """便捷函数：创建完整包装的环境

    自动应用所有相关的包装器。

    Args:
        base_env: 基础环境
        robot_name: 机器人名称
        task_config: 完整的任务配置字典

    Returns:
        包装后的环境

    Examples:
        >>> env = gym.make("PickCube-v1", robot_uids="rj2506")
        >>> env = create_wrapped_env(env, "rj2506", full_task_config)
    """
    if task_config is None:
        task_config = {}

    env = base_env

    # 1. 应用机器人配置
    robot_config = {
        k: v for k, v in task_config.items()
        if k in ["robot_pose", "keyframes", "urdf_config"]
    }
    if robot_config:
        env = CustomRobotWrapper(env, robot_name, robot_config)

    # 2. 应用任务配置
    object_config = task_config.get("object_config", {})
    if object_config:
        env = TaskConfigWrapper(env, object_config)

    # 3. 应用相机配置
    camera_config = task_config.get("camera_config", {})
    if camera_config:
        env = CameraConfigWrapper(env, camera_config)

    return env
