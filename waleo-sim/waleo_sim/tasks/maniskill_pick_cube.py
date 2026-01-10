"""ManiSkill 环境封装

基于 ManiSkill3 的 PickCube 任务，使用 Franka Panda 机械臂。
支持头部相机和腕部相机作为视觉输入。
"""

from typing import Optional, Tuple, Dict, Any, List
import numpy as np

from waleo_sim.base import RobotEnv
from waleo_config import EnvConfig, CameraConfig


class ManiSkillPickCubeEnv(RobotEnv):
    """ManiSkill3 PickCube 环境

    使用 Franka Panda 机械臂抓取立方体。

    特点：
    - Franka Panda 7-DOF 机械臂
    - 头部相机（外部观察）
    - 腕部相机（接近观察）
    - 关节位置控制

    观察空间：
    - robot_state: 7维（关节位置）
    - gripper_state: 2维（夹爪状态）
    - head_camera_rgb: (H, W, 3) 头部相机RGB图像
    - wrist_camera_rgb: (H, W, 3) 腕部相机RGB图像

    动作空间：
    - 8维（7关节位置 + 1夹爪）
    """

    # 机器人配置
    ROBOT_CONFIG = {
        "panda": {
            "action_dim": 8,  # 7关节 + 1夹爪
            "state_dim": 9,   # 机器人状态维度
            "joint_dim": 7,   # 关节数量
        }
    }

    def __init__(
        self,
        robot_type: str = "panda",
        render_mode: Optional[str] = None,
        headless: bool = False,
        image_size: Tuple[int, int] = (224, 224),
        use_cameras: bool = True,
    ):
        """
        Args:
            robot_type: 机器人类型（panda）
            render_mode: 渲染模式
            headless: 无头模式
            image_size: 相机图像大小 (width, height)
            use_cameras: 是否使用相机观察
        """
        self.image_size = image_size
        self.use_cameras = use_cameras

        super().__init__(
            task="pick_place",
            robot_type=robot_type,
            render_mode=render_mode,
            simulation_backend="maniskill",
            headless=headless,
        )

        # ManiSkill 特定配置
        self._init_maniskill_env()

    def _init_maniskill_env(self):
        """初始化 ManiSkill3 环境"""
        try:
            from mani_skill.envs import PickCubeEnv
        except ImportError as e:
            raise ImportError(
                f"ManiSkill3 import failed: {e}\n"
                "Install with: pip install mani_skill"
            ) from e

        # 创建 ManiSkill3 环境
        self.maniskill_env = PickCubeEnv(
            robot_uids="panda",
        )

        # 获取观察和动作空间
        self._action_space = self.maniskill_env.action_space

        # 定义观察空间
        width, height = self.image_size

        # 状态观察（机器人状态）
        state_dim = 7 + 2  # 关节位置 + 夹爪状态

        # 相机观察
        camera_channels = 3 if self.use_cameras else 0
        camera_obs_dim = width * height * camera_channels

        # 总观察维度
        self._observation_dim = state_dim + (2 * camera_obs_dim if self.use_cameras else 0)

        # 创建观察空间类
        from gymnasium import spaces

        self._observation_space_space = spaces.Dict({
            "robot_state": spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,)),
            "gripper_state": spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
        })

        if self.use_cameras:
            self._observation_space_space.spaces["head_camera_rgb"] = spaces.Box(
                low=0, high=255, shape=(height, width, 3), dtype=np.uint8
            )
            self._observation_space_space.spaces["wrist_camera_rgb"] = spaces.Box(
                low=0, high=255, shape=(height, width, 3), dtype=np.uint8
            )

    def _load_robot_model(self):
        """加载机器人模型（由 ManiSkill 处理）"""
        pass  # ManiSkill 自动加载模型

    def _setup_cameras(self):
        """设置相机（由 ManiSkill 处理）"""
        pass  # ManiSkill 自动设置相机

    def _compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> float:
        """计算奖励（由 ManiSkill 环境提供）"""
        # ManiSkill 环境内部计算奖励
        return 0.0

    def _check_termination(self) -> bool:
        """检查是否完成任务"""
        # ManiSkill 环境内部检查终止条件
        return False

    @property
    def robot_state(self) -> np.ndarray:
        """获取机器人状态"""
        obs = self._get_current_obs()
        if obs is None:
            return np.zeros(9)
        return self._extract_robot_state(obs)

    @property
    def camera_images(self) -> Dict[str, np.ndarray]:
        """获取所有相机图像"""
        if not self.use_cameras:
            return {}

        obs = self._get_current_obs()
        if obs is None:
            return {
                "head_camera_rgb": np.zeros((*self.image_size[::-1], 3), dtype=np.uint8),
                "wrist_camera_rgb": np.zeros((*self.image_size[::-1], 3), dtype=np.uint8),
            }

        return self._extract_camera_images(obs)

    def _get_current_obs(self) -> Optional[Dict]:
        """获取当前观察"""
        if not hasattr(self, "_last_obs") or self._last_obs is None:
            return None
        return self._last_obs

    def _extract_robot_state(self, obs: Dict) -> np.ndarray:
        """从观察中提取机器人状态"""
        # ManiSkill3 的观察格式：
        # obs 是一个包含 "agent" 和 "extra" 键的字典
        # 数据是 torch.Tensor 类型
        state = []

        if isinstance(obs, dict):
            # 处理 agent 观察（机器人本体状态）
            if "agent" in obs:
                agent_data = obs["agent"]
                # 提取关键数据
                if hasattr(agent_data, 'keys'):
                    # agent_data 也是字典
                    for key in ["qpos", "qvel", "tcp_pose"]:
                        if key in agent_data:
                            val = agent_data[key]
                            if hasattr(val, 'cpu'):
                                val = val.cpu().numpy()
                            if hasattr(val, 'flatten'):
                                val = val.flatten()
                            state.extend(val)
                else:
                    # agent_data 是 tensor
                    val = agent_data
                    if hasattr(val, 'cpu'):
                        val = val.cpu().numpy()
                    if hasattr(val, 'flatten'):
                        val = val.flatten()
                    state.extend(val[:9])  # 取前9个元素

            # 处理 extra 观察（任务相关状态）
            if "extra" in obs:
                extra_data = obs["extra"]
                # 提取目标位置等
                for key in ["goal_pos", "tcp_pose", "is_grasped"]:
                    if key in extra_data:
                        val = extra_data[key]
                        if hasattr(val, 'cpu'):
                            val = val.cpu().numpy()
                        if hasattr(val, 'flatten'):
                            val = val.flatten()
                        state.extend(val)

        # 如果没有提取到任何状态，返回默认值
        if len(state) == 0:
            return np.zeros(9, dtype=np.float32)

        # 转换为 numpy 数组
        state_array = np.array(state, dtype=np.float32)

        # 截断或填充到 9 维
        if len(state_array) > 9:
            state_array = state_array[:9]
        elif len(state_array) < 9:
            state_array = np.pad(state_array, (0, 9 - len(state_array)))

        return state_array

    def _extract_camera_images(self, obs: Dict) -> Dict[str, np.ndarray]:
        """从观察中提取相机图像"""
        images = {}
        height, width = self.image_size[::-1]

        if isinstance(obs, dict):
            # ManiSkill3 可能将图像数据放在 "sensor_data" 或其他键下
            # 检查可能的键
            for key in ["sensor_data", "images", "rgb", "rgbd"]:
                if key in obs:
                    img_data = obs[key]
                    if isinstance(img_data, dict):
                        # 遍历图像字典
                        for cam_name, cam_img in img_data.items():
                            if hasattr(cam_img, 'cpu'):
                                cam_img = cam_img.cpu().numpy()
                            if hasattr(cam_img, 'permute'):
                                # CHW -> HWC
                                cam_img = cam_img.permute(1, 2, 0)
                            if len(cam_img.shape) == 3 and cam_img.shape[2] >= 3:
                                # 取 RGB 通道
                                rgb_img = cam_img[:, :, :3]
                                # 归一化到 0-255
                                if rgb_img.max() <= 1.0:
                                    rgb_img = (rgb_img * 255).astype(np.uint8)
                                else:
                                    rgb_img = rgb_img.astype(np.uint8)

                                # 根据相机名称分配
                                if "base" in cam_name or "front" in cam_name:
                                    images["head_camera_rgb"] = rgb_img
                                elif "wrist" in cam_name or "hand" in cam_name:
                                    images["wrist_camera_rgb"] = rgb_img
                                else:
                                    # 默认第一个给头部，第二个给腕部
                                    if "head_camera_rgb" not in images:
                                        images["head_camera_rgb"] = rgb_img
                                    elif "wrist_camera_rgb" not in images:
                                        images["wrist_camera_rgb"] = rgb_img
                    break

        # 如果没有提取到图像，返回空白图像
        if "head_camera_rgb" not in images:
            images["head_camera_rgb"] = np.zeros((height, width, 3), dtype=np.uint8)
        if "wrist_camera_rgb" not in images:
            images["wrist_camera_rgb"] = np.zeros((height, width, 3), dtype=np.uint8)

        return images

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        """重置环境"""
        self._last_obs, info = self.maniskill_env.reset(seed=seed, options=options)

        # 构建观察字典
        obs = self._build_observation()

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """执行一步"""
        self._last_obs, reward, terminated, truncated, info = self.maniskill_env.step(action)

        # 构建观察字典
        obs = self._build_observation()

        return obs, reward, terminated, truncated, info

    def _build_observation(self) -> Dict:
        """构建观察字典"""
        obs = {}

        # 添加机器人状态
        obs["robot_state"] = self.robot_state
        obs["gripper_state"] = self.robot_state[-2:]  # 夹爪状态

        # 添加相机图像
        if self.use_cameras:
            camera_images = self.camera_images
            obs.update(camera_images)

        return obs

    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        """渲染环境"""
        if mode == "human":
            self.maniskill_env.render()
            return None
        elif mode == "rgb_array":
            # 返回头部相机的图像
            images = self.camera_images
            if "head_camera_rgb" in images:
                return images["head_camera_rgb"]
            else:
                return np.zeros((224, 224, 3), dtype=np.uint8)

        return None

    def close(self) -> None:
        """关闭环境"""
        if hasattr(self, "maniskill_env"):
            self.maniskill_env.close()

    @property
    def observation_space(self):
        """观察空间"""
        return self._observation_space_space

    @property
    def action_space(self):
        """动作空间"""
        return self._action_space

    def get_robot_config(self) -> dict:
        """获取机器人配置

        Returns:
            机器人配置字典，包含 action_dim, state_dim 等
        """
        robot_type = self.robot_type if self.robot_type in self.ROBOT_CONFIG else "panda"
        return self.ROBOT_CONFIG[robot_type]


def create_maniskill_env(
    image_size: Tuple[int, int] = (224, 224),
    use_cameras: bool = True,
    headless: bool = False,
) -> ManiSkillPickCubeEnv:
    """创建 ManiSkill PickCube 环境工厂函数

    Args:
        image_size: 相机图像大小 (width, height)
        use_cameras: 是否使用相机观察
        headless: 无头模式

    Returns:
        ManiSkill 环境实例
    """
    env = ManiSkillPickCubeEnv(
        robot_type="panda",
        image_size=image_size,
        use_cameras=use_cameras,
        headless=headless,
    )
    return env
