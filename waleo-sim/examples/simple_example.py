"""Waleo-Sim 使用示例

演示如何使用 waleo-sim 模块创建和使用仿真环境。
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from waleo_sim.base import BaseEnv, EnvWrapper, RobotEnv
from waleo_config import EnvConfig


# 示例 1: 创建自定义环境
class SimpleNavigationEnv(BaseEnv):
    """简单的导航环境示例

    目标: 控制智能体从起点移动到目标点
    """

    def __init__(self, goal_position=(5.0, 5.0)):
        self.goal_position = np.array(goal_position)
        self.agent_position = np.zeros(2)
        self.max_steps = 100
        self.current_step = 0

        # 定义观察空间和动作空间（简化版）
        class Space:
            def __init__(self, shape):
                self._shape = shape

            @property
            def shape(self):
                return self._shape

            def sample(self):
                return np.random.randn(*self._shape)

        self._observation_space = Space((4,))  # [agent_x, agent_y, goal_x, goal_y]
        self._action_space = Space((2,))  # [velocity_x, velocity_y]

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def reset(self, seed=None, options=None):
        """重置环境"""
        self.agent_position = np.zeros(2)
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        """执行一步"""
        # 更新智能体位置
        self.agent_position += action * 0.1
        self.current_step += 1

        # 计算奖励（负距离）
        distance = np.linalg.norm(self.agent_position - self.goal_position)
        reward = -distance

        # 检查是否到达目标
        terminated = distance < 0.5

        # 检查是否超时
        truncated = self.current_step >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {"distance": distance}

    def _get_obs(self):
        """获取观察"""
        return np.concatenate([self.agent_position, self.goal_position])


# 示例 2: 创建环境包装器
class NoisyObservationWrapper(EnvWrapper):
    """添加噪声的观察包装器"""

    def __init__(self, env, noise_level=0.1):
        super().__init__(env)
        self.noise_level = noise_level

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # 添加噪声
        noisy_obs = obs + np.random.randn(*obs.shape) * self.noise_level
        return noisy_obs, reward, terminated, truncated, info


# 示例 3: 创建机器人环境（框架）
class SimpleRobotEnv(RobotEnv):
    """简单的机器人环境框架"""

    def __init__(self, task="reach", **kwargs):
        super().__init__(task=task, **kwargs)
        # 初始化状态
        self.end_effector_pos = np.zeros(3)
        self.goal_pos = np.array([0.5, 0.0, 0.5])

    def _load_robot_model(self):
        """加载机器人模型"""
        print(f"加载 {self.robot_type} 机器人模型...")

    def _setup_cameras(self):
        """设置相机"""
        print("设置相机...")

    def _compute_reward(self, achieved_goal, desired_goal):
        """计算奖励"""
        return -np.linalg.norm(achieved_goal - desired_goal)

    @property
    def robot_state(self):
        """获取机器人状态"""
        return self.end_effector_pos

    def reset(self, seed=None, options=None):
        """重置环境"""
        # 重置末端执行器和目标位置
        self.end_effector_pos = np.zeros(3)
        self.goal_pos = np.random.rand(3) * 0.5
        return self.robot_state, {}

    def step(self, action):
        """执行一步"""
        # 更新末端执行器位置
        self.end_effector_pos += action * 0.05

        # 计算奖励
        reward = self._compute_reward(self.end_effector_pos, self.goal_pos)

        # 检查终止
        terminated = self._check_termination()
        truncated = False

        return self.robot_state, reward, terminated, truncated, {}


def run_example(env, num_episodes=3):
    """运行环境示例"""
    print(f"\n运行环境: {env.__class__.__name__}")
    print("=" * 50)

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0

        for step in range(100):
            # 随机动作（实际使用时应该用策略）
            action = env.action_space.sample()

            # 执行步骤
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1

            if terminated or truncated:
                break

        print(f"Episode {episode + 1}: "
              f"Reward={episode_reward:.2f}, "
              f"Length={episode_length}")

    print("=" * 50)


if __name__ == "__main__":
    print("Waleo-Sim 使用示例")
    print("=" * 50)

    # 示例 1: 简单导航环境
    print("\n1. 简单导航环境")
    nav_env = SimpleNavigationEnv(goal_position=(3.0, 3.0))
    run_example(nav_env, num_episodes=3)

    # 示例 2: 使用包装器
    print("\n2. 使用噪声包装器")
    noisy_nav_env = NoisyObservationWrapper(nav_env, noise_level=0.1)
    run_example(noisy_nav_env, num_episodes=3)

    # 示例 3: 机器人环境框架
    print("\n3. 机器人环境框架")
    from waleo_config import EnvConfig

    config = EnvConfig(
        task="reach",
        robot_type="so100",
        simulation_backend="mujoco"
    )
    print(f"配置: {config}")

    robot_env = SimpleRobotEnv(
        task="reach",
        robot_type="so100",
        simulation_backend="mujoco"
    )
    print(f"环境: {robot_env}")

    print("\n" + "=" * 50)
    print("示例运行完成！")
    print("=" * 50)
