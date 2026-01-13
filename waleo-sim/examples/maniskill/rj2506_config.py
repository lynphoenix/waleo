"""
RJ2506机器人自定义配置
用于覆盖ManiSkill默认配置
"""

# RJ2506在PickCube任务中的自定义配置
RJ2506_PICKCUBE_CONFIG = {
    # 机器人加载位置
    # 机器人离桌子足够远，避免嵌入桌子
    "robot_load_position": [-0.85, 0, -0.35],  # 离桌子更远，避免干涉

    # Cube生成配置
    # 把Cube放在机器人前方约0.35米处，确保在手臂可达范围内
    "cube_spawn_center": (-0.5, 0),  # 更靠近机器人
    "cube_spawn_half_size": 0.01,  # 缩小到1cm，更稳定的生成位置
    "cube_half_size": 0.008,  # 缩小到0.8cm (1.6cm立方体)，更容易抓取

    # Goal配置
    "goal_thresh": 0.025,
    "max_goal_height": 0.3,

    # 相机配置
    "sensor_cam_eye_pos": [-0.5, 0.2, 0.6],
    "sensor_cam_target_pos": [-0.62, 0.29, 0.1],
    # human相机：从侧面斜上方观察，能看到地面
    "human_cam_eye_pos": [-1.5, -1.5, 1.5],
    "human_cam_target_pos": [-0.5, 0, 0.2],
}


def apply_rj2506_config():
    """
    动态应用RJ2506配置到ManiSkill环境
    在创建环境前调用此函数
    """
    try:
        from mani_skill.envs.tasks.tabletop import pick_cube_cfgs
        from mani_skill.envs.tasks.tabletop import pick_cube
        from mani_skill.agents.robots.rj2506 import RJ2506
        import numpy as np

        # 1. 修改RJ2506机器人的rest pose（夹爪初始完全张开）
        original_rest_qpos = RJ2506.keyframes["rest"].qpos.copy()
        # 修改最后两个值（夹爪）为0.015（完全张开）
        modified_rest_qpos = original_rest_qpos.copy()
        modified_rest_qpos[-2] = 0.015
        modified_rest_qpos[-1] = 0.015

        # 创建新的Keyframe对象
        from mani_skill.agents.base_agent import Keyframe
        RJ2506.keyframes["rest"] = Keyframe(
            qpos=modified_rest_qpos,
            pose=RJ2506.keyframes["rest"].pose
        )
        print(f"  - 已修改RJ2506 rest pose: 夹爪位置从 [{original_rest_qpos[-2]:.3f}, {original_rest_qpos[-1]:.3f}] 改为 [0.015, 0.015]")

        # 2. 更新PickCube配置
        pick_cube_cfgs.PICK_CUBE_CONFIGS["rj2506"].update({
            "cube_half_size": RJ2506_PICKCUBE_CONFIG["cube_half_size"],
            "goal_thresh": RJ2506_PICKCUBE_CONFIG["goal_thresh"],
            "cube_spawn_half_size": RJ2506_PICKCUBE_CONFIG["cube_spawn_half_size"],
            "cube_spawn_center": RJ2506_PICKCUBE_CONFIG["cube_spawn_center"],
            "max_goal_height": RJ2506_PICKCUBE_CONFIG["max_goal_height"],
            "sensor_cam_eye_pos": RJ2506_PICKCUBE_CONFIG["sensor_cam_eye_pos"],
            "sensor_cam_target_pos": RJ2506_PICKCUBE_CONFIG["sensor_cam_target_pos"],
            "human_cam_eye_pos": RJ2506_PICKCUBE_CONFIG["human_cam_eye_pos"],
            "human_cam_target_pos": RJ2506_PICKCUBE_CONFIG["human_cam_target_pos"],
        })

        # 3. 修改机器人加载位置
        original_load_agent = pick_cube.PickCubeEnv._load_agent

        def custom_load_agent(self, options):
            pos = RJ2506_PICKCUBE_CONFIG["robot_load_position"]
            # 调用原始的_load_agent，但使用自定义位置
            import sapien
            super(type(self), self)._load_agent(options, sapien.Pose(p=pos))

        # 替换方法
        pick_cube.PickCubeEnv._load_agent = custom_load_agent

        # 4. 修改reset方法，确保夹爪完全张开（双重保险）
        original_reset = pick_cube.PickCubeEnv.reset

        def custom_reset(self, seed=None, options=None):
            obs, info = original_reset(self, seed=seed, options=options)

            # 强制设置夹爪位置到完全张开 (0.015m)
            if self.robot_uids == "rj2506":
                import torch
                # 获取机器人
                agent = self.agent
                # 获取qpos
                qpos = agent.robot.get_qpos()
                # 设置夹爪关节为完全张开 (0.015m)
                # qpos shape: (num_envs, num_dofs) 或 (num_dofs,)
                if len(qpos.shape) == 1:
                    qpos[-2] = 0.015
                    qpos[-1] = 0.015
                else:
                    qpos[:, -2] = 0.015
                    qpos[:, -1] = 0.015
                # 应用qpos
                agent.robot.set_qpos(qpos)
                # 更新observation
                obs = self.get_obs()

            return obs, info

        pick_cube.PickCubeEnv.reset = custom_reset

        print("✓ 已应用RJ2506自定义配置")
        print(f"  - robot_load_position: {RJ2506_PICKCUBE_CONFIG['robot_load_position']}")
        print(f"  - cube_spawn_center: {RJ2506_PICKCUBE_CONFIG['cube_spawn_center']}")
        print(f"  - cube_spawn_half_size: {RJ2506_PICKCUBE_CONFIG['cube_spawn_half_size']}")
        print(f"  - cube_half_size: {RJ2506_PICKCUBE_CONFIG['cube_half_size']}")
        print(f"  - 夹爪rest pose已修改为完全张开")
        print(f"  - reset时强制夹爪完全张开")

        return True

    except Exception as e:
        print(f"✗ 应用配置失败: {e}")
        return False


if __name__ == "__main__":
    # 测试配置
    print("RJ2506配置:")
    for key, value in RJ2506_PICKCUBE_CONFIG.items():
        print(f"  {key}: {value}")
