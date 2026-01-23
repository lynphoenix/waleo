"""
RJ2506机器人最小化配置 - 只包含必要的PickCube配置
不修改rest pose和load_agent
"""

RJ2506_PICKCUBE_CONFIG = {
    # Cube生成配置 - 方案1：机器人正前方稍偏左
    "cube_spawn_center": (-0.35, 0.10),
    "cube_spawn_half_size": 0.02,
    "cube_half_size": 0.008,
    "goal_thresh": 0.025,
    "max_goal_height": 0.3,
    # 相机配置 - 更新为指向新目标位置
    "sensor_cam_eye_pos": [-0.5, 0.2, 0.6],
    "sensor_cam_target_pos": [-0.35, 0.10, 0.1],
    "human_cam_eye_pos": [-1.5, -1.5, 1.5],
    "human_cam_target_pos": [-0.5, 0, 0.2],
}


def apply_rj2506_config_minimal():
    """
    最小化配置 - 只更新PickCube配置，不修改rest pose和load_agent
    """
    try:
        from mani_skill.envs.tasks.tabletop import pick_cube_cfgs

        # 只更新PickCube配置
        if "rj2506" not in pick_cube_cfgs.PICK_CUBE_CONFIGS:
            pick_cube_cfgs.PICK_CUBE_CONFIGS["rj2506"] = {}

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

        print("✓ 已应用RJ2506最小化配置（无rest pose/load_agent修改）")
        print(f"  - cube_spawn_center: {RJ2506_PICKCUBE_CONFIG['cube_spawn_center']}")
        print(f"  - cube_half_size: {RJ2506_PICKCUBE_CONFIG['cube_half_size']}")
        return True

    except Exception as e:
        print(f"✗ 应用配置失败: {e}")
        return False
