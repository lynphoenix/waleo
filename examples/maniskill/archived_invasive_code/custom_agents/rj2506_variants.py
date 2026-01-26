"""
RJ2506 Robot Variants with fixed base (no mobile platform).
These use the no-agv URDF to place the robot on the floor instead of on a mobile platform.
The robot is positioned to the side of the table to avoid collision.
Note: Pose adjustment is handled in train_ppo_configurable.py via gym.make wrapper.
"""
from copy import deepcopy
import numpy as np
import sapien
import torch
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor


@register_agent(override=True)
class RJ2506_LeftArm(BaseAgent):
    uid = "RJ2506_LeftArm"
    # Use the no-AGV URDF to place robot on floor
    urdf_path = "/root/data2/lyn/waleo/waleo-sim/assets/robots/RJ2506/urdf/RJ2506_leftarm_only_noagv.urdf"
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            left_hand_finger1=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            left_hand_finger2=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([0.0, 0.0, 0.0, -np.pi/4, 0.0, np.pi/2, 0.04, 0.04]),
            pose=sapien.Pose(),
        )
    )

    arm_joint_names = [
        "left_arm_joint0",
        "left_arm_joint1",
        "left_arm_joint2",
        "left_arm_joint3",
        "left_arm_joint4",
        "left_arm_joint5",
    ]

    gripper_joint_names = [
        "left_hand_finger1_joint",
        "left_hand_finger2_joint",
    ]

    ee_link_name = "left_hand_tcp"

    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 100

    @property
    def _sensor_configs(self):
        return []

    @property
    def _controller_configs(self):
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )

        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            lower=-0.01,
            upper=0.04,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            mimic={"left_hand_finger2_joint": {"joint": "left_hand_finger1_joint"}},
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_pos
            ),
        )

        return deepcopy(controller_configs)

    def _after_init(self):
        self.finger1_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "left_hand_finger1"
        )
        self.finger2_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "left_hand_finger2"
        )
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :-2]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @property
    def tcp_pos(self):
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        return self.tcp.pose


@register_agent(override=True)
class RJ2506_LeftArm_WithWristCam(RJ2506_LeftArm):
    uid = "RJ2506_LeftArm_WithWristCam"

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="wrist_camera",
                pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
                width=256,
                height=256,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["left_hand_base"],
            ),
        ]


# Alias for backward compatibility
RJ2506 = RJ2506_LeftArm
