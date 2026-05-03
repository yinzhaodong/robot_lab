# Copyright (c) 2024-2025 ArcLab
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
import robot_lab.tasks.locomotion.velocity.mdp as mdp
from robot_lab.tasks.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg, RewardsCfg,
)

##
# Pre-defined configs
##
# use cloud assets
# from isaaclab_assets.robots.unitree import ARCLAB_ARCDOG_CFG  # isort: skip
# use local assets
from robot_lab.assets.arclab import ARCLAB_ARCDOG_ADJUSTABLE_LEG_CFG  # isort: skip
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort:skip


@configclass
class ArcdogAdjustableLegRewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""

    rotate_joint_pos_penalty = RewTerm(
        func=mdp.joint_position_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_(thigh_joint|calf_joint)$"),
            "stand_still_scale": 1.8,
            "velocity_threshold": 0.3,
        },
    )

    box_joint_pos_penalty = RewTerm(
        func=mdp.joint_position_penalty,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_box_joint"),
            "stand_still_scale": 1.25,
            "velocity_threshold": 0.5,
        },
    )

    stand_still_flat = RewTerm(
        func=mdp.stand_still_flat_orientation_bonus,
        weight=0.0,  # 默认权重设为0，在 EnvCfg 中具体配置
        params={
            "command_name": "base_velocity",
            "std": 0.1,               # 控制对倾斜的敏感度，越小越严格
            "command_threshold": 0.1,  # 速度指令小于此值视为静止
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


    # 惩罚伸缩腿的剧烈加速度 (震荡的主要特征)
    box_joint_acc_penalty = RewTerm(
        func=mdp.joint_acc_l2,
        weight=0.0, # 在 EnvCfg 中激活
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_box_joint"),
        },
    )
    
    # 针对伸缩腿的关节速度惩罚
    box_joint_vel_penalty = RewTerm(
        func=mdp.joint_vel_l2,  # 使用关节速度，它支持 asset_cfg
        weight=-0.01,           # 权重建议：从 -0.01 到 -0.05 开始尝试，太大会导致腿动不了
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_box_joint"),
        },
    )

    box_joint_action_rate = RewTerm(
        func=mdp.action_rate_l2_by_name,
        weight=-0.0, 
        params={
            # 使用正则表达式匹配你的伸缩关节，比如包含 "box_joint" 的所有关节
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*box_joint.*")
        }
    )

    # 针对伸缩腿的专属限位惩罚
    box_joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,  # 复用同一个底层函数
        weight=0.0,                 
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_box_joint"),
        },
    )

    # Keep the public reward name as "feet_gait", but use the MGDP-style
    # motion_trot implementation: diagonal joint-pose symmetry penalty.
    feet_gait = RewTerm(
        func=mdp.motion_trot_joint_symmetry,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "velocity_threshold": 0.1,
            "stop_after_steps": None,
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_group_pairs": (
                (
                    ("FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"),
                    ("RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"),
                ),
                (
                    ("FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"),
                    ("RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"),
                ),
            ),
        },
    )

    # -------------------------------------------------------------------------
    # MGDP stage1 reward candidates.
    # Source reference:
    # https://github.com/arclab-hku/MGDP/blob/master/legged_gym/legged_gym/envs/random_dog/random_dog_config_stage1.py
    #
    # These terms mirror the MGDP reward names and are disabled by default
    # (weight = 0.0). Enable them in EnvCfg.__post_init__ only when you want to
    # test a specific MGDP-style reward. Most of them duplicate existing rewards,
    # so avoid enabling both the original term and its mgdp_* alias unless you
    # intentionally want double counting.
    # -------------------------------------------------------------------------
    mgdp_tracking_lin_vel = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=0.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    mgdp_tracking_ang_vel = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    mgdp_lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0)
    mgdp_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=0.0)
    mgdp_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    mgdp_dof_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    mgdp_action_rate = RewTerm(func=mdp.action_rate_l2, weight=0.0)
    mgdp_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    mgdp_collision = RewTerm(
        func=mdp.undesired_contacts,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "threshold": 1.0},
    )
    mgdp_motion_trot = RewTerm(
        func=mdp.motion_trot_joint_symmetry,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "velocity_threshold": 0.1,
            "stop_after_steps": None,
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_group_pairs": (
                (
                    ("FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"),
                    ("RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"),
                ),
                (
                    ("FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"),
                    ("RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"),
                ),
            ),
        },
    )
    mgdp_feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "threshold": 0.5,
            "command_threshold": 0.25,
        },
    )
    mgdp_feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="")},
    )
    mgdp_stand_still = RewTerm(
        func=mdp.stand_still_without_cmd,
        weight=0.0,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )



@configclass
class ArclabArcdogAdjustableLegBodyflatEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: ArcdogAdjustableLegRewardsCfg = ArcdogAdjustableLegRewardsCfg()


    base_link_name = "base"
    trunk_link_name = "trunk"
    hip_link_name = ".*_thigh"
    knee_link_name = ".*_calf"
    abad_link_name = ".*_hip"
    foot_link_name = ".*_foot"
    extension_link_name = ".*_box"

    # fmt: off
    joint_names = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint",
        "RR_hip_joint", "FL_thigh_joint", "FR_thigh_joint",
        "RL_thigh_joint", "RR_thigh_joint", "FL_calf_joint",
        "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
        "FL_box_joint", "FR_box_joint", "RL_box_joint", "RR_box_joint",
    ]
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        # switch robot to unitree a1
        self.scene.robot = ARCLAB_ARCDOG_ADJUSTABLE_LEG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.terrain.terrain_generator=ROUGH_TERRAINS_CFG
        # ------------------------------Observations------------------------------
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = (
            self.joint_names
        )
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = (
            self.joint_names
        )

        # ------------------------------Actions------------------------------
        # # 强制将 Action 的零位对齐到 0.075，这样网络输出 0 时，腿保持在 0.075
        # self.scene.robot.default_joint_angles = {
        #     "FL_hip_joint": 0.1, "FR_hip_joint": -0.1, 
        #     "RL_hip_joint": 0.1, "RR_hip_joint": -0.1,
        #     "FL_thigh_joint": 0.6, "FR_thigh_joint": 0.6, 
        #     "RL_thigh_joint": 0.6, "RR_thigh_joint": 0.6,
        #     "FL_calf_joint": -0.95, "FR_calf_joint": -0.95, 
        #     "RL_calf_joint": -0.95, "RR_calf_joint": -0.95,
        #     # 关键：这里必须与 init_state 一致
        #     "FL_box_joint": 0.1, "FR_box_joint": 0.1, 
        #     "RL_box_joint": 0.1, "RR_box_joint": 0.1,
        # }
        
        # reduce action scale
        # self.actions.joint_pos.scale = 0.1
        self.actions.joint_pos.scale = {
            ".*_box_joint": 0.02, 
            ".*_(hip_joint|thigh_joint|calf_joint)$": 0.1,
        }
        self.actions.joint_pos.clip = {".*": (-60.0, 60.0)}
        self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------Events------------------------------
        self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = [
            self.base_link_name
        ]
        self.events.randomize_apply_external_force_torque.params[
            "asset_cfg"
        ].body_names = [self.base_link_name]
        self.events.randomize_actuator_gains.params["asset_cfg"].joint_names = [".*"]
        self.events.randomize_joint_friction.params["asset_cfg"].joint_names = [".*"]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [
            self.base_link_name
        ]
        # self.events.randomize_rigid_body_material.params["asset_cfg"].body_names = [
        #     self.foot_link_name
        # ]
        self.events.randomize_screw_joints.params["asset_cfg"].joint_names = [".*_box_joint"]

        # ------------------------------Rewards------------------------------
        # Reward params. Final weights are set in reward_weights below.
        self.rewards.base_height_l2.params["target_height"] = 0.44
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [
            self.base_link_name
        ]
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [
            self.base_link_name
        ]

        # Joint penalty params.
        self.rewards.joint_pos_limits.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=".*_(hip|thigh|calf)_joint"
        )

        # Contact sensor
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            "base", ".*_thigh", ".*calf"
        ]

        # Foot and gait reward params.
        self.rewards.feet_air_time.params["threshold"] = 0.25
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [
            self.foot_link_name
        ]
        self.rewards.feet_contact.params["expect_contact_num"] = 2 # 确保期望值为 2
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [
            self.foot_link_name
        ]
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_exp.params["target_height"] = 0.08
        self.rewards.feet_height_exp.params["asset_cfg"].body_names = [
            self.foot_link_name
        ]
        self.rewards.feet_gait.params["velocity_threshold"] = 0.1
        self.rewards.feet_gait.params["joint_group_pairs"] = (
            (
                ("FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"),
                ("RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"),
            ),
            (
                ("FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"),
                ("RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"),
            ),
        )

        # ------------------------------Final active reward weights------------------------------
        # Put final weights here as the single source of truth for this task.
        # Order: MGDP rewards, extra Arcdog rewards, then disabled rewards.
        reward_weights = {
            # MGDP stage1 rewards.
            "track_lin_vel_xy_exp": 1.0,        # tracking_lin_vel
            "track_ang_vel_z_exp": 0.5,         # tracking_ang_vel
            "lin_vel_z_l2": -1.0,               # lin_vel_z
            "ang_vel_xy_l2": -0.05,             # ang_vel_xy
            "flat_orientation_l2": -0.2,        # orientation
            "stand_still_without_cmd": -0.1,    # stand_still
            "joint_torques_l2": -1.0e-5,        # torques
            "joint_acc_l2": -2.5e-7,            # dof_acc
            "action_rate_l2": -0.01,            # action_rate
            "undesired_contacts": -1.0,         # collision
            "feet_gait": -0.1,                  # motion_trot
            "feet_air_time": 1.0,
            "feet_stumble": -1.0,

            # Extra Arcdog adjustable-leg/bodyflat rewards.
            "box_joint_vel_penalty": -0.01,
            "box_joint_acc_penalty": -1.0e-5,
            "box_joint_pos_limits": -1.0,
            "box_joint_pos_penalty": 0,
            "joint_pos_limits": -0.05,
            "body_lin_acc_l2": -0.0,

            # Explicitly disabled for this task.
            "is_terminated": 0.0,
            "base_height_l2": 0.0,
            "stand_still_flat": 0.0,
            "joint_vel_l2": 0.0,
            "feet_height_exp": 0.0,
            "feet_contact": 0.0,
            "feet_slide": 0.0,
            "joint_vel_limits": 0.0,
            "box_joint_action_rate": 0.0,
            "joint_power": 0.0,
            "rotate_joint_pos_penalty": 0.0,
        }
        for reward_name, weight in reward_weights.items():
            getattr(self.rewards, reward_name).weight = weight

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "ArclabArcdogAdjustableLegBodyflatEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
            self.base_link_name,
        ]
        # self.terminations.illegal_contact = None
        # ------------------------------Curriculums------------------------------
        self.curriculum.command_levels.params["range_multiplier"] = (0.1,1.0)


        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.resampling_time_range = (5.0, 10.0)
