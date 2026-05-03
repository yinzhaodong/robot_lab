# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from typing import Optional
import robot_lab.tasks.locomotion.velocity.mdp as mdp
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward
    # # 计算竖直方向分量
    # uprightness = torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], -1.0, 1.0)
    # # 15°阈值，对应 cos(15°)≈0.966
    # threshold = torch.cos(torch.deg2rad(torch.tensor(15.0, device=uprightness.device)))
    # # 当姿态比15°更正时，直接取满额；超过15°才进入衰减
    # scale = torch.clamp((uprightness - threshold) / (1 - threshold), 0.0, 1.0)

    # reward *= scale
# def track_lin_vel_xy_exp(
#     env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
#     # extract the used quantities (to enable type-hinting)
#     asset: RigidObject = env.scene[asset_cfg.name]
#     # compute the error
#     lin_vel_error = torch.sum(
#         torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
#         dim=1,
#     )
#     reward = torch.exp(-lin_vel_error / std**2)
#     reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
#     return reward


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def track_lin_vel_x_world_exp(
    env: ManagerBasedRLEnv,
    command_name: str,  # 通常是 "base_velocity"
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]

    # 1) 取命令的 (vx^b, vy^b)，扩到 3D 向量，z=0
    cmd_b = env.command_manager.get_command(command_name)[:, :2]           # [N,2]
    cmd_b3 = torch.cat([cmd_b, torch.zeros_like(cmd_b[:, :1])], dim=1)     # [N,3]

    # 2) 仅用 yaw 把命令旋到世界系（忽略 pitch/roll）
    #    quat_apply_yaw 文档：只绕航向旋转向量
    quat_w = asset.data.root_link_quat_w                                   # [N,4], wxyz
    cmd_w3 = math_utils.quat_apply_yaw(quat_w, cmd_b3)                     # [N,3]
    v_cmd_x_w = cmd_w3[:, 0]                                               # 目标世界 x 速度

    # 3) 实际世界 x 速度（用 root_com_lin_vel_w）
    v_x_w = asset.data.root_com_lin_vel_w[:, 0]

    # 4) 指数核
    err = (v_cmd_x_w - v_x_w).pow(2)
    rew = torch.exp(-err / (std ** 2))

    # 5) 不要再用 “-projected_gravity_b[:,2]” 去关停奖励（会在站立时 → 0）
    #    如需门控，可改用与“竖直更友好”的 gate（见上文）
    return rew

def track_ang_vel_z_base_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint_power"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward


def stand_still_without_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    # Penalize motion when command is nearly zero.
    reward = mdp.joint_deviation_l1(env, asset_cfg)
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def wheels_stop_without_cmd(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    当没有速度命令时，惩罚轮子转动（基于关节速度）。
    """
    # 提取机器人 articulation
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取这些关节的速度
    wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]  # [num_envs, num_wheel_joints]

    # 判断命令是否为 "静止" （这里只看 base 线速度/角速度是否接近 0）
    command = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < 0.1

    # 计算惩罚：轮子速度越大，惩罚越大
    penalty = torch.sum(torch.abs(wheel_vel), dim=1)

    return penalty * command

def wheel_action_l2(env: ManagerBasedRLEnv, wheel_ids: list[int]) -> torch.Tensor:
    # env.action_manager.action: (num_envs, action_dim)
    actions = env.action_manager.action[:, wheel_ids]
    return torch.sum(actions**2, dim=1)


def wheel_slip_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    wheel_radius: float = 0.1,
    epsilon: float = 0.10,
    vel_body_frame: bool = True,
    # ---- 可选门控：与你的 wheels_stop_without_cmd 对齐 ----
    command_name: Optional[str] = None,   # 传入则可按“无命令”门控
    no_cmd_lin_thresh: float = 0.05,      # m/s
    no_cmd_ang_thresh: float = 0.05,      # rad/s
    gate_on_no_command: bool = False,     # True: 仅在“无命令”时启用惩罚
    # ---- 可选门控：接触（需要你在 scene 里有 contact_forces） ----
    contact_sensor_cfg: Optional[SceneEntityCfg] = None,
    contact_threshold: float = 1.0,
    gate_on_contact: bool = False         # True: 仅在“接地”时启用惩罚
) -> torch.Tensor:
    """
    轮滑率（longitudinal slip ratio）的 L1-mean 惩罚。
    s_i = (ω_i * R - v_x) / (|v_x| + eps)
    - 理想纯滚: v_x = ω * R -> s -> 0
    - 返回: 每个 env 的标量惩罚 [N]
    """
    # 取机器人与轮关节角速度 ω: [N, n_wheels]
    asset: Articulation = env.scene[asset_cfg.name]
    omega = asset.data.joint_vel[:, asset_cfg.joint_ids]  # rad/s, shape [N, n_wheels]

    # 取得机体前向线速度 v_x: 优先机体系，其次世界系
    if vel_body_frame and hasattr(asset.data, "root_lin_vel_b"):
        v = asset.data.root_lin_vel_b  # [N, 3]
        v_x = v[:, 0]
    elif hasattr(asset.data, "root_lin_vel_w"):
        # 若只有世界系速度，这里保守取 world x；更严谨可将速度投影到机体前向
        v_x = asset.data.root_lin_vel_w[:, 0]
    else:
        # 兜底：若无速度可用，则不产生惩罚
        return torch.zeros(omega.shape[0], device=omega.device, dtype=omega.dtype)

    # 纵向轮滑率 s: [N, n_wheels]
    slip = (omega * wheel_radius - v_x.unsqueeze(-1)) / (torch.abs(v_x).unsqueeze(-1) + epsilon)

    # L1-mean（对大 slip 更敏感，同时与轮数无关）
    penalty = torch.mean(torch.abs(slip), dim=1)  # [N]

    # ----- 可选门控 1：仅在“无命令”时启用 -----
    if gate_on_no_command and (command_name is not None):
        cmd = env.command_manager.get_command(command_name)  # [N, k]
        lin_cmd = cmd[:, :2]
        ang_cmd = cmd[:, 2] if cmd.shape[1] >= 3 else torch.zeros_like(lin_cmd[:, 0])
        no_lin = torch.norm(lin_cmd, dim=1) < no_cmd_lin_thresh
        no_ang = torch.abs(ang_cmd) < no_cmd_ang_thresh
        no_cmd_mask = (no_lin & no_ang).to(penalty.dtype)
        penalty = penalty * no_cmd_mask

    # ----- 可选门控 2：仅在接地时启用 -----
    if gate_on_contact and (contact_sensor_cfg is not None) and (contact_sensor_cfg.name in env.scene):
        cf = env.scene[contact_sensor_cfg.name].data.net_forces_w  # [N, B, 3] or [N,B,6]
        if cf.ndim == 3:
            contact_mask = (cf[..., 2].abs().max(dim=1).values > contact_threshold)
        else:
            contact_mask = (cf.abs().max(dim=1).values > contact_threshold)
        penalty = penalty * contact_mask.to(penalty.dtype)

    return penalty


def joint_position_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_com_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    return torch.where(
        torch.logical_or(cmd > 0.1, body_vel > velocity_threshold), reward, stand_still_scale * reward
    )  # * torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 1)
    # reward = torch.square(
    #     asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    # )
    # return torch.sum(reward, dim=1)

def wheel_vel_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    velocity_threshold: float,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    running_reward = torch.sum(in_air * joint_vel, dim=1)
    standing_reward = torch.sum(joint_vel, dim=1)
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        standing_reward,
    )
    return reward

class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        max_err: float,
        velocity_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        return torch.where(
            torch.logical_or(cmd > 0.1, body_vel > self.velocity_threshold), sync_reward * async_reward, 0.0
        )

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)

def motion_trot_joint_symmetry(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    joint_group_pairs: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...],
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
    velocity_threshold: float = 0.1,
    stop_after_steps: int | None = None,
) -> torch.Tensor:
    """MGDP-style trot motion penalty based on diagonal joint-pose symmetry.

    MGDP's ``motion_trot`` penalizes the absolute joint-position difference
    between diagonal legs. This is different from :class:`GaitReward`, which uses
    contact/air-time synchronization. Use a negative reward weight for this term.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "_motion_trot_joint_pair_ids"):
        env._motion_trot_joint_pair_ids = []
        for left_group, right_group in joint_group_pairs:
            left_ids = [asset.find_joints(joint_name)[0][0] for joint_name in left_group]
            right_ids = [asset.find_joints(joint_name)[0][0] for joint_name in right_group]
            env._motion_trot_joint_pair_ids.append((left_ids, right_ids))

    reward = torch.zeros(env.num_envs, device=env.device)
    for left_ids, right_ids in env._motion_trot_joint_pair_ids:
        left_pos = asset.data.joint_pos[:, left_ids]
        right_pos = asset.data.joint_pos[:, right_ids]
        reward += torch.sum(torch.abs(left_pos - right_pos), dim=1)

    cmd = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    active = torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold)
    reward = torch.where(active, reward, torch.zeros_like(reward))

    if stop_after_steps is not None and getattr(env, "common_step_counter", 0) > stop_after_steps:
        reward *= 0.0
    return reward

def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= (torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7) * torch.where(torch.abs(torch.atan2(env.scene["robot"].data.projected_gravity_b[:, 0], -env.scene["robot"].data.projected_gravity_b[:, 2])) > 0.3490658503988659, 0.1*torch.ones_like(reward), torch.ones_like(reward))
    return reward

def wheel_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "wheel_mirror_joints_cache") or env.wheel_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.wheel_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]

    # ---- 新增：本地超参（不改函数签名）----
    tau = 0.3           # 小差距阈值（rad/s）
    small_scale = 0.25  # 小差距区惩罚缩放
    exp_cap = 60.0      # 指数上限防爆
    eps = 1e-12
    beta = 2.0 * small_scale / (tau + eps)  # 保证在 d=tau 处函数值与一阶导连续
    # -----------------------------------

    # ... 前面保持不变（含 tau/d_max/scale 等） ...
    reward = torch.zeros(env.num_envs, device=env.device)

    per_pair_terms = []  # 收集每一对镜像轮的惩罚（负值）

    for joint_pair in env.wheel_mirror_joints_cache:
        left  = asset.data.joint_vel[:, joint_pair[0][0]]
        right = asset.data.joint_vel[:, joint_pair[1][0]]

        d = torch.abs(left - right)        # [N] or [N,K]
        d_norm = d / 30.0                  # 30 = 最大速度差
        scale = 50.0                       # d=3 → -0.5（单对）
        diff = -(scale * (d_norm ** 2))    # 负值=惩罚

        if diff.ndim > 1:
            diff = diff.sum(dim=-1)        # [N]
        per_pair_terms.append(diff)        # 记录每一对

    if len(per_pair_terms) > 0:
        terms = torch.stack(per_pair_terms, dim=-1)  # [N, P]

        # ===== 选择一种聚合方式（任选其一）=====

        # 1) 求和（不平均）：多个异常叠加更痛
        reward = terms.sum(dim=-1)

        # 2) 取“最差一对”（最负的那一列）：任意一对异常就很痛
        # reward, _ = terms.min(dim=-1)

        # 3) Top-k 平均（例如最差的2对）
        # k = min(2, terms.shape[-1])
        # reward = terms.topk(k, dim=-1, largest=False).values.mean(dim=-1)

        # 4) 平滑最小（smooth-min），兼顾可导与“抓最差”
        # tau_aggr = 0.5
        # reward = -tau_aggr * torch.logsumexp(-terms / tau_aggr, dim=-1)
    else:
        reward = torch.zeros(env.num_envs, device=env.device)

    # 姿态缩放保持不变
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward






def action_mirror(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    mirror_joints: list[list[str]],
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "action_mirror_joints_cache") or env.action_mirror_joints_cache is None:
        env.action_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair]
            for joint_pair in mirror_joints
        ]

    penalty = torch.zeros(env.num_envs, device=env.device)
    for joint_pair in env.action_mirror_joints_cache:
        diff = torch.abs(env.action_manager.action[:, joint_pair[0][0]]) - \
               torch.abs(env.action_manager.action[:, joint_pair[1][0]])
        penalty += torch.mean(torch.square(diff), dim=-1)

    penalty = penalty / (len(mirror_joints) + 1e-6)
    reward = torch.exp(-5.0 * penalty)

    upright_weight = torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
    reward *= upright_weight
    return reward


def action_sync(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_groups: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Cache joint indices if not already done
    if not hasattr(env, "action_sync_joint_cache") or env.action_sync_joint_cache is None:
        env.action_sync_joint_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_group] for joint_group in joint_groups
        ]

    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over each joint group
    for joint_group in env.action_sync_joint_cache:
        if len(joint_group) < 2:
            continue  # need at least 2 joints to compare

        # Get absolute actions for all joints in this group
        actions = torch.stack(
            [torch.abs(env.action_manager.action[:, joint[0]]) for joint in joint_group], dim=1
        )  # shape: (num_envs, num_joints_in_group)

        # Calculate mean action for each environment
        mean_actions = torch.mean(actions, dim=1, keepdim=True)

        # Calculate variance from mean for each joint
        variance = torch.mean(torch.square(actions - mean_actions), dim=1)

        # Add to reward (we want to minimize this variance)
        reward += variance.squeeze()
    reward *= 1 / len(joint_groups) if len(joint_groups) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float, command_threshold: float = 0.25 # 新增：指令速度阈值
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > command_threshold
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact(
    env: ManagerBasedRLEnv, command_name: str, expect_contact_num: int, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_continue_contact(env, command_name, expect_contact_num, sensor_cfg) -> torch.Tensor:
    s = env.scene.sensors[sensor_cfg.name]
    forces = s.data.net_forces_w  # [N, num_bodies, 3]
    # 每脚是否接触（法向力阈值可按需要调）
    contact = (forces[:, sensor_cfg.body_ids, 2].abs() > 20).float()  # [N, num_feet]

    # —— 惰性初始化 & 指数滑动平均占空比（强调“持续贴地”）——
    if not hasattr(env, "contact_ema"):
        env.contact_ema = torch.zeros_like(contact)
    alpha = 0.1  # 越小越强调“持续”；0.05~0.2 常用
    env.contact_ema = (1.0 - alpha) * env.contact_ema + alpha * contact  # [N, num_feet]

    # —— 占空比阈值：每脚是否达到“贴地占空比”要求 —— 
    target_duty = 0.85  # 平地建议 0.75~0.85；爬箱子可放宽到 ~0.65
    slack = 0.05
    duty_ok = (env.contact_ema >= (target_duty - slack)).float()        # [N, num_feet]
    reward = duty_ok.mean(dim=1)                                        # [N], 0~1

    # —— 速度权重（替代硬门控）：慢速也能有奖励，但快一点更赚 —— 
    cmd = env.command_manager.get_command(command_name)                 # [N, D]
    v_ref = 1.0  # 参考最大期望线速度，按你的命令分布调整
    w = (cmd[:, 0:2].norm(dim=1) / (v_ref + 1e-6)).clamp(0.2, 1.0)     # 避免静止时全没奖励
    return reward * w


# def feet_continue_contact(env, command_name, expect_contact_num, sensor_cfg) -> torch.Tensor:
#     s = env.scene.sensors[sensor_cfg.name]
#     forces = s.data.net_forces_w  # [N, num_bodies, 3]
#     # 每脚是否接触（法向力阈值可按需要调）
#     contact = (forces[:, sensor_cfg.body_ids, 2].abs() > 9.8).float()  # [N, num_feet]

#     # —— 惰性初始化 & 指数滑动平均占空比（强调“持续贴地”）——
#     if not hasattr(env, "contact_ema"):
#         env.contact_ema = torch.zeros_like(contact)
#     alpha = 0.05  # 越小越强调“持续”；0.05~0.2 常用
#     env.contact_ema = (1.0 - alpha) * env.contact_ema + alpha * contact  # [N, num_feet]

#     # —— 占空比阈值：每脚是否达到“贴地占空比”要求 —— 
#     target_duty = 0.65  # 平地建议 0.75~0.85；爬箱子可放宽到 ~0.65
#     slack = 0.05
#     duty_ok = (env.contact_ema >= (target_duty - slack)).float()        # [N, num_feet]
#     reward = duty_ok.mean(dim=1)                                        # [N], 0~1

#     # —— 速度权重（替代硬门控）：慢速也能有奖励，但快一点更赚 —— 
#     cmd = env.command_manager.get_command(command_name)                 # [N, D]
#     v_ref = 0.6  # 参考最大期望线速度，按你的命令分布调整
#     w = (cmd[:, 0].abs() / (v_ref + 1e-6)).clamp(0.2, 1.0)     # 避免静止时全没奖励
#     return reward * w


def feet_contact_without_cmd(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    reward = torch.sum(contact, dim=-1).float()
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    # Penalize feet hitting vertical surfaces
    return torch.any(contact & (forces_xy > forces_z), dim=1)


def feet_distance_y_exp(
    env: ManagerBasedRLEnv, stance_width: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)
    n_feet = len(asset_cfg.body_ids)
    footsteps_in_body_frame = torch.zeros(env.num_envs, n_feet, 3, device=env.device)
    for i in range(n_feet):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )
    side_sign = torch.tensor(
        [1.0 if i % 2 == 0 else -1.0 for i in range(n_feet)],
        device=env.device,
    )
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    desired_ys = stance_width_tensor / 2 * side_sign.unsqueeze(0)
    stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / (std**2))
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_xy_exp(
    env: ManagerBasedRLEnv,
    stance_width: float,
    stance_length: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute the current footstep positions relative to the root
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)

    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )

    # Desired x and y positions for each foot
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    stance_length_tensor = stance_length * torch.ones([env.num_envs, 1], device=env.device)

    desired_xs = torch.cat(
        [stance_length_tensor / 2, stance_length_tensor / 2, -stance_length_tensor / 2, -stance_length_tensor / 2],
        dim=1,
    )
    desired_ys = torch.cat(
        [stance_width_tensor / 2, -stance_width_tensor / 2, stance_width_tensor / 2, -stance_width_tensor / 2], dim=1
    )

    # Compute differences in x and y
    stance_diff_x = torch.square(desired_xs - footsteps_in_body_frame[:, :, 0])
    stance_diff_y = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])

    # Combine x and y differences and compute the exponential penalty
    stance_diff = stance_diff_x + stance_diff_y
    return torch.exp(-torch.sum(stance_diff, dim=1) / std)


def feet_height_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return torch.exp(-reward / std)


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    # feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # reward = torch.sum(feet_vel.norm(dim=-1) * contacts, dim=1)

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

# def smoothness_1(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - env.action_manager.prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     return torch.sum(diff, dim=1)


# def smoothness_2(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - 2 * env.action_manager.prev_action + env.action_manager.prev_prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     diff = diff * (env.action_manager.prev_prev_action[:, :] != 0)  # ignore second step
#     return torch.sum(diff, dim=1)


def wheel_spin_in_air_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    reward = torch.sum(in_air * joint_vel, dim=1)
    return reward


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def track_lin_vel_world_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_com_lin_vel_w[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_world_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - asset.data.root_com_ang_vel_w[:, 2]
    )
    return torch.exp(-ang_vel_error / std**2)


def feet_height_body_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def joint_pos_penalty_height_gated(
    env,
    command_name: str,          # 不再使用速度门槛，但保留签名兼容 RewardTermCfg
    asset_cfg,                  # SceneEntityCfg("robot", joint_names=腿部关节)
    sensor_cfg,                 # SceneEntityCfg("front_height")
    stand_still_scale: float = 5.0,
    # velocity_threshold / command_threshold 不再使用，但保留参数以兼容旧配置
    velocity_threshold: float = 0.5,
    command_threshold: float = 0.1,
    h_free_min: float = 0.10,
    h_free_max: float = 0.45,
    offset: float = 0.5,
    alpha: float = 0.2,
    # 可选：倾斜缩放的下限，避免被乘成 0 完全没梯度
    tilt_floor: float = 0.1,    # ∈[0,1]；0.1 表示至少保留 10%
):
    """
    连续高度扫描做门控：|前方高度变化| 越大，越“放开”；|变化|小则强制动。
    与机器人速度无关；全时生效。倾斜越大，惩罚越小（鼓励在倾斜时调整姿态）。
    """
    import torch
    from isaaclab.envs.mdp import observations as mdp  # 这里用 mdp.height_scan
    import math
    # 1) 基础量
    asset = env.scene[asset_cfg.name]
    pos_err = torch.linalg.norm(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids], dim=1
    )

    # 2) 连续高度扫描（不离散）
    H = mdp.height_scan(env, sensor_cfg=sensor_cfg, offset=offset)  # [N, K] or [N, 1]
    # height_scan 返回(传感器高度 - 命中点z - offset)；常见设置下上台阶为负、坑为正
    terrain_delta = -H

    # 3) 取代表性幅值（可换成 quantile 更稳）
    if terrain_delta.ndim == 2 and terrain_delta.shape[1] > 1:
        mag = terrain_delta.abs().max(dim=1).values
        # 也可用更鲁棒的分位数：mag = torch.quantile(terrain_delta.abs(), q=0.8, dim=1)
    else:
        mag = terrain_delta.abs().squeeze(-1)

    # 4) 连续门控：|变化| ≤ h_free_min 强制动；≥ h_free_max 基本放开
    gate = torch.clamp((mag - h_free_min) / max(1e-6, (h_free_max - h_free_min)), 0.0, 1.0)  # [0,1]

    # 5) EMA 平滑，避免门控抖动
    if not hasattr(env, "pose_gate_ema"):
        env.pose_gate_ema = gate
    env.pose_gate_ema = (1.0 - alpha) * env.pose_gate_ema + alpha * gate
    g = env.pose_gate_ema

    # 6) 制动倍数：g=0 → stand_still_scale；g=1 → 1
    s = stand_still_scale - (stand_still_scale - 1.0) * g   # 与速度无关

    # 7) 倾斜缩放（直立时因子≈1，越倾斜越小；带 floor 避免缩放为 0）
    # proj_gz = env.scene["robot"].data.projected_gravity_b[:, 2]  # 直立≈-1
    # grav = torch.clamp(-proj_gz, 0.0, 0.7) / 0.7                # [0,1]
    # grav = tilt_floor + (1.0 - tilt_floor) * grav               # [tilt_floor, 1]
    g_b = asset.data.projected_gravity_b  # [N,3], 直立≈[0,0,-1]
    # 用重力在机体系的 x、z 分量估计俯仰角：pitch = atan2(|gx|, -gz)
    pitch = torch.atan2(g_b[:, 0].abs(), (-g_b[:, 2]).clamp_min(1e-6))  # [rad]

    lo = math.radians(10.0)
    hi = math.radians(30.0)
    t = ((pitch - lo) / (hi - lo)).clamp(0.0, 1.0)          # [0,1]
    grav = 1.0 - t * (1.0 - tilt_floor)                     # 1 → tilt_floor
    scale = s * grav
    return pos_err * scale   # 外面配 weight 为负，使其成为惩罚项



# 安全的“高台/坑”双向门控版：平地强压制，遇高台或坑逐步放开并适度鼓励俯仰
def flat_orientation_height_gated(
    env,
    sensor_cfg=None,              # SceneEntityCfg("height_scanner")
    h_low: float = 0.10,          # 低于此(米/分位)→视作低风险，强压制倾斜
    h_high: float = 0.25,         # 高于此开始完全放开(线性-光滑过渡)
    encourage_scale: float = 0.5, # 鼓励俯仰强度
    use_disc: bool = True,        # 复用你的 height_scan_disc
    offset: float = 0.5,          # 与你的扫描一致
    alpha: float = 0.2,           # 门控EMA，抑制抖动(0.1~0.3)
    tilt_cap_rad: float = 0.35,   # 最多鼓励到 ~20° 的俯仰，防止过大倾斜
):
    import torch, math
    robot = env.scene["robot"]

    # 1) 倾斜度：XY分量的模（≈ sin(倾角)）；以及原始“平姿态”惩罚
    g = robot.data.projected_gravity_b  # [N,3]
    tilt_xy = torch.sqrt(torch.clamp(g[:, 0]**2 + g[:, 1]**2, min=1e-9))     # [N]
    upright_pen = tilt_xy**2                                                 # 与 isaaclab flat_orientation_l2 对齐

    # 2) 读取前向“高度/坑深”并做成对称的“危险度”hazard ∈ [0,1]
    hazard = None
    if isinstance(getattr(sensor_cfg, "name", None), str):
        try:
            _ = env.scene[sensor_cfg.name]
            if use_disc:
                # 约定：bins ∈ [-1,1]；绝对值越大说明越“极端”(高台 or 坑更明显/更近)
                bins = mdp.height_scan_disc(env, sensor_cfg=sensor_cfg, offset=offset).squeeze(-1)  # [N]
                hazard = bins.abs().clamp(0.0, 1.0)   # 同时覆盖“台阶高”和“坑深”
            else:
                # 连续高度：正=台阶高、负=坑深（若你语义相反，可取负号）
                h = mdp.height_scan(env, sensor_cfg=sensor_cfg, offset=offset).squeeze(-1)          # [N]
                # 统一成“危险度”：取绝对值，并按阈值映射
                hazard = h.abs()
        except KeyError:
            pass

    if hazard is None:
        gate = torch.zeros_like(upright_pen)  # 没传感器→当平地处理
    else:
        # 3) hazard→[0,1] 门控；支持“米值”或“分位值”，并用 smoothstep 让过渡更平滑
        gate_lin = torch.clamp((hazard - h_low) / max(1e-6, (h_high - h_low)), 0.0, 1.0)
        gate = gate_lin * gate_lin * (3.0 - 2.0 * gate_lin)  # smoothstep

    # 4) EMA平滑，避免相机/射线抖动引起奖励震荡
    if not hasattr(env, "_tilt_gate_ema"):
        env._tilt_gate_ema = gate
    else:
        env._tilt_gate_ema = (1.0 - alpha) * env._tilt_gate_ema + alpha * gate
    gate_smooth = env._tilt_gate_ema

    # 5) 只鼓励到一个安全上限，防止把机器人“教”到大仰角翻车
    tilt_cap = math.sin(tilt_cap_rad)
    encouraged_tilt = torch.clamp(tilt_xy, max=tilt_cap)

    # 6) 组合：平地(门控小)→强压制；遇高台/坑(门控大)→减惩罚并适度鼓励俯仰
    #    返回“代价”型项：权重大于0时，就是惩罚 − 奖励 的形式
    return (1.0 - gate_smooth) * upright_pen - encourage_scale * gate_smooth * encouraged_tilt


def upright_gate(env, asset_cfg):
    asset = env.scene[asset_cfg.name]
    # 直立度：-gz ∈ [0,1]；直立≈1，倾倒≈0
    upright = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 1.0)
    return upright

# 利用高度扫描做“障碍门控”：有台阶/平台时缩小这项的权重
def obstacle_gate(env, sensor_cfg, h_low=0.10, h_high=0.25, offset=0.5):
    # height_scan: 传感器高度 - 击中点z - offset，越大越“有台阶”
    h = mdp.height_scan(env, sensor_cfg=sensor_cfg, offset=offset)  # [N, M] 或 [N,1]
    h = h.abs().max(dim=1).values  # 取最大幅度
    gate = torch.clamp((h - h_low) / (h_high - h_low + 1e-6), 0.0, 1.0)
    return gate

def upward_for_climb(env, asset_cfg=SceneEntityCfg("robot"),
                     sensor_cfg=None, k_progress=0.5, relax_scale=0.3):
    asset = env.scene[asset_cfg.name]

    # 1) 门控：有明显台阶/平台时 gate→1，否则→0
    if sensor_cfg is not None:
        gate = obstacle_gate(env, sensor_cfg)  # 见上面实现
    else:
        gate = torch.zeros(asset.data.root_pos_w.shape[0], device=asset.device)

    # 2) 直立度惩罚（平地强，遇障放松）
    upright = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 1.0)  # 直立≈1
    pen_tilt = (1.0 - upright)**2
    pen_tilt *= (1.0 - gate + gate * relax_scale)

    # 3) 向上“进度”奖励（PBRS），只在 gate>0 时起作用
    z = asset.data.root_pos_w[:, 2]
    if not hasattr(env, "_last_z"): env._last_z = z.clone()
    dz = torch.clamp(z - env._last_z, min=0.0)
    env._last_z = z
    r_progress = k_progress * dz * gate

    # 4) 汇总（注意：惩罚在总奖励里给负权重）
    return r_progress - pen_tilt


def climb_progress_dyn_pbrs(env, asset_cfg=SceneEntityCfg("robot"),
                            sensor_cfg=None, k: float = 2.0, gamma_shape: float = 0.99,
                            h_low: float = 0.10, h_high: float = 0.25, offset: float = 0.5):
    asset = env.scene[asset_cfg.name]
    z = asset.data.root_pos_w[:, 2]

    # 门控：这里以“增强门控”为例（有障碍→更强调爬高），若要抑制，把  alpha 换成 (1-alpha)
    if sensor_cfg is not None:
        h = mdp.height_scan(env, sensor_cfg=sensor_cfg, offset=offset)
        h = h.abs().max(dim=1).values
        alpha = torch.clamp((h - h_low) / (h_high - h_low + 1e-6), 0.0, 1.0)
        g = alpha  # 有障碍→g↑
    else:
        g = torch.ones_like(z)

    phi = k * g * z
    if not hasattr(env, "_phi_prev"):
        env._phi_prev = phi.clone()
    if hasattr(env, "reset_buf"):
        env._phi_prev = torch.where(env.reset_buf.bool(), phi, env._phi_prev)
    rew = gamma_shape * phi - env._phi_prev
    env._phi_prev = phi
    return rew

def stand_still_flat_orientation_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Provides an extra reward for keeping the body flat specifically when the robot 
    is commanded to stand still. This helps stabilization on slopes.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # 1. Check if the command is "stand still"
    cmd_norm = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    is_static_cmd = cmd_norm < command_threshold
    
    # 2. Calculate flatness reward (same as above)
    gravity_b = asset.data.projected_gravity_b
    flat_error = torch.sum(torch.square(gravity_b[:, :2]), dim=1)
    flat_reward = torch.exp(-flat_error / std**2)
    
    # 3. Apply reward only when static
    reward = flat_reward * is_static_cmd.float()
    
    # Survival gating
    reward *= torch.clamp(-gravity_b[:, 2], 0.0, 0.7) / 0.7
    
    return reward

def blind_climbing_vel_z_bonus(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    pitch_threshold: float = 0.05, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    盲走爬台阶奖励：当机器人被指令向前移动，且车头抬起时，奖励其向上的 Z 轴速度。
    """
    # 获取机器人实体数据
    robot = env.scene[asset_cfg.name]

    # 1. 获取向前的速度指令 (X轴)
    commands = env.command_manager.get_command(command_name)
    cmd_vel_x = commands[:, 0]

    # 2. 估算机身仰角 (Pitch)
    # projected_gravity_b 是世界坐标系的重力向量 [0, 0, -1] 在机身局部坐标系下的投影。
    # 当车头抬起 (Nose up) 时，重力在机身局部坐标系下会指向斜后方，即局部 X 轴分量为负。
    # 因此，-projected_gravity_b[:, 0] 是一个正值，近似代表仰角的正弦值 (sin(pitch))。
    projected_gravity = robot.data.projected_gravity_b
    pitch_approx = -projected_gravity[:, 0]

    # 3. 获取世界坐标系下的实际 Z 轴线速度
    vel_z = robot.data.root_lin_vel_w[:, 2]

    # 4. 判断是否处于“爬台阶”状态：
    # 条件 A: 接收到足够大的向前指令 (例如 > 0.2 m/s)
    # 条件 B: 车头抬起超过一定阈值 (pitch_threshold 0.05 大约是 3 度)
    is_climbing = torch.logical_and(cmd_vel_x > 0.2, pitch_approx > pitch_threshold)

    # 5. 计算奖励：只奖励向上的速度 (vel_z > 0)，且只有在爬台阶状态下才给奖励
    climbing_vel_z = torch.clamp(vel_z, min=0.0)
    
    # 返回奖励值 (非爬台阶状态下，此项奖励为 0)
    return climbing_vel_z * is_climbing.float()

def climbing_pitch_up_bonus(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    【改进版：扬身奖励】
    鼓励机器人在遇到障碍物受阻时抬起前身，但防止其在平地起步时原地“翘头”作弊。
    """
    robot = env.scene[asset_cfg.name]
    
    # 1. 获取指令和实际速度
    cmd_vel_x = env.command_manager.get_command(command_name)[:, 0]
    actual_vel_x = robot.data.root_lin_vel_b[:, 0]
    actual_vel_z = robot.data.root_lin_vel_b[:, 2]  # 获取 Z 轴（上下）速度
    
    # 2. 计算 Pitch 角的替代指标 (抬头时为正)
    pitch_metric = -robot.data.projected_gravity_b[:, 0]
    
    # 3. 核心逻辑修改：增加多重限制条件
    
    # 条件 A: 强烈的向前意图
    intent_forward = cmd_vel_x > 0.4
    
    # 条件 B: 实际速度受阻，但必须大于一个下限！(防起步作弊核心)
    # actual_vel_x > 0.05: 确保机器人已经“动起来了”，而不是刚出生在原地静止。
    # actual_vel_x < 0.3: 速度明显低于预期，说明被障碍物挡住了。
    is_resisted = (actual_vel_x > 0.05) & (actual_vel_x < 0.3)
    
    # 条件 C: 确保机器人没有在往下掉 (防止下坡或下台阶时误触发抬头)
    not_falling = actual_vel_z > -0.1
    
    # 条件 D: 限制最大奖励值 (防后空翻核心)
    # 如果不限制，机器人会为了追求无限大的奖励而直接向后翻倒。
    # 限制最大值为 0.4 (大约对应 pitch 角 23.5 度，sin(23.5°) ≈ 0.4)
    # 将上限提高到 0.85，允许机器人仰角达到约 60 度时获得最大奖励
    capped_pitch = torch.clamp(pitch_metric, min=0.0, max=0.85)

    # 【新增】防翻车熔断锁：如果仰角超过约 75 度 (sin(75°) ≈ 0.96)，说明快要后空翻了，直接判定为不安全
    is_safe_pitch = pitch_metric < 0.95
    
    # 4. 组合所有条件：必须同时满足才给奖励，且要求已经有轻微的抬头趋势 (>0.05)
    # 必须满足：想往前走 + 速度受阻 + 没在下落 + 仰角大于0.05 + 仰角在安全范围内
    valid_climbing_state = intent_forward & is_resisted & not_falling & (pitch_metric > 0.05) & is_safe_pitch
    
    # 5. 发放奖励
    reward = torch.where(valid_climbing_state, capped_pitch, torch.zeros_like(pitch_metric))
    
    return reward

def front_legs_reach_bonus(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    【前腿搭台奖励】
    当机器人处于抬头状态时，奖励前腿（Z轴）抬高。
    鼓励它把前脚尽可能举高，去够高台的台面。
    """
    robot = env.scene[asset_cfg.name]
    
    # 获取前脚在世界坐标系下的 Z 轴高度
    # 注意：这里依赖于在 params 中传入 asset_cfg=SceneEntityCfg("robot", body_names=["FL_foot", "FR_foot"])
    front_feet_indices = asset_cfg.body_ids
    front_feet_z = robot.data.body_pos_w[:, front_feet_indices, 2] # shape: (num_envs, 2)
    mean_front_z = torch.mean(front_feet_z, dim=1)
    
    # 获取机身高度
    root_z = robot.data.root_pos_w[:, 2]
    
    # 计算前脚相对于机身的高度差 (鼓励前脚比机身抬得更高)
    relative_z = mean_front_z - root_z
    
    # 触发条件：机身必须处于抬头状态 (pitch_metric > 0.05，约 3度以上)
    pitch_metric = -robot.data.projected_gravity_b[:, 0]
    is_pitching = pitch_metric > 0.05 
    
    # 当抬头且前脚抬起时，奖励其相对高度
    reward = torch.where(is_pitching & (relative_z > -0.1), relative_z + 0.1, torch.zeros_like(relative_z))
    return reward

# ==========================================
# 1. 像马一样扬身 (Horse Rearing Posture) - 防破解版
# ==========================================
def horse_rearing_posture_bonus(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    asset_cfg: SceneEntityCfg, 
    front_foot_names: list,
    rear_foot_names: list,
    target_pitch_deg: float = 35.0,  # 目标仰角：35度 (不要让它竖直)
    target_height_diff: float = 0.35 # 目标高度差：0.35米 (根据你的台阶高度调整)
) -> torch.Tensor:
    """
    鼓励机器人扬起前身，但限制最大角度，并强制要求实际向前移动。
    """
    asset = env.scene[asset_cfg.name]
    
    # 1. 获取指令速度 和 真实速度
    velocity_command = env.command_manager.get_command(command_name)
    cmd_x = velocity_command[:, 0]
    # 获取机器人基座在机身坐标系下的真实线速度
    actual_vel_x = asset.data.root_lin_vel_b[:, 0] 
    
    # 2. 计算当前 Pitch 的 sin 值
    projected_gravity = asset.data.projected_gravity_b
    current_pitch_sin = -projected_gravity[:, 0] 
    
    # 将目标角度转换为 sin 值
    target_pitch_rad = math.radians(target_pitch_deg)
    target_pitch_sin = math.sin(target_pitch_rad)
    
    # 3. 计算高度差
    front_foot_ids = asset.find_bodies(front_foot_names)[0]
    rear_foot_ids = asset.find_bodies(rear_foot_names)[0]
    front_feet_z = asset.data.body_pos_w[:, front_foot_ids, 2].mean(dim=1)
    rear_feet_z = asset.data.body_pos_w[:, rear_foot_ids, 2].mean(dim=1)
    height_diff = front_feet_z - rear_feet_z
    
    # ================= 核心修改区 =================
    # 使用高斯核函数 (exp(-x^2))：越接近目标值，奖励越接近 1.0；偏离越远，奖励越趋近于 0
    
    # 仰角奖励：控制在目标角度附近
    pitch_reward = torch.exp(-5.0 * torch.square(current_pitch_sin - target_pitch_sin))
    
    # 高度差奖励：控制在目标高度差附近
    height_reward = torch.exp(-5.0 * torch.square(height_diff - target_height_diff))
    
    # 判定条件：有前进指令 且 真实速度也在前进 (防止原地罚站)
    is_commanding_forward = cmd_x > 0.1
    is_actually_moving = actual_vel_x > 0.15 # 必须有真实的向前速度
    
    # 综合奖励
    bonus = pitch_reward * height_reward * is_commanding_forward * is_actually_moving
    
    return bonus

# ==========================================
# 2. 前腿搭台后锁死 (Front Legs Quiet on Step) - 修改版
# ==========================================
def front_legs_quiet_on_step_penalty(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    front_foot_names: list, 
    front_knee_names: list,
    step_height_threshold: float = 0.30
) -> torch.Tensor:
    """
    当【两只前脚同时】接触到高于地面的平台时，严厉惩罚前腿膝盖（calf）的运动，迫使其“锁死”或保持稳定。
    """
    asset = env.scene[asset_cfg.name]
    
    # 找到前脚和前膝盖的索引
    front_foot_ids = asset.find_bodies(front_foot_names)[0]
    front_knee_ids = asset.find_joints(front_knee_names)[0]
    
    # 获取前脚的高度 (Z坐标)，形状为 (num_envs, 2)
    front_foot_z = asset.data.body_pos_w[:, front_foot_ids, 2]
    
    # 判定条件：两只前脚的高度【同时】大于绝对值 step_height_threshold
    # 使用 .all(dim=1) 确保两个脚都满足条件
    both_feet_on_step = (front_foot_z > step_height_threshold).all(dim=1)
    
    # 获取前膝盖的速度
    front_knee_vel = asset.data.joint_vel[:, front_knee_ids]
    
    # 如果双腿都搭上了高台，惩罚前膝盖的速度平方
    penalty = torch.sum(torch.square(front_knee_vel), dim=1) * both_feet_on_step
    return penalty


# ==========================================
# 3. 后腿发力蹬踏 (Rear Legs Power Drive) - 修改版
# ==========================================
def rear_legs_power_drive_bonus(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    rear_drive_joint_names: list
) -> torch.Tensor:
    """
    当机器人处于扬身状态时，奖励后腿的 thigh 和 calf 关节输出巨大的扭矩/功率，
    鼓励后腿把身体“推”上去。
    """
    asset = env.scene[asset_cfg.name]
    
    # 找到后腿发力关节（thigh 和 calf）的索引
    rear_joint_ids = asset.find_joints(rear_drive_joint_names)[0]
    
    # 获取当前 Pitch 角
    projected_gravity = asset.data.projected_gravity_b
    is_pitching_up = -projected_gravity[:, 0] > 0.1 # 仰角大于约5度
    
    # 获取后腿指定关节的输出扭矩和速度
    rear_torques = asset.data.applied_torque[:, rear_joint_ids]
    rear_vel = asset.data.joint_vel[:, rear_joint_ids]
    
    # 机械功率 = 扭矩 * 速度。我们奖励做正功（发力伸展）
    power = rear_torques * rear_vel
    # 过滤掉负功，只奖励正向发力
    positive_power = torch.clamp(power, min=0.0)
    
    # 只有在抬头爬升时，才奖励后腿发力
    bonus = torch.sum(positive_power, dim=1) * is_pitching_up
    return bonus

# ==========================================
# 4. 仅惩罚 Roll 和 Yaw，放开 Pitch (Roll-Yaw Only Penalty)
# ==========================================
def roll_yaw_orientation_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    替代原来的 flat_orientation_l2。
    允许机器人抬头(Pitch)，但严厉惩罚左右侧翻(Roll)和不必要的偏航(Yaw)。
    """
    asset = env.scene[asset_cfg.name]
    # projected_gravity_b = [sin(pitch), -sin(roll)*cos(pitch), -cos(roll)*cos(pitch)]
    # 我们只惩罚 Y 轴分量 (对应 Roll)
    roll_penalty = torch.square(asset.data.projected_gravity_b[:, 1])
    return roll_penalty


def action_rate_l2_by_name(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    通过 SceneEntityCfg 指定关节名称，只惩罚这些关节的动作变化率。
    """
    # 获取机器人资产
    asset = env.scene[asset_cfg.name]
    
    # 根据传入的正则表达式解析出具体的关节索引
    # 注意：这里获取的是在驱动关节列表中的索引，通常与 action 的索引一一对应
    joint_indices, _ = asset.find_joints(asset_cfg.joint_names)
    
    # 切片提取特定关节的动作
    current_action = env.action_manager.action[:, joint_indices]
    prev_action = env.action_manager.prev_action[:, joint_indices]
    
    return torch.sum(torch.square(current_action - prev_action), dim=1)

def stand_still_joint_vel_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    自定义惩罚：当机器人收到静止指令时，严厉惩罚任何关节的速度（即摇晃和抽搐）。
    不影响移动时的步态，也不限制静止时的具体关节角度。
    """
    # 获取机器人实体
    robot = env.scene[asset_cfg.name]
    
    # 获取速度指令 (通常是 [lin_x, lin_y, ang_z])
    command = env.command_manager.get_command(command_name)
    
    # 计算指令速度的绝对大小 (L2 Norm)
    # 取前三个维度计算模长，代表整体的运动意图
    command_norm = torch.norm(command[:, :3], dim=1)
    
    # 判断是否处于“静止状态” (指令速度小于阈值)
    is_standing_still = command_norm < command_threshold
    
    # 计算所有关节速度的平方和 (dof_vel^2)
    # 速度越大，平方后的惩罚越重
    joint_vel_sq = torch.sum(torch.square(robot.data.joint_vel), dim=1)
    
    # 只有在静止时才输出惩罚值，移动时输出 0
    return is_standing_still.float() * joint_vel_sq

def stand_still_base_ang_vel_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    惩罚静止时的机身角速度。
    允许机器人以任何姿态站立，但严厉惩罚机身的晃动（Roll, Pitch, Yaw 的变化率）。
    这能有效迫使策略学会“柔和纠正”，增加系统的阻尼，防止真机震荡发散。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # 判断是否处于静止指令
    command_norm = torch.norm(command[:, :3], dim=1)
    is_standing_still = command_norm < command_threshold
    
    # 获取机身在世界坐标系下的角速度 (root_ang_vel_w)
    # 也可以使用相对于机身坐标系的角速度 (root_ang_vel_b)，效果类似
    base_ang_vel_sq = torch.sum(torch.square(asset.data.root_ang_vel_w), dim=1)
    
    return is_standing_still.float() * base_ang_vel_sq


def stand_still_base_lin_vel_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    惩罚静止时的机身线速度。
    抑制机身的 X, Y, Z 平移晃动（例如前后左右平移或上下起伏）。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # 判断是否处于静止指令
    command_norm = torch.norm(command[:, :3], dim=1)
    is_standing_still = command_norm < command_threshold
    
    # 获取机身在世界坐标系下的线速度的平方和
    base_lin_vel_sq = torch.sum(torch.square(asset.data.root_lin_vel_w), dim=1)
    
    return is_standing_still.float() * base_lin_vel_sq

