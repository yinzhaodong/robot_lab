
from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

# import omni.log
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .actions_cfg import DelayedJointPositionActionCfg

class DelayedJointPositionAction(JointPositionAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: DelayedJointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: DelayedJointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self._action_history_buf = torch.zeros(self.num_envs, cfg.history_length, self._num_joints, device=self.device, dtype=torch.float)
        self._delay_update_global_steps = cfg.delay_update_global_steps
        self._action_delay_steps = list(cfg.action_delay_steps) if isinstance(cfg.action_delay_steps, list) else cfg.action_delay_steps
        self._use_delay = cfg.use_delay
        self._randomize_action_latency = cfg.randomize_action_latency
        self._latency_range = cfg.latency_range
        self._delay_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.env = env 
        self._set_delay_steps()

    def _latency_to_steps(self, latency: float) -> int:
        """Convert latency seconds to policy action delay steps."""
        step_dt = getattr(self.env, "step_dt", None)
        if step_dt is None:
            step_dt = getattr(self.env, "physics_dt", 0.02)
        return int(round(float(latency) / float(step_dt)))

    def _set_delay_steps(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        max_delay = self.cfg.history_length - 1

        if self._randomize_action_latency:
            min_latency, max_latency = self._latency_range
            min_steps = max(0, min(max_delay, self._latency_to_steps(min_latency)))
            max_steps = max(0, min(max_delay, self._latency_to_steps(max_latency)))
            if max_steps < min_steps:
                min_steps, max_steps = max_steps, min_steps
            shape = (self.num_envs,) if isinstance(env_ids, slice) else (len(env_ids),)
            self._delay_steps[env_ids] = torch.randint(
                min_steps, max_steps + 1, shape, device=self.device, dtype=torch.long
            )
        elif isinstance(self._action_delay_steps, int):
            self._delay_steps[env_ids] = max(0, min(max_delay, int(self._action_delay_steps)))
        elif len(self._action_delay_steps) != 0:
            self._delay_steps[env_ids] = max(0, min(max_delay, int(self._action_delay_steps.pop(0))))

    def apply_actions(self):
        # set position targets
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        if self.env.common_step_counter % self._delay_update_global_steps == 0:
            self._set_delay_steps()
        self._action_history_buf = torch.cat([self._action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)
        if self._use_delay:
            indices = self.cfg.history_length - 1 - self._delay_steps
            env_ids = torch.arange(self.num_envs, device=self.device)
            self._raw_actions[:] = self._action_history_buf[env_ids, indices]
        else:
            self._raw_actions[:] = actions
        # apply the affine transformations

        if self.cfg.clip is not None:
            self._raw_actions = torch.clamp(
                self._raw_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        self._processed_actions = self._raw_actions * self._scale + self._offset
        # clip actions

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._action_history_buf[env_ids, :, :] = 0.
        if self._randomize_action_latency:
            self._set_delay_steps(env_ids)

    @property
    def action_history_buf(self):
        return self._action_history_buf
