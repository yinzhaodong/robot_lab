# ==============================================================================
# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Modified by: Tianyang TANG
# ==============================================================================

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher
# from isaaclab.managers import SceneEntityCfg

# import json

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
parser.add_argument("--se2_gamepad", action="store_true", default=False, help="Whether to use se2_gamepad.")
parser.add_argument("--play_lin_vel_x", type=float, default=0.5, help="Fixed forward x velocity command for play mode.")
parser.add_argument("--play_terrain_size", type=float, default=4.0, help="Terrain tile size used in play mode.")
parser.add_argument(
    "--play_max_init_terrain_level",
    type=int,
    default=0,
    help="Maximum initial terrain row for play mode. 0 starts from the easiest row.",
)
parser.add_argument("--debug", action="store_true", default=False, help="Print debug information (env config, action and observation spaces).")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point.")
parser.add_argument("--moe", action="store_true", default=False, help="Whether to use MoE.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# args_cli = parser.parse_args()
args_cli, hydra_args = parser.parse_known_args()
if hydra_args:
    print("[INFO] Ignoring Hydra-style overrides in play.py:", hydra_args)

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import time
import torch

import rsl_rl_utils
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.devices import Se2Keyboard
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
try:
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ModuleNotFoundError:
    get_published_pretrained_checkpoint = None
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab.devices.keyboard.se2_keyboard import Se2KeyboardCfg 
import robot_lab.tasks  # noqa: F401
# --- MoE Actor that can drop-in replace the base policy's actor MLP ---
import torch
import torch.nn as nn
import torch.nn.functional as F

from isaaclab.devices import Se2Gamepad
from isaaclab.devices.gamepad.se2_gamepad import Se2GamepadCfg

def _make_mlp(in_dim: int, hidden: list[int], out_dim: int, act: nn.Module):
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), act()]
        last = h
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)

class _MoEActor(nn.Module):
    """Deterministic MoE actor head: softmax routing over expert MLPs.

    - No randomness inside the actor mean; all exploration still comes from base policy's Normal(mean, std).
    - Supports top-k sparse routing by zeroing non-topk logits before softmax (still deterministic).
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: list[int],
        act: type[nn.Module],
        num_experts: int = 4,
        topk: int = 1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self._in_features = int(in_dim)     # NEW: 供导出器探测
        self._out_features = int(out_dim)   # NEW: 供导出器探测
        assert num_experts >= 1
        assert 1 <= topk <= num_experts
        self.num_experts = num_experts
        self.topk = topk
        self.register_buffer("temperature", torch.tensor(float(temperature)))
        self.experts = nn.ModuleList([_make_mlp(in_dim, hidden, out_dim, act) for _ in range(num_experts)])
        # 一个小 gating MLP（与基类 actor 的规模同一量级即可）
        gate_hidden = max(64, (hidden[0] if hidden else 128) // 2)
        self.gate = _make_mlp(in_dim, [gate_hidden], num_experts, act)
    # --- 以下三个方法是“顺序模块”兼容探针 ---
    class _FakeLayer:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    def __len__(self):                      # NEW
        # 让导出器可以 len(self.actor)
        return 2

    def __getitem__(self, idx):             # NEW
        # 导出器会读 [0].in_features，可能也会读 [-1].out_features
        if idx in (0, -2):
            return self._FakeLayer(in_features=self._in_features)
        if idx in (-1, 1):
            return self._FakeLayer(out_features=self._out_features)
        raise IndexError(idx)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.gate(x)  # [B, E]
        if self.topk < self.num_experts:
            # 稀疏 top-k：非 top-k 位置置为 -inf，再 softmax
            topk_vals, topk_idx = logits.topk(self.topk, dim=-1)
            mask = torch.zeros_like(logits, dtype=torch.bool).scatter(1, topk_idx, True)
            logits = logits.masked_fill(~mask, float("-inf"))
        probs = F.softmax(logits / self.temperature.clamp(min=1e-6), dim=-1)  # [B, E]
        means = torch.stack([e(x) for e in self.experts], dim=1)              # [B, E, A]
        out = torch.einsum("be,bea->ba", probs, means)                        # [B, A]
        return out

# -------- Policy that reuses ALL rsl-rl logic and just swaps the actor --------
# RSL-RL 5.x removed rsl_rl.modules.actor_critic. Standard PPO configs do not
# use these legacy custom policies, so keep this file importable and fail only if
# ActorCriticMoE is explicitly selected.
try:
    from rsl_rl.modules.actor_critic import ActorCritic as _BaseActorCritic
except ModuleNotFoundError:
    import types
    import rsl_rl.modules as _rsl_modules

    class _UnavailableActorCritic(nn.Module):
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "ActorCriticMoE requires the old RSL-RL ActorCritic API. "
                "Current rsl-rl-lib uses the newer actor/critic MLPModel API."
            )

    _ac_mod = types.ModuleType("rsl_rl.modules.actor_critic")
    setattr(_ac_mod, "ActorCritic", _UnavailableActorCritic)
    sys.modules["rsl_rl.modules.actor_critic"] = _ac_mod
    setattr(_rsl_modules, "actor_critic", _ac_mod)
    _BaseActorCritic = _UnavailableActorCritic

import torch
import torch.nn as nn

class ActorCriticMoE(_BaseActorCritic):
    """RSL-RL v3 兼容：基于基类的 MoE 策略。
    - 复用基类：obs 归一化 / log_std / act() / evaluate() / evaluate_actions() / 导出等
    - 仅替换 actor 的 MLP 为 MoE 头（确定性均值；探索仍由基类的 Normal(mean, std) 完成）
    """
    def __init__(
        self,
        # ★ v3 签名：先给 obs、obs_groups，再给 num_actions 与其余 cfg ★
        obs,                      # dict 或 TensorDict：来自环境的一个样本观测（含各组）
        obs_groups: dict,         # 形如 {'actor': ['policy'], 'critic': ['critic']}
        num_actions: int,
        *,
        actor_hidden_dims = [256, 256],
        critic_hidden_dims = [512, 256],
        activation = "elu",
        init_noise_std = 0.8,
        noise_std_type = "scalar",
        actor_obs_normalization = True,
        critic_obs_normalization = True,
        # MoE 相关
        num_experts: int = 4,
        topk: int = 1,
        moe_temperature: float = 1.0,
        **kwargs,
    ):
        # 先让基类按 v3 流程把一切搭好（包含 obs 归一化、log_std 等）
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            actor_hidden_dims = actor_hidden_dims,
            critic_hidden_dims = critic_hidden_dims,
            activation = activation,
            init_noise_std = init_noise_std,
            noise_std_type = noise_std_type,
            actor_obs_normalization = actor_obs_normalization,
            critic_obs_normalization = critic_obs_normalization,
            **kwargs,
        )

        act = dict(relu=nn.ReLU, elu=nn.ELU, gelu=nn.GELU)[activation.lower()]

        # 1) 从基类已构建好的 actor MLP 里取首层 Linear 的 in_features（最稳妥）
        base_actor = self.actor
        actor_in_dim = None
        for m in base_actor.modules():
            if isinstance(m, nn.Linear):
                actor_in_dim = m.in_features
                break

        # 2) 兜底：如果意外没取到（几乎不会发生），再用 obs_groups 的 'policy' 写法；若还没有就取第一个键
        if actor_in_dim is None:
            def _lastdim(t):
                return int(t.shape[-1])
            keys_for_actor = obs_groups.get("policy", None)
            if keys_for_actor is None and "actor" in obs_groups:  # 兼容别处用过的命名
                keys_for_actor = obs_groups["actor"]
            if keys_for_actor is None:
                keys_for_actor = ["policy"] if (isinstance(obs, dict) and "policy" in obs) else [next(iter(obs.keys()))]
            actor_in_dim = sum(_lastdim(obs[k]) for k in keys_for_actor)

        # 3) 构建 MoE 头并替换
        self.actor = _MoEActor(
            in_dim = actor_in_dim,
            out_dim = num_actions,
            hidden = list(actor_hidden_dims),
            act    = act,
            num_experts = num_experts,
            topk        = topk,
            temperature = float(moe_temperature),
        )

        # 4) （可选）继续给 MoE 线性层加谱归一化，与你现有风格一致
        try:
            from torch.nn.utils.parametrizations import spectral_norm as _sn
            for m in self.actor.modules():
                if isinstance(m, nn.Linear):
                    _sn(m, n_power_iterations=1)
        except Exception:
            pass

        # （可选）给 MoE 里的 Linear 上谱归一化，与你之前的 SN 风格一致
        try:
            from torch.nn.utils.parametrizations import spectral_norm as _sn
            for m in self.actor.modules():
                if isinstance(m, nn.Linear):
                    _sn(m, n_power_iterations=1)
        except Exception:
            pass

    # 便于监控 gating 的平均熵（可 wandb.log）
# 让 Isaac Lab 用 getattr(...) 能找到：rsl_rl.modules.actor_critic.ActorCriticMoE

# === NEW: debug helper to print root pose & target height ===
def _print_root_and_target(env):
    """
    Print root_pos_w (x,y,z), configured target_height, and (if available)
    ground_z from height scanner plus adjusted target = ground_z + target_height.
    """
    try:
        asset = env.unwrapped.scene["robot"]
        rp = asset.data.root_pos_w[0].detach().cpu().numpy()
        msg = f"[ROOT_POS_W] x={rp[0]:+.3f}  y={rp[1]:+.3f}  z={rp[2]:+.3f}"
    except Exception as e:
        print(f"[WARN] Cannot read root_pos_w: {e}", flush=True)
        return

    # read target_height from env config if present
    th = None
    try:
        th = float(env.unwrapped.cfg.rewards.base_height_l2.params["target_height"])
        msg += f"  | target_height(cfg)={th:+.3f}"
    except Exception:
        pass

    # try to read ground estimate from a RayCaster named "height_scanner_base"
    try:
        sensor = env.unwrapped.scene.sensors.get("height_scanner_base", None)
        if sensor is not None:
            z_hits = sensor.data.ray_hits_w[0, :, 2]
            import torch
            valid = torch.isfinite(z_hits)
            if valid.any():
                ground_z = z_hits[valid].mean().item()
                msg += f"  | ground_z≈{ground_z:+.3f}"
                if th is not None:
                    msg += f"  | adjusted≈{ground_z + th:+.3f}"
    except Exception as e:
        msg += f"  | ground_z=N/A ({e})"

    print(msg, flush=True)


def _convert_legacy_ppo_cfg_for_rsl_rl_5(train_cfg: dict) -> dict:
    """Convert IsaacLab's legacy PPO policy config to the RSL-RL 5.x actor/critic layout."""
    if "actor" in train_cfg or "policy" not in train_cfg:
        return train_cfg

    policy_cfg = dict(train_cfg["policy"])

    def _is_missing(value) -> bool:
        return value is None or value == "???" or type(value).__name__ == "_MISSING_TYPE"

    def _get(name: str, default):
        value = policy_cfg.get(name, default)
        return default if _is_missing(value) else value

    policy_class_name = _get("class_name", "ActorCritic")
    model_class_name = "RNNModel" if "Recurrent" in policy_class_name else "MLPModel"
    obs_normalization = bool(train_cfg.get("empirical_normalization", False))
    actor_obs_normalization = bool(_get("actor_obs_normalization", obs_normalization))
    critic_obs_normalization = bool(_get("critic_obs_normalization", obs_normalization))

    distribution_class_name = (
        "HeteroscedasticGaussianDistribution"
        if bool(_get("state_dependent_std", False))
        else "GaussianDistribution"
    )
    distribution_cfg = {
        "class_name": distribution_class_name,
        "init_std": float(_get("init_noise_std", 1.0)),
        "std_type": _get("noise_std_type", "scalar"),
    }

    shared_model_cfg = {
        "class_name": model_class_name,
        "activation": _get("activation", "elu"),
    }
    if model_class_name == "RNNModel":
        shared_model_cfg.update(
            {
                "rnn_type": _get("rnn_type", "lstm"),
                "rnn_hidden_dim": int(_get("rnn_hidden_dim", 256)),
                "rnn_num_layers": int(_get("rnn_num_layers", 1)),
            }
        )

    train_cfg["actor"] = {
        **shared_model_cfg,
        "hidden_dims": _get("actor_hidden_dims", [256, 256, 256]),
        "obs_normalization": actor_obs_normalization,
        "distribution_cfg": distribution_cfg,
    }
    train_cfg["critic"] = {
        **shared_model_cfg,
        "hidden_dims": _get("critic_hidden_dims", [256, 256, 256]),
        "obs_normalization": critic_obs_normalization,
    }
    train_cfg.setdefault("obs_groups", {})
    return train_cfg


def _fill_default_obs_groups_for_rsl_rl_5(train_cfg: dict, env) -> dict:
    """Set explicit actor/critic observation groups for RSL-RL 5.x when legacy configs omit them."""
    obs_groups = train_cfg.setdefault("obs_groups", {})
    if "actor" in obs_groups and "critic" in obs_groups:
        return train_cfg

    obs_keys = set(env.get_observations().keys())
    if "actor" not in obs_groups:
        if "policy" in obs_keys:
            obs_groups["actor"] = ["policy"]
        elif "actor" in obs_keys:
            obs_groups["actor"] = ["actor"]
    if "critic" not in obs_groups:
        if "critic" in obs_keys:
            obs_groups["critic"] = ["critic"]
        elif "policy" in obs_keys:
            obs_groups["critic"] = ["policy"]
    return train_cfg


def _legacy_actor_state_for_rsl_rl_5(loaded_dict: dict, target_state: dict) -> dict:
    """Best-effort conversion of old ActorCritic checkpoints to RSL-RL 5.x actor keys."""
    source_state = None
    for key in ("actor_state_dict", "model_state_dict", "policy_state_dict", "actor_critic_state_dict"):
        if key in loaded_dict:
            source_state = loaded_dict[key]
            break
    if source_state is None:
        raise KeyError(
            "Checkpoint does not contain actor_state_dict or a known legacy policy state dict "
            f"(available keys: {list(loaded_dict.keys())})"
        )

    converted = {}
    for old_key, value in source_state.items():
        key = old_key.removeprefix("module.")
        candidates = [key]

        if key.startswith("actor."):
            suffix = key[len("actor.") :]
            candidates.extend((f"mlp.{suffix}", suffix))
        elif key.startswith("actor_mlp."):
            suffix = key[len("actor_mlp.") :]
            candidates.extend((f"mlp.{suffix}", suffix))
        elif key.startswith("actor_obs_normalizer."):
            suffix = key[len("actor_obs_normalizer.") :]
            candidates.append(f"obs_normalizer.{suffix}")
        elif key == "std":
            candidates.append("distribution.std_param")
        elif key == "log_std":
            candidates.append("distribution.log_std_param")

        for candidate in candidates:
            if candidate in target_state and target_state[candidate].shape == value.shape:
                converted[candidate] = value
                break

    if not converted:
        raise RuntimeError(
            "Could not map any checkpoint tensor to the current RSL-RL 5.x actor. "
            "The checkpoint may come from a different observation/action network architecture."
        )
    return converted


def _load_runner_checkpoint_for_play(runner, resume_path: str, device: str):
    """Load checkpoints for inference, supporting both RSL-RL 5.x and older ActorCritic saves."""
    loaded_dict = torch.load(resume_path, weights_only=False, map_location=device)

    if "actor_state_dict" in loaded_dict:
        # For play only the actor is required; skip critic/optimizer to avoid unnecessary format mismatches.
        runner.load(
            resume_path,
            load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False},
            strict=False,
            map_location=device,
        )
        return loaded_dict

    actor_state = _legacy_actor_state_for_rsl_rl_5(loaded_dict, runner.alg.actor.state_dict())
    missing, unexpected = runner.alg.actor.load_state_dict(actor_state, strict=False)
    print(
        "[INFO] Loaded legacy checkpoint actor for play "
        f"({len(actor_state)} tensors, missing={len(missing)}, unexpected={len(unexpected)})."
    )
    if len(missing) > 0:
        print(f"[INFO] Missing actor keys ignored for play: {list(missing)[:8]}")
    return loaded_dict


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # if args_cli.debug:
    #     print("\n==== [env_cfg 配置结构] ====\n")
    #     print_dict(env_cfg.to_dict(), nesting=4)
    # with open("env_cfg_debug.json", "w") as f:
    #     json.dump(env_cfg.to_dict(), f, indent=4)
    agent_cfg: RslRlBaseRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    
    # =========================================================================
    # 🌟 修改点 1：拦截并覆盖 agent_cfg (针对 symmetric_ppo_cfg)
    # =========================================================================
    if args_cli.agent == "symmetric_ppo_cfg":
        print("[INFO] Using Symmetric PPO Algorithm and Config for Playback!")
        from robot_lab.tasks.locomotion.velocity.config.quadruped.Arcdog_adjustable_leg.agents.symmetric_ppo_cfg import ArclabArcdogAdjustableLegBodyflatSymmetricPPORunnerCfg
        
        agent_cfg = ArclabArcdogAdjustableLegBodyflatSymmetricPPORunnerCfg()
        agent_cfg.class_name = "SymmetricOnPolicyRunner"
        
        # 如果需要重新应用 CLI 参数覆盖，可以取消下面这行的注释
        # if hasattr(cli_args, 'update_rsl_rl_cfg'):
        #     agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    # =========================================================================

    if args_cli.moe:
        import rsl_rl.modules.actor_critic as ac
        ac.ActorCriticMoE = ActorCriticMoE
        import rsl_rl.runners.on_policy_runner as _opr
        _opr.ActorCriticMoE = ActorCriticMoE
        agent_cfg.policy.class_name = "ActorCriticMoE"
        # 2) 强制对齐训练时的网络维度 / 激活 / 探索噪声
        #    （Hydra 风格的 'agent.policy.*' 在这个脚本里会被忽略，所以在代码里直接改 cfg）
        agent_cfg.policy.actor_hidden_dims = [512, 256, 128]
        agent_cfg.policy.critic_hidden_dims = [512, 256, 128]
        agent_cfg.policy.activation = "elu"
        agent_cfg.policy.init_noise_std = 0.8
    # make a smaller scene for play
    env_cfg.scene.num_envs = args_cli.num_envs
    # Start play from easy terrain by default. None would start across the full terrain difficulty range.
    env_cfg.scene.terrain.max_init_terrain_level = args_cli.play_max_init_terrain_level
    # reduce the number of terrains to save memory
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.size = (
            args_cli.play_terrain_size,
            args_cli.play_terrain_size,
        )
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # disable randomization for play
    env_cfg.observations.policy.enable_corruption = False
    # remove random pushing
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.randomize_push_robot = None
    env_cfg.curriculum.terrain_levels = None
    env_cfg.curriculum.command_levels = None

    # For visual evaluation, use a fixed forward-only command by default.
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.play_lin_vel_x, args_cli.play_lin_vel_x)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)

    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = True

        kb_cfg = Se2KeyboardCfg(
            v_x_sensitivity=float(env_cfg.commands.base_velocity.ranges.lin_vel_x[1]),
            v_y_sensitivity=float(env_cfg.commands.base_velocity.ranges.lin_vel_y[1]),
            omega_z_sensitivity=float(env_cfg.commands.base_velocity.ranges.ang_vel_z[1]),
            # sim_device 默认即可；需要的话可传 env_cfg.sim.device
        )
        controller = Se2Keyboard(kb_cfg)  # ← 用配置类构造

        # 返回形状 [1, 3] 的 (vx, vy, wz)
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: controller.advance().unsqueeze(0).to(env.device, dtype=torch.float32),
        )


        def reset_env_callback():
            print("[INFO] 'R' key pressed: Resetting environment.")
            nonlocal obs
            obs, _ = env.reset()
        controller.add_callback("R", reset_env_callback)


    if args_cli.se2_gamepad:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = True

        # 游戏手柄配置
        se2_gamepad_cfg = Se2GamepadCfg(
            v_x_sensitivity=2.0,
            v_y_sensitivity=1.5,
            omega_z_sensitivity=3.0,
            dead_zone=0.15,
        )
        se2_controller = Se2Gamepad(se2_gamepad_cfg)

        # 设置速度命令
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: se2_controller.advance().unsqueeze(0).to(env.device, dtype=torch.float32),
        )

        # 重置环境回调
        def reset_env_callback():
            print("[INFO] Resetting environment...")
            return env.reset()[0]  # 返回新的观测
        
        se2_controller.add_callback(7, reset_env_callback)  # Start按钮
        
        # 退出应用回调
        def exit_app_callback():
            print("[INFO] Exiting application...")
            exit(0)
        
        se2_controller.add_callback(6, exit_app_callback)  # Back/Select按钮


    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        if get_published_pretrained_checkpoint is None:
            raise RuntimeError(
                "This IsaacLab version does not provide published checkpoint lookup. "
                "Use --load_run/--checkpoint or pass a full --checkpoint path instead."
            )
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint and (os.path.isabs(args_cli.checkpoint) or os.path.dirname(args_cli.checkpoint)):
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    play_cfg = _convert_legacy_ppo_cfg_for_rsl_rl_5(agent_cfg.to_dict())
    play_cfg = _fill_default_obs_groups_for_rsl_rl_5(play_cfg, env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    
    # =========================================================================
    # 🌟 修改点 2：动态切换 RunnerClass (针对 SymmetricOnPolicyRunner)
    # =========================================================================
    if agent_cfg.class_name == "SymmetricOnPolicyRunner":
        from robot_lab.tasks.locomotion.velocity.config.quadruped.Arcdog_adjustable_leg.agents.symmetric_ppo import SymmetricOnPolicyRunner
        runner = SymmetricOnPolicyRunner(
            env, 
            play_cfg, 
            log_dir=None, 
            device=agent_cfg.device,
            config=env_cfg  # 传入自定义需要的 config 参数
        )
    elif agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, play_cfg, log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, play_cfg, log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # =========================================================================
    
    _load_runner_checkpoint_for_play(runner, resume_path, agent_cfg.device)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Extract the neural network module across RSL-RL versions.
    if hasattr(runner.alg, "get_policy"):
        policy_nn = runner.alg.get_policy()
    elif hasattr(runner.alg, "actor"):
        policy_nn = runner.alg.actor
    elif hasattr(runner.alg, "policy"):
        policy_nn = runner.alg.policy
    elif hasattr(runner.alg, "actor_critic"):
        policy_nn = runner.alg.actor_critic
    else:
        policy_nn = None

    # extract the normalizer
    if policy_nn is None:
        normalizer = None
    elif hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    elif hasattr(policy_nn, "obs_normalizer"):
        normalizer = policy_nn.obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # --- De-parametrize all layers before TorchScript export ---
    from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

    def _deparametrize_all(m):
        # 遍历所有子模块，若存在任何参数化(如 spectral_norm)则移除
        for mod in m.modules():
            if is_parametrized(mod):
                # 逐个把所有被参数化的参数（通常是 "weight"）恢复成普通参数
                if hasattr(mod, "parametrizations"):
                    for pname in list(mod.parametrizations.keys()):
                        try:
                            remove_parametrizations(mod, pname, leave_parametrized=False)
                        except Exception:
                            pass

    # Export is useful but should not block play across RSL-RL API/checkpoint formats.
    if policy_nn is not None:
        try:
            _deparametrize_all(policy_nn)
            export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
            export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx", verbose=True)
        except Exception as exc:
            print(f"[WARN] Policy export skipped: {exc}")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    # # --- 构建观测切片索引：名字 -> slice(start, end) ---
    # def build_group_index_map(obs_mgr, group_name="policy"):
    #     names = obs_mgr._group_obs_term_names[group_name]
    #     shapes = obs_mgr._group_obs_term_dim[group_name]  # e.g. (171,), (3,), ...
    #     idx_map, start = {}, 0
    #     for name, shape in zip(names, shapes):
    #         # shape 可能是 int 或 tuple，做个通用乘积
    #         if isinstance(shape, (list, tuple)):
    #             n = 1
    #             for s in shape:
    #                 n *= int(s)
    #         else:
    #             n = int(shape)
    #         idx_map[name] = slice(start, start + n)
    #         start += n
    #     return idx_map

    # # 在获取到第一帧 obs 之后构建一次映射
    # obs_mgr = env.unwrapped.observation_manager
    # idx_map = build_group_index_map(obs_mgr, group_name="policy")

    # # 取出并打印某个 env 的 height_scan（这里以 env_id = 0 为例）
    # env_id = 0
    # hs = obs[env_id, idx_map["height_scan"]].detach().cpu().numpy()
    # print(f"[height_scan] env#{env_id} len={hs.size}:")
    # print(hs)

    # # 如果你想按网格显示（171=9*19 很常见），可 reshape 看看
    # try:
    #     hs_grid = hs.reshape(9, 19)   # 若你的配置不是 9x19，把 9,19 换成你的行列
    #     print("[height_scan as grid 9x19]:")
    #     print(hs_grid)
    # except Exception:
    #     pass

    timestep = 0
    debug_print = False
    if args_cli.debug and not debug_print:
        # print obs & action dim
        print("\n========== [OBSERVATION / ACTION SHAPE INFO] ==========", flush=True)
        try:
            # print observation vector shape
            if isinstance(obs, dict):
                flat_obs_shape = sum([v.numel() for v in obs.values()])
                print(f"[OBS] Total flattened shape: {flat_obs_shape} (from {len(obs)} components)", flush=True)
            else:
                print(f"[OBS] shape: {tuple(obs.shape)}", flush=True)
            
            # print action vector shape
            action_tensor = env.unwrapped.action_manager.action
            print(f"[ACTION] shape: {tuple(action_tensor.shape)}", flush=True)
        except Exception as e:
            print(f"[WARN] Cannot access shape info: {e}", flush=True)
        print("========================================================\n", flush=True)

        # print obs group -> term list
        print("\n========== [OBS GROUP MEMBERS LIST & INFO] ==========", flush=True)
        try:
            obs_mgr = env.unwrapped.observation_manager
            for group_name, term_names in obs_mgr._group_obs_term_names.items():
                print(f"[OBS GROUP] {group_name}: {term_names}", flush=True)
                for idx, name in enumerate(term_names):
                    term_cfg = obs_mgr._group_obs_term_cfgs[group_name][idx]
                    shape = obs_mgr._group_obs_term_dim[group_name][idx]
                    func_name = getattr(term_cfg.func, '__name__', str(term_cfg.func))
                    noise_type = type(term_cfg.noise).__name__ if term_cfg.noise else None
                    # print detailed info
                    print(f"  [OBS NAME] {name}", flush=True)
                    print(f"    [FUNC]        {func_name}", flush=True)
                    print(f"    [SHAPE]       {shape}", flush=True)
                    print(f"    [HISTORY]     len={term_cfg.history_length} flatten={term_cfg.flatten_history_dim}", flush=True)
                    print(f"    [CLIP]        {term_cfg.clip}", flush=True)
                    print(f"    [SCALE]       {term_cfg.scale}", flush=True)
                    print(f"    [NOISE]       {noise_type}", flush=True)
                    # 专门处理 joint_pos 观测项
                    if name == "joint_pos":
                        # 打印额外的配置信息
                        print(f"    [SPECIFIC CONFIG FOR joint_pos]", flush=True)
                        # 获取关节名称
                        if hasattr(term_cfg, 'params') and 'asset_cfg' in term_cfg.params:
                            asset_cfg = term_cfg.params['asset_cfg']
                            print(f"      [JOINT_NAMES] {asset_cfg.joint_names}", flush=True)
                            # try:
                            #     asset = env.unwrapped.scene[asset_cfg.name]
                            #     joint_names = asset.joint_names
                            #     print(f"      [JOINT_NAMES] {joint_names}", flush=True)
                            # except Exception as e:
                            #     print(f"      [ERROR] Failed to get joint names: {str(e)}", flush=True)

        except Exception as e:
            print(f"[WARN] Observation manager terms not accessible: {e}", flush=True)
        print("======================================================\n", flush=True)

        # print action space vector
        print("\n====== [Action Vector Mapping] ======", flush=True)
        idx = 0
        for group_name, term in env.unwrapped.action_manager._terms.items():
            print(f"[ACTION GROUP] {group_name}", flush=True)
            joint_names = term._joint_names if hasattr(term, "_joint_names") else [f"joint_{i}" for i in range(term.action_dim)]
            term_actions = env.unwrapped.action_manager.action[0, idx : idx + term.action_dim].cpu().numpy()
            for i, val in enumerate(term_actions):
                joint_name = joint_names[i] if i < len(joint_names) else f"joint_{i}"
                print(f"  action[{idx+i:02d}] {joint_name:>12s}: {val:+.4f}", flush=True)
            idx += term.action_dim
        print("=====================================\n", flush=True)

        debug_print = True
        time.sleep(0.1)  # avoid stdout loss
    # # 导出后切 JIT
    # jit_path = os.path.join(export_model_dir, "policy.pt")
    # del runner           # 不再需要含 critic 的 runner
    # torch.cuda.empty_cache() # 回收显存

    # policy_jit = torch.jit.load(jit_path, map_location=env.unwrapped.device)
    # policy_jit.eval()
    # simulate environment
    while simulation_app.is_running():
        # print action space vector
        if args_cli.debug and args_cli.keyboard:
            print("\n====== [Action Vector Mapping] ======", flush=True)
            idx = 0
            for group_name, term in env.unwrapped.action_manager._terms.items():
                print(f"[ACTION GROUP] {group_name}", flush=True)
                joint_names = term._joint_names if hasattr(term, "_joint_names") else [f"joint_{i}" for i in range(term.action_dim)]
                term_actions = env.unwrapped.action_manager.action[0, idx : idx + term.action_dim].cpu().numpy()
                for i, val in enumerate(term_actions):
                    joint_name = joint_names[i] if i < len(joint_names) else f"joint_{i}"
                    print(f"  action[{idx+i:02d}] {joint_name:>12s}: {val:+.4f}", flush=True)
                idx += term.action_dim
            print("=====================================\n", flush=True)
            # === NEW: also print root_pos_w & (optional) ground/adjusted target
            _print_root_and_target(env)
            # # 取出并打印某个 env 的 height_scan（这里以 env_id = 0 为例）
            # env_id = 0
            # hs = obs[env_id, idx_map["height_scan"]].detach().cpu().numpy()
            # def print_height_scan_col_major(grid, precision=6, sep=" "):
            #     """
            #     按列打印：先 [0][0] [1][0] ... [8][0]，换行；
            #     然后 [0][1] [1][1] ... [8][1]，以此类推。
            #     """
            #     rows, cols = grid.shape
            #     fmt = f"{{:+.{precision}f}}"
            #     for c in range(cols):
            #         line = sep.join(fmt.format(float(grid[r, cols-1-c])) for r in range(rows))
            #         print(line, flush=True)

            # # 已有的 hs -> (9, 19)
            # hs_grid = hs.reshape(9, 19)
            # print_height_scan_col_major(hs_grid, precision=3)



        if args_cli.debug and args_cli.se2_gamepad:
            print("\n====== [Observatiion Information] ======", flush=True)
            idx = 0
            obs_mgr = env.unwrapped.observation_manager
            # "asset_cfg"= SceneEntityCfg("robot", joint_names=".*", preserve_order=True)
            asset_print = env.unwrapped.scene["robot"]  # 假设资产名为"robot"
            default_joint_pose = asset_print.data.default_joint_pos
            joint_names = asset_print.joint_names
            # default_joint_pose = env.unwrapped.cfg.scene[SceneEntityCfg("robot", joint_names=".*", preserve_order=True)].data.default_joint_pos[:, asset_cfg.joint_ids]
            for group_name, term_names in obs_mgr._group_obs_term_names.items():
                if group_name == "policy":
                    group_data = obs_mgr._obs_buffer[group_name].data
                    if isinstance(group_data, torch.Tensor):
                        joint_pos_rel_values = group_data.flatten()[9:27]
                        # 如果组数据是张量
                        # print(f"  Type: Tensor")
                        # print(f"  Shape: {group_data.shape}")
                        # # 打印部分值（避免打印过大张量）
                        # print(f"  base_ang_vel: {group_data.flatten()[0:3].tolist()}")
                        # print(f"  projected_gravity: {group_data.flatten()[3:6].tolist()}")
                        # print(f"  vel_command_obs: {group_data.flatten()[6:9].tolist()}")
                        # print(f"  joint_pos_rel: {group_data.flatten()[9:27].tolist()}")
                        # print(f"  joint_vel: {group_data.flatten()[27:45].tolist()}")
                        # print(f"  actions: {group_data.flatten()[45:63].tolist()}")
                        # print(f"  pose_command: {group_data.flatten()[63:70].tolist()}")
                        # for i, name in enumerate(joint_names):
                        #     default_joint_pose_val = default_joint_pose[0, i].item()
                        #     current_val = default_joint_pose_val + joint_pos_rel_values[i]
                        #     print(f"  {name:<25} | {current_val:10.6f}")

                        print(f"  base_ang_vel: {group_data.flatten()[0:3].tolist()}")
                        print(f"  projected_gravity: {group_data.flatten()[3:6].tolist()}")
                        print(f"  vel_command_obs: {group_data.flatten()[6:9].tolist()}")
                        print(f"  joint_pos_rel: {group_data.flatten()[9:27].tolist()}")
                        print(f"  joint_vel: {group_data.flatten()[27:45].tolist()}")
                        print(f"  actions: {group_data.flatten()[45:57].tolist()}")
                        print(f"  pose_command: {group_data.flatten()[57:64].tolist()}")
                    # if term_names == "action":
            # act_mgr = env.unwrapped.action_manager
            # act_data = act_mgr[]
            for group_name, term in env.unwrapped.action_manager._terms.items():
                print(f"[ACTION GROUP] {group_name}", flush=True)
                joint_names = term._joint_names if hasattr(term, "_joint_names") else [f"joint_{i}" for i in range(term.action_dim)]
                term_actions = env.unwrapped.action_manager.action[0, idx : idx + term.action_dim].cpu().numpy()
                for i, val in enumerate(term_actions):
                    joint_name = joint_names[i] if i < len(joint_names) else f"joint_{i}"
                    print(f"  action[{idx+i:02d}] {joint_name:>12s}: {val:+.4f}", flush=True)
                idx += term.action_dim
            print("=====================================\n", flush=True)

        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # actions = torch.zeros_like(actions)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        # if args_cli.keyboard:
        #     rsl_rl_utils.camera_follow(env)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()