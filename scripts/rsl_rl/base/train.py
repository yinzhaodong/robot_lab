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

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
import importlib.metadata as metadata
import platform
from packaging import version
from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args

# add obs&action dict
obs_action_info = {
    "observation_groups": {},
    "action_groups": {}
}

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--recovery_mode", action="store_true", default=False, help="Whether to use recovery mode.")
parser.add_argument("--debug", action="store_true", default=False, help="Print debug information (env config, action and observation spaces).")
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes.")

# ==========================================
# 🌟 修改点：在 argparse 中明确 agent 的 choices
# ==========================================
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", 
                    help="Name of the RL agent configuration entry point. Can be 'symmetric_ppo_cfg'.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# # always enable cameras to record video
# if args_cli.video:
#     args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# ---- torchrun read distributed----
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
IS_DISTRIBUTED = (WORLD_SIZE > 1) or bool(args_cli.distributed)
IS_MASTER = (LOCAL_RANK == 0)

# enable camera only on master
if args_cli.enable_cameras:
    args_cli.enable_cameras = True
elif args_cli.video:
    args_cli.enable_cameras = bool(args_cli.video and IS_MASTER)
# always enable cameras to record video


# disable W&B in slaves
if IS_DISTRIBUTED and not IS_MASTER:
    os.environ["WANDB_MODE"] = "disabled"  # 等价于 wandb.init(mode="disabled")

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)
"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime
import omni
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
# --- MoE Actor that can drop-in replace the base policy's actor MLP ---
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        assert num_experts >= 1
        assert 1 <= topk <= num_experts
        self.num_experts = num_experts
        self.topk = topk
        self.register_buffer("temperature", torch.tensor(float(temperature)))
        self.experts = nn.ModuleList([_make_mlp(in_dim, hidden, out_dim, act) for _ in range(num_experts)])
        # 一个小 gating MLP（与基类 actor 的规模同一量级即可）
        gate_hidden = max(64, (hidden[0] if hidden else 128) // 2)
        self.gate = _make_mlp(in_dim, [gate_hidden], num_experts, act)
        # 暴露一个路由熵指标，便于 wandb 打点
        self.last_router_probs: torch.Tensor | None = None

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
        self.last_router_probs = probs
        return out

# -------- Policy that reuses ALL rsl-rl logic and just swaps the actor --------
# RSL-RL 5.x removed rsl_rl.modules.actor_critic. The standard configs below do
# not use these legacy custom policies, so keep the file importable and fail only
# if a legacy policy is explicitly selected.
try:
    from rsl_rl.modules.actor_critic import ActorCritic as _BaseActorCritic
except ModuleNotFoundError:
    import types
    import rsl_rl.modules as _rsl_modules

    class _UnavailableActorCritic(nn.Module):
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "ActorCriticMoE/ActorCriticSN require the old RSL-RL ActorCritic API. "
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
    @property
    def routing_entropy(self):
        p = getattr(self.actor, "last_router_probs", None)
        if p is None:
            return None
        eps = 1e-8
        return (-(p * (p + eps).log()).sum(dim=-1)).mean()
# 让 Isaac Lab 用 getattr(...) 能找到：rsl_rl.modules.actor_critic.ActorCriticMoE
import rsl_rl.modules.actor_critic as ac
ac.ActorCriticMoE = ActorCriticMoE
import rsl_rl.runners.on_policy_runner as _opr
_opr.ActorCriticMoE = ActorCriticMoE
# --- CustomRecordVideo: PyAV + W&B---
from typing import Callable
try:
    import wandb
except Exception:
    wandb = None
try:
    import av  # optional
except Exception:
    av = None

from gymnasium.wrappers.rendering import RecordVideo
from gymnasium import logger

class CustomRecordVideo(RecordVideo):
    def __init__(
        self,
        env: gym.Env,
        video_folder: str,
        episode_trigger: Callable[[int], bool] | None = None,
        step_trigger: Callable[[int], bool] | None = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
        fps: int | None = None,
        disable_logger: bool = True,
        enable_wandb: bool = True,
        wandb_key: str = "train/video",
        video_resolution: tuple[int, int] = (1280, 720),
        video_crf: int = 30,
    ):
        # robustness
        super().__init__(
            env=env,
            video_folder=video_folder,
            episode_trigger=episode_trigger,
            step_trigger=step_trigger,
            video_length=video_length,
            name_prefix=name_prefix,
            disable_logger=disable_logger,
        )
        # Gymnasium  RecordVideoV0 will set self.frames_per_sec（if fps=None， env.metadata.render_fps & 30）
        if fps is not None:
            self.frames_per_sec = fps  

        self.enable_wandb = bool(enable_wandb and (wandb is not None))
        self.wandb_key = wandb_key
        self.video_resolution = tuple(video_resolution)
        self.video_crf = int(video_crf)

    def _write_with_pyav(self, frames, path):
        # PyAV->h264 + yuv420p（for web use）
        if av is None:
            raise RuntimeError("PyAV not available")
        container = av.open(path, "w")
        stream = container.add_stream("libx264", rate=round(self.frames_per_sec))
        stream.width, stream.height = self.video_resolution
        stream.pix_fmt = "yuv420p"
        # CRF defines video quality
        stream.options = {"crf": str(self.video_crf), "preset": "ultrafast"}
        for fr in frames:
            vf = av.VideoFrame.from_ndarray(fr, format="rgb24")
            vf = vf.reformat(width=self.video_resolution[0], height=self.video_resolution[1])
            packet = stream.encode(vf)
            if packet:
                container.mux(packet)
        # flush
        packet = stream.encode(None)
        if packet:
            container.mux(packet)
        container.close()

    def stop_recording(self):
        """write to disk then upload to W&B。"""
        assert self.recording, "stop_recording was called, but no recording was started"

        path = os.path.join(self.video_folder, f"{self._video_name}.mp4")

        if len(self.recorded_frames) == 0:
            logger.warn("Ignored saving a video as there were zero frames to save.")
        else:
            try:
                # PyAV 
                self._write_with_pyav(self.recorded_frames, path)
            except Exception:
                # Roll back to moviepy
                super().stop_recording()
            else:
                # Reset
                self.recorded_frames = []
                self.recording = False
                self._video_name = None

            # Try Upload
            if self.enable_wandb and os.path.exists(path) and (wandb is not None):
                try:
                    # key for bounding
                    wandb.log({self.wandb_key: wandb.Video(path, format="mp4")}, commit=True)
                    print(f"[W&B] Logged video: {path}")
                except Exception as e:
                    print(f"[WARN] wandb video log failed: {e}")

def make_serializable(info: dict):
    """Convert tensor and object fields into serializable types for YAML."""
    def tensor_to_list(val):
        if isinstance(val, torch.Tensor):
            return val.cpu().tolist()
        return val

    serializable_info = {}
    for key, value in info.items():
        if isinstance(value, dict):
            serializable_info[key] = make_serializable(value)
        elif isinstance(value, list):
            serializable_info[key] = [make_serializable(v) if isinstance(v, dict) else tensor_to_list(v) for v in value]
        else:
            serializable_info[key] = tensor_to_list(value)
    return serializable_info
# === Add: Spectral-Normalized ActorCritic defined inline in train.py ===
import torch.nn as nn
# 兼容两种导入路径（不同 PyTorch 版本）
try:
    from torch.nn.utils.parametrizations import spectral_norm as _spectral_norm
except Exception:
    from torch.nn.utils import spectral_norm as _spectral_norm

# rsl-rl 的 ActorCritic 基类
try:
    from rsl_rl.modules.actor_critic import ActorCritic as _BaseActorCritic
except Exception:
    import rsl_rl.modules.actor_critic as _ac_mod
    _BaseActorCritic = _ac_mod.ActorCritic

def _apply_sn(module: nn.Module, n_power_iterations: int = 1):
    """给模块里所有 Linear 施加谱归一化。"""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            _spectral_norm(m, n_power_iterations=n_power_iterations)
    return module

class ActorCriticSN(_BaseActorCritic):
    """Actor-Critic with Spectral Normalization on all Linear layers."""
    def __init__(self, *args, sn_on=("actor", "critic"), n_power_iterations: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        if "actor" in sn_on and hasattr(self, "actor"):
            _apply_sn(self.actor, n_power_iterations)
        if "critic" in sn_on and hasattr(self, "critic"):
            _apply_sn(self.critic, n_power_iterations)

# ---- 最关键的一行：把默认类名映射到我们的 SN 版本（无需改任何 cfg）----
# import rsl_rl.modules.actor_critic as _ac
# _ac.ActorCritic = ActorCriticSN
# （可选）如果你在 cfg 里把 class_name 改成了 "ActorCriticSN"：
# import rsl_rl.runners.on_policy_runner as _opr
# _opr.ActorCriticSN = ActorCriticSN
# 这样 eval("ActorCriticSN") 也能解析到这个类。
# === End Add ===


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

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    
    # =========================================================================
    # 🌟 修改点：根据 args_cli.agent 动态拦截并覆盖 agent_cfg
    # =========================================================================
    if args_cli.agent == "symmetric_ppo_cfg":
        print("[INFO] Using Symmetric PPO Algorithm and Config!")
        from robot_lab.tasks.locomotion.velocity.config.quadruped.Arcdog_adjustable_leg.agents.symmetric_ppo_cfg import ArclabArcdogAdjustableLegBodyflatSymmetricPPORunnerCfg
        
        # 覆盖 config
        agent_cfg = ArclabArcdogAdjustableLegBodyflatSymmetricPPORunnerCfg()
        # 强制指定 class_name，以便后续逻辑识别
        agent_cfg.class_name = "SymmetricOnPolicyRunner"
    else:
        print("[INFO] Using Standard RSL-RL Config!")
    # =========================================================================

    if IS_DISTRIBUTED:
        env_cfg.sim.device = f"cuda:{LOCAL_RANK}"
        agent_cfg.device = f"cuda:{LOCAL_RANK}"
        seed = (agent_cfg.seed or 0) + LOCAL_RANK
        env_cfg.seed = seed
        agent_cfg.seed = seed
        
    # override configurations with non-hydra CLI arguments
    # 注意：因为我们在上面覆盖了 agent_cfg，这里的 cli_args.update_rsl_rl_cfg 依然能正常工作！
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # multi-gpu / multi-node (torch.distributed via torchrun)
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # random seed for each gpu
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

        # （Optional）Average num_envs： If you want --num_envs means “Total envs”
        # if args_cli.num_envs is not None:
        #     # world_size = GPUs * nodes
        #     world_size = app_launcher.world_size
        #     per_rank_envs = max(1, args_cli.num_envs // world_size)
        #     env_cfg.scene.num_envs = per_rank_envs

    # set recovery mode
    if args_cli.recovery_mode:
        env_cfg.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 1.0),
                "roll": (-3.14, 3.14),
                "pitch": (-3.14, 3.14),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        env_cfg.rewards.upward.weight = 0.5
        env_cfg.terminations.illegal_contact = None
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video and IS_MASTER:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "enable_wandb": (agent_cfg.logger == "wandb"),
            "wandb_key": "train/video",
            "video_resolution": (640, 360),
            "video_crf": 30,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = CustomRecordVideo(env, **video_kwargs)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    # create runner from rsl-rl
    train_cfg = _convert_legacy_ppo_cfg_for_rsl_rl_5(agent_cfg.to_dict())
    train_cfg = _fill_default_obs_groups_for_rsl_rl_5(train_cfg, env)
    
    # =========================================================================
    # 🌟 修改点：动态切换 RunnerClass 
    # =========================================================================
    if args_cli.agent == "symmetric_ppo_cfg" or agent_cfg.class_name == "SymmetricOnPolicyRunner":
        from robot_lab.tasks.locomotion.velocity.config.quadruped.Arcdog_adjustable_leg.agents.symmetric_ppo import SymmetricOnPolicyRunner
        runner = SymmetricOnPolicyRunner(
            env, 
            train_cfg,
            config=env_cfg,       # <==== 补充缺失的 config 参数！
            log_dir=log_dir, 
            device=agent_cfg.device
        )
    elif agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, train_cfg, log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # =========================================================================

    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    if args_cli.debug:
        import time
        print("\n========== [OBSERVATION / ACTION SHAPE INFO] ==========", flush=True)
        try:
            obs, _ = runner.env.reset()
            if isinstance(obs, dict):
                total_shape = sum([v.numel() for v in obs.values()])
                print(f"[OBS] Total shape (dict): {total_shape}", flush=True)
                for k, v in obs.items():
                    print(f"  - {k:20s}: shape = {tuple(v.shape)}", flush=True)
            elif isinstance(obs, torch.Tensor):
                print(f"[OBS] shape: {tuple(obs.shape)}", flush=True)
            else:
                print(f"[OBS] type={type(obs)}, content={obs}", flush=True)
            # print action vector shape
            action_tensor = env.unwrapped.action_manager.action
            print(f"[ACTION] shape: {tuple(action_tensor.shape)}", flush=True)
        except Exception as e:
            print(f"[WARN] Cannot access shape info: {e}", flush=True)
        print("========================================================\n", flush=True)

        obs_action_info = {"observation_groups": {}, "action_groups": {}}

        print("\n========== [OBS GROUP MEMBERS LIST & INFO] ==========", flush=True)
        try:
            obs_mgr = runner.env.env.unwrapped.observation_manager
            for group_name, term_names in obs_mgr._group_obs_term_names.items():
                print(f"[OBS GROUP] {group_name}: {term_names}", flush=True)
                obs_action_info["observation_groups"][group_name] = []

                for idx, name in enumerate(term_names):
                    term_cfg = obs_mgr._group_obs_term_cfgs[group_name][idx]
                    shape = obs_mgr._group_obs_term_dim[group_name][idx]
                    func_name = getattr(term_cfg.func, '__name__', str(term_cfg.func))
                    noise_type = type(term_cfg.noise).__name__ if term_cfg.noise else None

                    info = {
                        "name": name,
                        "shape": shape,
                        "func": func_name,
                        "history_length": term_cfg.history_length,
                        "flatten_history_dim": term_cfg.flatten_history_dim,
                        "clip": term_cfg.clip,
                        "scale": term_cfg.scale,
                        "noise": noise_type
                    }

                    obs_action_info["observation_groups"][group_name].append(info)

                    # print detailed info
                    print(f"  [OBS NAME] {name}", flush=True)
                    print(f"    [FUNC]        {func_name}", flush=True)
                    print(f"    [SHAPE]       {shape}", flush=True)
                    print(f"    [HISTORY]     len={term_cfg.history_length} flatten={term_cfg.flatten_history_dim}", flush=True)
                    print(f"    [CLIP]        {term_cfg.clip}", flush=True)
                    print(f"    [SCALE]       {term_cfg.scale}", flush=True)
                    print(f"    [NOISE]       {noise_type}", flush=True)
        except Exception as e:
            print(f"[WARN] Observation manager terms not accessible: {e}", flush=True)
        print("======================================================\n", flush=True)


        print("\n====== [Action Vector Mapping] ======", flush=True)
        try:
            idx = 0
            for group_name, term in runner.env.unwrapped.action_manager._terms.items():
                action_group = {
                    "action_dim": term.action_dim,
                    "joint_names": getattr(term, "_joint_names", [f"joint_{i}" for i in range(term.action_dim)])
                }
                obs_action_info["action_groups"][group_name] = action_group
                print(f"[ACTION GROUP] {group_name}", flush=True)
                joint_names = getattr(term, "_joint_names", [f"joint_{i}" for i in range(term.action_dim)])
                term_actions = runner.env.unwrapped.action_manager.action[0, idx : idx + term.action_dim].cpu().numpy()
                for i, val in enumerate(term_actions):
                    joint_name = joint_names[i] if i < len(joint_names) else f"joint_{i}"
                    print(f"  action[{idx+i:02d}] {joint_name:>12s}: {val:+.4f}", flush=True)
                idx += term.action_dim
        except Exception as e:
            print(f"[WARN] Action manager info not available: {e}", flush=True)
        print("=====================================\n", flush=True)
        safe_obs_action_info = make_serializable(obs_action_info)
        dump_yaml(os.path.join(log_dir, "params", "obs_action.yaml"), safe_obs_action_info)
        time.sleep(0.1)
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    # Optional: Force commit
    try:
        import wandb
        if wandb and wandb.run is not None:
            wandb.log({}, commit=True)
            wandb.finish()
    except Exception:
        pass
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()