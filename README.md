# robot_lab

This repository is a project-specific robot learning extension based on
[fan-ziqi/robot_lab](https://github.com/fan-ziqi/robot_lab) and Isaac Lab. It keeps
the extension outside the Isaac Lab source tree, so tasks, assets, reward terms,
and training scripts can be developed independently.

## Version Dependency

The recommended runtime follows the upstream `robot_lab` compatibility table:

| Component | Version |
| --- | --- |
| Isaac Sim | `5.1.0` |
| Isaac Lab | `v2.3.2` |
| Python | `3.11` |
| RSL-RL | `rsl-rl-lib==5.0.1` |
| OS | Ubuntu 22.04 |

If you are using Isaac Sim `5.1.0`, use Isaac Lab `v2.3.2` instead of an arbitrary
`main` or `develop` branch.

```bash
cd ~/Noitom/isaaclab/5.1/IsaacLab
git fetch --tags origin
git checkout v2.3.2
```

## Installation

Activate the Python environment that can import Isaac Lab and Isaac Sim:

```bash
conda activate dog
```

Install this extension in editable mode from the repository root:

```bash
cd ~/code/robot_lab
python -m pip install -e source/robot_lab
```

Optional sanity checks:

```bash
python -c "import isaaclab, robot_lab; print('robot_lab ready')"
python scripts/tools/list_envs.py
```

## Arcdog Bodyflat Task

This repository currently documents only this verified stretch-leg task:

```text
RobotLab-Isaac-Velocity-Bodyflat-ArcdogAdjustableLeg-v0
```

All commands below are meant to be run from the repository root:

```bash
cd ~/code/robot_lab
conda activate dog
```

### Train

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/base/train.py \
  --task=RobotLab-Isaac-Velocity-Bodyflat-ArcdogAdjustableLeg-v0 \
  --headless \
  --num_envs=6 \
  --max_iterations=20000 \
  --run_name=rew
```

The Bodyflat task enables MGDP-style randomized action latency for sim-to-real.
The default range is `0.0` to `0.02` seconds, configured in:

```text
source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/bodyflat_env_cfg.py
```

Relevant parameters:

```python
use_delay=True
randomize_action_latency=True
latency_range=(0.0, 0.02)
```


### Resume

Resume uses the run folder under `logs/rsl_rl/arclab_arcdog_adjustable_leg_bodyflat/`.
Only pass the run folder name to `--load_run`, not the full path.

```bash
CUDA_VISIBLE_DEVICES=5 python scripts/rsl_rl/base/train.py \
  --task=RobotLab-Isaac-Velocity-Bodyflat-ArcdogAdjustableLeg-v0 \
  --headless \
  --num_envs=4096 \
  --max_iterations=20000 \
  --resume \
  --load_run=2026-05-03_12-06-26_rew \
  --checkpoint=model_1000.pt \
  --run_name=rew_resume
```

### Play

Play uses the same log root:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/base/play.py \
  --task=RobotLab-Isaac-Velocity-Bodyflat-ArcdogAdjustableLeg-v0 \
  --num_envs=16 \
  --play_lin_vel_x=0.5 \
  --load_run=2026-05-03_12-09-41_new \
  --checkpoint=model_2000.pt
```


```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/base/play.py \
  --task=RobotLab-Isaac-Velocity-Bodyflat-ArcdogAdjustableLeg-v0 \
  --num_envs=16 \
  --play_lin_vel_x=0.9 \
  --checkpoint=logs/rsl_rl/arclab_arcdog_adjustable_leg_bodyflat/2026-05-03_11-06-34_raw/model_4600.pt
```

