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

## Training Scripts

All commands below are meant to be run from the repository root:

```bash
cd ~/code/robot_lab
conda activate dog
```

### Arcdog Adjustable Leg Bodyflat

This is the currently verified stretch-leg training task.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/base/train.py \
  --task=RobotLab-Isaac-Velocity-Bodyflat-ArcdogAdjustableLeg-v0 \
  --headless \
  --num_envs=16 \
  --max_iterations=12000
```

Quick smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/base/train.py \
  --task=RobotLab-Isaac-Velocity-Bodyflat-ArcdogAdjustableLeg-v0 \
  --headless \
  --num_envs=4 \
  --max_iterations=1
```

Other registered Arcdog adjustable-leg tasks:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/base/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-ArcdogAdjustableLeg-v0 \
  --headless \
  --num_envs=16 \
  --max_iterations=12000
```

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/base/train.py \
  --task=RobotLab-Isaac-Velocity-Rough-ArcdogAdjustableLeg-v0 \
  --headless \
  --num_envs=16 \
  --max_iterations=12000
```

### G1 Inspire Manipulation

Use the same command style for the G1 Inspire manipulation training entry point:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/g1_inspire_manip/train.py \
  --task=G1_Inspire_TS1 \
  --headless \
  --num_envs=16 \
  --max_iterations=12000
```

For a quick run:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/g1_inspire_manip/train.py \
  --task=G1_Inspire_TS1 \
  --headless \
  --num_envs=4 \
  --max_iterations=1
```

## Useful Options

```bash
# Record video during training.
CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/base/train.py \
  --task=RobotLab-Isaac-Velocity-Bodyflat-ArcdogAdjustableLeg-v0 \
  --headless \
  --video \
  --video_length=200 \
  --video_interval=2000 \
  --num_envs=16 \
  --max_iterations=12000
```

```bash
# Resume from an existing run.
CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/base/train.py \
  --task=RobotLab-Isaac-Velocity-Bodyflat-ArcdogAdjustableLeg-v0 \
  --headless \
  --resume \
  --load_run=<RUN_FOLDER_NAME> \
  --checkpoint=<CHECKPOINT_NAME>
```

```bash
# Debug observation and action layout.
CUDA_VISIBLE_DEVICES=0 python scripts/rsl_rl/base/train.py \
  --task=RobotLab-Isaac-Velocity-Bodyflat-ArcdogAdjustableLeg-v0 \
  --headless \
  --num_envs=4 \
  --max_iterations=1 \
  --debug
```

## Logs

RSL-RL logs are written under:

```text
logs/rsl_rl/<experiment_name>/<timestamp>/
```

For the bodyflat Arcdog task, the default experiment directory is:

```text
logs/rsl_rl/arclab_arcdog_adjustable_leg_bodyflat/
```

## Notes

- Isaac Sim warnings such as `IOMMU is enabled`, CPU powersave warnings, and
  `FabricManager::initializePointInstancer mismatched prototypes` are usually
  simulator warnings, not training failures.
- The local `scripts/rsl_rl/base/train.py` has compatibility handling for
  `rsl-rl-lib==5.0.1`, where PPO uses separate `actor` and `critic` model
  configs instead of the older single `policy` / `ActorCritic` API.
- Keep this repository outside the Isaac Lab repository, then install it with
  `python -m pip install -e source/robot_lab`.
