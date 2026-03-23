# MuJoCo Tank Car Navigation (MJX + PPO)

This is a fresh MuJoCo project for a tiny tank-like car with:
- a custom `xacro`/URDF model,
- a MuJoCo scene with a compact 2m x 2m room and minimal wall obstacles,
- an MJX-backed RL environment using 4 front distance sensors plus explicit target-direction cues,
- a PPO-style training loop that runs time-budgeted CPU training episodes.

## Project layout

- `assets/tank_car.xacro` - tank car robot design (10 cm x 5 cm body, lifted 5 cm).
- `assets/generated/tank_car.urdf` - generated URDF output.
- `mjcf/tank_car_nav.xml` - MuJoCo navigation scene.
- `src/tank_env.py` - MJX env (reset/step/reward/sensors).
- `src/policy.py` - actor-critic policy and PPO losses.
- `scripts/build_urdf.py` - compile xacro -> URDF.
- `scripts/train_tank_mjx.py` - training loop with 30-minute episode budgets.
- `scripts/train_preset_gpu.sh` - high-throughput GPU training preset.
- `scripts/render_rollout.py` - rollout GIF + sensor dashboard.
- `scripts/live_viewer.py` - interactive MuJoCo viewer.
- `WORK.md` - running work log and progress.

## Sensor setup

The robot has only four front-facing distance sensors at:
- `pi/5`
- `2*pi/5`
- `3*pi/5`
- `4*pi/5`

No center ray is used. The policy input uses:
- a short temporal stack of these same 4 readings (3 frames), and
- target guidance features (`distance_to_target`, `cos(heading_error)`, `sin(heading_error)`).

Policy action has 2 outputs: left-track speed command and right-track speed command (both normalized in `[-1, 1]`).
Each action is held for about `0.48s` (`frame_skip=4`, `action_hold_steps=6`) before the next network decision.

The target is intentionally transparent to distance sensors; sensors only report obstacle distances.

## Scene proportions

- Target is a ground cube of size `5 cm x 5 cm x 5 cm`.
- Room interior is `2 m x 2 m`.
- Wall thickness is `5 cm`.
- Interior wall obstacles are `50 cm x 5 cm`.
- Car and target spawn at random valid positions inside the room each episode.

## Environment setup

Use your existing venv:

```bash
/Volumes/SSD/v/mujoco/bin/python -m pip install -r requirements.txt
```

## Build URDF from xacro

```bash
/Volumes/SSD/v/mujoco/bin/python scripts/build_urdf.py
```

## Train (CPU, MJX backend)

Default command uses 30-minute training episodes and repeats until target accuracy (95%) or max training episodes:

```bash
/Volumes/SSD/v/mujoco/bin/python scripts/train_tank_mjx.py \
  --episode-minutes 30 \
  --target-accuracy 0.95 \
  --max-training-episodes 6 \
  --output-dir outputs/main_run
```

Resume from a saved checkpoint:

```bash
/Volumes/SSD/v/mujoco/bin/python scripts/train_tank_mjx.py \
  --resume-policy outputs/main_run/policy.pkl \
  --output-dir outputs/main_run \
  --max-training-episodes 8
```

## Train (Remote GPU preset)

```bash
bash scripts/train_preset_gpu.sh
```

Useful overrides:

```bash
PYTHON_BIN=/opt/venv/bin/python OUT_DIR=outputs/remote_gpu_run EPISODE_MINUTES=30 MAX_EPISODES=8 ROLLOUT_EPISODES=64 EVAL_EPISODES=64 UPDATE_EPOCHS=10 bash scripts/train_preset_gpu.sh
```

Main outputs:
- `outputs/main_run/policy.pkl`
- `outputs/main_run/train_metrics.json`
- `outputs/main_run/episode_summaries.json`
- `outputs/main_run/training_curves.png`

## Quick smoke run

```bash
/Volumes/SSD/v/mujoco/bin/python scripts/train_tank_mjx.py \
  --episode-minutes 0.1 \
  --max-training-episodes 1 \
  --rollout-episodes 6 \
  --eval-episodes 8 \
  --max-steps 260 \
  --bc-episodes 0 \
  --bc-epochs 0 \
  --output-dir outputs/smoke
```

## Render rollout

```bash
/Volumes/SSD/v/mujoco/bin/python scripts/render_rollout.py \
  --policy outputs/main_run/policy.pkl \
  --camera topdown \
  --gif outputs/main_run/rollout.gif \
  --chart outputs/main_run/rollout_dashboard.png
```

## Live viewer

```bash
/Volumes/SSD/v/mujoco/bin/python scripts/live_viewer.py --camera topdown --loop
```
