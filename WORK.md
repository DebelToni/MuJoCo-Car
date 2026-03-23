# WORK LOG

## Goal
Build a fresh MuJoCo + MJX tank-car RL project in this empty directory, including:
- A custom `xacro` / URDF tank-like car.
- A MuJoCo scene and MJX environment.
- PPO-style RL training on CPU with time-based episodes.
- Progress tracking and artifacts.

## Design Notes
- Platform body is set to 10 cm x 5 cm and lifted 5 cm above ground.
- Four corner wheels are included, with diagonal motorized wheels:
  - front-left motorized
  - rear-right motorized
  - front-right free wheel
  - rear-left free wheel
- Two side track geoms represent chain contact with the ground.
- Sensors are implemented as four front-facing virtual range rays with angles:
  - `pi/5`, `2*pi/5`, `3*pi/5`, `4*pi/5`
- RL observation now includes these 4 range readings plus explicit target guidance features.
- Target object spawns 5-10 m away, with random obstacles spawned in scene.

## Implementation Steps
- [x] Inspect reference project (`../MuJoCo-Test`) for structure and conventions.
- [x] Create project skeleton and source layout.
- [x] Implement tank-car `xacro` and xacro->urdf build script.
- [x] Implement MuJoCo scene XML for tank navigation.
- [x] Implement MJX env with random target/obstacles and 4 front sensors.
- [x] Implement PPO training pipeline with time-budgeted training episodes.
- [x] Add rollout rendering utility.
- [x] Run smoke tests (build + short training + rollout).
- [x] Run an initial training pass and save outputs.
- [x] Document commands and usage in README.

## Command Log
- Inspected current directory and confirmed it was empty.
- Read key files from `../MuJoCo-Test` (`README`, env, policy, train scripts, xacro build script).
- Built URDF successfully: `assets/generated/tank_car.urdf`.
- Ran MJX env smoke check (`reset` + `step`) successfully.
- Ran short training smoke (`outputs/smoke`) and produced policy + metrics + chart.
- Ran longer initial training passes:
  - `outputs/initial_train`
  - `outputs/diagnostic`
  - `outputs/diagnostic_v2`
  - `outputs/easy_task_test`
  - `outputs/easy_reward_test`
- Rendered rollout artifacts:
  - `outputs/smoke/rollout.gif`
  - `outputs/smoke/rollout_dashboard.png`
  - `outputs/easy_reward_test/rollout.gif`
  - `outputs/easy_reward_test/rollout_dashboard.png`

## Training Status Snapshot
- MJX PPO pipeline runs end-to-end on CPU.
- Time-budgeted training episodes execute correctly and auto-evaluate after each episode.
- In short validation budgets used during development, policy improved distance metrics but did not yet reach 95% eval success.
- Full 30-minute episode runs are configured in `scripts/train_tank_mjx.py` and documented in `README.md`.
- Started a long-run background training job:
  - command target: `outputs/main_run`
  - process id: `77343`
  - log file: `outputs/main_run/train.log`
- Stopped long-run background job cleanly on user request (for system shutdown).
- Added resume-safe training support in `scripts/train_tank_mjx.py`:
  - `--resume-policy` to continue from saved weights.
  - auto-load existing `train_metrics.json` and `episode_summaries.json` from output dir.
  - `--checkpoint-every-iters` to snapshot progress during long episodes.
- Updated observation design after user requirement update:
  - kept 4 front sensors,
  - added target cue features (`distance`, `cos`/`sin` heading error),
  - updated rollout plotting to read the newest sensor slice correctly.
- Ran post-change smoke training to verify updated observation pipeline:
  - `outputs/target_cue_smoke`
- Refactored scene geometry proportions:
  - target changed to ground cube (`5 cm` side),
  - obstacles changed to proportionate boxes:
    - pillars (`10 cm x 10 cm` footprint),
    - walls (`1.0 m x 10 cm` footprint).
- Refactored environment sensing/collision model:
  - distance sensors now ignore target completely and only report obstacle distances,
  - obstacle sensing switched to box-ray intersections,
  - obstacle collision clearance switched to box signed-distance checks.
- Cleared all legacy run folders by deleting `outputs/` for a fresh start.
- Added top-down camera (`topdown`) in MJCF and updated render/viewer scripts to support `--camera`.
- Added GPU training preset launcher: `scripts/train_preset_gpu.sh`.
- Replaced local `assets/tank_car.xacro` with user-provided `~/Desktop/tank_car.xacro`.
- Applied requested Xacro tweak: moved both chain tracks farther from the platform by `+5 cm` per side.
- Refactored environment to a minimal room setup:
  - 2m x 2m room walls,
  - 5cm wall thickness,
  - minimal interior walls of 50cm x 5cm,
  - random car spawn and random target spawn inside the room.
- Sensors remain 4 front rays and continue to ignore target geometry.
- Updated control cadence to reduce jitter:
  - policy still outputs 2 wheel-speed actions (left/right),
  - each action is now held for ~0.48s (`action_hold_steps=6`) before next decision.

## Pending Validation
- Fresh training run with the new scene/sensor setup.
