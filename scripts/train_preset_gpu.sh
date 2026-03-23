#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"
OUT_DIR="${OUT_DIR:-outputs/remote_gpu_run}"
EPISODE_MINUTES="${EPISODE_MINUTES:-30}"
MAX_EPISODES="${MAX_EPISODES:-8}"
TARGET_ACCURACY="${TARGET_ACCURACY:-0.95}"
ROLLOUT_EPISODES="${ROLLOUT_EPISODES:-64}"
EVAL_EPISODES="${EVAL_EPISODES:-64}"
MAX_STEPS="${MAX_STEPS:-520}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-10}"
LR="${LR:-2.5e-4}"
BC_EPISODES="${BC_EPISODES:-0}"
BC_EPOCHS="${BC_EPOCHS:-0}"

export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

"${PYTHON_BIN}" scripts/train_tank_mjx.py \
  --episode-minutes "${EPISODE_MINUTES}" \
  --max-training-episodes "${MAX_EPISODES}" \
  --target-accuracy "${TARGET_ACCURACY}" \
  --rollout-episodes "${ROLLOUT_EPISODES}" \
  --eval-episodes "${EVAL_EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --update-epochs "${UPDATE_EPOCHS}" \
  --learning-rate "${LR}" \
  --bc-episodes "${BC_EPISODES}" \
  --bc-epochs "${BC_EPOCHS}" \
  --checkpoint-every-iters 1 \
  --output-dir "${OUT_DIR}"
