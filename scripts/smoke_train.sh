#!/usr/bin/env bash
# BASE_MODEL and TRAIN_PARQUET must be set. Checkpoints go under OUTPUT_DIR.
set -euo pipefail
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export TOTAL_TRAINING_STEPS=1 SAVE_FREQ=1
export RESUME_MODE="${RESUME_MODE:-disable}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-zoomgain_smoke}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/train_zoomgain.sh" "$@"
