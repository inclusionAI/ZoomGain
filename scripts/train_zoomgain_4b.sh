#!/usr/bin/env bash
set -euo pipefail
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-zoomgain_qwen35_4b}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/train_zoomgain.sh" "$@"
