#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON_BIN:-python3}" -u "${HERE}/eval_zoomgain_benchmark_vllm_multigpu.py" \
  --model-path "${MODEL_PATH:?Set MODEL_PATH to a local HF model directory}" \
  --annotations "${ANNOTATIONS:?Set ANNOTATIONS to ZoomGainBench annotations.jsonl}" \
  --output-dir "${OUTPUT_DIR:-${HERE}/outputs}" \
  --gpus "${GPUS:-0}" --tp "${TP:-1}" --port "${PORT:-18811}" \
  --lan_type "${LANGUAGE:-all}" \
  --generation-profile "${GENERATION_PROFILE:-qwen3_nothinking}" \
  --max-tokens "${MAX_TOKENS:-2048}" --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --per-engine-concurrency "${CONCURRENCY:-12}" \
  --vllm-bin "${VLLM_BIN:-vllm}" "$@"
