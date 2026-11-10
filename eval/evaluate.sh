#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a local HF directory or global_step_N directory}"
MODEL_PATH="${MODEL_PATH%/}"
has_weights() {
  [[ -f "$1/config.json" ]] && {
    [[ -f "$1/model.safetensors" || -f "$1/model.safetensors.index.json" ]] ||
      compgen -G "$1/model-*.safetensors" > /dev/null
  }
}
if ! has_weights "${MODEL_PATH}"; then
  for candidate in "${MODEL_PATH}/actor/huggingface" "${MODEL_PATH}/huggingface"; do
    if has_weights "${candidate}"; then MODEL_PATH="${candidate}"; break; fi
  done
fi
has_weights "${MODEL_PATH}" || { echo "No merged HF checkpoint found: ${MODEL_PATH}" >&2; exit 1; }
MODEL_PATH="$(cd "${MODEL_PATH}" && pwd)"
if [[ -z "${RUN_NAME:-}" ]]; then
  run_dir="${MODEL_PATH}"
  RUN_NAME="$(basename "${MODEL_PATH}")"
  while [[ "${run_dir}" != "/" ]]; do
    if [[ "$(basename "${run_dir}")" =~ ^global_step_[0-9]+$ ]]; then
      run_dir="$(dirname "${run_dir}")"
      if [[ "$(basename "${run_dir}")" == checkpoints ]]; then run_dir="$(dirname "${run_dir}")"; fi
      RUN_NAME="$(basename "${run_dir}")"; break
    fi
    run_dir="$(dirname "${run_dir}")"
  done
fi
EXTRA_ARGS=()
if [[ -n "${DATA_CONFIG:-}" ]]; then EXTRA_ARGS+=(--data-config "${DATA_CONFIG}"); fi
exec "${PYTHON_BIN:-python3}" -u "${HERE}/evaluate.py" \
  --model-path "${MODEL_PATH}" --run-name "${RUN_NAME}" \
  --gpus "${GPUS:-0}" --tp "${TP:-1}" --port "${PORT:-18840}" \
  --benchmarks "${BENCHMARKS:-zoombench,vstar_all,hrbench4k,hrbench8k,mmerealworld,treebench,visualprobe_all,cvbench,mmstar,mmvp,babyvision,perceptionbench,o3bench}" \
  --generation-profile "${GENERATION_PROFILE:-qwen35_nothinking}" \
  --answer-format answer_tag --use-cot true \
  --max-new-tokens "${MAX_TOKENS:-2048}" --extra-max-new-tokens "${EXTRA_MAX_TOKENS:-8192}" \
  --per-engine-concurrency "${CONCURRENCY:-12}" --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.6}" --preview-samples 0 \
  --vllm-bin "${VLLM_BIN:-vllm}" --output "${OUTPUT_DIR:-${HERE}/outputs/${RUN_NAME}}" \
  "${EXTRA_ARGS[@]}" "$@"
