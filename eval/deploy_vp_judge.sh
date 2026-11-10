#!/usr/bin/env bash
# Serve the VisualProbe answer judge.
#
# VisualProbe is scored by an LLM judge, not by string matching (Mini-o3,
# verl/utils/reward_score/general_qa_tool{,_qwen}.py -- GPT-4o or
# Qwen2.5-VL-72B).  p4_eval_vllm.py reproduces that protocol with the authors'
# verbatim prompt against whatever model this script serves; its default
# --vp-judge-endpoint points here.
#
# Run this on the current single-node judge host (default GPU 7, port 8031), e.g.
#   MODEL_PATH=/path/to/Qwen3.5-35B-A3B bash deploy_vp_judge.sh
# Override anything via the environment:
#   GPUS=6,7 VP_JUDGE_PORT=8041 bash deploy_vp_judge.sh
#   FOREGROUND=1 bash deploy_vp_judge.sh     # do not detach
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Original experiments used Qwen3.5-35B-A3B as the judge.
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the local judge model directory}"
SERVED_NAME="${SERVED_NAME:-judge}"
GPUS="${GPUS:-7}"
VP_JUDGE_PORT="${VP_JUDGE_PORT:-8031}"
HOST="${HOST:-0.0.0.0}"
TP="${TP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_UTIL="${GPU_UTIL:-0.90}"
VLLM_BIN="${VLLM_BIN:-$(command -v vllm || true)}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/outputs/vp_judge/logs}"
FOREGROUND="${FOREGROUND:-0}"
GDN_PREFILL_BACKEND="${GDN_PREFILL_BACKEND:-triton}"

# vLLM JIT kernels use tools from the selected runtime environment.
if [[ -n "${VLLM_BIN}" ]]; then
  export PATH="$(dirname "${VLLM_BIN}"):${PATH}"
fi
# CUDA_HOME is CUDA 12.4 while Torch/FlashInfer are CUDA 12.8 builds; the
# FlashInfer GDN JIT therefore cannot compile on this image. Use vLLM's supported
# Triton/FLA backend, which is already installed and avoids that JIT path.
# This judge is BF16 (not FP8-quantized). deep_gemm is optional and absent from
# the image; disabling it avoids vLLM 0.19.1 warmup probing a missing backend.
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"

# NOTE: --reasoning-parser is deliberately NOT set. Splitting the reply at a
# bare </think> is what silently zeroed a whole RL evaluation once (README
# 23.8); the judge client concatenates reasoning+content anyway, and requests
# pin chat_template_kwargs.enable_thinking=false, so the score line always
# lands in the ordinary content channel.
# --enable-prefix-caching is a straight win here: every request shares the same
# system prompt and the same instruction block.

[[ -x "${VLLM_BIN}" ]] || { echo "vllm not found at ${VLLM_BIN}" >&2; exit 1; }
[[ -d "${MODEL_PATH}" ]] || { echo "model not found at ${MODEL_PATH}" >&2; exit 1; }
mkdir -p "${LOG_DIR}"

if (echo >"/dev/tcp/127.0.0.1/${VP_JUDGE_PORT}") >/dev/null 2>&1; then
  echo "port ${VP_JUDGE_PORT} is already in use on $(hostname); pick another VP_JUDGE_PORT or stop the owner:" >&2
  pgrep -af "vllm serve" >&2 || true
  exit 1
fi

LOG="${LOG_DIR}/serve_$(basename "${MODEL_PATH}")_${VP_JUDGE_PORT}.log"
ARGS=(
  serve "${MODEL_PATH}"
  --served-model-name "${SERVED_NAME}"
  --host "${HOST}"
  --port "${VP_JUDGE_PORT}"
  --tensor-parallel-size "${TP}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_UTIL}"
  --enable-prefix-caching
  --gdn-prefill-backend "${GDN_PREFILL_BACKEND}"
)

echo "host=$(hostname)  gpus=${GPUS}  model=${MODEL_PATH}  served-as=${SERVED_NAME}"
echo "endpoint=http://${HOST}:${VP_JUDGE_PORT}/v1   log=${LOG}"

if [[ "${FOREGROUND}" == "1" ]]; then
  exec env CUDA_VISIBLE_DEVICES="${GPUS}" "${VLLM_BIN}" "${ARGS[@]}"
fi

# setsid so closing the SSH session or the tmux window cannot take the engine
# down mid-evaluation; the trade-off is that it must be stopped explicitly.
setsid nohup env CUDA_VISIBLE_DEVICES="${GPUS}" "${VLLM_BIN}" "${ARGS[@]}" \
  >"${LOG}" 2>&1 </dev/null &
SERVER_PID=$!
echo "launched, waiting for readiness (model loading and torch.compile)"

for _ in $(seq 1 180); do
  if curl -s --noproxy '*' --max-time 4 "http://127.0.0.1:${VP_JUDGE_PORT}/v1/models" >/dev/null 2>&1; then
    echo "READY: http://$(hostname -I | awk '{print $1}'):${VP_JUDGE_PORT}/v1  (model '${SERVED_NAME}')"
    exit 0
  fi

  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "STARTUP FAILED -- tail of ${LOG}:" >&2
    tail -25 "${LOG}" >&2
    exit 1
  fi
  sleep 5
done
echo "timed out waiting for the server; see ${LOG}" >&2
exit 1
