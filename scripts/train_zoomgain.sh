#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EASYR1_DIR="${ROOT}/EasyR1-Qwen35"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BASE_MODEL="${BASE_MODEL:?Set BASE_MODEL to the local Qwen3.5 model directory}"
TRAIN_PARQUET="${TRAIN_PARQUET:?Set TRAIN_PARQUET to the prepared ZoomGain train.parquet}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-zoomgain_qwen35}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/${EXPERIMENT_NAME}}"
BASE_MODEL="$(cd "${BASE_MODEL}" && pwd)"
TRAIN_PARQUET="$("${PYTHON_BIN}" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${TRAIN_PARQUET}")"
OUTPUT_DIR="$("${PYTHON_BIN}" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${OUTPUT_DIR}")"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${NUM_GPUS:-${#GPU_ARRAY[@]}}"
[[ "${NUM_GPUS}" -eq "${#GPU_ARRAY[@]}" ]] || { echo "NUM_GPUS does not match CUDA_VISIBLE_DEVICES" >&2; exit 2; }
FREEZE_VISION_TOWER=false
KEEP_VISION_MERGER_TRAINABLE=false
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
ROLLOUT_N="${ROLLOUT_N:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-2}"
LOGPROB_MICRO_BATCH_SIZE="${LOGPROB_MICRO_BATCH_SIZE:-4}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-49152}"
LOGPROB_MAX_TOKEN_LEN_PER_GPU="${LOGPROB_MAX_TOKEN_LEN_PER_GPU:-${PPO_MAX_TOKEN_LEN_PER_GPU}}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
ROLLOUT_AGENT_WORKERS="${ROLLOUT_AGENT_WORKERS:-16}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-25000}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
FORMAT_REWARD_WEIGHT="${FORMAT_REWARD_WEIGHT:-0.0}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
SAVE_FREQ="${SAVE_FREQ:-25}"
# Retain one resumable actor checkpoint; keep all validated HF exports.
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-1}"
CHECKPOINT_SAVE_CONTENTS="${CHECKPOINT_SAVE_CONTENTS:-['model','optimizer','extra','hf_model']}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-true}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-true}"
USE_ROLLOUT_LOG_PROBS="${USE_ROLLOUT_LOG_PROBS:-true}"
USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-true}"
RESPONSE_ONLY_LOGITS="${RESPONSE_ONLY_LOGITS:-true}"
# Actor master-weight/Adam-state dtype. Must stay fp32: at lr=1e-6 the AdamW
# step (~1e-6) is far below the BF16 ulp at typical |w|~1e-2 (~3e-5), so BF16
# masters silently round away nearly all updates (the pre-2026-08-04 10k run
# moved only ~2.6% of ViT elements vs ~16% under fp32 masters on Qwen3-VL).
# fsdp_config.dtype below stays bfloat16 (mixed-precision compute).
ACTOR_MODEL_DTYPE="${ACTOR_MODEL_DTYPE:-fp32}"
PARAM_OFFLOAD="${PARAM_OFFLOAD:-false}"
OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-false}"
OFFLOAD_POLICY="${OFFLOAD_POLICY:-false}"
RESUME_MODE="${RESUME_MODE:-auto}"


export PYTHONPATH="${ROOT}:${EASYR1_DIR}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES
export ZOOMGAIN_MAX_RESPONSE_TOKENS="${MAX_RESPONSE_LENGTH}"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY grpc_proxy GRPC_PROXY
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_GDN_PREFILL_BACKEND="${VLLM_GDN_PREFILL_BACKEND:-triton}"
if [[ ! -x "${CUDA_HOME:-/nonexistent}/bin/nvcc" ]]; then
    TORCH_CUDA="$("${PYTHON_BIN}" -c 'import torch; print(torch.version.cuda or "")')"
    for candidate in "/usr/local/cuda-${TORCH_CUDA}" /usr/local/cuda; do
        if [[ -x "${candidate}/bin/nvcc" ]]; then export CUDA_HOME="${candidate}"; break; fi
    done
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi
RUN_TEMP="$(mktemp -d /tmp/zoomgain.XXXXXXXX)"
PREPARED_PARQUET="${RUN_TEMP}/train.parquet"
trap 'rm -f -- "${PREPARED_PARQUET}"' EXIT
export TMPDIR="${RUN_TEMP}/tmp" RAY_TMPDIR="${RUN_TEMP}/ray" VERL_ZMQ_IPC_DIR="${RUN_TEMP}/ipc"
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}" "${VERL_ZMQ_IPC_DIR}" "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/rollouts"
"${PYTHON_BIN}" "${ROOT}/zoomgain/prepare_data.py" --input "${TRAIN_PARQUET}" --output "${PREPARED_PARQUET}"
if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then exit 0; fi
RUN_LOG="${OUTPUT_DIR}/logs/train_$(date +%Y%m%d_%H%M%S).log"
cd "${EASYR1_DIR}"
"${PYTHON_BIN}" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    algorithm.rollout_correction.bypass_mode="${USE_ROLLOUT_LOG_PROBS}" \
    algorithm.rollout_correction.loss_type=ppo_clip \
    data.train_files="${PREPARED_PARQUET}" \
    data.val_files="${PREPARED_PARQUET}" \
    data.prompt_key=prompt \
    data.reward_fn_key=data_source \
    data.image_key=images \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.train_max_samples="${TRAIN_MAX_SAMPLES}" \
    data.val_max_samples=8 \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=false \
    data.truncation=error \
    data.image_patch_size=16 \
    data.shuffle=true \
    data.seed=42 \
    data.dataloader_num_workers="${DATALOADER_NUM_WORKERS}" \
    data.return_multi_modal_inputs=true \
    +data.apply_chat_template_kwargs.enable_thinking=false \
    reward.custom_reward_function.path="${ROOT}/zoomgain/reward.py" \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs.format_reward_weight="${FORMAT_REWARD_WEIGHT}" \
    reward.reward_manager.name=naive \
    reward.num_workers="${NUM_GPUS}" \
    actor_rollout_ref.model.path="${BASE_MODEL}" \
    actor_rollout_ref.model.trust_remote_code=false \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.freeze_vision_tower="${FREEZE_VISION_TOWER}" \
    actor_rollout_ref.model.keep_vision_merger_trainable="${KEEP_VISION_MERGER_TRAINABLE}" \
    actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}" \
    actor_rollout_ref.model.use_fused_kernels=true \
    actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
    actor_rollout_ref.model.response_only_logits="${RESPONSE_ONLY_LOGITS}" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.freeze_vision_tower="${FREEZE_VISION_TOWER}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=1e-2 \
    "actor_rollout_ref.actor.optim.override_optimizer_config={foreach: false}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.calculate_entropy=false \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.3 \
    actor_rollout_ref.actor.use_torch_compile="${USE_TORCH_COMPILE}" \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size="${NUM_GPUS}" \
    actor_rollout_ref.actor.fsdp_config.model_dtype="${ACTOR_MODEL_DTYPE}" \
    actor_rollout_ref.actor.fsdp_config.dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=true \
    actor_rollout_ref.actor.fsdp_config.param_offload="${PARAM_OFFLOAD}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${OPTIMIZER_OFFLOAD}" \
    actor_rollout_ref.actor.fsdp_config.offload_policy="${OFFLOAD_POLICY}" \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile="${USE_TORCH_COMPILE}" \
    actor_rollout_ref.actor.checkpoint.save_contents="${CHECKPOINT_SAVE_CONTENTS}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.calculate_log_probs="${USE_ROLLOUT_LOG_PROBS}" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    +actor_rollout_ref.rollout.limit_images=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.rollout.enable_prefix_caching=false \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.enforce_eager=false \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOGPROB_MICRO_BATCH_SIZE}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${LOGPROB_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.rollout.agent.num_workers="${ROLLOUT_AGENT_WORKERS}" \
    trainer.project_name=zoomgain \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger="['console']" \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.val_before_train=false \
    trainer.test_freq=-1 \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP}" \
    trainer.default_local_dir="${OUTPUT_DIR}/checkpoints" \
    trainer.rollout_data_dir="${OUTPUT_DIR}/rollouts" \
    trainer.resume_mode="${RESUME_MODE}" \
    hydra.run.dir="${OUTPUT_DIR}/logs/hydra" \
    trainer.use_v1=true \
    "$@" 2>&1 | tee "${RUN_LOG}"

echo "ZOOMGAIN_TRAINING_COMPLETE output=${OUTPUT_DIR}"
