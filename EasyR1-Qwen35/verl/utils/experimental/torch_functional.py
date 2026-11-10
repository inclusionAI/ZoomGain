# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch

try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

    _FLASH_ATTN_CROSS_ENTROPY_AVAILABLE = True
except ImportError:
    _FLASH_ATTN_CROSS_ENTROPY_AVAILABLE = False


# Qwen3.5-9B has a 248320 x 4096 LM head.  Its fused PPO backward otherwise
# creates several vocabulary-sized temporaries for dlogits and a full BF16
# dWeight temporary.  The 4B model has hidden size 2560 and deliberately does
# not match this guard, so its established backward path remains unchanged.
_QWEN35_9B_VOCAB_SIZE = 248320
_QWEN35_9B_HIDDEN_SIZE = 4096
_QWEN35_9B_VOCAB_GRAD_CHUNK_SIZE = 4096


def _is_qwen35_9b_vocab_weight(vocab_weights: torch.Tensor) -> bool:
    return (
        vocab_weights.ndim == 2
        and vocab_weights.shape[0] == _QWEN35_9B_VOCAB_SIZE
        and vocab_weights.shape[1] == _QWEN35_9B_HIDDEN_SIZE
    )


def _accumulate_qwen35_9b_vocab_gradient_(
    output: torch.Tensor,
    dlogits: torch.Tensor,
    hidden_states: torch.Tensor,
) -> None:
    """Accumulate the 9B LM-head gradient without a full-size temporary.

    Splitting rows of ``dlogits.T @ hidden_states`` is mathematically
    independent across vocabulary entries.  Keeping the matmul and add as two
    separate BF16 operations also preserves the existing rounding boundary.
    """

    for start in range(0, dlogits.shape[1], _QWEN35_9B_VOCAB_GRAD_CHUNK_SIZE):
        end = min(start + _QWEN35_9B_VOCAB_GRAD_CHUNK_SIZE, dlogits.shape[1])
        update = dlogits[:, start:end].t() @ hidden_states
        output[start:end].add_(update)
        del update


def _build_qwen35_9b_logprob_gradient_in_place_(
    probs: torch.Tensor,
    dlog_probs: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Turn softmax probabilities into the log-prob gradient in place.

    For each token, d(log p_label) / d(logits) is ``one_hot(label) - probs``.
    Reusing ``probs`` avoids allocating the full-size one-hot, subtraction,
    and multiplication outputs.  This helper is used only for the exact 9B
    LM-head shape when entropy has no gradient.
    """

    probs.neg_()
    label_updates = torch.ones_like(input_ids, dtype=probs.dtype).unsqueeze(-1)
    probs.scatter_add_(-1, input_ids.unsqueeze(-1), label_updates)
    probs.mul_(dlog_probs.to(torch.float32).unsqueeze(-1))
    return probs


def _fused_linear_for_ppo_fwd(
    hidden_states: torch.FloatTensor,
    vocab_weights: torch.FloatTensor,
    input_ids: torch.LongTensor,
    temperature: float = 1.0,
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    logits = (hidden_states @ vocab_weights.t()) / temperature
    orig_dtype = logits.dtype
    logits = logits.to(torch.float32)

    probs = logits.softmax(dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)

    if _FLASH_ATTN_CROSS_ENTROPY_AVAILABLE:
        per_token_entropy_loss = cross_entropy_loss(logits, input_ids)[0]
        token_log_probs = -per_token_entropy_loss
    else:
        # Fallback to original PyTorch implementation
        log_probs = logits.log_softmax(dim=-1)
        token_log_probs = log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)

    assert token_log_probs.dtype == torch.float32
    return token_log_probs, entropy.to(orig_dtype)


def _fused_linear_for_ppo_bwd(
    dlog_probs: Optional[torch.FloatTensor],
    dentropy: Optional[torch.FloatTensor],
    hidden_states: torch.FloatTensor,
    vocab_weights: torch.FloatTensor,
    input_ids: torch.LongTensor,
    temperature: float = 1.0,
    vocab_gradient_buffer: Optional[torch.FloatTensor] = None,
) -> tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
    logits = (hidden_states @ vocab_weights.t()) / temperature
    orig_dtype = logits.dtype
    logits = logits.to(torch.float32)

    probs = logits.softmax(dim=-1)

    use_qwen35_9b_in_place_logprob_grad = (
        dlog_probs is not None and dentropy is None and _is_qwen35_9b_vocab_weight(vocab_weights)
    )
    if use_qwen35_9b_in_place_logprob_grad:
        # The PPO launchers set calculate_entropy=false.  Once softmax has
        # completed, logits are no longer needed and probs can become dlogits.
        del logits
        dlogits = _build_qwen35_9b_logprob_gradient_in_place_(
            probs=probs,
            dlog_probs=dlog_probs,
            input_ids=input_ids,
        )
    else:
        dlogits = 0

        # Gradient from log_probs
        if dlog_probs is not None:
            one_hot_input = torch.zeros_like(logits).scatter_(-1, input_ids.unsqueeze(-1), 1)
            dlogits += dlog_probs.to(torch.float32).unsqueeze(-1) * (one_hot_input - probs)

        # Gradient from entropy
        if dentropy is not None:
            log_probs = logits.log_softmax(dim=-1)
            entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
            dlogits += probs * (log_probs + entropy.unsqueeze(-1)) * (-dentropy.unsqueeze(-1))

    dlogits = dlogits.to(orig_dtype) / temperature

    dhidden_states = dlogits @ vocab_weights
    if vocab_gradient_buffer is None:
        dvocab_weights = dlogits.t() @ hidden_states
    else:
        _accumulate_qwen35_9b_vocab_gradient_(
            output=vocab_gradient_buffer,
            dlogits=dlogits,
            hidden_states=hidden_states,
        )
        dvocab_weights = None

    return dhidden_states, dvocab_weights


class FusedLinearForPPOFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.FloatTensor,
        vocab_weights: torch.FloatTensor,
        input_ids: torch.LongTensor,
        temperature: float = 1.0,
        chunk_size: int = 512,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        ctx.set_materialize_grads(False)

        # Cast to a 2D tensor of the shape [T, D] for ease of working
        orig_ndim = hidden_states.ndim
        assert orig_ndim in (2, 3), f"Invalid hidden_states shape, received {hidden_states.shape}"

        orig_batch_size = -1
        if orig_ndim == 3:
            assert input_ids.ndim == 2, f"input_ids shape doesn't match, {hidden_states.shape} {input_ids.shape}"
            orig_batch_size = hidden_states.shape[0]
            hidden_states_requires_grad = hidden_states.requires_grad
            hidden_states = hidden_states.flatten(0, 1)
            hidden_states.requires_grad_(hidden_states_requires_grad)
            input_ids = input_ids.flatten(0, 1)

        T = hidden_states.shape[0]

        # Allocate memory for outputs
        output_requires_grad = hidden_states.requires_grad or vocab_weights.requires_grad
        # Logits are upcasted to fp32 before computing log_probs, which are also fp32
        log_probs = torch.zeros(T, device=hidden_states.device, dtype=torch.float32, requires_grad=output_requires_grad)
        entropy = hidden_states.new_zeros(T, requires_grad=output_requires_grad)

        # Perform forward one chunk at a time
        for chunk_start in range(0, T, chunk_size):
            chunk_end = min(chunk_start + chunk_size, T)

            chunk_log_probs, chunk_entropy = _fused_linear_for_ppo_fwd(
                hidden_states=hidden_states[chunk_start:chunk_end],
                vocab_weights=vocab_weights,
                input_ids=input_ids[chunk_start:chunk_end],
                temperature=temperature,
            )
            log_probs[chunk_start:chunk_end] = chunk_log_probs
            entropy[chunk_start:chunk_end] = chunk_entropy

        # Cast the output back to the original input dimension
        if orig_ndim == 3:
            log_probs = log_probs.view(orig_batch_size, -1)
            entropy = entropy.view(orig_batch_size, -1)

        ctx.save_for_backward(hidden_states, vocab_weights, input_ids)
        ctx.orig_batch_size = orig_batch_size
        ctx.orig_ndim = orig_ndim
        ctx.temperature = temperature
        ctx.chunk_size = chunk_size

        return log_probs, entropy

    @staticmethod
    def backward(ctx, dlog_probs: Optional[torch.FloatTensor], dentropy: Optional[torch.FloatTensor]):
        assert dlog_probs is not None or dentropy is not None

        hidden_states, vocab_weights, input_ids = ctx.saved_tensors
        orig_batch_size = ctx.orig_batch_size
        orig_ndim = ctx.orig_ndim
        temperature = ctx.temperature
        chunk_size = ctx.chunk_size

        # Here orig_ndim refers to the orig_ndim of hidden_states
        if orig_ndim == 3:
            if dlog_probs is not None:
                dlog_probs = dlog_probs.flatten()
            if dentropy is not None:
                dentropy = dentropy.flatten()

        T = hidden_states.shape[0]

        # Allocate memory for outputs
        dhidden_states = None
        if hidden_states.requires_grad:
            dhidden_states = torch.zeros_like(hidden_states)
        dvocab_weights = None
        if vocab_weights.requires_grad:
            dvocab_weights = torch.zeros_like(vocab_weights)
        use_qwen35_9b_vocab_grad_chunking = vocab_weights.requires_grad and _is_qwen35_9b_vocab_weight(
            vocab_weights
        )

        # Perform backward one chunk at a time
        for chunk_start in range(0, T, chunk_size):
            chunk_end = min(chunk_start + chunk_size, T)
            chunk_dlog_probs = None
            if dlog_probs is not None:
                chunk_dlog_probs = dlog_probs[chunk_start:chunk_end]
            chunk_dentropy = None
            if dentropy is not None:
                chunk_dentropy = dentropy[chunk_start:chunk_end]

            h, v = _fused_linear_for_ppo_bwd(
                dlog_probs=chunk_dlog_probs,
                dentropy=chunk_dentropy,
                hidden_states=hidden_states[chunk_start:chunk_end],
                vocab_weights=vocab_weights,
                input_ids=input_ids[chunk_start:chunk_end],
                temperature=temperature,
                vocab_gradient_buffer=dvocab_weights if use_qwen35_9b_vocab_grad_chunking else None,
            )

            if hidden_states.requires_grad:
                dhidden_states[chunk_start:chunk_end] += h
            if vocab_weights.requires_grad and v is not None:
                dvocab_weights += v

        # Cast the output back to the original input dimension
        if orig_ndim == 3 and hidden_states.requires_grad:
            hidden_size = hidden_states.shape[-1]
            dhidden_states = dhidden_states.view(orig_batch_size, -1, hidden_size)

        return (
            dhidden_states,  # hidden_states
            dvocab_weights,  # vocab_weights
            None,  # input_ids
            None,  # temperature
            None,  # chunk_size
        )


class FusedLinearForPPO(torch.nn.Module):
    def __init__(self, chunk_size: int = 512):
        super().__init__()

        self.chunk_size = chunk_size

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        vocab_weights: torch.FloatTensor,
        input_ids: torch.LongTensor,
        temperature: float = 1.0,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        input_ids = input_ids.to(torch.int64)
        return FusedLinearForPPOFunction.apply(
            hidden_states,
            vocab_weights,
            input_ids,
            temperature,
            self.chunk_size,
        )
