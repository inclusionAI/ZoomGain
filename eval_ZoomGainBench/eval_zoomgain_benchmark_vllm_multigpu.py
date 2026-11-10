#!/usr/bin/env python3
"""Evaluate ZoomGainBench with local vLLM, one engine per tensor-parallel GPU group."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import queue
import re
import signal
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

import benchmark_core as api_eval


LOCAL_ENDPOINT_ID = "local-vllm://replicated"
# Verbatim R1_SYSTEM_PROMPT from OpenGVLab's recommended VLMEvalKit adapter:
# https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/vlm/internvl/internvl_chat.py
R1_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different \
angles, potential solutions, and reason through the problem step-by-step. \
Enclose this entire thinking process within <think> and </think> tags.
2. After the thinking section, provide a clear, concise, and direct answer to \
the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the \
query. The final answer should be standalone and not reference the thinking \
section.
""".strip()

GENERATION_PROFILES = ("qwen3_nothinking", "default", "kimi_thinking", "kimi_nothinking", "internvl_thinking")
MODEL_ADAPTER_PROFILES = ("mimo_nothinking", "minicpm45_nothinking", "minicpm46_instruct")
GENERATION_PROFILES += MODEL_ADAPTER_PROFILES


def system_prompt_for_profile(profile: str) -> str:
    return R1_SYSTEM_PROMPT if profile == "internvl_thinking" else api_eval.ZWZ_SYSTEM_PROMPT


@dataclass
class Replica:
    gpu: str
    port: int
    endpoint: str
    process: subprocess.Popen
    log_handle: Any
    log_path: Path


_LIVE_REPLICAS: list[Replica] = []
_THREAD_LOCAL = threading.local()


def parse_gpus(spec: str) -> list[str]:
    compact = str(spec).strip()
    if not compact:
        raise ValueError("--gpus must not be empty")
    if re.search(r"[,\s]", compact):
        gpus = [part for part in re.split(r"[,\s]+", compact) if part]
    elif compact.isdigit():
        gpus = list(compact)
    else:
        raise ValueError(
            f"invalid --gpus {spec!r}; use compact '4567' or '4,5,6,7'"
        )
    if any(not gpu.isdigit() for gpu in gpus):
        raise ValueError(f"invalid GPU identifier in --gpus {spec!r}")
    if len(gpus) != len(set(gpus)):
        raise ValueError(f"duplicate GPU identifier in --gpus {spec!r}")
    return gpus


def group_gpus(gpus: list[str], tp: int) -> list[str]:
    if tp < 1:
        raise ValueError("--tp must be a positive integer")
    if not gpus or len(gpus) % tp:
        raise ValueError(
            f"GPU count ({len(gpus)}) must be divisible by --tp ({tp})"
        )
    return [",".join(gpus[i:i + tp]) for i in range(0, len(gpus), tp)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument(
        "--run-name",
        help=(
            "Optional output/model label. Default: the final component of "
            "--model-path, e.g. Qwen3.5-9B."
        ),
    )
    parser.add_argument("--gpus", "--gpu", dest="gpus", required=True, help="Compact 4567 or comma list")
    parser.add_argument("--tp", type=int, default=1, help="GPUs per replica; GPU count must be divisible by TP")
    parser.add_argument("--port", type=int, required=True, help="First replica port")
    parser.add_argument("--annotations", required=True, type=Path, help="ZoomGainBench annotations.jsonl")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--generation-profile",
        choices=GENERATION_PROFILES,
        default="qwen3_nothinking",
        help=(
            "qwen3_nothinking sends chat_template_kwargs.enable_thinking=false. "
            "internvl_thinking uses the verbatim official R1 system prompt."
        ),
    )
    parser.add_argument("--lan_type", "--lan-type", choices=("all", "cn", "en"), default="all", dest="lan_type")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument(
        "--rollouts",
        type=int,
        default=1,
        help="Number of sampled completions requested per question (OpenAI n).",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--per-engine-concurrency", type=int, default=12)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--minicpm-max-slice-nums", type=int, help="Native image slice budget: default 9 for 4.5, 36 for 4.6")
    parser.add_argument("--minicpm-downsample-mode", choices=("4x", "16x"), help="4.6 only; default 4x for fine-detail evaluation")
    parser.add_argument(
        "--enable-chunked-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable vLLM chunked prefill (default: enabled); --no-enable-chunked-prefill explicitly disables it.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--vllm-bin", default=shutil.which("vllm") or "vllm")
    parser.add_argument(
        "--gdn-prefill-backend",
        choices=("triton", "flashinfer"),
        default="triton",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--preview-samples", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-missing-tags-from", type=Path,
                        help="Read a completed run, rerun only missing-tag language tasks, and merge in a NEW output directory")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def require_model(path: Path) -> Path:
    model = path.expanduser().resolve()
    if not model.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model}")
    if not (model / "config.json").is_file():
        raise FileNotFoundError(f"model is missing config.json: {model}")
    weight_files = sorted(model.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"model has no .safetensors weights: {model}")
    return model


def model_signature(model: Path) -> str:
    """Cheap mutation guard for resume without hashing every weight byte."""

    digest = hashlib.sha256()
    for filename in (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer_config.json",
        "processor_config.json",
        "preprocessor_config.json",
        "chat_template.json",
    ):
        path = model / filename
        if path.is_file():
            digest.update(filename.encode())
            digest.update(path.read_bytes())
    for path in sorted(model.glob("*.safetensors")):
        stat_result = path.stat()
        digest.update(path.name.encode())
        digest.update(str(stat_result.st_size).encode())
        digest.update(str(stat_result.st_mtime_ns).encode())
    return digest.hexdigest()


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not cleaned:
        raise ValueError("--run-name must contain a usable character")
    return cleaned


def resolve_run_name(value: str | None, model: Path) -> str:
    run_name = value if value is not None else model.name
    if not run_name:
        raise ValueError(
            "cannot infer run name from --model-path; pass --run-name explicitly"
        )
    safe_name(run_name)
    return run_name


def served_model_name(run_name: str, generation_profile: str) -> str:
    prefix = "Qwen3-local" if generation_profile == "qwen3_nothinking" else "local"
    return f"{prefix}-{safe_name(run_name)[:80]}"


def ensure_ports_free(ports: list[int]) -> None:
    sockets: list[socket.socket] = []
    try:
        for port in ports:
            handle = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            handle.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            handle.bind(("127.0.0.1", port))
            sockets.append(handle)
    except OSError as exc:
        raise RuntimeError(f"vLLM replica port is unavailable: {exc}") from exc
    finally:
        for handle in sockets:
            handle.close()


def open_unique_log(path: Path) -> tuple[Path, Any]:
    attempt = 1
    while True:
        candidate = (
            path
            if attempt == 1
            else path.with_name(f"{path.stem}_attempt{attempt}{path.suffix}")
        )
        try:
            return candidate, candidate.open("x", encoding="utf-8")
        except FileExistsError:
            attempt += 1


def vllm_serve_command(
    *,
    vllm_bin: str,
    model: Path,
    served_name: str,
    port: int,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        vllm_bin,
        "serve",
        str(model),
        "--served-model-name",
        served_name,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(getattr(args, "tp", 1)),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--trust-remote-code",
        "--gdn-prefill-backend",
        args.gdn_prefill_backend,
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens or args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
    ]
    command.append("--enable-chunked-prefill" if args.enable_chunked_prefill else "--no-enable-chunked-prefill")
    if args.generation_profile == "minicpm46_instruct":
        # New-runtime option compatibility; this flag was removed upstream.
        index = command.index("--gdn-prefill-backend")
        del command[index:index + 2]
    if getattr(args, "mm_processor_kwargs", None):
        command.extend(["--mm-processor-kwargs", json.dumps(args.mm_processor_kwargs)])
    if args.generation_profile == "qwen3_nothinking":
        command.extend(["--reasoning-parser", "qwen3"])
    elif args.generation_profile.startswith("kimi_"):
        command.extend(["--reasoning-parser", "kimi_k2", "--mm-encoder-tp-mode", "data"])
    return command


def start_replicas(
    model: Path,
    served_name: str,
    gpus: list[str],
    first_port: int,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[Replica]:
    gpu_groups = group_gpus(gpus, getattr(args, "tp", 1))
    ports = [first_port + offset for offset in range(len(gpu_groups))]
    ensure_ports_free(ports)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    replicas: list[Replica] = []
    try:
        for gpu, port in zip(gpu_groups, ports):
            log_path, log_handle = open_unique_log(
                logs_dir / f"serve_{safe_name(args.run_name)}_gpu{gpu}_{port}.log"
            )
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
            if args.generation_profile == "minicpm45_nothinking":
                compat = Path(__file__).resolve().parent / "minicpm45_compat"
                if not (compat / "sitecustomize.py").is_file():
                    raise RuntimeError("Missing MiniCPM 4.5 tokenizer compatibility helper")
                environment["ZOOMGAIN_MINICPM45_COMPAT"] = "1"
                environment["PYTHONPATH"] = str(compat) + os.pathsep + environment.get("PYTHONPATH", "")
            environment["PATH"] = (
                f"{Path(args.vllm_bin).parent}:{environment.get('PATH', '')}"
            )
            environment.setdefault("VLLM_USE_DEEP_GEMM", "0")
            try:
                process = subprocess.Popen(
                    vllm_serve_command(
                        vllm_bin=args.vllm_bin,
                        model=model,
                        served_name=served_name,
                        port=port,
                        args=args,
                    ),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    preexec_fn=os.setsid,
                )
            except BaseException:
                log_handle.close()
                raise
            replica = Replica(
                gpu=gpu,
                port=port,
                endpoint=f"http://127.0.0.1:{port}/v1",
                process=process,
                log_handle=log_handle,
                log_path=log_path,
            )
            replicas.append(replica)
            _LIVE_REPLICAS.append(replica)
            print(
                f"[{served_name}] launch gpu={gpu} port={port} log={log_path}",
                flush=True,
            )
    except BaseException:
        teardown_replicas(replicas)
        raise
    return replicas


def wait_replicas_ready(replicas: list[Replica], timeout: int) -> None:
    for replica in replicas:
        deadline = time.monotonic() + timeout
        url = f"{replica.endpoint}/models"
        while time.monotonic() < deadline:
            if replica.process.poll() is not None:
                raise RuntimeError(
                    f"vLLM gpu={replica.gpu} exited with code "
                    f"{replica.process.returncode}; inspect {replica.log_path}"
                )
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200 and response.json().get("data"):
                    print(
                        f"  ready gpu={replica.gpu} endpoint={replica.endpoint}",
                        flush=True,
                    )
                    break
            except Exception:
                pass
            time.sleep(5)
        else:
            raise TimeoutError(
                f"vLLM gpu={replica.gpu} port={replica.port} not ready "
                f"within {timeout}s; inspect {replica.log_path}"
            )


def teardown_replicas(replicas: list[Replica]) -> None:
    for replica in replicas:
        if replica.process.poll() is None:
            try:
                os.killpg(os.getpgid(replica.process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if all(replica.process.poll() is not None for replica in replicas):
            break
        time.sleep(0.2)
    for replica in replicas:
        if replica.process.poll() is None:
            try:
                os.killpg(os.getpgid(replica.process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                replica.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        replica.log_handle.close()
        try:
            _LIVE_REPLICAS.remove(replica)
        except ValueError:
            pass


def handle_terminating_signal(signum: int, _frame: Any) -> None:
    print(
        f"\nreceived {signal.Signals(signum).name}; releasing "
        f"{len(_LIVE_REPLICAS)} vLLM replica(s)",
        flush=True,
    )
    teardown_replicas(list(_LIVE_REPLICAS))
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def install_signal_handlers() -> None:
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, handle_terminating_signal)


def thread_session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        _THREAD_LOCAL.session = session
    return session


def call_rollout_api(
    *,
    session: requests.Session,
    endpoint: str,
    model: str,
    row: dict[str, Any],
    max_tokens: int,
    timeout: int,
    max_attempts: int,
    rollouts: int,
    temperature: float,
    top_p: float,
    top_k: int,
    generation_profile: str = "qwen3_nothinking",
    mm_processor_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Request and score multiple completions while sharing one image prefill."""

    started = time.monotonic()
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt_for_profile(generation_profile)},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": api_eval.image_data_url(
                                Path(str(row["image_path"]))
                            )
                        },
                    },
                    {"type": "text", "text": row["question"]},
                ],
            },
        ],
        "n": rollouts,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "chat_template_kwargs": ({"thinking": generation_profile == "kimi_thinking"}
                                 if generation_profile.startswith("kimi_") else {"enable_thinking": False}),
    }
    if generation_profile == "internvl_thinking":
        payload.pop("chat_template_kwargs")
    if generation_profile == "mimo_nothinking":
        payload.pop("chat_template_kwargs")
        payload["messages"][1]["content"][-1]["text"] += "\n/no_think"
    if mm_processor_kwargs:
        payload["mm_processor_kwargs"] = mm_processor_kwargs
    if generation_profile == "minicpm46_instruct":
        payload["stop_token_ids"] = [248044, 248046]
    elif generation_profile == "minicpm45_nothinking":
        payload["stop_token_ids"] = [1, 151645]
    last_error = "unknown request failure"
    response_body: dict[str, Any] | None = None
    attempts_used = 0
    for attempt in range(1, max_attempts + 1):
        attempts_used = attempt
        response: requests.Response | None = None
        try:
            response = session.post(
                endpoint,
                json=payload,
                headers={
                    "Authorization": "Bearer local-vllm",
                    "Content-Type": "application/json",
                },
                timeout=timeout,
            )
            if response.status_code >= 400:
                last_error = f"HTTP {response.status_code}: {response.text[:2000]}"
                if (
                    response.status_code in api_eval.RETRYABLE_HTTP_CODES
                    and attempt < max_attempts
                ):
                    time.sleep(api_eval.retry_delay(response, attempt))
                    continue
                break
            parsed = response.json()
            choices = parsed.get("choices") if isinstance(parsed, dict) else None
            if not isinstance(choices, list) or len(choices) != rollouts:
                last_error = (
                    f"expected {rollouts} response choices, got "
                    f"{len(choices) if isinstance(choices, list) else 'invalid'}"
                )
                if attempt < max_attempts:
                    time.sleep(api_eval.retry_delay(response, attempt))
                    continue
                break
            response_body = parsed
            break
        except Exception as exc:  # noqa: BLE001 - preserve diagnostics
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_attempts:
                time.sleep(api_eval.retry_delay(response, attempt))

    common = {
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "attempts": attempts_used,
        "rollouts_requested": rollouts,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "chat_template_kwargs": payload.get("chat_template_kwargs"),
        "mm_processor_kwargs": payload.get("mm_processor_kwargs"),
        "stop_token_ids": payload.get("stop_token_ids"),
        "question_suffix": "/no_think" if generation_profile == "mimo_nothinking" else None,
    }
    if response_body is None:
        return {
            **common,
            "status": "request_error",
            "error": last_error,
            "score": 0.0,
            "full_credit": False,
            "answer_tag_valid": False,
        }

    rollout_rows = []
    try:
        for fallback_index, choice in enumerate(response_body["choices"]):
            message = choice["message"]
            content = api_eval.stringify_message_part(message.get("content"))
            reasoning = api_eval.stringify_message_part(
                message.get("reasoning_content")
                or message.get("reasoning")
                or message.get("analysis")
            )
            combined = "\n".join(
                part for part in (reasoning, content) if part
            ).strip()
            tagged_answer, tag_count = api_eval.extract_last_answer_tag(combined)
            scored = api_eval.score_answer(tagged_answer, row)
            rollout_rows.append({
                "rollout_index": int(choice.get("index", fallback_index)),
                "finish_reason": choice.get("finish_reason"),
                "content": content,
                "reasoning_content": reasoning,
                "content_chars": len(content),
                "reasoning_chars": len(reasoning),
                "answer_tag_count": tag_count,
                "answer_tag_valid": tagged_answer is not None,
                "answer_tag_answer": tagged_answer,
                **scored,
            })
    except Exception as exc:  # noqa: BLE001 - malformed provider response
        return {
            **common,
            "status": "malformed_response",
            "error": f"{type(exc).__name__}: {exc}",
            "response_body": response_body,
            "score": 0.0,
            "full_credit": False,
            "answer_tag_valid": False,
        }

    reward_sum = sum(float(item["score"]) for item in rollout_rows)
    full_credit_count = sum(bool(item["full_credit"]) for item in rollout_rows)
    valid_tag_count = sum(bool(item["answer_tag_valid"]) for item in rollout_rows)
    result = {
        **common,
        "status": "completed",
        "response_model": response_body.get("model"),
        "response_id": response_body.get("id"),
        "rollouts": rollout_rows,
        "rollouts_completed": len(rollout_rows),
        "reward_sum": reward_sum,
        "full_credit_count": full_credit_count,
        "valid_answer_tag_count": valid_tag_count,
        # Keep the existing evaluator's aggregate summary on a 0..1 scale.
        "score": reward_sum / rollouts,
        "full_credit": full_credit_count == rollouts,
        "answer_tag_valid": valid_tag_count == rollouts,
        "usage": response_body.get("usage"),
    }
    result[f"reward_sum_{rollouts}"] = reward_sum
    result[f"full_credit_count_{rollouts}"] = full_credit_count
    result[f"valid_answer_tag_count_{rollouts}"] = valid_tag_count
    if rollouts in (8, 16):
        scale_to_16 = 16 / rollouts
        result["equivalent_reward_sum_16"] = reward_sum * scale_to_16
        result["equivalent_full_credit_count_16"] = int(
            full_credit_count * scale_to_16
        )
    return result


def glm_box_fallback(result: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Keep raw text intact; accept a complete GLM box only without answer tags."""
    combined = "\n".join(str(result.get(k) or "") for k in ("reasoning_content", "content"))
    answer, count = api_eval.extract_last_answer_tag(combined)
    result["strict_answer_tag_valid"] = answer is not None
    if answer is not None or result.get("status") not in (None, "completed"):
        return result
    boxes = re.findall(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", combined, re.S)
    if not boxes or not boxes[-1].strip():
        return result
    answer = boxes[-1].strip()
    result.update(api_eval.score_answer(answer, row))
    result.update(answer_tag_valid=True, answer_tag_answer=answer,
                  answer_tag_count=count, answer_extraction_source="glm_box_fallback",
                  glm_box_count=len(boxes))
    return result


def load_retry_source(source: Path, expected: dict[str, Any], task_ids: set[str]):
    saved = json.loads((source / "run_config.json").read_text())
    comparable = dict(expected, max_tokens=saved["max_tokens"])
    validate_local_resume_config(source / "run_config.json", comparable)
    rows = {}
    for line in (source / "responses.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["task_id"]] = row
    if set(rows) != task_ids or any(r.get("status") != "completed" for r in rows.values()):
        raise RuntimeError("Retry source must contain all completed language tasks")
    targets = sorted(k for k, r in rows.items() if not r.get("answer_tag_valid"))
    return {k:r for k,r in rows.items() if k not in targets}, targets, saved


def result_for_task(
    row: dict[str, Any],
    *,
    run_name: str,
    served_name: str,
    replica: Replica,
    slot_id: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    try:
        if (getattr(args, "rollouts", 1) == 1
                and not args.generation_profile.startswith("kimi_")
                and args.generation_profile not in MODEL_ADAPTER_PROFILES
                and args.generation_profile != "internvl_thinking"):
            api_result = api_eval.call_api(
                session=thread_session(),
                endpoint=f"{replica.endpoint}/chat/completions",
                api_key="local-vllm",
                model=served_name,
                row=row,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                max_attempts=args.max_attempts,
            )
        else:
            api_result = call_rollout_api(
                session=thread_session(),
                endpoint=f"{replica.endpoint}/chat/completions",
                model=served_name,
                row=row,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                max_attempts=args.max_attempts,
                rollouts=args.rollouts,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                generation_profile=args.generation_profile,
                mm_processor_kwargs=getattr(args, "mm_processor_kwargs", None),
            )
            if args.rollouts == 1:
                api_result = {**api_result, **api_result.get("rollouts", [{}])[0]}
                api_result.pop("rollouts", None)
        if getattr(args, "glm_box_fallback", False):
            if "rollouts" not in api_result:
                api_result = glm_box_fallback(api_result, row)
            else:
                items = [glm_box_fallback(item, row) for item in api_result["rollouts"]]
                n = len(items)
                reward = sum(float(item["score"]) for item in items)
                credit = sum(bool(item["full_credit"]) for item in items)
                valid = sum(bool(item["answer_tag_valid"]) for item in items)
                api_result.update(rollouts=items, reward_sum=reward, score=reward/n,
                                  full_credit_count=credit, full_credit=credit == n,
                                  valid_answer_tag_count=valid, answer_tag_valid=valid == n)
                api_result.update({f"reward_sum_{n}": reward, f"full_credit_count_{n}": credit,
                                   f"valid_answer_tag_count_{n}": valid})
                if n in (8, 16):
                    api_result.update(equivalent_reward_sum_16=reward*16/n,
                                      equivalent_full_credit_count_16=int(credit*16/n))
        return {
            "schema_version": 2,
            "annotation_index": row["_annotation_index"],
            "task_index": row["_task_index"],
            "task_id": row["_task_id"],
            "id": row["id"],
            "language": row["language"],
            "model": run_name,
            "backend": "local-vllm-replicated",
            "worker_id": slot_id,
            "key_slot": None,
            "replica_gpu": replica.gpu,
            "replica_port": replica.port,
            "question_type": row["question_type"],
            "question_subset": row["question_subset"],
            "answer_format": row["answer_format"],
            "question": row["question"],
            "ground_truth": row["answer"],
            "ground_truth_text": row["answer_text"],
            "image_path": row["image_path"],
            **api_result,
            "finished_at": api_eval.utc_now(),
        }
    except BaseException as exc:
        result = api_eval.error_result(row, run_name, slot_id, 0, exc)
        result.update(
            {
                "backend": "local-vllm-replicated",
                "key_slot": None,
                "replica_gpu": replica.gpu,
                "replica_port": replica.port,
            }
        )
        return result


def validate_local_resume_config(
    path: Path, expected: dict[str, Any]
) -> dict[str, Any]:
    saved = api_eval.validate_resume_config(path, expected)
    fields = (
        "backend",
        "model_path",
        "model_signature",
        "generation_profile",
        "served_model_name",
        "max_model_len",
        "max_num_batched_tokens",
        "system_prompt",
        "qwen3_enable_thinking",
        "image_payload",
        "rollouts_per_question",
        "temperature",
        "top_p",
        "top_k",
        "max_num_seqs",
        "enable_chunked_prefill",
    )
    mismatches = [field for field in fields if saved.get(field) != expected.get(field)]
    for field in ("mm_processor_kwargs", "model_adapter"):
        if saved.get(field) != expected.get(field):
            mismatches.append(field)
    if saved.get("tensor_parallel_size", 1) != expected.get("tensor_parallel_size", 1):
        mismatches.append("tensor_parallel_size")
    if mismatches:
        raise RuntimeError(
            "Resume local-vLLM configuration differs for: "
            + ", ".join(mismatches)
        )
    return saved


def preview_results(results: list[dict[str, Any]], count: int) -> None:
    if count <= 0:
        return
    selected: list[dict[str, Any]] = []
    for predicate in (
        lambda row: not row.get("answer_tag_valid"),
        lambda row: bool(row.get("full_credit")),
        lambda row: not bool(row.get("full_credit")),
    ):
        candidate = next(
            (row for row in results if row not in selected and predicate(row)), None
        )
        if candidate is not None:
            selected.append(candidate)
        if len(selected) >= count:
            break
    for row in results:
        if len(selected) >= count:
            break
        if row not in selected:
            selected.append(row)
    for index, row in enumerate(selected, start=1):
        compact = {
            "task_id": row.get("task_id"),
            "question_type": row.get("question_type"),
            "language": row.get("language"),
            "ground_truth": row.get("ground_truth"),
            "parsed_answer": row.get("parsed_answer"),
            "score": row.get("score"),
            "answer_tag_valid": row.get("answer_tag_valid"),
            "finish_reason": row.get("finish_reason"),
        }
        print(f"[preview {index}] {json.dumps(compact, ensure_ascii=False)}")


def main() -> int:
    args = build_parser().parse_args()
    if args.resume and args.overwrite:
        raise RuntimeError("--resume and --overwrite are mutually exclusive")
    positive = {
        "--max-tokens": args.max_tokens,
        "--timeout": args.timeout,
        "--max-attempts": args.max_attempts,
        "--per-engine-concurrency": args.per_engine_concurrency,
        "--max-model-len": args.max_model_len,
        "--startup-timeout": args.startup_timeout,
        "--rollouts": args.rollouts,
        "--max-num-seqs": args.max_num_seqs,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise RuntimeError(f"must be positive: {', '.join(invalid)}")
    if args.max_num_batched_tokens is not None and args.max_num_batched_tokens <= 0:
        raise RuntimeError("--max-num-batched-tokens must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise RuntimeError("--gpu-memory-utilization must be in (0, 1]")
    if args.limit is not None and args.limit <= 0:
        raise RuntimeError("--limit must be positive")
    if args.preview_samples < 0:
        raise RuntimeError("--preview-samples must be non-negative")
    if args.temperature < 0:
        raise RuntimeError("--temperature must be non-negative")
    if not 0 < args.top_p <= 1:
        raise RuntimeError("--top-p must be in (0, 1]")

    gpus = parse_gpus(args.gpus)
    gpu_groups = group_gpus(gpus, args.tp)
    model = require_model(args.model_path)
    model_cfg = json.loads((model / "config.json").read_text())
    profile = args.generation_profile
    args.mm_processor_kwargs = None
    if profile.startswith("minicpm"):
        expected_type = "minicpmv" if profile == "minicpm45_nothinking" else "minicpmv4_6"
        if model_cfg.get("model_type") != expected_type:
            raise RuntimeError(f"{profile} requires model_type={expected_type}")
        if profile == "minicpm45_nothinking" and str(model_cfg.get("version")) != "4.5":
            raise RuntimeError("minicpm45_nothinking requires MiniCPM-V 4.5")
        slices = args.minicpm_max_slice_nums or (9 if profile == "minicpm45_nothinking" else 36)
        if slices < 1 or args.minicpm_max_slice_nums == 0:
            raise RuntimeError("--minicpm-max-slice-nums must be positive")
        args.mm_processor_kwargs = {"max_slice_nums": slices}
        if profile == "minicpm46_instruct":
            args.mm_processor_kwargs["downsample_mode"] = args.minicpm_downsample_mode or "4x"
        elif args.minicpm_downsample_mode:
            raise RuntimeError("--minicpm-downsample-mode requires MiniCPM-V 4.6")
    elif args.minicpm_max_slice_nums is not None or args.minicpm_downsample_mode:
        raise RuntimeError("MiniCPM image options require a MiniCPM profile")
    if profile == "mimo_nothinking" and (model_cfg.get("model_type") != "qwen2_5_vl" or "mimo" not in model.name.lower()):
        raise RuntimeError("mimo_nothinking requires the MiMo-VL checkpoint")
    if model_cfg.get("model_type") == "kimi_k25" and not args.generation_profile.startswith("kimi_"):
        raise RuntimeError("Kimi requires --generation-profile kimi_thinking or kimi_nothinking")
    if "k2.7" in model.name.lower() and args.generation_profile == "kimi_nothinking":
        raise RuntimeError("Kimi-K2.7-Code supports thinking mode only")
    args.glm_box_fallback = str(model_cfg.get("model_type", "")).lower().startswith("glm")
    args.run_name = resolve_run_name(args.run_name, model)
    ports = [args.port + offset for offset in range(len(gpu_groups))]
    if any(not 1 <= port <= 65535 for port in ports):
        raise RuntimeError("replica port is outside 1..65535")
    served_name = served_model_name(args.run_name, args.generation_profile)
    annotations_path = args.annotations.expanduser().resolve()
    output_dir = args.output_dir or (
        annotations_path.parent / "eval_runs_localVLLM" / safe_name(args.run_name)
    )
    output_dir = output_dir.expanduser().resolve()
    annotations = api_eval.load_annotations(annotations_path, args.limit)
    if args.retry_missing_tags_from and args.output_dir is None:
        output_dir = output_dir.with_name(output_dir.name + f"_retry_missing_{args.max_tokens}")
    if args.retry_missing_tags_from and output_dir.resolve() == args.retry_missing_tags_from.resolve():
        raise RuntimeError("Retry output must differ from the original run directory")
    tasks = api_eval.expand_language_tasks(annotations, args.lan_type)
    annotation_sha = api_eval.sha256_file(annotations_path)
    selected_task_ids = {row["_task_id"] for row in tasks}
    if len(selected_task_ids) != len(tasks):
        raise RuntimeError("Duplicate language task IDs")

    resolved_max_batched = args.max_num_batched_tokens or args.max_model_len
    config = {
        "schema_version": 2,
        "created_at": api_eval.utc_now(),
        "model": args.run_name,
        "lan_type": args.lan_type,
        "languages": list(
            api_eval.LANGUAGE_ORDER
            if args.lan_type == "all"
            else (args.lan_type,)
        ),
        "endpoint": LOCAL_ENDPOINT_ID,
        "backend": "local-vllm-replicated",
        "model_path": str(model),
        "model_signature": model_signature(model),
        "served_model_name": served_name,
        "generation_profile": args.generation_profile,
        "model_adapter": ({"profile": profile, "version": 1,
                           **({"tokenizer_compat_sha256": hashlib.sha256((Path(__file__).resolve().parent / "minicpm45_compat/sitecustomize.py").read_bytes()).hexdigest()} if profile == "minicpm45_nothinking" else {}),
                           "question_suffix": "/no_think" if profile == "mimo_nothinking" else None,
                           "stop_token_ids": [248044, 248046] if profile == "minicpm46_instruct" else ([1, 151645] if profile == "minicpm45_nothinking" else None)}
                          if profile in MODEL_ADAPTER_PROFILES else None),
        "mm_processor_kwargs": args.mm_processor_kwargs,
        "kimi_thinking": (args.generation_profile == "kimi_thinking" if args.generation_profile.startswith("kimi_") else None),
        "annotations_path": str(annotations_path),
        "annotations_sha256": annotation_sha,
        "selected_annotation_count": len(annotations),
        "selected_count": len(tasks),
        "limit": args.limit,
        "gpus": gpus,
        "gpu_groups": gpu_groups,
        "tensor_parallel_size": args.tp,
        "ports": ports,
        "replica_count": len(gpu_groups),
        "per_engine_concurrency": args.per_engine_concurrency,
        "request_slots": len(gpu_groups) * args.per_engine_concurrency,
        "max_tokens": args.max_tokens,
        "rollouts_per_question": args.rollouts,
        "timeout": args.timeout,
        "max_attempts": args.max_attempts,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": resolved_max_batched,
        "max_num_seqs": args.max_num_seqs,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "enable_prefix_caching": False,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "system_prompt": system_prompt_for_profile(args.generation_profile),
        "qwen3_enable_thinking": (
            False if args.generation_profile == "qwen3_nothinking" else None
        ),
        "image_payload": "original_file_bytes_data_url",
        "answer_extraction": "last_case_insensitive_answer_tag_from_reasoning_plus_content",
        "glm_box_fallback": args.glm_box_fallback,
        "ocr_scoring": {
            "metric": "thresholded_anls",
            "similarity_threshold": 0.7,
            "threshold_inclusive": True,
            "case_sensitive": True,
            "space_sensitive": False,
        },
        "vllm_serve_commands": [
            vllm_serve_command(
                vllm_bin=args.vllm_bin,
                model=model,
                served_name=served_name,
                port=port,
                args=args,
            )
            for port in ports
        ],
    }

    retained_results = {}
    if args.retry_missing_tags_from:
        source = args.retry_missing_tags_from.resolve()
        retained_results, retry_ids, original_config = load_retry_source(source, config, selected_task_ids)
        config["retry_missing_tags"] = {
            "source": str(source), "source_responses_sha256": api_eval.sha256_file(source / "responses.jsonl"),
            "target_task_ids": retry_ids, "retained_count": len(retained_results),
            "original_max_tokens": original_config["max_tokens"], "retry_max_tokens": args.max_tokens,
            "mixed_generation_settings": True,
        }
        print(f"Missing-tag retry: {len(retry_ids)} requests; retain {len(retained_results)} original results unchanged.", flush=True)
    print(
        f"Validated {len(annotations)} annotations -> {len(tasks)} language "
        f"tasks (lan_type={args.lan_type}); {len(gpus)} GPUs -> "
        f"{len(gpu_groups)} vLLM replicas (TP={args.tp}) x {args.per_engine_concurrency} slots.",
        flush=True,
    )
    print(
        f"Parity: original image bytes; {'official InternVL R1' if args.generation_profile == 'internvl_thinking' else 'exact hosted ZwZ'} system prompt; "
        f"temperature={args.temperature}; n={args.rollouts}; "
        f"top_p={args.top_p}; top_k={args.top_k}; "
        f"max_tokens={args.max_tokens}; "
        f"enable_thinking={config['qwen3_enable_thinking']}; shared scoring.",
        flush=True,
    )
    if args.dry_run:
        print(f"Model: {model}")
        print(f"Run name: {args.run_name}")
        print(f"Annotations: {annotations_path}")
        print(f"Output: {output_dir}")
        for gpu, command in zip(gpu_groups, config["vllm_serve_commands"]):
            print(f"GPU {gpu}: {' '.join(command)}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    existing_outputs = [
        output_dir / filename
        for filename in api_eval.OUTPUT_FILENAMES
        if (output_dir / filename).exists()
    ]
    if existing_outputs and not args.resume and not args.overwrite:
        raise RuntimeError(
            f"Output already exists: {existing_outputs[0]}; use --resume or "
            "--overwrite"
        )
    if args.overwrite:
        api_eval.clean_known_outputs(output_dir)

    completed_by_task_id: dict[str, dict[str, Any]] = dict(retained_results)
    if args.resume:
        saved = validate_local_resume_config(output_dir / "run_config.json", config)
        config["created_at"] = saved.get("created_at", config["created_at"])
        config["resumed_at"] = api_eval.utc_now()
        if saved.get("retry_missing_tags") != config.get("retry_missing_tags"):
            raise RuntimeError("Retry source or target set changed since the previous invocation")
        completed_by_task_id.update(api_eval.load_resume_results(
            output_dir / "responses.jsonl", selected_task_ids, args.run_name
        ))

    if args.retry_missing_tags_from and not args.resume:
        backup = output_dir / "source_backup"
        backup.mkdir(exist_ok=False)
        for name in ("run_config.json", "responses.jsonl", "results.json", "summary.json", "report.md"):
            src = args.retry_missing_tags_from / name
            if src.is_file():
                shutil.copy2(src, backup / name)
        api_eval.atomic_write_json(output_dir / "retry_plan.json", config["retry_missing_tags"])

    pending = [
        row for row in tasks if row["_task_id"] not in completed_by_task_id
    ]
    api_eval.atomic_write_json(output_dir / "run_config.json", config)
    print(
        f"Queued {len(pending)} pending language tasks in annotation order "
        f"({len(completed_by_task_id)} already complete).",
        flush=True,
    )

    replicas: list[Replica] = []
    executor: concurrent.futures.ThreadPoolExecutor | None = None
    futures: dict[
        concurrent.futures.Future[dict[str, Any]], dict[str, Any]
    ] = {}
    if pending:
        install_signal_handlers()
        slots: queue.Queue[tuple[int, Replica]] = queue.Queue()
        try:
            replicas = start_replicas(
                model, served_name, gpus, args.port, output_dir, args
            )
            wait_replicas_ready(replicas, args.startup_timeout)
            slot_id = 0
            for replica in replicas:
                for _ in range(args.per_engine_concurrency):
                    slot_id += 1
                    slots.put((slot_id, replica))

            def scheduled(row: dict[str, Any]) -> dict[str, Any]:
                current_slot, replica = slots.get()
                try:
                    return result_for_task(
                        row,
                        run_name=args.run_name,
                        served_name=served_name,
                        replica=replica,
                        slot_id=current_slot,
                        args=args,
                    )
                finally:
                    slots.put((current_slot, replica))

            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=len(replicas) * args.per_engine_concurrency,
                thread_name_prefix="zoomgain-vllm",
            )
            futures = {executor.submit(scheduled, row): row for row in pending}
            initial_rows = list(completed_by_task_id.values())
            initial_score = sum(float(row.get("score", 0.0)) for row in initial_rows)
            initial_errors = sum(
                row.get("status") != "completed" for row in initial_rows
            )
            initial_missing = sum(
                not row.get("answer_tag_valid") for row in initial_rows
            )
            log_mode = "a" if args.resume else "w"
            with (output_dir / "responses.jsonl").open(
                log_mode, encoding="utf-8", buffering=1
            ) as log_handle, tqdm(
                total=len(tasks),
                initial=len(initial_rows),
                desc="ZoomGain local vLLM",
                unit="q",
                dynamic_ncols=True,
            ) as bar:
                done = len(initial_rows)
                score_sum = initial_score
                errors = initial_errors
                missing_tags = initial_missing
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    log_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    log_handle.flush()
                    completed_by_task_id[result["task_id"]] = result
                    done += 1
                    score_sum += float(result.get("score", 0.0))
                    errors += result.get("status") != "completed"
                    missing_tags += not result.get("answer_tag_valid")
                    bar.update(1)
                    bar.set_postfix(
                        score=f"{score_sum / done:.4f}",
                        errors=errors,
                        missing_tags=missing_tags,
                        refresh=False,
                    )
        except BaseException:
            if executor is not None:
                for future in futures:
                    future.cancel()
            raise
        finally:
            teardown_replicas(replicas)
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)

    missing = [
        row["_task_id"]
        for row in tasks
        if row["_task_id"] not in completed_by_task_id
    ]
    if missing:
        raise RuntimeError(f"Missing final results for {len(missing)} language tasks")
    ordered_results = [
        completed_by_task_id[row["_task_id"]] for row in tasks
    ]
    summary = api_eval.summarize(ordered_results, args.run_name, args.lan_type)
    summary["backend"] = "local-vllm-replicated"
    summary["model_path"] = str(model)
    summary["generation_profile"] = args.generation_profile
    if args.retry_missing_tags_from:
        summary["retry_missing_tags"] = config["retry_missing_tags"]
        summary["glm_box_fallback_count"] = sum(r.get("answer_extraction_source") == "glm_box_fallback" for r in ordered_results)
    api_eval.atomic_write_json(
        output_dir / "results.json",
        {
            "schema_version": 2,
            "model": args.run_name,
            "backend": "local-vllm-replicated",
            "model_path": str(model),
            "generation_profile": args.generation_profile,
            "lan_type": args.lan_type,
            "annotations_sha256": annotation_sha,
            "annotation_count": len(annotations),
            "task_count": len(tasks),
            "results": ordered_results,
        },
    )
    api_eval.atomic_write_json(output_dir / "summary.json", summary)
    api_eval.write_report(output_dir / "report.md", summary, output_dir)
    if args.retry_missing_tags_from:
        with (output_dir / "report.md").open("a") as note:
            note.write("\n## Targeted retry provenance\n\nOnly original missing-tag tasks were rerun. "
                       f"Retained {len(retained_results)} original results; "
                       f"original max_tokens={config['retry_missing_tags']['original_max_tokens']}, retry max_tokens={args.max_tokens}. "
                       "This is a mixed-budget merged evaluation. GLM complete boxed answers are accepted only when answer tags are absent.\n")
    config["finished_at"] = api_eval.utc_now()
    config["summary_path"] = str(output_dir / "summary.json")
    api_eval.atomic_write_json(output_dir / "run_config.json", config)

    print(
        f"Finished {len(ordered_results)} language tasks from "
        f"{len(annotations)} annotations.",
        flush=True,
    )
    api_eval.print_score_breakdown(summary)
    preview_results(ordered_results, args.preview_samples)
    print(f"Artifacts: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
