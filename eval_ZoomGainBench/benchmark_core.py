"""ZoomGainBench scoring, annotation loading, and local vLLM HTTP transport. No hosted-model CLI."""

from __future__ import annotations

import argparse

import base64

import hashlib

import json

import math

import mimetypes

import multiprocessing as mp

import os

import queue

import random

import re

import sys

import time

import uuid

from collections import Counter, defaultdict

from datetime import datetime, timezone

from decimal import Decimal, InvalidOperation

from pathlib import Path

from typing import Any, Iterable

import requests
from PIL import Image

ZWZ_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it. The assistant first thinks about the "
    "reasoning process in the mind and then provides the user with the answer. "
    "The answer are enclosed within <answer> </answer> tags, respectively, "
    "i.e., reasoning process here <answer> answer here </answer>"
)

LANGUAGE_ORDER = ("cn", "en")

LANGUAGE_NAMES = {"cn": "Chinese", "en": "English"}

ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

OPTION_PATTERN = re.compile(r"^([A-Z])\.\s+", re.MULTILINE)

NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")

RETRYABLE_HTTP_CODES = frozenset({404, 408, 409, 425, 429, 500, 502, 503, 504})

MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

OUTPUT_FILENAMES = (
    "responses.jsonl",
    "results.json",
    "summary.json",
    "report.md",
    "run_config.json",
)

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def load_annotations(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"Expected object at {path}:{line_number}")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise RuntimeError(f"No annotations found in {path}")

    required = {
        "id",
        "question",
        "question_zh",
        "evidence_bbox",
        "answer",
        "answer_text",
        "answer_format",
        "question_type",
        "scoring",
        "image_path",
    }
    seen: set[str] = set()
    image_sizes: dict[Path, tuple[int, int]] = {}
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise RuntimeError(f"Annotation row {index} is missing: {sorted(missing)}")
        annotation_id = row["id"]
        if not isinstance(annotation_id, str) or not annotation_id:
            raise RuntimeError(f"Annotation row {index} has an invalid id")
        if annotation_id in seen:
            raise RuntimeError(f"Duplicate annotation id: {annotation_id}")
        seen.add(annotation_id)
        if not isinstance(row["question"], str) or not row["question"].strip():
            raise RuntimeError(f"{annotation_id}: invalid question")
        image_path = Path(str(row["image_path"])).expanduser()
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        row["image_path"] = str(image_path.resolve())
        if not image_path.is_file():
            raise RuntimeError(f"{annotation_id}: missing image: {image_path}")
        question_zh = row.get("question_zh")
        if not isinstance(question_zh, str) or not question_zh.strip():
            raise RuntimeError(f"{annotation_id}: invalid or missing question_zh")
        bbox = row.get("evidence_bbox")
        if not isinstance(bbox, list) or len(bbox) != 4 or any(type(v) is not int for v in bbox):
            raise RuntimeError(f"{annotation_id}: evidence_bbox must contain four pixel coordinates")
        if image_path not in image_sizes:
            with Image.open(image_path) as image:
                image_sizes[image_path] = image.size
        width, height = image_sizes[image_path]
        x1, y1, x2, y2 = bbox
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise RuntimeError(
                f"{annotation_id}: evidence_bbox {bbox} is invalid for image size {width}x{height}"
            )
        validate_scoring_spec(row)
        row["_annotation_index"] = index
    return rows

def question_subset(question_type: str) -> str:
    """Map source question types onto the three reported benchmark subsets."""

    if question_type in {"attribute", "action", "spatial", "color"}:
        return "attr"
    if question_type in {"counting", "ocr"}:
        return question_type
    raise RuntimeError(f"Unsupported question type: {question_type!r}")

def expand_language_tasks(
    annotations: list[dict[str, Any]],
    lan_type: str,
) -> list[dict[str, Any]]:
    if lan_type not in {"all", *LANGUAGE_ORDER}:
        raise RuntimeError(f"Unsupported lan_type: {lan_type!r}")
    languages = LANGUAGE_ORDER if lan_type == "all" else (lan_type,)
    tasks: list[dict[str, Any]] = []
    for row in annotations:
        localized_questions = {"en": row["question"], "cn": row["question_zh"]}
        subset = question_subset(str(row["question_type"]))
        for language in languages:
            task = dict(row)
            task["question"] = localized_questions[language]
            task["language"] = language
            task["question_subset"] = subset
            task["_task_id"] = f"{row['id']}::{language}"
            task["_task_index"] = len(tasks)
            tasks.append(task)
    return tasks

def validate_scoring_spec(row: dict[str, Any]) -> None:
    annotation_id = row.get("id", "<unknown>")
    scoring = row.get("scoring")
    if not isinstance(scoring, dict):
        raise RuntimeError(f"{annotation_id}: scoring must be an object")
    metric = scoring.get("metric")
    if row.get("answer_format") == "multiple_choice":
        if metric != "exact_match":
            raise RuntimeError(f"{annotation_id}: MCQ must use exact_match")
        return
    if row.get("question_type") == "counting":
        if metric != "numeric_exact_match":
            raise RuntimeError(f"{annotation_id}: counting must use numeric_exact_match")
        return
    if row.get("question_type") == "ocr":
        expected = {
            "metric": "thresholded_anls",
            "similarity_threshold": 0.7,
            "threshold_inclusive": True,
            "case_sensitive": True,
            "space_sensitive": False,
        }
        if scoring != expected:
            raise RuntimeError(
                f"{annotation_id}: OCR scoring does not match case-sensitive "
                f"whitespace-insensitive ANLS@0.7: {scoring!r}"
            )
        return
    raise RuntimeError(
        f"{annotation_id}: unsupported open-ended type {row.get('question_type')!r}"
    )

def stringify_message_part(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif item is not None:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "".join(parts)
    return str(value)

def extract_last_answer_tag(text: str) -> tuple[str | None, int]:
    matches = ANSWER_TAG_PATTERN.findall(text)
    if not matches:
        return None, 0
    answer = matches[-1].strip()
    return (answer or None), len(matches)

def levenshtein_distance(left: str, right: str) -> int:
    """Character-level Levenshtein distance with O(min(m, n)) memory."""

    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]

def thresholded_anls(
    prediction: str,
    ground_truth: str,
    *,
    threshold: float = 0.7,
    case_sensitive: bool = True,
) -> tuple[float, dict[str, Any]]:
    """Return whitespace-insensitive ANLS at or above the threshold."""

    # OCR spacing is often visually ambiguous (for example, ``60g`` versus
    # ``60 g``). Remove every Unicode whitespace character from both strings
    # before computing the otherwise character-level metric.
    prediction = "".join(prediction.split())
    ground_truth = "".join(ground_truth.split())
    if not case_sensitive:
        prediction = prediction.casefold()
        ground_truth = ground_truth.casefold()
    denominator = max(len(prediction), len(ground_truth))
    if denominator == 0:
        similarity = 1.0
        distance = 0
    else:
        distance = levenshtein_distance(prediction, ground_truth)
        similarity = 1.0 - distance / denominator
    score = similarity if similarity >= threshold else 0.0
    return score, {
        "edit_distance": distance,
        "normalization_length": denominator,
        "similarity": similarity,
        "similarity_threshold": threshold,
        "threshold_inclusive": True,
        "case_sensitive": case_sensitive,
        "space_sensitive": False,
        "whitespace_normalization": "remove_all_unicode_whitespace",
    }

def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())

def parse_mcq_answer(tagged_answer: str, row: dict[str, Any]) -> str | None:
    """Read an explicit option without Unicode-boundary or GT-dependent guesses."""
    labels = sorted(set(OPTION_PATTERN.findall(row["question"])))
    if not labels:
        return None
    choices = re.escape("".join(labels))
    text = tagged_answer.strip().replace("**", "").replace("__", "")
    exact = re.fullmatch(
        rf"\s*[（\(\[]?([{choices}])[）\)\].:。]?\s*",
        text,
        flags=re.IGNORECASE,
    )
    if exact:
        return exact.group(1).upper()
    # Compare with ALL options, never just the ground-truth answer_text.
    options = list(OPTION_PATTERN.finditer(row["question"]))
    option_texts = {}
    for index, match in enumerate(options):
        end = options[index + 1].start() if index + 1 < len(options) else len(row["question"])
        option_texts[match.group(1)] = row["question"][match.end():end].strip()

    def text_key(value: str) -> str:
        return normalize_text(value).strip(' "\'“”‘’。.!！;；')

    def exact_option(value: str) -> str | None:
        matches = [label for label, option in option_texts.items()
                   if text_key(value) == text_key(option)]
        return matches[0] if len(matches) == 1 else None

    def read_label(value: str) -> str | None:
        value = value.lstrip()
        value = re.sub(r"^(?:对应\s*)?选项\s*(?:是|为)?\s*[：:]?\s*", "", value)
        option = exact_option(value)
        if option is not None:
            return option
        match = re.match(rf"[（(\[]?([{choices}])[）)\]]?(?![A-Za-z0-9_])",
                         value, flags=re.IGNORECASE)
        if not match:
            return None
        label = match.group(1).upper()
        tail = value[match.end():]
        # Reject explicitly multiple/uncertain choices rather than picking one.
        if re.match(rf"\s*(?:或(?:者)?|和|与|、|/|or\b|and\b)\s*[{choices}](?![A-Za-z])",
                    tail, flags=re.IGNORECASE):
            return None
        # A followed by an English word is potentially the indefinite article.
        if label == "A" and re.match(r"\s+[A-Za-z]{2,}", tail):
            return None
        return label

    # Explicit conclusions take precedence over labels mentioned while comparing
    # options. The final explicit conclusion wins, independently of the GT.
    marker = re.compile(
        r"(?:最终)?答案\s*(?:应该是|应当是|应为|应是|是|为|选择|选)?\s*[：:]?\s*"
        r"|(?:正确|最终|对应|即|为)\s*选项\s*(?:应为|是|为)?\s*[：:]?\s*"
        r"|(?:因此|所以|故|综上)(?:[，,：:\s]*(?:应当|应该|应))?[，,：:\s]*(?:选择|选)\s*"
        r"|(?<![A-Za-z])(?:final\s+answer|correct\s+(?:answer|choice|option)|answer|choice|option)"
        r"(?![A-Za-z])\s*(?:is|would be|should be)?\s*[：:]?\s*",
        flags=re.IGNORECASE,
    )
    conclusions = list(marker.finditer(text))
    if conclusions:
        tail = text[conclusions[-1].end():]
        prediction = read_label(tail)
        if prediction is not None:
            return prediction
        # An explicit ambiguous/out-of-range letter or a retraction must not
        # resurrect an earlier answer. A prose explanation, however, does not
        # invalidate a clearly labelled answer at the beginning (B. Green...).
        if re.match(r"\s*[（(\[]?[A-Za-z](?![A-Za-z])", tail) or re.match(
            r"\s*(?:不知道|不确定|无法|不能确定|未知|unknown|uncertain|cannot|not sure)",
            tail, re.IGNORECASE,
        ):
            return None

    option = exact_option(text)
    if option is not None:
        return option
    # A leading numbered-option form is clear; arbitrary letters in prose are not.
    if re.match(rf"^[（(\[]?[{choices}](?:[）)\]]|[.、:：])", text, re.IGNORECASE):
        return read_label(text)
    return None

def parse_numeric_answer(tagged_answer: str) -> str | None:
    match = NUMBER_PATTERN.search(tagged_answer)
    return match.group(0) if match else None

def numeric_equal(left: str, right: Any) -> bool:
    try:
        return Decimal(left) == Decimal(str(right).strip())
    except (InvalidOperation, ValueError):
        return False

def score_answer(tagged_answer: str | None, row: dict[str, Any]) -> dict[str, Any]:
    if tagged_answer is None:
        return {
            "parsed_answer": None,
            "score": 0.0,
            "full_credit": False,
            "score_details": {"reason": "missing_or_empty_answer_tag"},
        }

    if row["answer_format"] == "multiple_choice":
        prediction = parse_mcq_answer(tagged_answer, row)
        full_credit = prediction == str(row["answer"]).upper()
        return {
            "parsed_answer": prediction,
            "score": float(full_credit),
            "full_credit": full_credit,
            "score_details": {"metric": "exact_match"},
        }

    if row["question_type"] == "counting":
        prediction = parse_numeric_answer(tagged_answer)
        full_credit = prediction is not None and numeric_equal(prediction, row["answer"])
        return {
            "parsed_answer": prediction,
            "score": float(full_credit),
            "full_credit": full_credit,
            "score_details": {"metric": "numeric_exact_match"},
        }

    if row["question_type"] == "ocr":
        scoring = row["scoring"]
        score, details = thresholded_anls(
            tagged_answer,
            str(row["answer"]),
            threshold=float(scoring["similarity_threshold"]),
            case_sensitive=bool(scoring["case_sensitive"]),
        )
        details["metric"] = "thresholded_anls"
        return {
            "parsed_answer": tagged_answer,
            "score": score,
            "full_credit": math.isclose(score, 1.0),
            "score_details": details,
        }

    return {
        "parsed_answer": tagged_answer,
        "score": 0.0,
        "full_credit": False,
        "score_details": {"reason": "unsupported_question_type"},
    }

def image_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime_type = MIME_BY_SUFFIX.get(suffix) or mimetypes.guess_type(path.name)[0]
    if not mime_type or not mime_type.startswith("image/"):
        raise RuntimeError(f"Unsupported image type: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"

def retry_delay(response: requests.Response | None, attempt: int) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        try:
            return min(120.0, max(1.0, float(retry_after)))
        except (TypeError, ValueError):
            pass
    return min(60.0, float(2 ** (attempt - 1))) + random.uniform(0.0, 0.5)

def call_api(
    *,
    session: requests.Session,
    endpoint: str,
    api_key: str,
    model: str,
    row: dict[str, Any],
    max_tokens: int,
    timeout: int,
    max_attempts: int,
) -> dict[str, Any]:
    started = time.monotonic()
    image_url = image_data_url(Path(str(row["image_path"])))
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": ZWZ_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": row["question"]},
                ],
            },
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    qwen3_no_thinking = model.startswith("Qwen3")
    if qwen3_no_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    last_error = "unknown request failure"
    response: requests.Response | None = None
    response_body: dict[str, Any] | None = None
    attempts_used = 0
    for attempt in range(1, max_attempts + 1):
        attempts_used = attempt
        response = None
        try:
            response = session.post(
                endpoint,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=timeout,
            )
            if response.status_code >= 400:
                compact = response.text[:2000].replace(api_key, "[REDACTED]")
                last_error = f"HTTP {response.status_code}: {compact}"
                if (
                    response.status_code in RETRYABLE_HTTP_CODES
                    and attempt < max_attempts
                ):
                    time.sleep(retry_delay(response, attempt))
                    continue
                break
            parsed = response.json()
            if not isinstance(parsed, dict):
                raise RuntimeError("Response JSON is not an object")
            response_body = parsed
            break
        except Exception as exc:  # noqa: BLE001 - preserve request diagnostics
            last_error = f"{type(exc).__name__}: {exc}".replace(api_key, "[REDACTED]")
            if attempt < max_attempts:
                time.sleep(retry_delay(response, attempt))

    elapsed = round(time.monotonic() - started, 3)
    common = {
        "elapsed_seconds": elapsed,
        "attempts": attempts_used,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": (
            {"enable_thinking": False} if qwen3_no_thinking else None
        ),
    }
    if response_body is None:
        return {**common, "status": "request_error", "error": last_error}

    try:
        choice = response_body["choices"][0]
        message = choice["message"]
        content = stringify_message_part(message.get("content"))
        reasoning = stringify_message_part(
            message.get("reasoning_content")
            or message.get("reasoning")
            or message.get("analysis")
        )
        combined = "\n".join(part for part in (reasoning, content) if part).strip()
        tagged_answer, tag_count = extract_last_answer_tag(combined)
        scored = score_answer(tagged_answer, row)
        return {
            **common,
            "status": "completed",
            "response_model": response_body.get("model"),
            "response_id": response_body.get("id"),
            "finish_reason": choice.get("finish_reason"),
            "content": content,
            "reasoning_content": reasoning,
            "content_chars": len(content),
            "reasoning_chars": len(reasoning),
            "answer_tag_count": tag_count,
            "answer_tag_valid": tagged_answer is not None,
            "answer_tag_answer": tagged_answer,
            **scored,
            "usage": response_body.get("usage"),
        }
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

def error_result(
    row: dict[str, Any], model: str, worker_id: int, key_slot: int, exc: BaseException
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "annotation_index": row["_annotation_index"],
        "task_index": row["_task_index"],
        "task_id": row["_task_id"],
        "id": row["id"],
        "language": row["language"],
        "model": model,
        "worker_id": worker_id,
        "key_slot": key_slot,
        "question_type": row["question_type"],
        "question_subset": row["question_subset"],
        "answer_format": row["answer_format"],
        "ground_truth": row["answer"],
        "status": "worker_error",
        "error": f"{type(exc).__name__}: {exc}",
        "score": 0.0,
        "full_credit": False,
        "answer_tag_valid": False,
        "finished_at": utc_now(),
    }

def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.writing-{uuid.uuid4().hex}"
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path, json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    )

def load_resume_results(
    path: Path, selected_task_ids: set[str], model: str
) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"Cannot resume without {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    completed: dict[str, dict[str, Any]] = {}
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1:
                print(
                    f"warning: ignoring incomplete final line in {path}",
                    file=sys.stderr,
                )
                continue
            raise RuntimeError(f"Invalid JSONL at {path}:{index + 1}: {exc}") from exc
        if row.get("model") != model:
            raise RuntimeError(f"Resume log contains a different model at line {index + 1}")
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            annotation_id = row.get("id")
            language = row.get("language")
            if isinstance(annotation_id, str) and language in LANGUAGE_ORDER:
                task_id = f"{annotation_id}::{language}"
            elif (
                isinstance(annotation_id, str)
                and f"{annotation_id}::en" in selected_task_ids
            ):
                # Schema-v1 runs predate bilingual evaluation.  Their rows do
                # not contain task_id/language, but every legacy question was
                # the English annotation question.
                language = "en"
                task_id = f"{annotation_id}::en"
            else:
                raise RuntimeError(
                    f"Resume log lacks task_id/language at line {index + 1}"
                )
            row["task_id"] = task_id
            row["language"] = language
        if "question_subset" not in row:
            question_type = row.get("question_type")
            if isinstance(question_type, str):
                row["question_subset"] = question_subset(question_type)
        if task_id not in selected_task_ids:
            continue
        if row.get("status") == "completed":
            completed[task_id] = row
    return completed

def validate_resume_config(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Cannot resume without {path}")
    saved = json.loads(path.read_text(encoding="utf-8"))
    # Schema-v1 runs were English-only and predate these bilingual bookkeeping
    # fields.  Normalize them in memory so an unchanged legacy run can resume
    # with --lan_type=en (or be extended safely with --lan_type=all).
    comparable_saved = dict(saved)
    legacy_monolingual = (
        "lan_type" not in saved
        and "languages" not in saved
        and "selected_annotation_count" not in saved
    )
    if legacy_monolingual:
        comparable_saved["lan_type"] = "en"
        comparable_saved["languages"] = ["en"]
        comparable_saved["selected_annotation_count"] = saved.get("selected_count")
    fields = (
        "model",
        "endpoint",
        "annotations_sha256",
        "selected_annotation_count",
        "limit",
        "max_tokens",
        "timeout",
        "max_attempts",
    )
    mismatches = [
        field
        for field in fields
        if comparable_saved.get(field) != expected.get(field)
    ]

    saved_languages = comparable_saved.get("languages")
    expected_languages = expected.get("languages")
    if not isinstance(saved_languages, list):
        saved_lan_type = comparable_saved.get("lan_type")
        saved_languages = (
            list(LANGUAGE_ORDER)
            if saved_lan_type == "all"
            else [saved_lan_type]
        )
    if not isinstance(expected_languages, list):
        expected_lan_type = expected.get("lan_type")
        expected_languages = (
            list(LANGUAGE_ORDER)
            if expected_lan_type == "all"
            else [expected_lan_type]
        )
    saved_language_set = set(saved_languages)
    expected_language_set = set(expected_languages)
    valid_languages = set(LANGUAGE_ORDER)
    language_expansion = (
        bool(saved_language_set)
        and saved_language_set < expected_language_set
        and expected_language_set <= valid_languages
    )
    if saved_language_set == expected_language_set:
        if comparable_saved.get("lan_type") != expected.get("lan_type"):
            mismatches.append("lan_type")
        if comparable_saved.get("selected_count") != expected.get("selected_count"):
            mismatches.append("selected_count")
    elif language_expansion:
        annotation_count = expected.get("selected_annotation_count")
        saved_count = comparable_saved.get("selected_count")
        expected_count = expected.get("selected_count")
        if not isinstance(annotation_count, int):
            mismatches.append("selected_annotation_count")
        else:
            if saved_count != annotation_count * len(saved_language_set):
                mismatches.append("selected_count")
            if expected_count != annotation_count * len(expected_language_set):
                mismatches.append("selected_count")
    else:
        mismatches.extend(("lan_type", "selected_count"))

    mismatches = list(dict.fromkeys(mismatches))
    if mismatches:
        raise RuntimeError(
            "Resume configuration differs for: " + ", ".join(mismatches)
        )
    if legacy_monolingual:
        print("Resuming legacy English-only run as lan_type=en.", flush=True)
    if language_expansion:
        print(
            "Extending resumed language tasks: "
            f"{','.join(saved_languages)} -> {','.join(expected_languages)}.",
            flush=True,
        )
    return saved

def summarize(
    results: Iterable[dict[str, Any]], model: str, lan_type: str
) -> dict[str, Any]:
    rows = list(results)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_subset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        question_type = str(row["question_type"])
        subset = str(row.get("question_subset") or question_subset(question_type))
        language = row.get("language")
        if language not in LANGUAGE_ORDER:
            raise RuntimeError(f"Result has invalid language: {language!r}")
        if subset not in {"attr", "counting", "ocr"}:
            raise RuntimeError(f"Result has invalid question subset: {subset!r}")
        by_type[question_type].append(row)
        by_subset[subset].append(row)
        by_language[language].append(row)

    def group_summary(group: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(group)
        return {
            "count": count,
            "mean_score": (
                sum(float(row.get("score", 0.0)) for row in group) / count
                if count
                else 0.0
            ),
            "full_credit": sum(bool(row.get("full_credit")) for row in group),
            "api_completed": sum(row.get("status") == "completed" for row in group),
            "answer_tag_valid": sum(
                bool(row.get("answer_tag_valid")) for row in group
            ),
        }

    overall = group_summary(rows)
    status_counts = Counter(str(row.get("status")) for row in rows)
    finish_reason_counts = Counter(
        str(row.get("finish_reason"))
        for row in rows
        if row.get("finish_reason") is not None
    )
    language_summaries: dict[str, dict[str, Any]] = {}
    for language in LANGUAGE_ORDER:
        language_rows = by_language.get(language)
        if not language_rows:
            continue
        language_overall = group_summary(language_rows)
        language_summaries[language] = {
            "name": LANGUAGE_NAMES[language],
            **language_overall,
            "by_subset": {
                subset: group_summary(
                    [
                        row
                        for row in language_rows
                        if (row.get("question_subset") or question_subset(
                            str(row["question_type"])
                        ))
                        == subset
                    ]
                )
                for subset in ("attr", "counting", "ocr")
            },
            "by_question_type": {
                question_type: group_summary(
                    [
                        row
                        for row in language_rows
                        if row["question_type"] == question_type
                    ]
                )
                for question_type in sorted(
                    {str(row["question_type"]) for row in language_rows}
                )
            },
        }
    return {
        "schema_version": 2,
        "model": model,
        "lan_type": lan_type,
        "created_at": utc_now(),
        "total": len(rows),
        "mean_score": overall["mean_score"],
        "full_credit": overall["full_credit"],
        "api_completed": overall["api_completed"],
        "answer_tag_valid": overall["answer_tag_valid"],
        "status_counts": dict(status_counts),
        "finish_reason_counts": dict(finish_reason_counts),
        "by_question_type": {
            question_type: group_summary(group)
            for question_type, group in sorted(by_type.items())
        },
        "by_subset": {
            subset: group_summary(by_subset.get(subset, []))
            for subset in ("attr", "counting", "ocr")
        },
        "by_language": language_summaries,
        "scoring": {
            "multiple_choice": "exact option-label match",
            "counting": "numeric exact match",
            "ocr": {
                "metric": "thresholded_anls",
                "similarity_threshold": 0.7,
                "threshold_inclusive": True,
                "case_sensitive": True,
                "space_sensitive": False,
            },
        },
    }

def write_report(path: Path, summary: dict[str, Any], output_dir: Path) -> None:
    total = summary["total"]
    lines = [
        f"# ZoomGain evaluation · {summary['model']}",
        "",
        f"- Questions: `{total}`",
        f"- Mean score: `{summary['mean_score']:.6f}`",
        f"- Full credit: `{summary['full_credit']}/{total}`",
        f"- API completed: `{summary['api_completed']}/{total}`",
        f"- Valid final answer tag: `{summary['answer_tag_valid']}/{total}`",
        "- OCR metric: case-sensitive, whitespace-insensitive `ANLS@0.7`",
        "",
    ]
    for language in LANGUAGE_ORDER:
        language_summary = summary["by_language"].get(language)
        if language_summary is None:
            continue
        lines.extend(
            [
                f"## {LANGUAGE_NAMES[language]} (`{language}`)",
                "",
                f"- Mean score: `{language_summary['mean_score']:.6f}`",
                f"- Requests: `{language_summary['count']}`",
                "",
                "| Subset | N | Mean score | Full credit | API completed | Valid tag |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for subset in ("attr", "counting", "ocr"):
            item = language_summary["by_subset"][subset]
            lines.append(
                f"| {subset} | {item['count']} | {item['mean_score']:.6f} | "
                f"{item['full_credit']} | {item['api_completed']} | "
                f"{item['answer_tag_valid']} |"
            )
        lines.append("")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Raw append-only responses: `{output_dir / 'responses.jsonl'}`",
            f"- Final ordered results: `{output_dir / 'results.json'}`",
            f"- Machine-readable summary: `{output_dir / 'summary.json'}`",
        ]
    )
    atomic_write_text(path, "\n".join(lines) + "\n")

def print_score_breakdown(summary: dict[str, Any]) -> None:
    for language in LANGUAGE_ORDER:
        language_summary = summary["by_language"].get(language)
        if language_summary is None:
            continue
        print(
            f"{LANGUAGE_NAMES[language]} ({language}) score: "
            f"{language_summary['mean_score']:.6f} "
            f"(n={language_summary['count']})",
            flush=True,
        )
        for subset in ("attr", "counting", "ocr"):
            item = language_summary["by_subset"][subset]
            print(
                f"  {subset}: {item['mean_score']:.6f} (n={item['count']})",
                flush=True,
            )
    print(
        f"Overall score: {summary['mean_score']:.6f} (n={summary['total']})",
        flush=True,
    )

def clean_known_outputs(output_dir: Path) -> None:
    for filename in OUTPUT_FILENAMES:
        (output_dir / filename).unlink(missing_ok=True)
