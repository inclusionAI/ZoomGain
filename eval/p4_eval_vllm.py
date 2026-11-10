#!/usr/bin/env python3
"""vLLM-served box-free evaluation for the RegionLens benchmarks.

Talks to a `vllm serve` endpoint (one model per GPU). Benchmarks: zoombench
(845, generate + official parser), VStar direct-attributes/relative-position
(letter MCQ; the dataset stores the ground truth as options[0]), hrbench4k/8k
(cyclic-option MCQ), and MME-RealWorld-Lite (official five-choice prompt and
hierarchical metrics), TreeBench (A--K answer-letter accuracy only),
CV-Bench (official source-balanced 2D/3D answer-letter accuracy), MMStar
(A--D answer-letter accuracy with six capability and 18 fine-grained slices),
and MMVP (150 paired visual questions; both questions must be correct).
VisualProbe Easy/Medium/Hard are exposed as three independent open-answer
benchmarks and scored with a conservative deterministic VQA normalizer.
Greedy decoding is used throughout. The Qwen3-VL profile defaults to 8 generated
tokens for direct evaluation and 512 for boxed/answer-tag CoT. The explicit
Qwen3.5 thinking profile defaults to 8192 generated tokens.
Per-sample predictions are dumped alongside the summary row.

Serve (one per GPU):
  CUDA_VISIBLE_DEVICES=2 vllm serve \
    /path/to/materialized_model --served-model-name eval-model --port 8302 \
    --max-model-len 24576 --gpu-memory-utilization 0.9

Then:
  python3 p4_eval_vllm.py \
    --endpoint http://127.0.0.1:8302/v1 --model eval-model \
    --arm mask --step 400 --benchmarks zoombench,hrbench4k \
    --output RegionLens/results/p4_eval/vllm_points.jsonl
"""

from __future__ import annotations

from eval_paths import DATA_ROOT

import argparse
import base64
import csv
import io
import json
import re
import time as _time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue

from PIL import Image
from tqdm.auto import tqdm

import babyvision_data as babyvision
import run_zoombench_generate_eval as zoom_eval
import visualprobe_answer as visualprobe_judge
import vp_llm_judge as vp_llm


HRBENCH_MCQ_INSTRUCTION = "Answer with the option's letter from the given choices directly."
MME_REALWORLD_INSTRUCTION = (
    "Select the best answer to the above multiple-choice question based on the image. "
    "Respond with only the letter (A, B, C, D, or E) of the correct option.\n"
    "The best answer is:"
)
TREEBENCH_EXPECTED_N = 405
TREEBENCH_ANSWER_LETTERS = frozenset("ABCDEFGHIJK")
TREEBENCH_MCQ_INSTRUCTION = HRBENCH_MCQ_INSTRUCTION
CVBENCH_EXPECTED_N = 2638
CVBENCH_SPLITS = (
    ("test_2d.parquet", "2D", 1438),
    ("test_3d.parquet", "3D", 1200),
)
CVBENCH_EXPECTED_TASK_COUNTS = {
    "Count": 788,
    "Relation": 650,
    "Depth": 600,
    "Distance": 600,
}
CVBENCH_EXPECTED_SOURCE_COUNTS = {
    "ADE20K": 633,
    "COCO": 805,
    "Omni3D": 1200,
}
CVBENCH_ANSWER_LETTERS = frozenset("ABCDEF")
CVBENCH_MCQ_INSTRUCTION = HRBENCH_MCQ_INSTRUCTION
CVBENCH_LIMIT_STRATA = (
    ("Count", "ADE20K"),
    ("Relation", "ADE20K"),
    ("Depth", "Omni3D"),
    ("Distance", "Omni3D"),
    ("Count", "COCO"),
    ("Relation", "COCO"),
)
MMSTAR_EXPECTED_N = 1500
MMSTAR_ANSWER_LETTERS = frozenset("ABCD")
MMSTAR_MCQ_INSTRUCTION = HRBENCH_MCQ_INSTRUCTION
MMSTAR_CATEGORY_L2 = {
    "coarse perception": (
        "image scene and topic",
        "image style & quality",
        "image emotion",
    ),
    "fine-grained perception": (
        "recognition",
        "object counting",
        "localization",
    ),
    "instance reasoning": (
        "single-instance reasoning",
        "cross-instance attribute reasoning",
        "cross-instance relation reasoning",
    ),
    "logical reasoning": (
        "diagram reasoning",
        "common reasoning",
        "code & sequence reasoning",
    ),
    "math": (
        "geometry",
        "statistical reasoning",
        "numeric commonsense and calculation",
    ),
    "science & technology": (
        "biology & chemistry & physics",
        "geography & earth science & agriculture",
        "electronics & energy & mechanical eng.",
    ),
}
MMSTAR_EXPECTED_CATEGORY_COUNTS = {
    category: 250 for category in MMSTAR_CATEGORY_L2
}
MMSTAR_EXPECTED_L2_COUNTS = {
    "biology & chemistry & physics": 146,
    "image scene and topic": 141,
    "recognition": 118,
    "geometry": 116,
    "diagram reasoning": 110,
    "common reasoning": 101,
    "single-instance reasoning": 99,
    "object counting": 92,
    "cross-instance attribute reasoning": 89,
    "statistical reasoning": 86,
    "image style & quality": 78,
    "cross-instance relation reasoning": 62,
    "geography & earth science & agriculture": 58,
    "numeric commonsense and calculation": 48,
    "electronics & energy & mechanical eng.": 46,
    "localization": 40,
    "code & sequence reasoning": 39,
    "image emotion": 31,
}
MMSTAR_EXPECTED_SOURCE_COUNTS = {
    "SEEDBench_IMG": 424,
    "MMBench": 366,
    "MathVista": 345,
    "AI2D_TEST": 196,
    "MMMU": 100,
    "ScienceQA_TEST": 69,
}
MMVP_EXPECTED_N = 300
MMVP_EXPECTED_PAIRS = 150
MMVP_ANSWER_LETTERS = frozenset("AB")
MMVP_MCQ_INSTRUCTION = HRBENCH_MCQ_INSTRUCTION
# The first six selections cover all top-level capabilities; the first 18
# cover every fine-grained axis. Further samples cycle through these strata.
MMSTAR_LIMIT_STRATA = tuple(
    (category, MMSTAR_CATEGORY_L2[category][round_index])
    for round_index in range(3)
    for category in MMSTAR_CATEGORY_L2
)
VISUALPROBE_BENCHMARKS = (
    "visualprobe_all",
    "visualprobe_easy",
    "visualprobe_medium",
    "visualprobe_hard",
)
VISUALPROBE_SPLITS = {
    "visualprobe_easy": {
        "directory": "VisualProbe_Easy",
        "data_source": "visual_probe_easy",
        "expected_n": 141,
    },
    "visualprobe_medium": {
        "directory": "VisualProbe_Medium",
        "data_source": "visual_probe_medium",
        "expected_n": 268,
    },
    "visualprobe_hard": {
        "directory": "VisualProbe_Hard",
        "data_source": "visual_probe_hard",
        "expected_n": 106,
    },
}
VISUALPROBE_SHORT_ANSWER_INSTRUCTION = (
    "Answer the question using a single word or short phrase."
)
VISUALPROBE_DEFAULT_MAX_NEW_TOKENS = 32
# VisualProbe's authors score with an LLM judge, not string matching (Mini-o3,
# verl/utils/reward_score/general_qa_tool{,_qwen}.py).  Our deterministic rule
# rejects paraphrase and elaboration, which penalises trained arms unevenly --
# measured +11..+21 points recovered, with the sign of several "regressions"
# flipping.  The judge is therefore the primary metric; the deterministic score
# is retained alongside it in every record and summary.
# Serve the default endpoint with scripts/deploy_vp_judge.sh.
VISUALPROBE_JUDGE_ENDPOINT = "http://127.0.0.1:8031/v1"
VISUALPROBE_JUDGE_MODEL = "judge"
# 512 truncates this judge mid-rationale; truncated replies score as unparsed.
VISUALPROBE_JUDGE_MAX_TOKENS = 1024
VISUALPROBE_JUDGE_CONCURRENCY = 32
# cascade skips rows the deterministic rule already accepts (~40% of a split)
# and is provably equivalent: 0 overturns in ~39,000 audited rows. full judges
# everything and is kept for audits and for reproducing the 2026-08-10 backfill.
VISUALPROBE_JUDGE_MODES = ("cascade", "full")
VISUALPROBE_JUDGE_MODE = "cascade"
# Fraction of deterministic hits judged anyway, purely to keep the
# strict_yes_judge_no monitor alive under cascade. Observational only.
VISUALPROBE_JUDGE_AUDIT_FRAC = 0.1
VISUALPROBE_JUDGE_METRIC_PROFILE = "visualprobe_llm_judge_minio3_prompt"
VISUALPROBE_DETERMINISTIC_METRIC_PROFILE = "visualprobe_conservative_vqa_normalized"
ANSWER_FORMATS = ("direct", "boxed", "answer_tag", "direct_answer_tag")
GENERATION_PROFILES = ("qwen3vl", "qwen35_nothinking", "qwen35_thinking", "glm53_default")
QWEN3VL_GENERATION_PROFILE = "qwen3vl"
QWEN35_NOTHINKING_GENERATION_PROFILE = "qwen35_nothinking"
QWEN35_GENERATION_PROFILE = "qwen35_thinking"
QWEN35_GENERATION_PROFILES = frozenset(
    (QWEN35_NOTHINKING_GENERATION_PROFILE, QWEN35_GENERATION_PROFILE)
)
QWEN35_DEFAULT_MAX_NEW_TOKENS = 8192
BOXED_ANSWER_INSTRUCTION = (
    r"Output your final answer in the format: \boxed{answer here}."
)
QWEN35_ANSWER_TAG_INSTRUCTION = (
    "Put your final answer inside <answer> and </answer>. "
    "Put only the answer itself inside the tag, without reasoning."
)
DIRECT_ANSWER_TAG_INSTRUCTION = (
    "Put the final answer within <answer> </answer> tags . i.e., "
    "<answer>answer here</answer>"
)
ZWZ_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it. The assistant first thinks about the "
    "reasoning process in the mind and then provides the user with the answer. "
    "The answer are enclosed within <answer> </answer> tags, respectively, "
    "i.e., reasoning process here <answer> answer here </answer>"
)
GLM53_ANSWER_TAG_REMINDER = (
    "At the end, output the final answer inside exactly one pair of <answer> and </answer> tags. "
    "Inside the tags, give only the answer: the option letter for multiple choice, the number for counting, "
    "or the short answer text. Do not use any other final-answer delimiter."
)
ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL)
MME_REALWORLD_ANSWER_PREFIXES = (
    "The best answer is",
    "The correct answer is",
    "The answer is",
    "The answer",
    "The best option is",
    "The correct option is",
    "Best answer:",
    "Best option:",
)
IMAGE_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def parallel_map_with_progress(function, items, max_workers: int, description: str) -> list:
    """Run requests concurrently, report true completions, preserve input order.

    ``Executor.map`` only yields in input order, so one slow early image can
    make a progress bar appear frozen even when many later requests have
    finished.  Futures are therefore consumed with ``as_completed`` while
    their results are written back to their original positions.
    """
    work_items = list(items)
    if max_workers <= 0:
        raise ValueError("request concurrency must be positive")
    if not work_items:
        return []
    ordered_results = [None] * len(work_items)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_positions = {
            pool.submit(function, item): position
            for position, item in enumerate(work_items)
        }
        with tqdm(
            total=len(work_items),
            desc=description,
            unit="req",
            dynamic_ncols=True,
            mininterval=0.5,
        ) as progress:
            try:
                for future in as_completed(future_positions):
                    position = future_positions[future]
                    ordered_results[position] = future.result()
                    progress.update(1)
            except BaseException:
                for pending in future_positions:
                    pending.cancel()
                raise
    return ordered_results


def parallel_map_across_endpoints_with_progress(
    function,
    items,
    endpoints: list[str],
    per_endpoint_concurrency: int,
    description: str,
) -> list:
    """Dynamically dispatch work to whichever vLLM endpoint becomes free.

    Each endpoint contributes ``per_endpoint_concurrency`` independent request
    slots.  A slot is returned to the queue only after its request finishes,
    so the next waiting sample naturally goes to the engine that completed
    first. Results are still restored to input order by
    :func:`parallel_map_with_progress`.
    """
    if not endpoints:
        raise ValueError("at least one vLLM endpoint is required")
    if per_endpoint_concurrency <= 0:
        raise ValueError("per-endpoint concurrency must be positive")
    available: Queue[str] = Queue()
    for endpoint in endpoints:
        for _ in range(per_endpoint_concurrency):
            available.put(endpoint)

    def scheduled(item):
        endpoint = available.get()
        try:
            return function(item, endpoint)
        finally:
            available.put(endpoint)

    return parallel_map_with_progress(
        scheduled,
        items,
        max_workers=len(endpoints) * per_endpoint_concurrency,
        description=description,
    )


def eval_endpoints(args) -> list[str]:
    endpoints = getattr(args, "endpoints", None)
    if endpoints:
        return [str(endpoint).rstrip("/") for endpoint in endpoints]
    endpoint = getattr(args, "endpoint", None)
    if not endpoint:
        raise ValueError("evaluation requires --endpoint or a non-empty endpoints list")
    return [str(endpoint).rstrip("/")]


def parallel_eval_requests(
    function,
    items,
    args,
    description: str,
    single_endpoint_concurrency: int,
) -> list:
    """Use legacy concurrency for one server and slot scheduling for replicas."""
    endpoints = eval_endpoints(args)
    per_endpoint_concurrency = (
        int(single_endpoint_concurrency)
        if len(endpoints) == 1
        else int(getattr(args, "per_engine_concurrency", 1))
    )
    return parallel_map_across_endpoints_with_progress(
        function,
        items,
        endpoints=endpoints,
        per_endpoint_concurrency=per_endpoint_concurrency,
        description=description,
    )


def parse_bool(value: str | bool) -> bool:
    """Parse explicit shell booleans such as ``--use_cot true``."""

    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(
        f"expected true or false, got {value!r}"
    )


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True, help="served-model-name")
    parser.add_argument("--arm", required=True, help="Label for the output row")
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument(
        "--benchmarks",
        default="zoombench,hrbench4k",
        help=(
            "Comma list from zoombench,vstar,vstar_relpos,vstar_all,"
            "hrbench4k,hrbench8k,mmerealworld,treebench,babyvision,cvbench,mmstar,mmvp,"
            "visualprobe_easy,visualprobe_medium,visualprobe_hard"
        ),
    )
    parser.add_argument(
        "--zoombench-annotation",
        default=str(DATA_ROOT / 'zoombench/annotation_zoombench.json'),
    )
    parser.add_argument("--zoombench-image-root", default=str(DATA_ROOT / 'zoombench'))
    parser.add_argument(
        "--vstar-annotation",
        default=str(DATA_ROOT / 'vstar/test_questions.jsonl'),
    )
    parser.add_argument(
        "--vstar-image-root",
        default=str(DATA_ROOT / 'vstar/images'),
    )
    parser.add_argument("--hrbench-root", default=str(DATA_ROOT / 'hrbench'))
    parser.add_argument(
        "--mmerealworld-annotation",
        default=str(DATA_ROOT / 'MME-RW/data/MME-RealWorld-Lite.json'),
    )
    parser.add_argument(
        "--mmerealworld-image-root",
        default=str(DATA_ROOT / 'MME-RW/data/imgs'),
    )
    parser.add_argument(
        "--treebench-annotation",
        default=str(DATA_ROOT / 'treebench/TreeBench.tsv'),
    )
    parser.add_argument(
        "--cvbench-root",
        default=str(DATA_ROOT / 'cvbench'),
        help="Root containing test_2d.parquet and test_3d.parquet.",
    )
    parser.add_argument(
        "--mmstar-root",
        default=str(DATA_ROOT / 'mmstar'),
        help="Root containing the downloaded mmstar.parquet.",
    )
    parser.add_argument(
        "--mmvp-root",
        default=str(DATA_ROOT / 'mmvp'),
        help="Root containing Questions.csv and the 'MMVP Images' directory.",
    )
    parser.add_argument(
        "--visualprobe-root",
        default=str(DATA_ROOT / 'visualprobe'),
        help=(
            "Root containing VisualProbe_Easy, VisualProbe_Medium, and "
            "VisualProbe_Hard."
        ),
    )
    parser.add_argument(
        "--babyvision-root",
        default=str(DATA_ROOT / 'babyvision/babyvision_data'),
        help="Directory holding BabyVision meta_data.jsonl and images/.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help=(
            "Default: 8192 for either Qwen3.5 profile; otherwise 8 for direct "
            "and 512 for boxed/answer-tag CoT."
        ),
    )
    parser.add_argument(
        "--visualprobe-max-new-tokens",
        type=int,
        default=VISUALPROBE_DEFAULT_MAX_NEW_TOKENS,
        help=(
            "Minimum generation budget for direct VisualProbe answers "
            f"(default: {VISUALPROBE_DEFAULT_MAX_NEW_TOKENS}). "
            "Boxed/answer-tag CoT uses --max-new-tokens."
        ),
    )
    parser.add_argument(
        "--vp-judge",
        type=parse_bool,
        default=True,
        help=(
            "Score VisualProbe with the LLM judge (VisualProbe's own protocol) "
            "instead of deterministic string matching. Both scores are always "
            "recorded; this only selects which one 'accuracy' reports. The "
            "endpoint is pinged before generation so an unreachable judge fails "
            "in seconds rather than after a full evaluation."
        ),
    )
    parser.add_argument(
        "--vp-judge-endpoint",
        default=VISUALPROBE_JUDGE_ENDPOINT,
        help=f"OpenAI-compatible base URL of the judge (default: {VISUALPROBE_JUDGE_ENDPOINT}).",
    )
    parser.add_argument(
        "--vp-judge-model",
        default=VISUALPROBE_JUDGE_MODEL,
        help=f"Served model name of the judge (default: {VISUALPROBE_JUDGE_MODEL}).",
    )
    parser.add_argument(
        "--vp-judge-max-tokens",
        type=int,
        default=VISUALPROBE_JUDGE_MAX_TOKENS,
        help=f"Judge reply budget (default: {VISUALPROBE_JUDGE_MAX_TOKENS}).",
    )
    parser.add_argument(
        "--vp-judge-concurrency",
        type=int,
        default=VISUALPROBE_JUDGE_CONCURRENCY,
        help="Concurrent judge requests; judging runs after generation.",
    )
    parser.add_argument(
        "--vp-judge-mode",
        choices=VISUALPROBE_JUDGE_MODES,
        default=VISUALPROBE_JUDGE_MODE,
        help=(
            "cascade (default) only judges rows the deterministic rule "
            "rejected -- ~40%% fewer calls, and equivalent in practice (0 "
            "overturned deterministic hits in ~39,000 rows). full judges every "
            "row; use it to audit that equivalence or to reproduce the "
            "2026-08-10 backfill."
        ),
    )
    parser.add_argument(
        "--vp-judge-audit-frac",
        type=float,
        default=VISUALPROBE_JUDGE_AUDIT_FRAC,
        help=(
            "Under cascade, the stable fraction of deterministic hits judged "
            "anyway so strict_yes_judge_no keeps monitoring. Observational: an "
            "audit verdict never changes a score. 0 disables the monitor."
        ),
    )
    parser.add_argument(
        "--answer-format",
        choices=ANSWER_FORMATS,
        default=None,
        help=(
            "direct preserves historical prompts/parsers; boxed preserves the "
            "older local EasyR1 profile; answer_tag uses ZwZ's original system "
            "prompt and scores only the final <answer> tag. Default: direct, "
            "or answer_tag when --use_cot true."
        ),
    )
    parser.add_argument(
        "--generation-profile",
        choices=GENERATION_PROFILES,
        default=QWEN3VL_GENERATION_PROFILE,
        help=(
            "qwen3vl preserves the existing Qwen3-VL request and prompt "
            "behavior. qwen35_nothinking changes only the Qwen3.5 chat-template "
            "switch; benchmark prompt construction remains exactly on the "
            "Qwen3-VL path. qwen35_thinking uses native "
            "thinking with no ZwZ "
            "system prompt and appends the final <answer> instruction to the "
            "user message."
        ),
    )
    parser.add_argument(
        "--use_cot",
        "--use-cot",
        type=parse_bool,
        default=False,
        help=(
            "Allow chain-of-thought for any served VLM by removing benchmark "
            "direct/only directives. This selects answer_tag output unless an "
            "explicit non-direct --answer-format is supplied."
        ),
    )
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument(
        "--mme-concurrency",
        type=int,
        default=4,
        help="MME-RealWorld request concurrency; kept low because images reach 86 MP/120 MB.",
    )
    parser.add_argument("--output", required=True)
    return parser


def encode_image(path: Path) -> str:
    # Lossless PNG so the served pixels match what the HF processor would load
    # from disk (JPEG re-encoding perturbs borderline MCQ decisions).
    with Image.open(path) as raw:
        image = raw.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def encode_original_image(path: Path) -> tuple[str, str]:
    """Send the file's own bytes -- no decode, no re-encode.

    This is *not* the lossy "JPEG re-encoding" that :func:`encode_image`'s
    comment warns about: passing the original bytes through means the server
    decodes exactly the file the HF processor would have opened, so the pixels
    are bit-identical to the PNG detour while costing ~1000x less CPU (a 4K
    HRBench JPEG is 1.7 MB and ~4 ms here, versus a 14.8 MB PNG and ~5.7 s).
    Prefer this for any on-disk image; :func:`encode_image` remains only for
    callers that must normalise an unknown/alpha-bearing format to RGB.
    """
    media_type = IMAGE_MEDIA_TYPES.get(path.suffix.lower())
    if media_type is None:
        raise ValueError(f"unsupported image type for {path}: {path.suffix!r}")
    return base64.b64encode(path.read_bytes()).decode(), media_type


# When set (by an entry point's --min-tokens), every request forbids EOS/stop
# before this many generated tokens.  Diagnostic knob for the empty-output
# (first-token <|im_end|>) failure mode of some RL checkpoints; vLLM accepts
# ``min_tokens`` as a sampling-params extension of the OpenAI chat API.
MIN_TOKENS: int | None = None
# When non-empty, chat() prefixes the user text with this literal string so
# vLLM does not insert its own "<placeholder>\n" join. Deliberately a raw
# string rather than a boolean: the value is model-family specific
# ("<|vision_start|><|image_pad|><|vision_end|>" for Qwen3-VL/Qwen3.5) and
# must be stated explicitly at the call site rather than inferred. Default
# None keeps every existing evaluation byte-identical.
MM_TEXT_PLACEHOLDER: str | None = None
QWEN_VISION_TEXT_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


class GeneratedText(str):
    """String-compatible generation retaining response metadata for sample logs."""
    def __new__(cls, text, metadata):
        value = super().__new__(cls, text)
        value.response_metadata = metadata
        return value


def chat(
    endpoint: str,
    model: str,
    image_b64: str,
    prompt: str,
    max_tokens: int,
    retries: int = 5,
    image_media_type: str = "image/png",
    system_prompt: str | None = None,
    generation_profile_name: str = QWEN3VL_GENERATION_PROFILE,
) -> str:
    # Train/eval prompt-parity switch (default off; see MM_TEXT_PLACEHOLDER).
    # vLLM's chat_utils.py builds the prompt as
    #     "\n".join(missing_placeholders + [text_prompt])
    # so when the client sends the image as a separate content part, vLLM
    # PREPENDS the vision placeholder and joins it to the text with a newline.
    # Training never sees that newline: verl renders prompts with
    # processor.apply_chat_template on structured content parts, which emits
    # "<|vision_end|>What is ..." with no separator (verified against a real
    # 9B RL run: 512/512 dumped rollout prompts have no extra newline).
    # Writing the placeholder into the text ourselves leaves
    # missing_placeholders empty, so the join degenerates to the identity and
    # the prompt matches training token-for-token.
    if MM_TEXT_PLACEHOLDER:
        prompt = f"{MM_TEXT_PLACEHOLDER}{prompt}"
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_media_type};base64,{image_b64}"
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }
    )
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if MIN_TOKENS is not None:
        payload["min_tokens"] = int(MIN_TOKENS)
    if generation_profile_name == QWEN35_GENERATION_PROFILE:
        payload["chat_template_kwargs"] = {"enable_thinking": True}
    elif generation_profile_name == QWEN35_NOTHINKING_GENERATION_PROFILE:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    elif generation_profile_name not in (QWEN3VL_GENERATION_PROFILE, "glm53_default"):
        raise ValueError(
            "generation_profile_name must be one of "
            f"{GENERATION_PROFILES}, got {generation_profile_name!r}"
        )
    data = json.dumps(payload).encode()
    last_exc = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                f"{endpoint}/chat/completions", data=data,
                headers={"Content-Type": "application/json"},
            )
            response = json.loads(urllib.request.urlopen(request, timeout=600).read())
            # With a reasoning parser, Qwen3.5 returns ``content=null`` when
            # generation exhausts its token budget before closing </think>.
            # That is a normal missing-final-answer result, not a transient
            # transport failure worth retrying five times.
            message = response["choices"][0]["message"]
            content = message.get("content")
            reasoning = message.get("reasoning") or message.get(
                "reasoning_content"
            )
            # Score the raw generation the training reward saw.  A server-side
            # reasoning parser can route a spontaneously opened <think> block
            # -- answer tag included -- into `reasoning`, leaving content
            # empty (observed on the fp32 ref-train RL checkpoint: 121/1500
            # MMStar answers hidden this way).  Concatenation is an identity
            # when reasoning is empty, and downstream answer-tag parsing takes
            # the LAST tag, so ordinary responses are unaffected.
            return GeneratedText((str(reasoning or "") + str(content or "")).strip(), {
                "content": content, "reasoning_content": reasoning,
                "evaluation_refusal": response.get("_evaluation_refusal"),
                "finish_reason": response["choices"][0].get("finish_reason"),
                "usage": response.get("usage"), "response_model": response.get("model"),
                "response_id": response.get("id"), "image_transport": response.get("_evaluation_image_transport")})
        except urllib.error.HTTPError as exc:
            try:
                body = json.loads(exc.read(16384))
                error = body.get("error", {})
                detail = str(error.get("message", "") if isinstance(error, dict) else error)[:2000]
                detail = re.sub(r'(?i)(?:ak_|sk-)[a-z0-9_-]{12,}', '[REDACTED]', detail)
                detail = re.sub(r'data:[^\s"<>]+', '[IMAGE_DATA_REDACTED]', detail)
            except (ValueError, AttributeError):
                detail = "non-JSON error body omitted"
            last_exc = RuntimeError(f"HTTP {exc.code}: {detail}")
            if exc.code in (400, 401, 403, 404, 413, 422):
                raise RuntimeError(f"Non-retryable API error: {last_exc}") from None
            _time.sleep(min(2 ** attempt, 30))
        except Exception as exc:  # noqa: BLE001 — transient 502/broken-pipe during warmup
            last_exc = exc
            _time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"chat failed after {retries} retries: {last_exc}")


def answer_format(args) -> str:
    """Return the requested answer profile while preserving legacy callers."""

    value = str(getattr(args, "answer_format", "direct"))
    if value not in ANSWER_FORMATS:
        raise ValueError(
            f"answer_format must be one of {ANSWER_FORMATS}, got {value!r}"
        )
    return value


def generation_profile_for(args) -> str:
    """Return an explicit request profile, defaulting to Qwen3-VL."""

    value = str(
        getattr(args, "generation_profile", QWEN3VL_GENERATION_PROFILE)
    )
    if value not in GENERATION_PROFILES:
        raise ValueError(
            f"generation_profile must be one of {GENERATION_PROFILES}, "
            f"got {value!r}"
        )
    return value


def use_cot(args) -> bool:
    """Return the model-agnostic CoT prompt switch."""

    value = getattr(args, "use_cot", False)
    if not isinstance(value, bool):
        raise TypeError(f"use_cot must be bool, got {type(value).__name__}")
    return value


def default_max_new_tokens(args) -> int:
    """Resolve the generation cap without changing Qwen3-VL defaults."""

    if generation_profile_for(args) in QWEN35_GENERATION_PROFILES or generation_profile_for(args) == "glm53_default":
        return QWEN35_DEFAULT_MAX_NEW_TOKENS
    return 8 if answer_format(args) == "direct" else 512


def validate_generation_options(args) -> None:
    """Reject profile combinations whose prompts cannot be scored safely."""

    if answer_format(args) == "direct_answer_tag" and use_cot(args):
        raise ValueError(
            "answer_format=direct_answer_tag requires --use_cot false"
        )
    profile = generation_profile_for(args)
    if profile != QWEN35_GENERATION_PROFILE:
        return
    if answer_format(args) != "answer_tag" or not use_cot(args):
        raise ValueError(
            "generation_profile=qwen35_thinking requires "
            "--answer-format answer_tag and --use_cot true"
        )


def output_path_for_profile(output: str | Path, cot_enabled: bool) -> Path:
    """Add a visible ``cot`` filename tag without duplicating an existing one."""

    path = Path(output)
    if not cot_enabled:
        return path
    stem = path.stem
    if re.search(r"(?:^|[-_])cot(?:$|[-_])", stem, flags=re.IGNORECASE):
        return path
    if re.search(r"(?:^|[-_])nocot$", stem, flags=re.IGNORECASE):
        stem = re.sub(r"nocot$", "cot", stem, flags=re.IGNORECASE)
    else:
        stem = f"{stem}_cot"
    return path.with_name(f"{stem}{path.suffix}")


def format_evaluation_prompt(prompt: str, args) -> str:
    """Apply the selected output profile to cleaned benchmark content."""

    profile = answer_format(args)
    if use_cot(args) and profile == "direct":
        raise ValueError("use_cot=true requires a non-direct answer_format")
    if profile == "answer_tag" and not use_cot(args):
        raise ValueError("answer_format=answer_tag requires use_cot=true")
    if profile == "direct":
        return prompt
    if profile == "direct_answer_tag":
        return (
            f"{str(prompt).rstrip()}\n{DIRECT_ANSWER_TAG_INSTRUCTION}"
        )
    if profile == "answer_tag":
        cleaned = str(prompt).strip()
        if generation_profile_for(args) == 'glm53_default':
            return f"{cleaned}\n\n{GLM53_ANSWER_TAG_REMINDER}"
        if generation_profile_for(args) == QWEN35_GENERATION_PROFILE:
            return f"{cleaned}\n\n{QWEN35_ANSWER_TAG_INSTRUCTION}"
        return cleaned
    # perception.jinja applies ``content | trim``, emits one blank line, then
    # this exact instruction.  Mirror that transformation byte-for-byte.
    return f"{str(prompt).strip()}\n\n{BOXED_ANSWER_INSTRUCTION}"


def system_prompt_for(args) -> str | None:
    """Return the true system-role prompt for the aligned ZwZ profile."""

    if generation_profile_for(args) == QWEN35_GENERATION_PROFILE:
        return None
    return ZWZ_SYSTEM_PROMPT if answer_format(args) == "answer_tag" else None


def prompt_profile_for(args) -> str:
    """Return a stable summary label for the exact prompt construction."""

    generation_profile = generation_profile_for(args)
    if generation_profile == QWEN35_GENERATION_PROFILE:
        return "qwen35_native_thinking_answer_tag"
    if generation_profile == QWEN35_NOTHINKING_GENERATION_PROFILE:
        if answer_format(args) == "direct_answer_tag":
            return "qwen35_nothinking_qwen3vl_direct_answer_tag"
        if answer_format(args) == "answer_tag":
            return "qwen35_nothinking_zwz_system_answer_tag"
        if use_cot(args):
            return "qwen35_nothinking_question_options_boxed"
        if answer_format(args) == "boxed":
            return "qwen35_nothinking_canonical_instruction_boxed"
        return "qwen35_nothinking_canonical_direct"
    if answer_format(args) == "answer_tag":
        return "zwz_system_answer_tag"
    if answer_format(args) == "direct_answer_tag":
        return "qwen3vl_direct_answer_tag"
    if use_cot(args):
        return "question_options_boxed"
    if answer_format(args) == "boxed":
        return "canonical_instruction_boxed"
    return "canonical_direct"


def _matching_brace(text: str, opening_brace: int) -> int | None:
    depth = 0
    for index in range(opening_brace, len(text)):
        char = text[index]
        if char == "{" and (index == 0 or text[index - 1] != "\\"):
            depth += 1
        elif char == "}" and (index == 0 or text[index - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return index
            if depth < 0:
                return None
    return None


def extract_final_boxed_answer(response: str) -> str | None:
    """Match the RL reward: return the last non-empty balanced ``\\boxed``."""

    if not isinstance(response, str):
        return None
    prefix = r"\boxed{"
    boxes: list[str] = []
    search_from = 0
    while True:
        start = response.find(prefix, search_from)
        if start < 0:
            break
        opening_brace = start + len(prefix) - 1
        closing_brace = _matching_brace(response, opening_brace)
        if closing_brace is not None:
            boxes.append(response[opening_brace + 1 : closing_brace])
        search_from = start + len(prefix)
    if not boxes:
        return None
    answer = boxes[-1].strip()
    return answer or None


def extract_final_answer_tag(response: str) -> str | None:
    """Return the last non-empty complete ``<answer>`` tag content."""

    if not isinstance(response, str):
        return None
    answers = ANSWER_TAG_PATTERN.findall(response)
    if not answers:
        return None
    answer = answers[-1].strip()
    return answer or None


def scoring_payload(generated: str, args) -> tuple[str, str | None]:
    """Return parser input plus the extracted final answer for diagnostics."""

    profile = answer_format(args)
    if profile == "direct":
        return generated, None
    if profile == "boxed":
        extracted = extract_final_boxed_answer(generated)
    else:
        extracted = extract_final_answer_tag(generated)
    return (extracted or ""), extracted


def add_boxed_record_fields(
    record: dict,
    *,
    args,
    prompt: str,
    boxed_answer: str | None,
) -> None:
    """Attach non-direct format diagnostics; keep the legacy public name."""

    record.update(getattr(record.get("generated"), "response_metadata", {}))
    if record.get('evaluation_refusal'):
        record.update(correct=False, score=0.0, judge_status='content_policy_violation')
    profile = answer_format(args)
    if profile == "direct":
        return
    valid = boxed_answer is not None
    record.update(
        {
            "answer_format": profile,
            "prompt": prompt,
            "format_valid": valid,
            "use_cot": use_cot(args),
        }
    )
    if profile == "boxed":
        record.update(
            {
                "boxed_answer": boxed_answer,
                "boxed_format_valid": valid,
            }
        )
    else:
        record.update(
            {
                "answer_tag_answer": boxed_answer,
                "answer_tag_format_valid": valid,
            }
        )


def add_boxed_summary(summary: dict, records: list[dict], args) -> dict:
    """Add boxed or answer-tag format metrics under compatible field names."""

    profile = answer_format(args)
    if profile == "direct":
        return summary
    field = (
        "boxed_format_valid"
        if profile == "boxed"
        else "answer_tag_format_valid"
    )
    valid = sum(bool(record[field]) for record in records)
    summary.update(
        {
            "answer_format": profile,
            "format_valid": valid,
            "format_missing": len(records) - valid,
            "format_rate": valid / len(records) if records else 0.0,
            "use_cot": use_cot(args),
        }
    )
    prefix = "boxed" if profile == "boxed" else "answer_tag"
    summary.update(
        {
            f"{prefix}_format_valid": valid,
            f"{prefix}_format_missing": len(records) - valid,
            f"{prefix}_format_rate": (
                valid / len(records) if records else 0.0
            ),
        }
    )
    return summary


def _without_terminal_instruction(prompt: str, instruction: str, dataset: str) -> str:
    """Remove one known answer directive without touching question content."""

    stripped = prompt.rstrip()
    if not stripped.endswith(instruction):
        raise ValueError(
            f"{dataset} prompt does not end with its expected answer instruction"
        )
    content = stripped[: -len(instruction)].rstrip()
    if not content:
        raise ValueError(f"{dataset} prompt became empty after removing instruction")
    return content


def build_zoombench_prompt(sample: dict, args) -> str:
    """Build the legacy prompt or its CoT-permitted content-only form."""

    canonical = zoom_eval.build_prompt(sample)
    if not use_cot(args):
        return canonical
    question_type = str(sample.get("question_type"))
    if question_type == "mcq":
        instruction = zoom_eval.MCQ_INSTRUCTION
    elif question_type == "blank":
        instruction = zoom_eval.NUMBER_INSTRUCTION
    else:
        raise ValueError(f"unknown ZoomBench question_type: {question_type!r}")
    return _without_terminal_instruction(canonical, instruction, "ZoomBench")


def eval_zoombench(args) -> tuple[dict, list[dict]]:
    with Path(args.zoombench_annotation).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if args.limit:
        data = data[: args.limit]

    def one(item, endpoint):
        idx, sample = item
        image_b64 = encode_image(Path(args.zoombench_image_root) / sample["image"])
        prompt = format_evaluation_prompt(build_zoombench_prompt(sample, args), args)
        text = chat(
            endpoint, args.model, image_b64,
            prompt, args.max_new_tokens,
            system_prompt=system_prompt_for(args),
            generation_profile_name=generation_profile_for(args),
        )
        qtype = str(sample["question_type"])
        payload, boxed = scoring_payload(text, args)
        pred = zoom_eval.parse_prediction(payload, qtype)
        gt = zoom_eval.normalize_answer(sample["response"], qtype)
        record = {
            "idx": idx,
            "qtype": qtype,
            "pred": pred,
            "gt": gt,
            "correct": pred == gt,
            "generated": text,
        }
        add_boxed_record_fields(
            record,
            args=args,
            prompt=prompt,
            boxed_answer=boxed,
        )
        return record

    records = parallel_eval_requests(
        one,
        enumerate(data),
        args,
        description="zoombench",
        single_endpoint_concurrency=args.concurrency,
    )
    by_type: dict[str, list[int]] = {}
    for record in records:
        by_type.setdefault(record["qtype"], []).append(int(record["correct"]))
    summary = {
        "accuracy": sum(r["correct"] for r in records) / len(records),
        "n": len(records),
        "by_type": {k: sum(v) / len(v) for k, v in by_type.items()},
    }
    return add_boxed_summary(summary, records, args), records


VSTAR_BENCHMARKS = {
    # Keep ``vstar`` compatible with the historical P4 name: it denotes the
    # 115 direct-attribute examples.  The two extra names expose the other
    # subset and the full 191-example aggregate to the same vLLM worker.
    "vstar": ("direct_attributes", 115),
    "vstar_relpos": ("relative_position", 76),
    "vstar_all": (None, 191),
}


def build_vstar_prompt(sample: dict, args=None) -> str:
    """Return canonical VStar text, optionally without its answer directive."""

    prompt = sample.get("text")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("VStar sample has an empty text prompt")
    if use_cot(args):
        return _without_terminal_instruction(
            prompt,
            zoom_eval.MCQ_INSTRUCTION,
            "VStar",
        )
    return prompt


def eval_vstar(args, benchmark: str) -> tuple[dict, list[dict]]:
    """Evaluate a VStar subset by greedy letter generation through vLLM."""
    if benchmark not in VSTAR_BENCHMARKS:
        raise ValueError(f"unknown VStar benchmark {benchmark!r}")
    subset, expected_n = VSTAR_BENCHMARKS[benchmark]
    annotation = Path(args.vstar_annotation)
    image_root = Path(args.vstar_image_root)
    all_data = []
    with annotation.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                all_data.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid VStar JSONL at {annotation}:{line_number}: {exc}"
                ) from exc
    data = (
        all_data
        if subset is None
        else [sample for sample in all_data if sample.get("category") == subset]
    )
    if len(data) != expected_n:
        raise RuntimeError(
            f"expected {expected_n} VStar {subset or 'all'} samples, got {len(data)}"
        )
    if args.limit is not None:
        data = data[: args.limit]
    if not data:
        raise ValueError("VStar evaluation selected zero samples")

    def one(item, endpoint):
        idx, sample = item
        # In the Qwen3-VL profile the JSONL text is sent verbatim. The CoT
        # profile removes only its known terminal answer directive.
        prompt = format_evaluation_prompt(build_vstar_prompt(sample, args), args)
        image_path = image_root / str(sample["image"])
        if not image_path.is_file():
            raise FileNotFoundError(f"VStar image not found: {image_path}")
        image_b64, image_media_type = encode_original_image(image_path)
        text = chat(
            endpoint,
            args.model,
            image_b64,
            prompt,
            args.max_new_tokens,
            image_media_type=image_media_type,
            system_prompt=system_prompt_for(args),
            generation_profile_name=generation_profile_for(args),
        )
        payload, boxed = scoring_payload(text, args)
        pred = zoom_eval.parse_prediction(payload, "mcq")
        gt = str(sample.get("label", "")).strip().upper()
        if gt not in {"A", "B", "C", "D"}:
            raise ValueError(
                f"invalid VStar label {sample.get('label')!r} for "
                f"question_id={sample.get('question_id')!r}"
            )
        record = dict(sample)
        record.update({
            "idx": idx,
            "image_path": str(image_path),
            "prompt": prompt,
            "pred": pred,
            "gt": gt,
            "correct": pred == gt,
            "generated": text,
        })
        add_boxed_record_fields(
            record,
            args=args,
            prompt=prompt,
            boxed_answer=boxed,
        )
        return record

    records = parallel_eval_requests(
        one,
        enumerate(data),
        args,
        description=benchmark,
        single_endpoint_concurrency=args.concurrency,
    )
    correct = sum(bool(record["correct"]) for record in records)
    by_type: dict[str, list[int]] = defaultdict(list)
    for record in records:
        by_type[record["category"]].append(int(record["correct"]))
    summary = {
        "accuracy": correct / len(records),
        "correct": correct,
        "n": len(records),
        "by_type": {key: sum(values) / len(values) for key, values in by_type.items()},
    }
    return add_boxed_summary(summary, records, args), records


def eval_hrbench(args, version: str) -> tuple[dict, list[dict]]:
    base = Path(args.hrbench_root) / f"hr-bench_{version}"
    with (base / f"annotation_hr-bench_{version}.json").open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if args.limit:
        data = data[: args.limit]

    tasks = []
    for sample in data:
        for cycle, (options, gt_letter) in enumerate(zip(sample["options"], sample["answer"])):
            tasks.append((sample, cycle, options, gt_letter))

    # Encode every distinct image exactly once, up front.  HRBench asks four
    # cycle questions per image, and the previous cache released its lock
    # before encoding, so all four threads missed together and re-encoded the
    # same 4K/8K file.  Worse, that encode ran *while holding a vLLM endpoint
    # slot*, so engines sat idle waiting on client CPU.  Pre-encoding also
    # keeps the (short) work off the request path entirely.
    image_cache: dict[str, tuple[str, str]] = {
        key: encode_original_image(base / key)
        for key in dict.fromkeys(sample["input_image"] for sample in data)
    }

    def one(task, endpoint):
        sample, cycle, options, gt_letter = task
        image_b64, image_media_type = image_cache[sample["input_image"]]
        prompt = format_evaluation_prompt(
            build_hrbench_prompt(sample, options, args),
            args,
        )
        text = chat(
            endpoint,
            args.model,
            image_b64,
            prompt,
            args.max_new_tokens,
            image_media_type=image_media_type,
            system_prompt=system_prompt_for(args),
            generation_profile_name=generation_profile_for(args),
        )
        payload, boxed = scoring_payload(text, args)
        pred = zoom_eval.parse_prediction(payload, "mcq")
        record = {
            "index": sample["index"],
            "cycle": cycle,
            "category": str(sample.get("category")),
            "pred": pred,
            "gt": str(gt_letter),
            "correct": pred == str(gt_letter),
            "generated": text,
        }
        add_boxed_record_fields(
            record,
            args=args,
            prompt=prompt,
            boxed_answer=boxed,
        )
        return record

    records = parallel_eval_requests(
        one,
        tasks,
        args,
        description=f"hrbench{version}",
        single_endpoint_concurrency=args.concurrency,
    )
    by_category: dict[str, list[int]] = {}
    for record in records:
        by_category.setdefault(record["category"], []).append(int(record["correct"]))
    summary = {
        "accuracy": sum(r["correct"] for r in records) / len(records),
        "n": len(records),
        "by_type": {k: sum(v) / len(v) for k, v in by_category.items()},
    }
    return add_boxed_summary(summary, records, args), records


def build_hrbench_prompt(sample: dict, options, args) -> str:
    """Build HRBench question/options, with legacy directive unless CoT is on."""

    content = f"{str(sample['question']).strip()}\n{str(options).strip()}"
    if use_cot(args):
        return content
    return f"{content}\n{HRBENCH_MCQ_INSTRUCTION}"


def load_treebench(annotation: str | Path) -> list[dict]:
    """Load the local TSV without trusting its auxiliary A--K columns.

    TreeBench's ``image`` field is an already encoded JPEG.  The authoritative
    option text is the combined ``multi-choice options`` field: all 68 OCR
    samples leave A--K empty, and a few other rows split same-line options
    incorrectly across those helper columns.
    """

    path = Path(annotation)
    if not path.is_file():
        raise FileNotFoundError(f"TreeBench annotation not found: {path}")
    csv.field_size_limit(100_000_000)
    required = {
        "index",
        "image",
        "question",
        "answer",
        "multi-choice options",
        "category",
        "l2-category",
        "target_instances",
    }
    rows: list[dict] = []
    seen_indices: set[int] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"TreeBench TSV is missing required columns: {sorted(missing)}"
            )
        for line_number, raw in enumerate(reader, start=2):
            if None in raw:
                raise ValueError(
                    f"TreeBench TSV has extra fields at {path}:{line_number}"
                )
            try:
                index = int(str(raw["index"]).strip())
            except ValueError as exc:
                raise ValueError(
                    f"invalid TreeBench index at {path}:{line_number}: "
                    f"{raw['index']!r}"
                ) from exc
            if index in seen_indices:
                raise ValueError(f"duplicate TreeBench index: {index}")
            seen_indices.add(index)

            image = str(raw["image"]).strip()
            question = str(raw["question"]).strip()
            options = str(raw["multi-choice options"]).strip()
            answer = str(raw["answer"]).strip().upper()
            category = str(raw["category"]).strip()
            l2_category = str(raw["l2-category"]).strip()
            if not image:
                raise ValueError(f"TreeBench index {index} has an empty image")
            if not question:
                raise ValueError(f"TreeBench index {index} has an empty question")
            if not options:
                raise ValueError(f"TreeBench index {index} has empty options")
            if answer not in TREEBENCH_ANSWER_LETTERS:
                raise ValueError(
                    f"TreeBench index {index} has invalid answer {answer!r}"
                )
            if not category:
                raise ValueError(f"TreeBench index {index} has an empty category")
            try:
                target_instances = json.loads(str(raw["target_instances"]))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"TreeBench index {index} has invalid target_instances"
                ) from exc

            rows.append(
                {
                    "index": index,
                    "image": image,
                    "question": question,
                    "options": options,
                    "answer": answer,
                    "category": category,
                    "l2_category": l2_category,
                    "target_instances": target_instances,
                }
            )
    return rows


def build_treebench_prompt(sample: dict, args=None) -> str:
    """Build the shared answer-only TreeBench MCQ prompt."""

    question = str(sample["question"]).strip()
    options = str(sample["options"]).strip()
    if not question or not options:
        raise ValueError("TreeBench prompt requires a question and options")
    content = f"{question}\nOptions:\n{options}"
    if use_cot(args):
        return content
    return f"{content}\n{TREEBENCH_MCQ_INSTRUCTION}"


def parse_treebench_prediction(text: str) -> str:
    """Extract one generated TreeBench option letter (A--K)."""

    match = re.search(r"\b([A-K])\b", str(text), flags=re.IGNORECASE)
    return match.group(1).upper() if match else ""


def eval_treebench(args) -> tuple[dict, list[dict]]:
    """Evaluate TreeBench as answer-letter accuracy only (no box IoU)."""

    data = load_treebench(args.treebench_annotation)
    if len(data) != TREEBENCH_EXPECTED_N:
        raise RuntimeError(
            f"expected {TREEBENCH_EXPECTED_N} TreeBench samples, got {len(data)}"
        )
    expected_indices = list(range(TREEBENCH_EXPECTED_N))
    actual_indices = [sample["index"] for sample in data]
    if actual_indices != expected_indices:
        raise RuntimeError(
            "TreeBench indices must be sequential from 0 through "
            f"{TREEBENCH_EXPECTED_N - 1}"
        )
    if args.limit is not None:
        data = data[: args.limit]
    if not data:
        raise ValueError("TreeBench evaluation selected zero samples")

    def one(sample, endpoint):
        prompt = format_evaluation_prompt(
            build_treebench_prompt(sample, args),
            args,
        )
        text = chat(
            endpoint,
            args.model,
            sample["image"],
            prompt,
            args.max_new_tokens,
            image_media_type="image/jpeg",
            system_prompt=system_prompt_for(args),
            generation_profile_name=generation_profile_for(args),
        )
        payload, boxed = scoring_payload(text, args)
        pred = parse_treebench_prediction(payload)
        gt = sample["answer"]
        record = {
            # Deliberately omit the 0.3--1.7 MB base64 image from result dumps.
            "index": sample["index"],
            "category": sample["category"],
            "l2_category": sample["l2_category"],
            "question": sample["question"],
            "options": sample["options"],
            "target_instances": sample["target_instances"],
            "prompt": prompt,
            "pred": pred,
            "gt": gt,
            "correct": pred == gt,
            "generated": text,
        }
        add_boxed_record_fields(
            record,
            args=args,
            prompt=prompt,
            boxed_answer=boxed,
        )
        return record

    records = parallel_eval_requests(
        one,
        data,
        args,
        description="treebench",
        single_endpoint_concurrency=args.concurrency,
    )
    correct = sum(bool(record["correct"]) for record in records)
    by_category: dict[str, list[int]] = defaultdict(list)
    for record in records:
        by_category[record["category"]].append(int(record["correct"]))
    summary = {
        "accuracy": correct / len(records),
        "correct": correct,
        "n": len(records),
        "metric_profile": "answer_letter_accuracy_only",
        "by_type": {
            category: sum(values) / len(values)
            for category, values in by_category.items()
        },
    }
    return add_boxed_summary(summary, records, args), records


def _image_media_type_from_bytes(
    image_bytes: bytes,
    *,
    sample_id: int,
    dataset: str = "CV-Bench",
) -> str:
    """Identify an embedded image by its bytes, not its misleading suffix."""

    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if (
        len(image_bytes) >= 12
        and image_bytes[:4] == b"RIFF"
        and image_bytes[8:12] == b"WEBP"
    ):
        return "image/webp"
    raise ValueError(
        f"{dataset} sample {sample_id} has an unsupported embedded image"
    )


def _cvbench_relation_axis(choices: list[str], *, sample_id: int) -> str | None:
    normalized = {choice.strip().lower() for choice in choices}
    if normalized == {"left", "right"}:
        return "horizontal_left_right"
    if normalized == {"above", "below"}:
        return "vertical_above_below"
    if normalized and len(normalized) == 2:
        raise ValueError(
            f"CV-Bench Relation sample {sample_id} has unexpected choices "
            f"{choices!r}"
        )
    return None


def load_cvbench(root: str | Path) -> list[dict]:
    """Load and fully validate the official CV-Bench parquet files."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "CV-Bench evaluation requires pyarrow to read its parquet files"
        ) from exc

    dataset_root = Path(root)
    required_columns = {
        "idx",
        "type",
        "task",
        "image",
        "question",
        "choices",
        "answer",
        "prompt",
        "filename",
        "source",
        "source_dataset",
        "source_filename",
        "target_class",
        "target_size",
        "bbox",
    }
    rows: list[dict] = []
    for filename, expected_type, expected_n in CVBENCH_SPLITS:
        path = dataset_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"CV-Bench parquet not found: {path}")
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != expected_n:
            raise RuntimeError(
                f"expected {expected_n} rows in {path}, got "
                f"{parquet.metadata.num_rows}"
            )
        missing = required_columns.difference(parquet.schema_arrow.names)
        if missing:
            raise ValueError(
                f"CV-Bench parquet {path} is missing columns: {sorted(missing)}"
            )

        # Keep only one small Arrow batch live at a time. The final Python
        # records retain the original bytes, avoiding a 33% all-dataset
        # base64 expansion before requests are sent.
        for batch in parquet.iter_batches(batch_size=64):
            for raw in batch.to_pylist():
                sample_id = int(raw["idx"])
                sample_type = str(raw["type"]).strip()
                task = str(raw["task"]).strip()
                source = str(raw["source"]).strip()
                question = str(raw["question"]).strip()
                prompt = str(raw["prompt"])
                choices = [str(choice).strip() for choice in raw["choices"]]
                answer_raw = str(raw["answer"]).strip().upper()
                answer_match = re.fullmatch(r"\(([A-F])\)", answer_raw)

                if sample_type != expected_type:
                    raise ValueError(
                        f"CV-Bench sample {sample_id} in {filename} has type "
                        f"{sample_type!r}, expected {expected_type!r}"
                    )
                if not question or not prompt.strip():
                    raise ValueError(
                        f"CV-Bench sample {sample_id} has an empty question/prompt"
                    )
                if not 2 <= len(choices) <= len(CVBENCH_ANSWER_LETTERS):
                    raise ValueError(
                        f"CV-Bench sample {sample_id} has invalid choices "
                        f"{choices!r}"
                    )
                if any(not choice for choice in choices):
                    raise ValueError(
                        f"CV-Bench sample {sample_id} has an empty choice"
                    )
                if answer_match is None:
                    raise ValueError(
                        f"CV-Bench sample {sample_id} has invalid answer "
                        f"{answer_raw!r}"
                    )
                answer = answer_match.group(1)
                answer_index = ord(answer) - ord("A")
                if answer_index >= len(choices):
                    raise ValueError(
                        f"CV-Bench sample {sample_id} answer {answer!r} is "
                        f"outside {len(choices)} choices"
                    )

                image = raw["image"]
                if not isinstance(image, dict) or not image.get("bytes"):
                    raise ValueError(
                        f"CV-Bench sample {sample_id} has no embedded image bytes"
                    )
                image_bytes = bytes(image["bytes"])
                media_type = _image_media_type_from_bytes(
                    image_bytes,
                    sample_id=sample_id,
                )
                relation_axis = (
                    _cvbench_relation_axis(choices, sample_id=sample_id)
                    if task == "Relation"
                    else None
                )
                rows.append(
                    {
                        "idx": sample_id,
                        "type": sample_type,
                        "task": task,
                        "_image_bytes": image_bytes,
                        "_image_media_type": media_type,
                        "question": question,
                        "choices": choices,
                        "answer": answer,
                        "answer_raw": answer_raw,
                        "prompt": prompt,
                        "filename": str(raw["filename"]),
                        "source": source,
                        "source_dataset": str(raw["source_dataset"]),
                        "source_filename": str(raw["source_filename"]),
                        "target_class": raw["target_class"],
                        "target_size": raw["target_size"],
                        "bbox": raw["bbox"],
                        "relation_axis": relation_axis,
                    }
                )

    if len(rows) != CVBENCH_EXPECTED_N:
        raise RuntimeError(
            f"expected {CVBENCH_EXPECTED_N} total CV-Bench rows, got {len(rows)}"
        )
    indices = [sample["idx"] for sample in rows]
    if indices != list(range(CVBENCH_EXPECTED_N)):
        raise RuntimeError(
            "CV-Bench indices must be unique and sequential from 0 through "
            f"{CVBENCH_EXPECTED_N - 1}"
        )
    task_counts = Counter(sample["task"] for sample in rows)
    if task_counts != Counter(CVBENCH_EXPECTED_TASK_COUNTS):
        raise RuntimeError(
            "CV-Bench task counts do not match the official release: "
            f"{dict(task_counts)}"
        )
    source_counts = Counter(sample["source"] for sample in rows)
    if source_counts != Counter(CVBENCH_EXPECTED_SOURCE_COUNTS):
        raise RuntimeError(
            "CV-Bench source counts do not match the official release: "
            f"{dict(source_counts)}"
        )
    return rows


def select_cvbench_samples(data: list[dict], limit: int | None) -> list[dict]:
    """Apply a deterministic task/source-stratified smoke-test limit."""

    if limit is None or limit >= len(data):
        return list(data)
    if limit <= 0:
        raise ValueError("CV-Bench limit must be positive")

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for sample in data:
        grouped[(sample["task"], sample["source"])].append(sample)
    missing = [key for key in CVBENCH_LIMIT_STRATA if not grouped[key]]
    if missing:
        raise RuntimeError(f"CV-Bench is missing sampling strata: {missing}")

    selected: list[dict] = []
    positions = {key: 0 for key in CVBENCH_LIMIT_STRATA}
    while len(selected) < limit:
        progressed = False
        for key in CVBENCH_LIMIT_STRATA:
            position = positions[key]
            group = grouped[key]
            if position < len(group):
                selected.append(group[position])
                positions[key] += 1
                progressed = True
                if len(selected) == limit:
                    break
        if not progressed:
            break
    if len(selected) != min(limit, len(data)):
        raise RuntimeError(
            f"CV-Bench stratified limit selected {len(selected)} of "
            f"{min(limit, len(data))} requested samples"
        )
    return selected


def build_cvbench_prompt(sample: dict, args=None) -> str:
    """Use the official prompt, optionally adding its direct-letter suffix."""

    canonical = str(sample.get("prompt", ""))
    if not canonical.strip():
        raise ValueError(
            f"CV-Bench sample {sample.get('idx')!r} has an empty prompt"
        )
    if use_cot(args):
        return canonical
    return f"{canonical.rstrip()}\n{CVBENCH_MCQ_INSTRUCTION}"


def parse_cvbench_prediction(text: str) -> str:
    """Extract only an A--F option letter from the final scoring payload."""

    match = re.search(r"\b([A-F])\b", str(text), flags=re.IGNORECASE)
    return match.group(1).upper() if match else ""


def _cvbench_group_accuracy(
    records: list[dict],
    field: str,
) -> tuple[dict[str, float], dict[str, int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for record in records:
        value = record.get(field)
        if value is not None:
            groups[str(value)].append(int(record["correct"]))
    accuracies = {
        key: sum(values) / len(values)
        for key, values in sorted(groups.items())
    }
    counts = {key: len(values) for key, values in sorted(groups.items())}
    return accuracies, counts


def summarize_cvbench(records: list[dict], *, full_n: int) -> dict:
    """Compute the official source-balanced score plus diagnostic slices."""

    if not records:
        raise ValueError("CV-Bench evaluation selected zero samples")
    correct = sum(bool(record["correct"]) for record in records)
    micro_accuracy = correct / len(records)
    by_source, source_counts = _cvbench_group_accuracy(records, "source")
    by_task, task_counts = _cvbench_group_accuracy(records, "task")
    micro_by_type, type_counts = _cvbench_group_accuracy(records, "type")
    by_source_dataset, source_dataset_counts = _cvbench_group_accuracy(
        records,
        "source_dataset",
    )
    relation_records = [
        record for record in records if record.get("relation_axis") is not None
    ]
    by_relation_axis, relation_axis_counts = _cvbench_group_accuracy(
        relation_records,
        "relation_axis",
    )

    required_sources = {"ADE20K", "COCO", "Omni3D"}
    official_available = required_sources.issubset(by_source)
    if official_available:
        accuracy_2d = (by_source["ADE20K"] + by_source["COCO"]) / 2
        accuracy_3d = by_source["Omni3D"]
        official_accuracy = (accuracy_2d + accuracy_3d) / 2
        accuracy = official_accuracy
        metric_profile = (
            "cvbench_official_source_balanced_answer_letter_accuracy"
        )
        by_type = {"2D": accuracy_2d, "3D": accuracy_3d}
    else:
        accuracy_2d = None
        accuracy_3d = by_source.get("Omni3D")
        official_accuracy = None
        accuracy = micro_accuracy
        metric_profile = "cvbench_partial_micro_answer_letter_accuracy"
        by_type = micro_by_type

    return {
        "accuracy": accuracy,
        "official_accuracy": official_accuracy,
        "micro_accuracy": micro_accuracy,
        "correct": correct,
        "n": len(records),
        "full_n": full_n,
        "limited_diagnostic": len(records) != full_n,
        "official_score_available": official_available,
        "accuracy_2d": accuracy_2d,
        "accuracy_3d": accuracy_3d,
        "metric_profile": metric_profile,
        "by_type": by_type,
        "micro_by_type": micro_by_type,
        "by_source": by_source,
        "by_task": by_task,
        "by_relation_axis": by_relation_axis,
        "by_source_dataset": by_source_dataset,
        "group_counts": {
            "type": type_counts,
            "source": source_counts,
            "task": task_counts,
            "relation_axis": relation_axis_counts,
            "source_dataset": source_dataset_counts,
        },
    }


def eval_cvbench(args) -> tuple[dict, list[dict]]:
    """Evaluate CV-Bench using answer letters and its official aggregation."""

    full_data = load_cvbench(args.cvbench_root)
    data = select_cvbench_samples(full_data, args.limit)

    def one(sample, endpoint):
        prompt = format_evaluation_prompt(
            build_cvbench_prompt(sample, args),
            args,
        )
        image_b64 = base64.b64encode(sample["_image_bytes"]).decode("ascii")
        text = chat(
            endpoint,
            args.model,
            image_b64,
            prompt,
            args.max_new_tokens,
            image_media_type=sample["_image_media_type"],
            system_prompt=system_prompt_for(args),
            generation_profile_name=generation_profile_for(args),
        )
        payload, boxed = scoring_payload(text, args)
        pred = parse_cvbench_prediction(payload)
        gt = sample["answer"]
        pred_index = ord(pred) - ord("A") if pred else -1
        gt_index = ord(gt) - ord("A")
        record = {
            "idx": sample["idx"],
            "type": sample["type"],
            "task": sample["task"],
            "source": sample["source"],
            "source_dataset": sample["source_dataset"],
            "source_filename": sample["source_filename"],
            "filename": sample["filename"],
            "question": sample["question"],
            "choices": sample["choices"],
            "answer_raw": sample["answer_raw"],
            "relation_axis": sample["relation_axis"],
            "target_class": sample["target_class"],
            "target_size": sample["target_size"],
            "bbox": sample["bbox"],
            "image_media_type": sample["_image_media_type"],
            "prompt": prompt,
            "pred": pred,
            "pred_choice": (
                sample["choices"][pred_index]
                if 0 <= pred_index < len(sample["choices"])
                else None
            ),
            "gt": gt,
            "gt_choice": sample["choices"][gt_index],
            "correct": pred == gt,
            "generated": text,
        }
        add_boxed_record_fields(
            record,
            args=args,
            prompt=prompt,
            boxed_answer=boxed,
        )
        return record

    records = parallel_eval_requests(
        one,
        data,
        args,
        description="cvbench",
        single_endpoint_concurrency=args.concurrency,
    )
    summary = summarize_cvbench(records, full_n=len(full_data))
    return add_boxed_summary(summary, records, args), records


def load_mmstar(root: str | Path) -> list[dict]:
    """Load and fully validate the official 1,500-row MMStar parquet."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "MMStar evaluation requires pyarrow to read mmstar.parquet"
        ) from exc

    dataset_root = Path(root)
    path = dataset_root / "mmstar.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"MMStar parquet not found: {path}")

    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != MMSTAR_EXPECTED_N:
        raise RuntimeError(
            f"expected {MMSTAR_EXPECTED_N} MMStar rows, got "
            f"{parquet.metadata.num_rows}"
        )
    required_columns = {
        "index",
        "question",
        "answer",
        "category",
        "l2_category",
        "image",
        "meta_info",
    }
    missing = required_columns.difference(parquet.schema_arrow.names)
    if missing:
        raise ValueError(
            f"MMStar parquet is missing columns: {sorted(missing)}"
        )

    rows: list[dict] = []
    for batch in parquet.iter_batches(batch_size=64):
        for raw in batch.to_pylist():
            try:
                sample_id = int(raw["index"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"MMStar has an invalid index {raw.get('index')!r}"
                ) from exc

            question_raw = raw["question"]
            answer_raw = raw["answer"]
            category_raw = raw["category"]
            l2_raw = raw["l2_category"]
            if not isinstance(question_raw, str) or not question_raw.strip():
                raise ValueError(
                    f"MMStar sample {sample_id} has an empty question"
                )
            if not isinstance(answer_raw, str):
                raise ValueError(
                    f"MMStar sample {sample_id} has a non-string answer"
                )
            answer = answer_raw.strip().upper()
            if answer not in MMSTAR_ANSWER_LETTERS:
                raise ValueError(
                    f"MMStar sample {sample_id} has invalid answer {answer!r}"
                )
            if not isinstance(category_raw, str):
                raise ValueError(
                    f"MMStar sample {sample_id} has a non-string category"
                )
            if not isinstance(l2_raw, str):
                raise ValueError(
                    f"MMStar sample {sample_id} has a non-string l2_category"
                )
            category = category_raw.strip()
            l2_category = l2_raw.strip()
            expected_l2 = MMSTAR_CATEGORY_L2.get(category)
            if expected_l2 is None:
                raise ValueError(
                    f"MMStar sample {sample_id} has unknown category "
                    f"{category!r}"
                )
            if l2_category not in expected_l2:
                raise ValueError(
                    f"MMStar sample {sample_id} maps l2_category "
                    f"{l2_category!r} to unexpected category {category!r}"
                )

            meta = raw["meta_info"]
            if not isinstance(meta, dict):
                raise ValueError(
                    f"MMStar sample {sample_id} has invalid meta_info"
                )
            source = str(meta.get("source") or "").strip()
            split = str(meta.get("split") or "").strip()
            image_path = str(meta.get("image_path") or "").strip()
            if not source or not image_path:
                raise ValueError(
                    f"MMStar sample {sample_id} has incomplete meta_info"
                )
            if split != "val":
                raise ValueError(
                    f"MMStar sample {sample_id} has split {split!r}, "
                    "expected 'val'"
                )

            image_raw = raw["image"]
            if not isinstance(image_raw, (bytes, bytearray, memoryview)):
                raise ValueError(
                    f"MMStar sample {sample_id} has no embedded image bytes"
                )
            image_bytes = bytes(image_raw)
            if not image_bytes:
                raise ValueError(
                    f"MMStar sample {sample_id} has empty image bytes"
                )
            media_type = _image_media_type_from_bytes(
                image_bytes,
                sample_id=sample_id,
                dataset="MMStar",
            )

            rows.append(
                {
                    "index": sample_id,
                    "question": question_raw.strip(),
                    "answer": answer,
                    "category": category,
                    "l2_category": l2_category,
                    "source": source,
                    "split": split,
                    "image_path": image_path,
                    "_image_bytes": image_bytes,
                    "_image_media_type": media_type,
                }
            )

    if len(rows) != MMSTAR_EXPECTED_N:
        raise RuntimeError(
            f"expected {MMSTAR_EXPECTED_N} total MMStar rows, got {len(rows)}"
        )
    indices = [sample["index"] for sample in rows]
    if indices != list(range(MMSTAR_EXPECTED_N)):
        raise RuntimeError(
            "MMStar indices must be unique and sequential from 0 through "
            f"{MMSTAR_EXPECTED_N - 1}"
        )
    category_counts = Counter(sample["category"] for sample in rows)
    if category_counts != Counter(MMSTAR_EXPECTED_CATEGORY_COUNTS):
        raise RuntimeError(
            "MMStar category counts do not match the official release: "
            f"{dict(category_counts)}"
        )
    l2_counts = Counter(sample["l2_category"] for sample in rows)
    if l2_counts != Counter(MMSTAR_EXPECTED_L2_COUNTS):
        raise RuntimeError(
            "MMStar l2_category counts do not match the official release: "
            f"{dict(l2_counts)}"
        )
    source_counts = Counter(sample["source"] for sample in rows)
    if source_counts != Counter(MMSTAR_EXPECTED_SOURCE_COUNTS):
        raise RuntimeError(
            "MMStar source counts do not match the official release: "
            f"{dict(source_counts)}"
        )
    media_types = {sample["_image_media_type"] for sample in rows}
    if media_types != {"image/jpeg"}:
        raise RuntimeError(
            "MMStar embedded-image formats do not match the official release: "
            f"{sorted(media_types)}"
        )
    return rows


def select_mmstar_samples(data: list[dict], limit: int | None) -> list[dict]:
    """Apply a deterministic category/l2-stratified diagnostic limit."""

    if limit is None or limit >= len(data):
        return list(data)
    if limit <= 0:
        raise ValueError("MMStar limit must be positive")

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for sample in data:
        grouped[(sample["category"], sample["l2_category"])].append(sample)
    missing = [key for key in MMSTAR_LIMIT_STRATA if not grouped[key]]
    if missing:
        raise RuntimeError(f"MMStar is missing sampling strata: {missing}")

    selected: list[dict] = []
    positions = {key: 0 for key in MMSTAR_LIMIT_STRATA}
    while len(selected) < limit:
        progressed = False
        for key in MMSTAR_LIMIT_STRATA:
            position = positions[key]
            group = grouped[key]
            if position < len(group):
                selected.append(group[position])
                positions[key] += 1
                progressed = True
                if len(selected) == limit:
                    break
        if not progressed:
            break
    if len(selected) != min(limit, len(data)):
        raise RuntimeError(
            f"MMStar stratified limit selected {len(selected)} of "
            f"{min(limit, len(data))} requested samples"
        )
    return selected


def build_mmstar_prompt(sample: dict, args=None) -> str:
    """Preserve MMStar's question/options and add only the output contract."""

    question = str(sample.get("question", "")).strip()
    if not question:
        raise ValueError(
            f"MMStar sample {sample.get('index')!r} has an empty question"
        )
    # Every MathVista-derived row starts with a source answer-format hint
    # followed by ``Question:``. It is not semantic question content and can
    # conflict with MMStar's letter-only target (three rows even ask for a
    # numeric value, while the converted ground truth is A--D). Remove only
    # this tightly identified first line; retain the question and choices
    # byte-for-byte otherwise.
    if str(sample.get("source", "")).strip() == "MathVista":
        first_line, separator, remainder = question.partition("\n")
        if (
            separator
            and remainder.startswith("Question:")
            and "Please answer the question" in first_line
            and first_line.endswith("at the end.")
        ):
            question = remainder.strip()
    if use_cot(args):
        return question
    return f"{question}\n{MMSTAR_MCQ_INSTRUCTION}"


def parse_mmstar_prediction(text: str) -> str:
    """Extract an A--D label without treating the article ``a`` as option A."""

    answer = str(text).strip()
    if not answer:
        return ""

    # The common direct, boxed, and answer-tag payloads are just ``A`` or
    # ``(A)``. Lowercase is safe only in this whole-payload form.
    exact = re.fullmatch(
        r"[\(\[]?\s*([A-Da-d])\s*[\)\]]?\s*[.。:!?]?",
        answer,
    )
    if exact:
        return exact.group(1).upper()
    latex = re.fullmatch(
        r"\\(?:text|mathrm)\s*\{\s*([A-Da-d])\s*\}\s*[.。:!?]?",
        answer,
    )
    if latex:
        return latex.group(1).upper()
    leading = re.match(
        r"^\s*(?:\(([A-D])\)|([A-D])\s*[\)\].。,:：;；!?！-])"
        r"(?:\s|$)",
        answer,
    )
    if leading:
        return (leading.group(1) or leading.group(2)).upper()

    # ``option b`` is unambiguous even in lowercase. Keep ``answer is`` to
    # uppercase labels so prose such as "the answer is a square" cannot
    # silently become option A.
    option_matches = re.findall(
        r"(?i:\b(?:final\s+)?(?:option|choice)\s*(?:is|:|=)?\s*)"
        r"[\(\[]?\s*([A-Da-d])\b",
        answer,
    )
    if option_matches:
        return option_matches[-1].upper()
    selection_matches = re.findall(
        r"(?i:\b(?:choose|select|pick)\s*(?:(?:option|choice)\s*)?)"
        r"[\(\[]?\s*([A-Da-d])\b",
        answer,
    )
    if selection_matches:
        return selection_matches[-1].upper()
    answer_matches = re.findall(
        r"(?i:\b(?:final\s+)?answer\s*(?:is|:|=)?\s*)"
        r"[\(\[]?\s*([A-D])\b",
        answer,
    )
    if answer_matches:
        return answer_matches[-1]

    # Permit a conventional final uppercase label, but never scan all
    # case-insensitive single-letter words in arbitrary reasoning.
    final_uppercase = re.search(
        r"(?<![A-Za-z])([A-D])(?:\s*[\)\]}])?\s*[.。:!?]?\s*$",
        answer,
    )
    return final_uppercase.group(1) if final_uppercase else ""


def summarize_mmstar(records: list[dict], *, full_n: int) -> dict:
    """Compute MMStar image accuracy and its official diagnostic slices."""

    if not records:
        raise ValueError("MMStar evaluation selected zero samples")
    correct = sum(bool(record["correct"]) for record in records)
    accuracy = correct / len(records)
    by_category, category_counts = _cvbench_group_accuracy(
        records,
        "category",
    )
    by_l2_category, l2_category_counts = _cvbench_group_accuracy(
        records,
        "l2_category",
    )
    by_source, source_counts = _cvbench_group_accuracy(records, "source")
    valid_predictions = sum(
        record["pred"] in MMSTAR_ANSWER_LETTERS for record in records
    )
    macro_category_accuracy = (
        sum(by_category.values()) / len(by_category)
        if by_category
        else 0.0
    )
    return {
        "accuracy": accuracy,
        "micro_accuracy": accuracy,
        "macro_category_accuracy": macro_category_accuracy,
        "correct": correct,
        "n": len(records),
        "full_n": full_n,
        "limited_diagnostic": len(records) != full_n,
        "valid_predictions": valid_predictions,
        "invalid_predictions": len(records) - valid_predictions,
        "valid_prediction_rate": valid_predictions / len(records),
        "metric_profile": "mmstar_answer_letter_accuracy",
        "by_type": by_category,
        "by_category": by_category,
        "by_l2_category": by_l2_category,
        "by_source": by_source,
        "group_counts": {
            "category": category_counts,
            "l2_category": l2_category_counts,
            "source": source_counts,
        },
    }


def eval_mmstar(args) -> tuple[dict, list[dict]]:
    """Evaluate MMStar using embedded parquet images and answer letters."""

    full_data = load_mmstar(args.mmstar_root)
    data = select_mmstar_samples(full_data, args.limit)

    def one(sample, endpoint):
        prompt = format_evaluation_prompt(
            build_mmstar_prompt(sample, args),
            args,
        )
        image_b64 = base64.b64encode(sample["_image_bytes"]).decode("ascii")
        text = chat(
            endpoint,
            args.model,
            image_b64,
            prompt,
            args.max_new_tokens,
            image_media_type=sample["_image_media_type"],
            system_prompt=system_prompt_for(args),
            generation_profile_name=generation_profile_for(args),
        )
        payload, boxed = scoring_payload(text, args)
        pred = parse_mmstar_prediction(payload)
        gt = sample["answer"]
        record = {
            "index": sample["index"],
            "category": sample["category"],
            "l2_category": sample["l2_category"],
            "source": sample["source"],
            "split": sample["split"],
            "image_path": sample["image_path"],
            "image_media_type": sample["_image_media_type"],
            "question": sample["question"],
            "prompt": prompt,
            "pred": pred,
            "gt": gt,
            "correct": pred == gt,
            "generated": text,
        }
        add_boxed_record_fields(
            record,
            args=args,
            prompt=prompt,
            boxed_answer=boxed,
        )
        return record

    records = parallel_eval_requests(
        one,
        data,
        args,
        description="mmstar",
        single_endpoint_concurrency=args.concurrency,
    )
    summary = summarize_mmstar(records, full_n=len(full_data))
    return add_boxed_summary(summary, records, args), records


def _parse_mmvp_options(options: str, *, sample_id: int) -> list[str]:
    """Parse MMVP's canonical ``(a) ... (b) ...`` two-choice field."""

    match = re.fullmatch(
        r"\(a\)\s*(.+?)\s+\(b\)\s*(.+)",
        str(options).strip(),
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        raise ValueError(
            f"MMVP sample {sample_id} has invalid options {options!r}; "
            "expected '(a) ... (b) ...'"
        )
    choices = [match.group(1).strip(), match.group(2).strip()]
    if any(not choice for choice in choices):
        raise ValueError(f"MMVP sample {sample_id} has an empty option")
    return choices


def load_mmvp(root: str | Path) -> list[dict]:
    """Load and validate the official 300-question / 150-pair MMVP release."""

    dataset_root = Path(root).expanduser().resolve()
    annotation = dataset_root / "Questions.csv"
    image_root = dataset_root / "MMVP Images"
    if not annotation.is_file():
        raise FileNotFoundError(f"MMVP Questions.csv not found: {annotation}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"MMVP image directory not found: {image_root}")

    required = {"Index", "Question", "Options", "Correct Answer"}
    rows: list[dict] = []
    seen_indices: set[int] = set()
    with annotation.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"MMVP CSV is missing required columns: {sorted(missing)}"
            )
        for line_number, raw in enumerate(reader, start=2):
            if None in raw:
                raise ValueError(
                    f"MMVP CSV has extra fields at {annotation}:{line_number}"
                )
            try:
                index = int(str(raw["Index"]).strip())
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid MMVP index at {annotation}:{line_number}: "
                    f"{raw.get('Index')!r}"
                ) from exc
            if index in seen_indices:
                raise ValueError(f"duplicate MMVP index: {index}")
            seen_indices.add(index)

            question = str(raw["Question"] or "").strip()
            options = str(raw["Options"] or "").strip()
            answer_match = re.fullmatch(
                r"\(([abAB])\)", str(raw["Correct Answer"] or "").strip()
            )
            if not question:
                raise ValueError(f"MMVP sample {index} has an empty question")
            choices = _parse_mmvp_options(options, sample_id=index)
            if answer_match is None:
                raise ValueError(
                    f"MMVP sample {index} has invalid answer "
                    f"{raw.get('Correct Answer')!r}"
                )
            answer = answer_match.group(1).upper()

            image_path = image_root / f"{index}.jpg"
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"MMVP sample {index} image not found: {image_path}"
                )
            image_bytes = image_path.read_bytes()
            media_type = _image_media_type_from_bytes(
                image_bytes,
                sample_id=index,
                dataset="MMVP",
            )
            if media_type != "image/jpeg":
                raise ValueError(
                    f"MMVP sample {index} expected JPEG, got {media_type}"
                )

            pair_id = (index + 1) // 2
            rows.append(
                {
                    "index": index,
                    "pair_id": pair_id,
                    "pair_position": 1 if index % 2 else 2,
                    "paired_index": index + 1 if index % 2 else index - 1,
                    "question": question,
                    "options": options,
                    "choices": choices,
                    "answer": answer,
                    "image_path": str(image_path),
                    "image_filename": image_path.name,
                    "_image_media_type": media_type,
                }
            )

    if len(rows) != MMVP_EXPECTED_N:
        raise RuntimeError(
            f"expected {MMVP_EXPECTED_N} MMVP questions, got {len(rows)}"
        )
    indices = [sample["index"] for sample in rows]
    if indices != list(range(1, MMVP_EXPECTED_N + 1)):
        raise RuntimeError(
            "MMVP indices must be unique and sequential from 1 through "
            f"{MMVP_EXPECTED_N}"
        )
    answer_counts = Counter(sample["answer"] for sample in rows)
    if answer_counts != Counter({"A": MMVP_EXPECTED_PAIRS, "B": MMVP_EXPECTED_PAIRS}):
        raise RuntimeError(
            "MMVP answer counts do not match the official release: "
            f"{dict(answer_counts)}"
        )
    for offset in range(0, len(rows), 2):
        first, second = rows[offset : offset + 2]
        if (
            first["pair_id"] != second["pair_id"]
            or first["pair_position"] != 1
            or second["pair_position"] != 2
            or first["paired_index"] != second["index"]
            or second["paired_index"] != first["index"]
        ):
            raise RuntimeError(
                f"MMVP pair structure is invalid at indices "
                f"{first['index']}/{second['index']}"
            )
        if first["question"] != second["question"]:
            raise RuntimeError(
                f"MMVP pair {first['pair_id']} does not share one question"
            )
        if first["options"] != second["options"]:
            raise RuntimeError(
                f"MMVP pair {first['pair_id']} does not share one option set"
            )
        if {first["answer"], second["answer"]} != MMVP_ANSWER_LETTERS:
            raise RuntimeError(
                f"MMVP pair {first['pair_id']} does not have complementary answers"
            )
    return rows


def select_mmvp_samples(data: list[dict], limit: int | None) -> list[dict]:
    """Apply a prefix diagnostic limit without ever splitting an MMVP pair."""

    if limit is None or limit >= len(data):
        return list(data)
    if limit <= 0:
        raise ValueError("MMVP limit must be positive")
    if limit % 2:
        raise ValueError(
            "MMVP --limit counts questions and must be even so every pair is complete"
        )
    return list(data[:limit])


def build_mmvp_prompt(sample: dict, args=None) -> str:
    """Preserve the official question/options and use the shared MCQ contract."""

    question = str(sample.get("question", "")).strip()
    options = str(sample.get("options", "")).strip()
    if not question or not options:
        raise ValueError("MMVP prompt requires a question and options")
    content = f"{question}\n{options}"
    if use_cot(args):
        return content
    return f"{content}\n{MMVP_MCQ_INSTRUCTION}"


def parse_mmvp_prediction(text: str) -> str:
    """Extract A/B while accepting MMVP's native ``(a) option text`` form."""

    prediction = parse_mmstar_prediction(text)
    if prediction in MMVP_ANSWER_LETTERS:
        return prediction
    answer = str(text).strip()
    if not answer:
        return ""
    # MMVP publishes lowercase ``(a) ... (b) ...`` choices, and aligned models
    # commonly repeat the chosen choice verbatim inside <answer>. Requiring
    # punctuation around a leading label keeps ordinary prose such as
    # ``a butterfly`` from becoming option A.
    leading = re.match(
        r"^\s*(?:\(\s*([ab])\s*\)|([ab])\s*[\)\].。,:：;；!?！-])(?:\s|$)",
        answer,
        flags=re.IGNORECASE,
    )
    if leading:
        return (leading.group(1) or leading.group(2)).upper()
    explicit = re.findall(
        r"(?i:\b(?:final\s+)?answer\s*(?:is|:|=)\s*)"
        r"[\(\[]?\s*([ab])(?:\s*[\)\]])?\b",
        answer,
    )
    return explicit[-1].upper() if explicit else ""


def summarize_mmvp(records: list[dict], *, full_question_n: int) -> dict:
    """Compute MMVP's official all-or-nothing score over adjacent pairs."""

    if not records:
        raise ValueError("MMVP evaluation selected zero samples")
    if len(records) % 2:
        raise ValueError("MMVP scoring requires complete two-question pairs")

    pair_groups: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        pair_groups[int(record["pair_id"])].append(record)
    pair_correct = 0
    for pair_id, pair in sorted(pair_groups.items()):
        if len(pair) != 2 or {row["pair_position"] for row in pair} != {1, 2}:
            raise ValueError(f"MMVP pair {pair_id} is incomplete")
        is_correct = all(bool(row["correct"]) for row in pair)
        pair_correct += int(is_correct)
        for row in pair:
            row["pair_correct"] = is_correct

    question_correct = sum(bool(record["correct"]) for record in records)
    question_n = len(records)
    pair_n = len(pair_groups)
    pair_accuracy = pair_correct / pair_n
    question_accuracy = question_correct / question_n
    full_pair_n = full_question_n // 2
    official_score_available = (
        question_n == full_question_n and pair_n == full_pair_n
    )
    valid_predictions = sum(
        record["pred"] in MMVP_ANSWER_LETTERS for record in records
    )
    return {
        # ``accuracy`` is deliberately the pair score: that is MMVP's headline
        # metric, not the easier per-question accuracy.
        "accuracy": pair_accuracy,
        "official_accuracy": pair_accuracy if official_score_available else None,
        "pair_accuracy": pair_accuracy,
        "question_accuracy": question_accuracy,
        "correct": pair_correct,
        "n": pair_n,
        "pair_correct": pair_correct,
        "pair_n": pair_n,
        "question_correct": question_correct,
        "question_n": question_n,
        "full_n": full_pair_n,
        "full_pair_n": full_pair_n,
        "full_question_n": full_question_n,
        "limited_diagnostic": not official_score_available,
        "official_score_available": official_score_available,
        "valid_predictions": valid_predictions,
        "invalid_predictions": question_n - valid_predictions,
        "valid_prediction_rate": valid_predictions / question_n,
        "metric_profile": (
            "mmvp_official_pair_accuracy"
            if official_score_available
            else "mmvp_partial_pair_accuracy"
        ),
        "by_type": {"pair": pair_accuracy, "question": question_accuracy},
    }


def eval_mmvp(args) -> tuple[dict, list[dict]]:
    """Evaluate MMVP with the existing prompt/system-role profile machinery."""

    full_data = load_mmvp(args.mmvp_root)
    data = select_mmvp_samples(full_data, args.limit)

    def one(sample, endpoint):
        prompt = format_evaluation_prompt(
            build_mmvp_prompt(sample, args),
            args,
        )
        image_b64, media_type = encode_original_image(Path(sample["image_path"]))
        if media_type != sample["_image_media_type"]:
            raise RuntimeError(
                f"MMVP sample {sample['index']} image media type changed"
            )
        text = chat(
            endpoint,
            args.model,
            image_b64,
            prompt,
            args.max_new_tokens,
            image_media_type=media_type,
            system_prompt=system_prompt_for(args),
            generation_profile_name=generation_profile_for(args),
        )
        payload, boxed = scoring_payload(text, args)
        pred = parse_mmvp_prediction(payload)
        gt = sample["answer"]
        pred_index = ord(pred) - ord("A") if pred else -1
        gt_index = ord(gt) - ord("A")
        record = {
            "index": sample["index"],
            "pair_id": sample["pair_id"],
            "pair_position": sample["pair_position"],
            "paired_index": sample["paired_index"],
            "image_filename": sample["image_filename"],
            "image_media_type": media_type,
            "question": sample["question"],
            "options": sample["options"],
            "choices": sample["choices"],
            "prompt": prompt,
            "pred": pred,
            "pred_choice": (
                sample["choices"][pred_index]
                if 0 <= pred_index < len(sample["choices"])
                else None
            ),
            "gt": gt,
            "gt_choice": sample["choices"][gt_index],
            "correct": pred == gt,
            "generated": text,
        }
        add_boxed_record_fields(
            record,
            args=args,
            prompt=prompt,
            boxed_answer=boxed,
        )
        return record

    records = parallel_eval_requests(
        one,
        data,
        args,
        description="mmvp",
        single_endpoint_concurrency=args.concurrency,
    )
    summary = summarize_mmvp(records, full_question_n=len(full_data))
    return add_boxed_summary(summary, records, args), records


def load_visualprobe(
    root: str | Path,
    benchmark: str,
) -> list[dict]:
    """Load and fully validate one downloaded VisualProbe difficulty split."""

    if benchmark == "visualprobe_all":
        rows = []
        for name in VISUALPROBE_SPLITS:
            for sample in load_visualprobe(root, name):
                rows.append(dict(sample, subset=name, subset_index=sample['index'],
                                 index=len(rows), sample_id=name + '::' + str(sample['doc_id'])))
        return rows
    if benchmark not in VISUALPROBE_SPLITS:
        raise ValueError(
            f"unknown VisualProbe benchmark {benchmark!r}; "
            f"expected one of {VISUALPROBE_BENCHMARKS}"
        )
    split = VISUALPROBE_SPLITS[benchmark]
    dataset_root = Path(root).expanduser().resolve()
    annotation = dataset_root / split["directory"] / "val.json"
    if not annotation.is_file():
        raise FileNotFoundError(
            f"{benchmark} annotation not found: {annotation}"
        )
    with annotation.open("r", encoding="utf-8") as handle:
        raw_rows = json.load(handle)
    if not isinstance(raw_rows, list):
        raise ValueError(
            f"{benchmark} annotation must contain a JSON list: {annotation}"
        )
    expected_n = int(split["expected_n"])
    if len(raw_rows) != expected_n:
        raise RuntimeError(
            f"expected {expected_n} {benchmark} samples, got {len(raw_rows)}"
        )

    required = {"images", "doc_id", "problem", "solution", "data_source"}
    seen_doc_ids: set[str] = set()
    rows: list[dict] = []
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, dict):
            raise ValueError(
                f"{benchmark} row {index} must be an object, "
                f"got {type(raw).__name__}"
            )
        missing = required.difference(raw)
        if missing:
            raise ValueError(
                f"{benchmark} row {index} is missing fields: {sorted(missing)}"
            )

        doc_id = str(raw["doc_id"]).strip()
        if not doc_id:
            raise ValueError(f"{benchmark} row {index} has an empty doc_id")
        if doc_id in seen_doc_ids:
            raise ValueError(f"{benchmark} has duplicate doc_id {doc_id!r}")
        seen_doc_ids.add(doc_id)

        data_source = str(raw["data_source"]).strip()
        if data_source != split["data_source"]:
            raise ValueError(
                f"{benchmark} row {index} has data_source {data_source!r}; "
                f"expected {split['data_source']!r}"
            )

        images = raw["images"]
        if not isinstance(images, list) or len(images) != 1:
            raise ValueError(
                f"{benchmark} row {index} must reference exactly one image"
            )
        relative_image = Path(str(images[0]).strip())
        if not str(relative_image) or relative_image.is_absolute():
            raise ValueError(
                f"{benchmark} row {index} has invalid relative image path "
                f"{images[0]!r}"
            )
        image_path = (dataset_root / relative_image).resolve()
        try:
            image_path.relative_to(dataset_root)
        except ValueError as exc:
            raise ValueError(
                f"{benchmark} row {index} image escapes dataset root: "
                f"{relative_image}"
            ) from exc
        if not image_path.is_file():
            raise FileNotFoundError(
                f"{benchmark} image not found for {doc_id}: {image_path}"
            )

        problem = str(raw["problem"]).strip()
        image_prefix = re.match(r"^\s*<image>\s*", problem)
        if image_prefix is None:
            raise ValueError(
                f"{benchmark} row {index} problem must begin with <image>"
            )
        question = problem[image_prefix.end() :].strip()
        if not question or "<image>" in question:
            raise ValueError(
                f"{benchmark} row {index} has an invalid question body"
            )
        solution = str(raw["solution"]).strip()
        if not solution:
            raise ValueError(
                f"{benchmark} row {index} has an empty solution"
            )

        rows.append(
            {
                "index": index,
                "doc_id": doc_id,
                "data_source": data_source,
                "problem": problem,
                "question": question,
                "solution": solution,
                "image": str(image_path),
                "image_relative": relative_image.as_posix(),
            }
        )
    return rows


def build_visualprobe_prompt(sample: dict, args=None) -> str:
    """Build VisualProbe's concise-answer prompt, or bare question for CoT."""

    question = str(sample["question"]).strip()
    if not question:
        raise ValueError("VisualProbe prompt requires a non-empty question")
    if use_cot(args):
        return question
    return f"{question}\n{VISUALPROBE_SHORT_ANSWER_INSTRUCTION}"


def visualprobe_generation_max_tokens(args) -> int:
    """Give direct short answers enough room without changing other datasets."""

    configured = int(args.max_new_tokens)
    if answer_format(args) != "direct":
        return configured
    visualprobe_minimum = int(
        getattr(
            args,
            "visualprobe_max_new_tokens",
            VISUALPROBE_DEFAULT_MAX_NEW_TOKENS,
        )
    )
    if visualprobe_minimum <= 0:
        raise ValueError("--visualprobe-max-new-tokens must be positive")
    return max(configured, visualprobe_minimum)


def visualprobe_judge_args(source=None) -> dict:
    """The judge settings, forwarded from ``source`` or defaulted.

    The worker entry points build their own ``Namespace`` for the evaluation
    functions instead of reusing this module's parser. A missing field there
    would silently disable judging and quietly emit deterministic scores under
    a judge-labelled pipeline, so every such Namespace is populated from here.
    """

    def pick(name, default):
        value = getattr(source, name, None)
        return default if value is None else value

    return {
        "vp_judge": pick("vp_judge", True),
        "vp_judge_endpoint": pick("vp_judge_endpoint", VISUALPROBE_JUDGE_ENDPOINT),
        "vp_judge_model": pick("vp_judge_model", VISUALPROBE_JUDGE_MODEL),
        "vp_judge_max_tokens": int(
            pick("vp_judge_max_tokens", VISUALPROBE_JUDGE_MAX_TOKENS)
        ),
        "vp_judge_concurrency": int(
            pick("vp_judge_concurrency", VISUALPROBE_JUDGE_CONCURRENCY)
        ),
        "vp_judge_mode": pick("vp_judge_mode", VISUALPROBE_JUDGE_MODE),
        "vp_judge_audit_frac": float(
            pick("vp_judge_audit_frac", VISUALPROBE_JUDGE_AUDIT_FRAC)
        ),
    }


def add_visualprobe_judge_arguments(parser) -> None:
    """Expose the judge knobs on a worker's own parser (same names/defaults)."""

    parser.add_argument("--vp-judge", type=parse_bool, default=True)
    parser.add_argument("--vp-judge-endpoint", default=VISUALPROBE_JUDGE_ENDPOINT)
    parser.add_argument("--vp-judge-model", default=VISUALPROBE_JUDGE_MODEL)
    parser.add_argument(
        "--vp-judge-max-tokens", type=int, default=VISUALPROBE_JUDGE_MAX_TOKENS
    )
    parser.add_argument(
        "--vp-judge-concurrency", type=int, default=VISUALPROBE_JUDGE_CONCURRENCY
    )
    parser.add_argument(
        "--vp-judge-mode",
        choices=VISUALPROBE_JUDGE_MODES,
        default=VISUALPROBE_JUDGE_MODE,
    )
    parser.add_argument(
        "--vp-judge-audit-frac", type=float, default=VISUALPROBE_JUDGE_AUDIT_FRAC
    )


def visualprobe_judge_enabled(args) -> bool:
    return bool(getattr(args, "vp_judge", False)) and bool(
        getattr(args, "vp_judge_endpoint", "")
    )


def check_visualprobe_judge(args) -> None:
    """Fail fast if the judge is unreachable, before any generation happens."""

    endpoint = str(args.vp_judge_endpoint).rstrip("/")
    try:
        request = urllib.request.Request(f"{endpoint}/models")
        with vp_llm._OPENER.open(request, timeout=20) as response:
            served = {m["id"] for m in json.load(response).get("data", [])}
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(
            f"VisualProbe judge unreachable at {endpoint}: {error!r}. Start it "
            "with scripts/deploy_vp_judge.sh, point --vp-judge-endpoint "
            "elsewhere, or pass --vp-judge false to score deterministically."
        ) from error
    if args.vp_judge_model not in served:
        raise RuntimeError(
            f"judge endpoint {endpoint} serves {sorted(served)}, not "
            f"{args.vp_judge_model!r}; set --vp-judge-model accordingly."
        )


def _judge_audit_selected(question: str, gt: str, pred: str, fraction: float) -> bool:
    """Stable per-triple audit sampling, so a re-run audits the same rows."""

    if fraction <= 0:
        return False
    if fraction >= 1:
        return True
    digest = vp_llm.triple_key(question, gt, pred)
    return int(digest[:8], 16) < fraction * 0x1_0000_0000


def run_visualprobe_judge(records: list[dict], args) -> None:
    """Attach LLM-judge verdicts to records in place.

    Identical (question, gt, pred) triples are judged once: models agree far
    more often than they differ, so deduplication removes ~80% of the calls on
    a multi-arm sweep and makes the verdict identical wherever the text is.
    (Within one split it removes ~nothing -- every question appears once.)

    ``cascade`` mode additionally skips rows the deterministic rule already
    accepts. That is not a shortcut with a quality cost: the judge overturned a
    deterministic hit in 0 of ~39,000 audited rows, so the two modes provably
    agree, and it removes ~40% of the calls in a live single-arm evaluation.
    Because skipping them would also silence the ``strict_yes_judge_no`` health
    monitor, a stable ~``--vp-judge-audit-frac`` sample of accepted rows is
    judged anyway -- observed, never used to change a verdict.
    """

    mode = getattr(args, "vp_judge_mode", "full")
    audit_fraction = float(getattr(args, "vp_judge_audit_frac", 0.0) or 0.0)

    unique: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in records:
        key = (
            str(record.get("question", "")),
            str(record.get("gt", "")),
            str(record.get("pred", "")),
        )
        if mode == "cascade" and record["deterministic_correct"]:
            if not _judge_audit_selected(*key, audit_fraction):
                continue
            record["judge_audited"] = True
        unique[key].append(record)
    if not unique:
        return

    def work(key):
        question, gt, pred = key
        reply = vp_llm.judge_call(
            args.vp_judge_endpoint,
            args.vp_judge_model,
            question,
            gt,
            pred,
            max_tokens=args.vp_judge_max_tokens,
            enable_thinking=False,
            timeout=180.0,
            retries=4,
        )
        return key, vp_llm.parse_score(reply)[0], reply

    with ThreadPoolExecutor(max_workers=max(1, args.vp_judge_concurrency)) as pool:
        futures = [pool.submit(work, key) for key in unique]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="visualprobe judge",
            leave=False,
        ):
            key, score, reply = future.result()
            for record in unique[key]:
                record["judge_score"] = score
                record["judge_raw"] = reply


def eval_visualprobe(
    args,
    benchmark: str,
) -> tuple[dict, list[dict]]:
    """Evaluate one VisualProbe split.

    ``accuracy``/``correct`` follow the LLM judge when it is enabled (the
    benchmark's own protocol); the deterministic normalized-match score is
    always kept as ``accuracy_deterministic``/``correct_deterministic`` so both
    metrics stay comparable across the archive.
    """

    if visualprobe_judge_enabled(args):
        check_visualprobe_judge(args)
    data = load_visualprobe(args.visualprobe_root, benchmark)
    if args.limit is not None:
        data = data[: args.limit]
    if not data:
        raise ValueError(f"{benchmark} evaluation selected zero samples")
    generation_max_tokens = visualprobe_generation_max_tokens(args)

    def one(sample, endpoint):
        image_b64, media_type = encode_original_image(Path(sample["image"]))
        prompt = format_evaluation_prompt(
            build_visualprobe_prompt(sample, args),
            args,
        )
        generated = chat(
            endpoint,
            args.model,
            image_b64,
            prompt,
            generation_max_tokens,
            image_media_type=media_type,
            system_prompt=system_prompt_for(args),
            generation_profile_name=generation_profile_for(args),
        )
        payload, boxed = scoring_payload(generated, args)
        judgement = visualprobe_judge.judge_visualprobe_answer(
            payload,
            sample["solution"],
            sample["question"],
        )
        record = {
            **sample,
            "prompt": prompt,
            "pred": payload.strip(),
            "gt": sample["solution"],
            "correct": judgement.correct,
            "deterministic_correct": judgement.correct,
            "strict_correct": judgement.strict_correct,
            "match_rule": judgement.match_rule,
            "normalized_pred": judgement.strict_prediction,
            "normalized_gt": judgement.strict_ground_truth,
            "compared_pred": judgement.compared_prediction,
            "compared_gt": judgement.compared_ground_truth,
            "text_reading_question": judgement.text_reading_question,
            "generated": generated,
        }
        add_boxed_record_fields(
            record,
            args=args,
            prompt=prompt,
            boxed_answer=boxed,
        )
        return record

    records = parallel_eval_requests(
        one,
        data,
        args,
        description=benchmark,
        single_endpoint_concurrency=args.concurrency,
    )
    judged = visualprobe_judge_enabled(args)
    judge_mode = getattr(args, "vp_judge_mode", "full")
    if judged:
        run_visualprobe_judge(records, args)
        for record in records:
            # An unparsed judge reply scores wrong, so an accuracy carrying a
            # non-zero judge_unparsed is a lower bound. Under cascade the
            # deterministic hit stands on its own and an audit verdict is
            # observational only -- it must never flip an accepted row.
            if judge_mode == "cascade" and record["deterministic_correct"]:
                record["correct"] = True
            else:
                record["correct"] = record.get("judge_score") == 1

    deterministic_correct = sum(
        bool(record["deterministic_correct"]) for record in records
    )
    correct = sum(bool(record["correct"]) for record in records)
    strict_correct = sum(
        bool(record["strict_correct"]) for record in records
    )
    match_rule_counts: dict[str, int] = defaultdict(int)
    for record in records:
        match_rule_counts[record["match_rule"]] += 1
    summary = {
        "accuracy": correct / len(records),
        "correct": correct,
        "accuracy_deterministic": deterministic_correct / len(records),
        "correct_deterministic": deterministic_correct,
        "strict_accuracy": strict_correct / len(records),
        "strict_correct": strict_correct,
        "n": len(records),
        "metric_profile": (
            VISUALPROBE_JUDGE_METRIC_PROFILE
            if judged
            else VISUALPROBE_DETERMINISTIC_METRIC_PROFILE
        ),
        "by_type": {
            source: sum(bool(r['correct']) for r in records if r['data_source'] == source)
                    / sum(r['data_source'] == source for r in records)
            for source in sorted({r['data_source'] for r in records})
        },
        "match_rule_counts": dict(sorted(match_rule_counts.items())),
        "generation_max_new_tokens": generation_max_tokens,
    }
    if judged:
        judged_rows = [r for r in records if "judge_score" in r]
        checked = [r for r in judged_rows if r["deterministic_correct"]]
        summary.update(
            {
                "judge_model": args.vp_judge_model,
                "judge_endpoint": args.vp_judge_endpoint,
                "judge_prompt_sha": vp_llm.PROMPT_SHA,
                "judge_max_tokens": args.vp_judge_max_tokens,
                "judge_mode": judge_mode,
                "judge_scored_rows": len(judged_rows),
                "judge_unparsed": sum(
                    1 for r in judged_rows if r.get("judge_score") is None
                ),
                # Health monitor: the judge overturning a deterministic hit has
                # never been observed (0 in ~39,000 rows). Under cascade only
                # the audit sample can see it, so the denominator is reported
                # next to it -- a bare 0 out of 0 would mean nothing.
                "strict_yes_judge_no": sum(
                    1 for r in checked if r.get("judge_score") == 0
                ),
                "strict_hits_checked": len(checked),
            }
        )
    if benchmark == 'visualprobe_all':
        from visualprobe_groups import subset_summaries
        summary['by_subset'] = subset_summaries(records)
        summary['aggregation'] = 'micro_accuracy'
    return add_boxed_summary(summary, records, args), records


def eval_babyvision(args) -> tuple[dict, list[dict]]:
    """Evaluate BabyVision under this project's uniform answer wrapper.

    Deviation from upstream, deliberate: the question, the lettered choices,
    the ground truth and the judge prompt are upstream's, but the answer is
    requested through this project's shared ``answer_format`` wrapper rather
    than upstream's ``\\boxed{}`` sentence, so BabyVision stays comparable with
    the other nine benchmarks in a sweep. That makes the absolute score
    non-comparable with the public leaderboard, which also averages three
    sampled passes; single-pass greedy here is an internal instrument.

    Judging is always ``full``: 65% of the tasks are free-form, so there is no
    deterministic rule trustworthy enough to short-circuit the judge.
    """

    if not visualprobe_judge_enabled(args):
        raise ValueError(
            "BabyVision has no reliable deterministic metric (65% free-form "
            "answers); it requires the LLM judge. Point --vp-judge-endpoint at "
            "a live judge instead of passing --vp-judge false."
        )
    check_visualprobe_judge(args)

    data = babyvision.load_babyvision(args.babyvision_root)
    if args.limit is not None:
        data = data[: args.limit]

    def one(sample, endpoint):
        image_b64, media_type = babyvision.encode_babyvision_image(sample["image"])
        prompt = format_evaluation_prompt(
            babyvision.build_babyvision_question(sample), args
        )
        text = chat(
            endpoint,
            args.model,
            image_b64,
            prompt,
            args.max_new_tokens,
            image_media_type=media_type,
            system_prompt=system_prompt_for(args),
            generation_profile_name=generation_profile_for(args),
        )
        payload, boxed = scoring_payload(text, args)
        pred = babyvision.parse_babyvision_prediction(payload, sample)
        record = {
            "index": sample["index"],
            "type": sample["type"],
            "subtype": sample["subtype"],
            "ans_type": sample["ans_type"],
            "question": babyvision.build_babyvision_question(sample),
            "options": sample["options"],
            "prompt": prompt,
            "pred": payload.strip(),
            "parsed_pred": pred,
            "gt": sample["answer"],
            "deterministic_correct": pred == sample["answer"],
            "generated": text,
        }
        add_boxed_record_fields(record, args=args, prompt=prompt, boxed_answer=boxed)
        return record

    records = parallel_eval_requests(
        one, data, args, description="babyvision",
        single_endpoint_concurrency=args.concurrency,
    )

    def judge(record):
        reply = vp_llm.judge_call(
            args.vp_judge_endpoint,
            args.vp_judge_model,
            record["question"],
            record["gt"],
            record["pred"],
            max_tokens=args.vp_judge_max_tokens,
            enable_thinking=False,
            timeout=180.0,
            retries=4,
            system_prompt=None,
            query_template=babyvision.BABYVISION_JUDGE_PROMPT,
        )
        record["judge_raw"] = reply
        verdict = babyvision.parse_babyvision_verdict(reply)
        record["judge_score"] = None if verdict is None else int(verdict)
        record["correct"] = verdict is True
        return record

    with ThreadPoolExecutor(max_workers=max(1, args.vp_judge_concurrency)) as pool:
        list(
            tqdm(
                pool.map(judge, records),
                total=len(records),
                desc="babyvision judge",
                leave=False,
            )
        )

    correct = sum(bool(r["correct"]) for r in records)
    deterministic = sum(bool(r["deterministic_correct"]) for r in records)
    by_type: dict[str, list[dict]] = defaultdict(list)
    by_subtype: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_type[record["type"]].append(record)
        by_subtype[record["subtype"]].append(record)
    summary = {
        "accuracy": correct / len(records),
        "correct": correct,
        "accuracy_deterministic": deterministic / len(records),
        "correct_deterministic": deterministic,
        "n": len(records),
        "metric_profile": "babyvision_llm_judge_official_prompt",
        "by_type": {k: sum(r["correct"] for r in v) / len(v) for k, v in by_type.items()},
        "by_subtype": {
            k: sum(r["correct"] for r in v) / len(v) for k, v in sorted(by_subtype.items())
        },
        "by_ans_type": {
            k: sum(r["correct"] for r in v) / len(v)
            for k, v in {
                t: [r for r in records if r["ans_type"] == t]
                for t in babyvision.BABYVISION_ANSWER_TYPES
            }.items()
            if v
        },
        "judge_model": args.vp_judge_model,
        "judge_endpoint": args.vp_judge_endpoint,
        "judge_mode": "full",
        "judge_unparsed": sum(1 for r in records if r.get("judge_score") is None),
        "generation_max_new_tokens": args.max_new_tokens,
    }
    return add_boxed_summary(summary, records, args), records


def build_mmerealworld_prompt(sample: dict, args=None) -> str:
    question = str(sample["Text"]).strip()
    choices = [str(choice).strip() for choice in sample["Answer choices"]]
    if not choices:
        raise ValueError(f"question {sample.get('Question_id')!r} has no answer choices")
    choices_text = "\n".join(choices)
    content = f"{question}\nThe choices are listed below:\n{choices_text}"
    if use_cot(args):
        return content
    return f"{content}\n{MME_REALWORLD_INSTRUCTION}"


def parse_mmerealworld_prediction(text: str, choices: list[str]) -> str:
    """Mirror the benchmark's official answer-letter extraction."""
    answer = text.strip()
    # ``"" in choice`` is true in Python. Without this guard, a missing
    # answer tag (including a Qwen3.5 reasoning-only truncation) silently
    # becomes the first option, A, in the choice-text fallback below.
    if not answer:
        return ""
    for prefix in MME_REALWORLD_ANSWER_PREFIXES:
        answer = answer.replace(prefix, "")
    if len(answer.split()) > 10 and not re.search(r"[ABCDE]", answer):
        return ""
    match = re.search(r"[ABCDE]", answer)
    if match is not None:
        return match.group(0)
    for choice in choices:
        if answer.lower() in choice.lower() and len(choice) > 1:
            return choice[1]
    return ""


def normalize_mmerealworld_category(category: str) -> str:
    """Apply the category normalization used by the official evaluator."""
    normalized = str(category).lower()
    if "attribute" in normalized:
        normalized = normalized.split("/")[0] + "/attribute"
    return normalized


def _mmerealworld_group_stats(records: list[dict], fields: tuple[str, ...]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        key = "/".join(str(record[field]) for field in fields)
        groups[key].append(record)
    result = {}
    for key, rows in groups.items():
        correct = sum(bool(row["correct"]) for row in rows)
        result[key] = {
            "accuracy": correct / len(rows),
            "correct": correct,
            "n": len(rows),
            "e_selected": sum(row["pred"] == "E" for row in rows),
        }
    return result


def summarize_mmerealworld(records: list[dict]) -> dict:
    if not records:
        raise ValueError("MME-RealWorld evaluation selected zero samples")
    correct = sum(bool(record["correct"]) for record in records)
    return {
        "accuracy": correct / len(records),
        "correct": correct,
        "n": len(records),
        "e_selected": sum(record["pred"] == "E" for record in records),
        "by_task": _mmerealworld_group_stats(records, ("task",)),
        "by_subtask": _mmerealworld_group_stats(records, ("task", "subtask")),
        "by_category": _mmerealworld_group_stats(
            records, ("task", "subtask", "category")
        ),
    }


def eval_mmerealworld(args) -> tuple[dict, list[dict]]:
    annotation = Path(args.mmerealworld_annotation)
    image_root = Path(args.mmerealworld_image_root)
    with annotation.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if args.limit is not None:
        data = data[: args.limit]
    if not data:
        raise ValueError("MME-RealWorld evaluation selected zero samples")
    workers = int(args.mme_concurrency)
    if workers <= 0:
        raise ValueError("--mme-concurrency must be positive")

    def one(item, endpoint):
        idx, sample = item
        image_path = image_root / str(sample["Image"])
        if not image_path.is_file():
            raise FileNotFoundError(f"MME-RealWorld image not found: {image_path}")
        image_b64, media_type = encode_original_image(image_path)
        prompt = format_evaluation_prompt(
            build_mmerealworld_prompt(sample, args),
            args,
        )
        text = chat(
            endpoint,
            args.model,
            image_b64,
            prompt,
            args.max_new_tokens,
            image_media_type=media_type,
            system_prompt=system_prompt_for(args),
            generation_profile_name=generation_profile_for(args),
        )
        choices = [str(choice) for choice in sample["Answer choices"]]
        payload, boxed = scoring_payload(text, args)
        pred = parse_mmerealworld_prediction(payload, choices)
        gt = str(sample["Ground truth"]).strip().upper()
        record = dict(sample)
        record.update({
            # Retain the official fields and add Output so the dump can also
            # be consumed directly by MME-RealWorld's eval_your_results.py.
            "Output": text,
            "idx": idx,
            "question_id": str(sample["Question_id"]),
            "image": str(image_path),
            "task": str(sample["Task"]),
            "subtask": str(sample["Subtask"]),
            "category": normalize_mmerealworld_category(sample["Category"]),
            "raw_category": str(sample["Category"]),
            "pred": pred,
            "gt": gt,
            "correct": pred == gt,
            "generated": text,
        })
        add_boxed_record_fields(
            record,
            args=args,
            prompt=prompt,
            boxed_answer=boxed,
        )
        return record

    records = parallel_eval_requests(
        one,
        enumerate(data),
        args,
        description="mmerealworld",
        single_endpoint_concurrency=workers,
    )
    return add_boxed_summary(summarize_mmerealworld(records), records, args), records


def main() -> None:
    import time

    args = build_parser().parse_args()
    if args.answer_format is None:
        args.answer_format = "answer_tag" if args.use_cot else "direct"
    if args.max_new_tokens is None:
        args.max_new_tokens = default_max_new_tokens(args)
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.visualprobe_max_new_tokens <= 0:
        raise ValueError("--visualprobe-max-new-tokens must be positive")
    if args.use_cot and args.answer_format == "direct":
        raise ValueError("--use_cot true requires a non-direct answer format")
    if args.answer_format == "answer_tag" and not args.use_cot:
        raise ValueError("--answer-format answer_tag requires --use_cot true")
    validate_generation_options(args)

    requested_output = Path(args.output)
    output = output_path_for_profile(requested_output, args.use_cot)
    args.output = str(output)
    if output != requested_output:
        print(
            f"use_cot=true: output renamed to {output}",
            flush=True,
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    for benchmark in [b.strip() for b in args.benchmarks.split(",")]:
        start = time.perf_counter()
        if benchmark == "zoombench":
            summary, records = eval_zoombench(args)
        elif benchmark in VSTAR_BENCHMARKS:
            summary, records = eval_vstar(args, benchmark)
        elif benchmark in ("hrbench4k", "hrbench8k"):
            summary, records = eval_hrbench(args, benchmark[-2:])
        elif benchmark == "mmerealworld":
            summary, records = eval_mmerealworld(args)
        elif benchmark == "treebench":
            summary, records = eval_treebench(args)
        elif benchmark == "babyvision":
            summary, records = eval_babyvision(args)
        elif benchmark == "cvbench":
            summary, records = eval_cvbench(args)
        elif benchmark == "mmstar":
            summary, records = eval_mmstar(args)
        elif benchmark == "mmvp":
            summary, records = eval_mmvp(args)
        elif benchmark in VISUALPROBE_BENCHMARKS:
            summary, records = eval_visualprobe(args, benchmark)
        else:
            raise ValueError(f"unknown benchmark {benchmark}")
        elapsed = time.perf_counter() - start
        row = {
            "arm": args.arm, "step": args.step, "benchmark": benchmark,
            "engine": "vllm",
            "answer_format": answer_format(args),
            "use_cot": use_cot(args),
            "generation_profile": generation_profile_for(args),
            "prompt_profile": prompt_profile_for(args),
            "system_prompt": system_prompt_for(args),
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0.0,
            "elapsed_sec": round(elapsed, 1),
            **summary,
        }
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        dump = output.with_name(
            output.stem + f"_{args.arm}_{args.step}_{benchmark}_samples.jsonl"
        )
        with dump.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"{benchmark}: {summary['accuracy']:.2%} (n={summary['n']}) "
            f"in {elapsed/60:.1f} min -> {dump.name}",
            flush=True,
        )


if __name__ == "__main__":
    main()
