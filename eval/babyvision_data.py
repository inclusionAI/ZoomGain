"""BabyVision loading, prompt construction and answer judging.

BabyVision (UniPat-AI, 388 tasks) asks for visual abilities a three-year-old
has and current MLLMs mostly lack: find-the-different, mazes, cube unfolding,
pattern completion.  Human accuracy is 94.1% while the best reported model is
under 50%, so it is the most vision-centric instrument available to this
project -- and also the one most likely to sit on the floor for a 4B model,
which is why it should be probed before being trusted as a metric.

Three properties of the released data drive everything here:

1. **The options carry no letters.**  ``options`` is a plain list and
   ``choiceAns`` is a 0-based index into it, so the letters we print and the
   letter we score against must be generated from one convention or the whole
   choice subset silently misaligns.
2. **Option counts vary from 2 to 6.**  A hard-coded A--D letter set would both
   miss the E/F answers and accept an out-of-range ``D`` on a 3-option
   question, so the valid letters are derived per sample.
3. **65% of the tasks are free-form ``blank`` answers** (coordinates like
   ``(4,7)``, counts, short phrases).  There is no dependable deterministic
   rule for those, which is why BabyVision is judged in ``full`` mode: the
   cascade shortcut used for VisualProbe has no safe first stage here.

The field names in the released ``meta_data.jsonl`` are lower-case
(``type``/``subtype``); the upstream ``evaluate_model.py`` reads ``Type``/
``Subtype`` and would raise on this file, so it is not a reference
implementation to copy from.  The judge prompt below, however, *is* copied from
upstream ``babyvision_eval/utils.py`` verbatim.
"""

from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

BABYVISION_EXPECTED_N = 388
BABYVISION_TYPES = (
    "Fine-grained Discrimination",
    "Visual Tracking",
    "Spatial Perception",
    "Visual Pattern Recognition",
)
BABYVISION_ANSWER_TYPES = ("choice", "blank")

# --- verbatim from BabyVision babyvision_eval/utils.py (LLM_JUDGE_PROMPT) ---
BABYVISION_JUDGE_PROMPT = """You are a careful and strict evaluator. You will be given:

1. **Question**
2. **Ground Truth Answer** (correct answer)
3. **Model Output** (answer from another model)

**Your goal:** Determine if the Model Output **accurately matches** the Ground Truth Answer in meaning.

* Matching means: the facts, entities, and key details are equivalent, even if phrasing differs.
* Not matching means: the Model Output is wrong, incomplete, contains extra incorrect facts, or changes the meaning.

**Process (internal reasoning):**

1. Read and understand the Question, Ground Truth Answer, and Model Output.
2. Ignore small wording differences, formatting, or synonyms.
3. If all factual content matches, conclude `1`. Otherwise, conclude `0`.

**Important:**

* Think through your decision step-by-step **internally** before responding.
* In your final output, return **only** True or False, with no extra text or explanation.

**Output format:**

True

or

False

**Input:**

Question: {question},
Ground Truth Answer: {ground_truth},
Model Output: {prediction}
"""
# --- end verbatim (only the placeholder names are adapted to judge_call) ---

# Upstream decides with ``'true' in reply`` before ``'false' in reply``, which
# any chatty judge can trip. We require a standalone token and take the last
# one, reporting anything else as unparsed instead of defaulting to False.
_VERDICT_RE = re.compile(r"\b(true|false)\b", re.IGNORECASE)


def parse_babyvision_verdict(reply: str) -> bool | None:
    matches = _VERDICT_RE.findall(str(reply or ""))
    if not matches:
        return None
    return matches[-1].lower() == "true"


def encode_babyvision_image(path: str | Path) -> tuple[str, str]:
    """Normalise to upright RGB PNG and return ``(base64, media_type)``.

    Raw-byte passthrough is wrong for this release. Its images are JPEG/PNG/
    **WebP**, RGB/**RGBA**/**palette**, ten files carry an extension that
    disagrees with their real format (so a media type taken from the suffix
    would be a lie), and four carry a rotation EXIF tag. ``exif_transpose``
    comes first because those four would otherwise reach the model in a
    different orientation than the human annotator saw -- on a benchmark about
    spatial reasoning that is answer-changing, not cosmetic. Re-encoding costs
    nothing here: the median image is 0.29 MP, unlike the 4K HR-Bench files
    that motivated passthrough in the first place.
    """

    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode(), "image/png"


def format_babyvision_choices(options: list[str]) -> str:
    """``(A) first`` / ``(B) second`` ... exactly as upstream renders them."""

    return "\n".join(f"({chr(65 + i)}) {choice}" for i, choice in enumerate(options))


def babyvision_letters(sample: dict) -> str:
    """The letters that are actually valid for this sample."""

    return "".join(chr(65 + i) for i in range(len(sample.get("options") or [])))


def load_babyvision(root: str | Path) -> list[dict[str, Any]]:
    """Load and fully validate the released BabyVision benchmark directory."""

    base = Path(root).expanduser().resolve()
    annotation = base / "meta_data.jsonl"
    if not annotation.is_file():
        raise FileNotFoundError(f"BabyVision annotation not found: {annotation}")

    samples: list[dict[str, Any]] = []
    seen: set[Any] = set()
    with annotation.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            context = f"{annotation}:{line_number}"
            for key in ("taskId", "question", "image", "ansType", "type", "subtype"):
                if key not in row:
                    raise ValueError(f"{context}: missing {key}")
            task_id = row["taskId"]
            if task_id in seen:
                raise ValueError(f"{context}: duplicate taskId {task_id}")
            seen.add(task_id)

            answer_type = str(row["ansType"])
            if answer_type not in BABYVISION_ANSWER_TYPES:
                raise ValueError(f"{context}: unexpected ansType {answer_type!r}")
            if str(row["type"]) not in BABYVISION_TYPES:
                raise ValueError(f"{context}: unexpected type {row['type']!r}")

            options = list(row.get("options") or [])
            if answer_type == "choice":
                index = row.get("choiceAns")
                if not isinstance(index, int) or not 0 <= index < len(options):
                    raise ValueError(
                        f"{context}: choiceAns {index!r} outside its "
                        f"{len(options)} options"
                    )
                answer = chr(65 + index)
            else:
                answer = str(row.get("blankAns", "")).strip()
                if not answer:
                    raise ValueError(f"{context}: blank task has no blankAns")

            image = base / row["image"]
            if not image.is_file():
                raise FileNotFoundError(f"{context}: image not found: {image}")

            samples.append(
                {
                    # ``coT`` holds a reference solution and is deliberately not
                    # carried here: it must never reach a prompt.
                    "index": task_id,
                    "question": str(row["question"]).strip(),
                    "options": options,
                    "answer": answer,
                    "ans_type": answer_type,
                    "type": str(row["type"]),
                    "subtype": str(row["subtype"]),
                    "image": str(image),
                }
            )

    if len(samples) != BABYVISION_EXPECTED_N:
        raise RuntimeError(
            f"expected {BABYVISION_EXPECTED_N} BabyVision tasks, got {len(samples)}"
        )
    return samples


def build_babyvision_question(sample: dict) -> str:
    """Question plus lettered choices; the answer wrapper is applied later."""

    question = sample["question"]
    if sample["ans_type"] != "choice":
        return question
    return f"{question}\nChoices:\n{format_babyvision_choices(sample['options'])}"


def parse_babyvision_prediction(payload: str, sample: dict) -> str:
    """Deterministic reading of the extracted answer.

    Only feeds the secondary ``accuracy_deterministic`` column -- the LLM judge
    is the primary metric -- but it still respects the per-sample letter range
    so that column is not quietly inflated by out-of-range guesses.
    """

    text = str(payload or "").strip()
    if sample["ans_type"] != "choice":
        return text
    letters = babyvision_letters(sample)
    if not letters:
        return ""
    match = re.search(rf"\b([{letters}])\b", text, flags=re.IGNORECASE)
    return match.group(1).upper() if match else ""
