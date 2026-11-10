"""Pure ZoomGain answer parsing and scoring; extracted from the original evaluator."""
from __future__ import annotations
import math
import re
from decimal import Decimal, InvalidOperation
from typing import Any
ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
OPTION_PATTERN = re.compile(r"^([A-Z])\.\s+", re.MULTILINE)
NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")

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


