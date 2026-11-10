"""Deterministic answer matching for the VisualProbe benchmark.

VisualProbe mixes ordinary VQA answers (objects, colors, directions and
counts) with exact-reading answers (sign text, identifiers, phone numbers and
license plates).  The standard VQAv2/LLaVA normalizer is too destructive for
the latter because it removes articles and internal punctuation globally.

This module therefore exposes two deliberately different notions:

* ``strict`` matching normalizes presentation only (Unicode, case, surrounding
  quotes, whitespace and sentence-final punctuation).
* ``vqa`` matching additionally accepts conservative semantic variants:
  leading articles for non-text-reading questions, a few unambiguous spelling
  variants, and whole numeric expressions such as ``one``/``1`` or
  ``twenty-one``/``21``.

The input ``prediction`` must already be the final candidate answer.  Callers
using a boxed/CoT response should extract the last ``\\boxed{...}`` first.
No model or network-based judge is used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import re
import unicodedata
from typing import Any


_LATEX_WRAPPERS = (
    r"\text",
    r"\mathrm",
    r"\mathbf",
    r"\textbf",
    r"\operatorname",
)
_LATEX_SPACES = (
    r"\,",
    r"\;",
    r"\:",
    r"\!",
    r"\quad",
    r"\qquad",
    r"\enspace",
    r"\ ",
)
_PAIRED_QUOTES = {
    "`": "`",
    "'": "'",
    '"': '"',
    "“": "”",
    "‘": "’",
}
_DASH_TRANSLATION = str.maketrans(
    {
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "―": "-",
        "−": "-",
    }
)
_TRAILING_SENTENCE_PUNCTUATION = ".!?。！？;；:："

_TEXT_READING_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"\b(?:written|printed|spelled|spelt|inscription)\b",
        r"\bwhat\s+does\b.+\bsay\b",
        r"\b(?:what|which)\s+(?:(?:is|are)\s+)?(?:the\s+)?"
        r"(?:texts?|words?|letters?|digits?|numbers?|names?|codes?)\b",
        r"\b(?:texts?|words?|letters?|digits?|numbers?)\s+"
        r"(?:above|below|under|over|before|after|left|right|on|in)\b",
        r"\b(?:license|licence)\s+plate\b",
        r"\b(?:phone|telephone|house|page|stall)\s+number\b",
        r"\bfirst\s+(?:line|word|letter)\b",
        r"\benglish\s+(?:text|word|line|letter)\b",
    )
)
_NUMERIC_QUESTION_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"\bhow\s+many\b",
        r"\bhow\s+much\b",
        r"\b(?:what|which)\s+(?:(?:is|are)\s+)?(?:the\s+)?"
        r"(?:number|count|price|weight|quantity|amount|percentage|discount)\b",
        r"\bnumber\s+of\b",
        r"\b(?:price|weight|quantity|amount|percentage|discount)\b",
    )
)

_ONES = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_DIGIT_WORDS = {word: value for word, value in _ONES.items() if value < 10}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_UNIT_ALIASES = {
    "%": "%",
    "percent": "%",
    "percentage": "%",
    "g": "g",
    "gram": "g",
    "grams": "g",
    "kg": "kg",
    "kilogram": "kg",
    "kilograms": "kg",
    "mm": "mm",
    "millimeter": "mm",
    "millimeters": "mm",
    "millimetre": "mm",
    "millimetres": "mm",
    "cm": "cm",
    "centimeter": "cm",
    "centimeters": "cm",
    "centimetre": "cm",
    "centimetres": "cm",
    "m": "m",
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "km": "km",
    "kilometer": "km",
    "kilometers": "km",
    "kilometre": "km",
    "kilometres": "km",
    "degree": "degree",
    "degrees": "degree",
    "yuan": "yuan",
    "rmb": "yuan",
    "dollar": "usd",
    "dollars": "usd",
    "usd": "usd",
    "euro": "eur",
    "euros": "eur",
    "pound": "gbp",
    "pounds": "gbp",
}
_CURRENCY_PREFIXES = {
    "$": "usd",
    "¥": "yuan",
    "￥": "yuan",
    "€": "eur",
    "£": "gbp",
}

_WHOLE_ANSWER_ALIASES = {
    "to the left": "left",
    "toward the left": "left",
    "towards the left": "left",
    "leftward": "left",
    "leftwards": "left",
    "to the right": "right",
    "toward the right": "right",
    "towards the right": "right",
    "rightward": "right",
    "rightwards": "right",
    "upward": "up",
    "upwards": "up",
    "downward": "down",
    "downwards": "down",
}


@dataclass(frozen=True)
class _NumericAnswer:
    """Canonical scalar or range used only during pairwise matching."""

    values: tuple[Decimal, ...]
    unit: str
    used_number_words: bool
    significant_leading_zero: bool
    kind: str = "scalar"

    @property
    def key(self) -> tuple[str, tuple[Decimal, ...], str]:
        return self.kind, self.values, self.unit

    @property
    def display(self) -> str:
        values = ",".join(_decimal_text(value) for value in self.values)
        suffix = f"|{self.unit}" if self.unit else ""
        return f"{self.kind}:{values}{suffix}"


@dataclass(frozen=True)
class VisualProbeAnswerJudgement:
    """Transparent verdict and normalization diagnostics."""

    correct: bool
    strict_correct: bool
    match_rule: str
    prediction: str
    ground_truth: str
    strict_prediction: str
    strict_ground_truth: str
    compared_prediction: str
    compared_ground_truth: str
    text_reading_question: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def _unwrap_latex_wrapper(text: str) -> str:
    value = text.strip()
    while True:
        unwrapped = False
        for command in _LATEX_WRAPPERS:
            prefix = command + "{"
            if not value.startswith(prefix):
                continue
            opening_brace = len(command)
            closing_brace = _matching_brace(value, opening_brace)
            if closing_brace == len(value) - 1:
                value = value[opening_brace + 1 : closing_brace].strip()
                unwrapped = True
                break
        if not unwrapped:
            return value


def _strip_paired_quotes(text: str) -> str:
    value = text.strip()
    while len(value) >= 2:
        expected = _PAIRED_QUOTES.get(value[0])
        if expected is None or value[-1] != expected:
            break
        value = value[1:-1].strip()
    return value


def normalize_visualprobe_answer(answer: Any) -> str:
    """Apply presentation-only normalization while preserving OCR content."""

    if answer is None:
        return ""
    value = unicodedata.normalize("NFKC", str(answer)).strip()
    value = _strip_paired_quotes(value)
    if len(value) >= 2 and value.startswith("$") and value.endswith("$"):
        value = value[1:-1].strip()

    value = value.replace(r"\left", "").replace(r"\right", "")
    value = _unwrap_latex_wrapper(value)
    for latex_space in _LATEX_SPACES:
        value = value.replace(latex_space, " ")

    value = value.translate(_DASH_TRANSLATION)
    value = value.replace("\n", " ").replace("\t", " ")
    value = re.sub(r"\s+", " ", value).strip()

    # Whitespace around punctuation is typographic, but the punctuation itself
    # is content for OCR identifiers and must remain.
    value = re.sub(r"\s*([,/:+@#&=_-])\s*", r"\1", value)
    value = re.sub(r"\(\s+", "(", value)
    value = re.sub(r"\s+\)", ")", value)

    value = value.rstrip()
    while value and value[-1] in _TRAILING_SENTENCE_PUNCTUATION:
        value = value[:-1].rstrip()
    return value.casefold()


def is_text_reading_question(question: Any) -> bool:
    """Return whether punctuation/articles are likely part of the target text."""

    value = unicodedata.normalize("NFKC", str(question or ""))
    return any(pattern.search(value) for pattern in _TEXT_READING_PATTERNS)


def is_numeric_question(question: Any) -> bool:
    value = unicodedata.normalize("NFKC", str(question or ""))
    return any(pattern.search(value) for pattern in _NUMERIC_QUESTION_PATTERNS)


def _parse_below_100(tokens: list[str]) -> int | None:
    if not tokens:
        return 0
    if len(tokens) == 1:
        if tokens[0] in _ONES:
            return _ONES[tokens[0]]
        return _TENS.get(tokens[0])
    if (
        len(tokens) == 2
        and tokens[0] in _TENS
        and tokens[1] in _ONES
        and 0 < _ONES[tokens[1]] < 10
    ):
        return _TENS[tokens[0]] + _ONES[tokens[1]]
    return None


def _parse_below_1000(tokens: list[str]) -> int | None:
    if not tokens:
        return 0
    if tokens.count("hundred") > 1:
        return None
    if "hundred" not in tokens:
        return _parse_below_100(tokens)

    index = tokens.index("hundred")
    left = tokens[:index]
    right = tokens[index + 1 :]
    if len(left) != 1 or left[0] not in _ONES:
        return None
    multiplier = _ONES[left[0]]
    if not 0 < multiplier < 10:
        return None
    remainder = _parse_below_100(right)
    if remainder is None:
        return None
    return multiplier * 100 + remainder


def _parse_integer_words(tokens: list[str]) -> int | None:
    tokens = [token for token in tokens if token != "and"]
    if not tokens:
        return None
    if "a" in tokens:
        if (
            len(tokens) < 2
            or tokens[0] != "a"
            or not any(
                scale in tokens for scale in ("hundred", "thousand", "million")
            )
        ):
            return None
        tokens = ["one", *tokens[1:]]

    total = 0
    remaining = list(tokens)
    for scale_name, scale_value in (("million", 1_000_000), ("thousand", 1_000)):
        if remaining.count(scale_name) > 1:
            return None
        if scale_name not in remaining:
            continue
        index = remaining.index(scale_name)
        left = remaining[:index]
        remaining = remaining[index + 1 :]
        multiplier = _parse_below_1000(left)
        if multiplier is None or multiplier <= 0:
            return None
        total += multiplier * scale_value

    remainder = _parse_below_1000(remaining)
    if remainder is None:
        return None
    return total + remainder


def _parse_scalar_number(
    text: str,
    *,
    allow_none: bool,
) -> tuple[Decimal, bool, bool] | None:
    value = text.strip()
    if not value:
        return None
    if allow_none and value == "none":
        return Decimal(0), True, False

    numeric_pattern = re.compile(
        r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$"
    )
    if numeric_pattern.fullmatch(value):
        unsigned = value.lstrip("+-").replace(",", "")
        integer_part = unsigned.split(".", 1)[0]
        leading_zero = len(integer_part) > 1 and integer_part.startswith("0")
        try:
            return Decimal(value.replace(",", "")), False, leading_zero
        except InvalidOperation:
            return None

    tokens = value.replace("-", " ").split()
    if not tokens:
        return None

    sign = 1
    if tokens[0] in {"minus", "negative"}:
        sign = -1
        tokens = tokens[1:]
    if not tokens:
        return None

    if tokens.count("point") > 1:
        return None
    if "point" in tokens:
        point = tokens.index("point")
        integer_tokens = tokens[:point]
        fractional_tokens = tokens[point + 1 :]
        if not fractional_tokens:
            return None

        if integer_tokens and all(token.isdigit() for token in integer_tokens):
            if len(integer_tokens) != 1:
                return None
            integer_value = int(integer_tokens[0])
            integer_used_words = False
        else:
            integer_value = _parse_integer_words(integer_tokens or ["zero"])
            integer_used_words = True
        if integer_value is None:
            return None

        fractional_digits: list[str] = []
        fractional_used_words = False
        for token in fractional_tokens:
            if token in _DIGIT_WORDS:
                fractional_digits.append(str(_DIGIT_WORDS[token]))
                fractional_used_words = True
            elif token.isdigit():
                fractional_digits.append(token)
            else:
                return None
        decimal_text = f"{integer_value}.{''.join(fractional_digits)}"
        try:
            number = Decimal(decimal_text) * sign
        except InvalidOperation:
            return None
        return number, integer_used_words or fractional_used_words, False

    # Reject malformed mixtures such as ``twenty 1``.  Decimal expressions
    # such as ``3 point five`` are handled explicitly above.
    if any(token.isdigit() for token in tokens):
        return None
    integer_value = _parse_integer_words(tokens)
    if integer_value is None:
        return None
    return Decimal(integer_value * sign), True, False


def _split_unit(text: str) -> tuple[str, str]:
    value = text.strip()
    prefix_unit = ""
    if value and value[0] in _CURRENCY_PREFIXES:
        prefix_unit = _CURRENCY_PREFIXES[value[0]]
        value = value[1:].strip()

    compact = re.fullmatch(
        r"([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)([a-z%]+)",
        value,
    )
    if compact and compact.group(2) in _UNIT_ALIASES:
        suffix_unit = _UNIT_ALIASES[compact.group(2)]
        if prefix_unit and prefix_unit != suffix_unit:
            return text, ""
        return compact.group(1), prefix_unit or suffix_unit

    tokens = value.split()
    if tokens and tokens[-1] in _UNIT_ALIASES:
        suffix_unit = _UNIT_ALIASES[tokens[-1]]
        if prefix_unit and prefix_unit != suffix_unit:
            return text, ""
        return " ".join(tokens[:-1]), prefix_unit or suffix_unit
    return value, prefix_unit


def _canonical_numeric_answer(
    text: str,
    *,
    question: Any,
) -> _NumericAnswer | None:
    core, unit = _split_unit(text)
    allow_none = is_numeric_question(question)

    range_match = re.fullmatch(r"between\s+(.+?)\s+and\s+(.+)", core)
    if range_match:
        left = _parse_scalar_number(range_match.group(1), allow_none=allow_none)
        right = _parse_scalar_number(range_match.group(2), allow_none=allow_none)
        if left is None or right is None:
            return None
        return _NumericAnswer(
            values=(left[0], right[0]),
            unit=unit,
            used_number_words=left[1] or right[1],
            significant_leading_zero=left[2] or right[2],
            kind="range",
        )

    scalar = _parse_scalar_number(core, allow_none=allow_none)
    if scalar is None:
        return None
    return _NumericAnswer(
        values=(scalar[0],),
        unit=unit,
        used_number_words=scalar[1],
        significant_leading_zero=scalar[2],
    )


def _decimal_text(value: Decimal) -> str:
    if value == 0:
        return "0"
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _semantic_text(value: str, *, text_reading: bool) -> str:
    if not value:
        return ""
    result = value
    if not text_reading:
        article = re.fullmatch(r"(?:a|an|the)\s+(.+)", result)
        if article and article.group(1).strip():
            result = article.group(1).strip()

        result = re.sub(r"\bgrey\b", "gray", result)
        result = re.sub(r"\bblonde\b", "blond", result)
        result = _WHOLE_ANSWER_ALIASES.get(result, result)
    return result


def judge_visualprobe_answer(
    prediction: Any,
    ground_truth: Any,
    question: Any = "",
) -> VisualProbeAnswerJudgement:
    """Judge one final candidate answer using strict and conservative VQA rules."""

    raw_prediction = "" if prediction is None else str(prediction)
    raw_ground_truth = "" if ground_truth is None else str(ground_truth)
    strict_prediction = normalize_visualprobe_answer(raw_prediction)
    strict_ground_truth = normalize_visualprobe_answer(raw_ground_truth)
    text_reading = is_text_reading_question(question)

    if (
        strict_prediction
        and strict_ground_truth
        and strict_prediction == strict_ground_truth
    ):
        return VisualProbeAnswerJudgement(
            correct=True,
            strict_correct=True,
            match_rule="strict",
            prediction=raw_prediction,
            ground_truth=raw_ground_truth,
            strict_prediction=strict_prediction,
            strict_ground_truth=strict_ground_truth,
            compared_prediction=strict_prediction,
            compared_ground_truth=strict_ground_truth,
            text_reading_question=text_reading,
        )

    numeric_prediction = _canonical_numeric_answer(
        strict_prediction,
        question=question,
    )
    numeric_ground_truth = _canonical_numeric_answer(
        strict_ground_truth,
        question=question,
    )
    if (
        numeric_prediction is not None
        and numeric_ground_truth is not None
        and numeric_prediction.key == numeric_ground_truth.key
        and not numeric_prediction.significant_leading_zero
        and not numeric_ground_truth.significant_leading_zero
        and (
            numeric_prediction.used_number_words
            or numeric_ground_truth.used_number_words
            or not text_reading
        )
    ):
        return VisualProbeAnswerJudgement(
            correct=True,
            strict_correct=False,
            match_rule="numeric",
            prediction=raw_prediction,
            ground_truth=raw_ground_truth,
            strict_prediction=strict_prediction,
            strict_ground_truth=strict_ground_truth,
            compared_prediction=numeric_prediction.display,
            compared_ground_truth=numeric_ground_truth.display,
            text_reading_question=text_reading,
        )

    semantic_prediction = _semantic_text(
        strict_prediction,
        text_reading=text_reading,
    )
    semantic_ground_truth = _semantic_text(
        strict_ground_truth,
        text_reading=text_reading,
    )
    if (
        semantic_prediction
        and semantic_ground_truth
        and semantic_prediction == semantic_ground_truth
    ):
        return VisualProbeAnswerJudgement(
            correct=True,
            strict_correct=False,
            match_rule="semantic",
            prediction=raw_prediction,
            ground_truth=raw_ground_truth,
            strict_prediction=strict_prediction,
            strict_ground_truth=strict_ground_truth,
            compared_prediction=semantic_prediction,
            compared_ground_truth=semantic_ground_truth,
            text_reading_question=text_reading,
        )

    compared_prediction = (
        numeric_prediction.display
        if numeric_prediction is not None
        else semantic_prediction
    )
    compared_ground_truth = (
        numeric_ground_truth.display
        if numeric_ground_truth is not None
        else semantic_ground_truth
    )
    return VisualProbeAnswerJudgement(
        correct=False,
        strict_correct=False,
        match_rule="mismatch",
        prediction=raw_prediction,
        ground_truth=raw_ground_truth,
        strict_prediction=strict_prediction,
        strict_ground_truth=strict_ground_truth,
        compared_prediction=compared_prediction,
        compared_ground_truth=compared_ground_truth,
        text_reading_question=text_reading,
    )


def visualprobe_answers_match(
    prediction: Any,
    ground_truth: Any,
    question: Any = "",
) -> bool:
    """Boolean convenience wrapper around :func:`judge_visualprobe_answer`."""

    return judge_visualprobe_answer(
        prediction=prediction,
        ground_truth=ground_truth,
        question=question,
    ).correct


def visualprobe_strict_answers_match(
    prediction: Any,
    ground_truth: Any,
) -> bool:
    """Match presentation-normalized answers without VQA equivalences."""

    normalized_prediction = normalize_visualprobe_answer(prediction)
    normalized_ground_truth = normalize_visualprobe_answer(ground_truth)
    return bool(
        normalized_prediction
        and normalized_ground_truth
        and normalized_prediction == normalized_ground_truth
    )
