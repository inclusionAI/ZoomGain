"""ZoomBench prompt and answer parsing helpers."""

from __future__ import annotations

import argparse

import importlib.util

import json

import re

import sys

from pathlib import Path

from typing import Any

from PIL import Image

MCQ_INSTRUCTION = "Answer with the option's letter from the given choices directly."

NUMBER_INSTRUCTION = "Answer the question using a single number."

MCQ_INSTRUCTION_RE = re.compile(
    r"\s*Answer with the option's letter from the given choices\.?\s*$",
    re.IGNORECASE,
)

NUMERIC_INSTRUCTION_RE = re.compile(
    r"\s*Please answer using Arabic numerals\.?\s*$",
    re.IGNORECASE,
)

def build_prompt(sample: dict[str, Any]) -> str:
    query = str(sample["query"]).strip()
    question_type = sample.get("question_type")
    if question_type == "mcq":
        query = MCQ_INSTRUCTION_RE.sub("", query).strip()
        return f"{query}\n{MCQ_INSTRUCTION}"
    if question_type == "blank":
        query = NUMERIC_INSTRUCTION_RE.sub("", query).strip()
        return f"{query}\n{NUMBER_INSTRUCTION}"
    raise ValueError(f"unknown question_type: {question_type!r}")

def parse_prediction(text: str, question_type: str) -> str:
    text = text.strip()
    if question_type == "mcq":
        match = re.search(r"\b([A-D])\b", text, flags=re.IGNORECASE)
        return match.group(1).upper() if match else ""
    if question_type == "blank":
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            return ""
        pred = match.group(0)
        if pred.endswith(".0"):
            pred = pred[:-2]
        return pred
    return text

def normalize_answer(answer: Any, question_type: str) -> str:
    answer_text = str(answer).strip()
    if question_type == "mcq":
        return answer_text.upper()
    if question_type == "blank" and answer_text.endswith(".0"):
        return answer_text[:-2]
    return answer_text
