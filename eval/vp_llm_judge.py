#!/usr/bin/env python3
"""Re-score archived VisualProbe dumps with Mini-o3's official LLM judge prompt.

Our production metric (``visualprobe_answer.py``) is a deterministic
normalized-exact-match rule.  VisualProbe's authors instead score with an LLM
judge (``verl/utils/reward_score/general_qa_tool{,_qwen}.py`` in
github.com/Mini-o3/Mini-o3): the prediction, the ground truth and the question
go to GPT-4o / Qwen2.5-VL-72B as text -- no image -- with an instruction to
"consider synonyms or paraphrases as valid matches", and the reply's ``Score:``
digit is the reward.  ``SYSTEM_PROMPT`` and ``QUERY_PROMPT`` below are copied
from that file byte-for-byte.

Because every archived ``*_visualprobe_*_samples.jsonl`` already stores the
extracted ``pred``, the ``gt`` and the ``question``, re-scoring needs no GPU
generation: it is a pure function of text we already have.  That makes three
metrics derivable from one pass:

``strict``    what we report today (record's own ``correct`` flag)
``judge``     the official semantics: every item decided by the LLM
``cascade``   strict-correct items kept, only strict-wrong items sent to the
              judge -- i.e. ``strict OR judge``, which is >= both of the above

Judgements are cached by (question, gt, pred) so identical triples across arms
cost one call, and a re-run is free.  The raw judge reply is stored with every
record: scoring can be re-derived under a different parse without re-calling
the endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --- verbatim from Mini-o3 verl/utils/reward_score/general_qa_tool_qwen.py ---
SYSTEM_PROMPT = "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs.\nYour task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:\n------\n##INSTRUCTIONS:\n- Focus on the meaningful match between the predicted answer and the correct answer.\n- Consider synonyms or paraphrases as valid matches.\n- Evaluate the correctness of the prediction compared to the answer."

QUERY_PROMPT = """I will give you a question related to an image and the following text as inputs:\n\n1. **Question Related to the Image**: {question}\n2. **Ground Truth Answer**: {ground_truth}\n3. **Model Predicted Answer**: {prediction}\n\nYour task is to evaluate the model's predicted answer against the ground truth answer, based on the context provided by the question related to the image. Consider the following criteria for evaluation:\n- **Relevance**: Does the predicted answer directly address the question posed, considering the information provided by the given question?\n- **Accuracy**: Compare the predicted answer to the ground truth answer. You need to evaluate from the following two perspectives:\n(1) If the ground truth answer is open-ended, consider whether the prediction accurately reflects the information given in the ground truth without introducing factual inaccuracies. If it does, the prediction should be considered correct.\n(2) If the ground truth answer is a definitive answer, strictly compare the model's prediction to the actual answer. Pay attention to unit conversions such as length and angle, etc. As long as the results are consistent, the model's prediction should be deemed correct.\n**Output Format**:\nYour response should include an integer score indicating the correctness of the prediction: 1 for correct and 0 for incorrect. Note that 1 means the model's prediction strictly aligns with the ground truth, while 0 means it does not.\nThe format should be \"Score: 0 or 1\""""
# --- end verbatim ---

PROMPT_SHA = hashlib.sha256((SYSTEM_PROMPT + QUERY_PROMPT).encode()).hexdigest()[:16]

# The endpoint must be reached directly; this host exports an http_proxy that
# would otherwise intercept and 502 the request (README 23.7 records the same
# trap for the difficulty annotator).
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

_SCORE_RE = re.compile(r"score\s*[:：]\s*\**\s*([01])", re.IGNORECASE)


def triple_key(question: str, gt: str, pred: str) -> str:
    payload = f"{question}\x00{gt}\x00{pred}\x00{PROMPT_SHA}"
    return hashlib.sha256(payload.encode()).hexdigest()


def parse_score(reply: str) -> tuple[int | None, int | None]:
    """Return ``(strict_parse, official_parse)``.

    ``strict_parse`` requires a literal ``Score: 0|1``.  ``official_parse``
    reproduces Mini-o3's own ``1.0 if '1' in response else 0.0``, kept so the
    fidelity gap between the two parses is measurable rather than assumed.
    """

    official = 1 if "1" in reply else 0
    match = _SCORE_RE.search(reply)
    if match:
        return int(match.group(1)), official
    tail = reply.strip().split()[-1] if reply.strip() else ""
    if tail in {"0", "1"}:
        return int(tail), official
    return None, official


def judge_call(
    endpoint: str,
    model: str,
    question: str,
    gt: str,
    pred: str,
    *,
    max_tokens: int,
    enable_thinking: bool,
    timeout: float,
    retries: int,
    system_prompt: str | None = None,
    query_template: str | None = None,
) -> str:
    """One judge verdict as raw text.

    ``system_prompt``/``query_template`` default to VisualProbe's (Mini-o3's)
    wording. Another benchmark that ships its own judge prompt -- BabyVision
    does -- passes it here rather than being scored under a foreign rubric;
    only the transport, retry and thinking-disabled contract are shared.
    ``system_prompt=None`` with a template that needs no system turn sends a
    single user message, matching such a benchmark's own call shape.
    """

    template = QUERY_PROMPT if query_template is None else query_template
    system = SYSTEM_PROMPT if (system_prompt is None and query_template is None) else system_prompt
    messages = [{"role": "system", "content": system}] if system else []
    messages.append(
        {
            "role": "user",
            "content": template.format(
                question=question, ground_truth=gt, prediction=pred
            ),
        }
    )
    body: dict = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                f"{endpoint.rstrip('/')}/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with _OPENER.open(request, timeout=timeout) as response:
                message = json.load(response)["choices"][0]["message"]
            reasoning = message.get("reasoning") or message.get("reasoning_content")
            return (str(reasoning or "") + str(message.get("content") or "")).strip()
        except Exception as error:  # noqa: BLE001
            last = error
            time.sleep(min(2.0 ** attempt, 8.0))
    raise RuntimeError(f"judge failed after {retries} attempts: {last!r}")


class Cache:
    """Append-only judgement cache keyed by (question, gt, pred, prompt)."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.data: dict[str, dict] = {}
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        self.data[row["key"]] = row
        self.handle = path.open("a", encoding="utf-8")

    def get(self, key: str) -> dict | None:
        return self.data.get(key)

    def put(self, key: str, row: dict) -> None:
        with self.lock:
            self.data[key] = row
            self.handle.write(json.dumps({"key": key, **row}, ensure_ascii=False) + "\n")
            self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def collect_rows(dumps: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for dump in dumps:
        arm = dump.name.replace("_samples.jsonl", "")
        for line in dump.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            rows.append(
                {
                    "dump": str(dump),
                    "arm": arm,
                    "index": record.get("index", record.get("doc_id")),
                    "question": str(record.get("question", "")),
                    "gt": str(record.get("gt", "")),
                    "pred": str(record.get("pred", "")),
                    "strict": bool(record.get("correct")),
                    "match_rule": record.get("match_rule"),
                }
            )
    return rows


def summarize(rows: list[dict]) -> dict:
    by_arm: dict[str, list[dict]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
    out = {}
    for arm, group in sorted(by_arm.items()):
        n = len(group)
        strict = sum(r["strict"] for r in group)
        judged = [r for r in group if r.get("judge") is not None]
        judge = sum(r["judge"] == 1 for r in judged)
        cascade = sum(1 for r in group if r["strict"] or r.get("judge") == 1)
        out[arm] = {
            "n": n,
            "judged": len(judged),
            "unparsed": sum(1 for r in group if r.get("judge") is None),
            "strict_acc": 100 * strict / n,
            "judge_acc": 100 * judge / len(judged) if judged else None,
            "cascade_acc": 100 * cascade / n,
            "strict_yes_judge_no": sum(
                1 for r in group if r["strict"] and r.get("judge") == 0
            ),
            "strict_no_judge_yes": sum(
                1 for r in group if not r["strict"] and r.get("judge") == 1
            ),
            "parse_disagree_official": sum(
                1
                for r in group
                if r.get("judge") is not None
                and r.get("judge_official") is not None
                and r["judge"] != r["judge_official"]
            ),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL")
    parser.add_argument("--model", default="judge")
    parser.add_argument(
        "--dumps",
        nargs="+",
        required=True,
        help="glob(s) over archived *_visualprobe_*_samples.jsonl",
    )
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--cache", default=None, help="cache jsonl (default <out>/cache.jsonl)")
    parser.add_argument("--concurrency", type=int, default=32)
    # 128 truncates this judge mid-analysis: it writes ~650 chars of reasoning
    # before the "Score:" line, so a short budget silently produces unparsed
    # rows (and Mini-o3's own `'1' in response` parse would score those 1 off a
    # stray "1." in the numbered rationale).
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="cap rows (smoke)")
    parser.add_argument(
        "--strict-wrong-only",
        action="store_true",
        help="cascade mode: only call the judge on strict-wrong rows",
    )
    args = parser.parse_args()

    dumps: list[Path] = []
    for pattern in args.dumps:
        matched = sorted(Path().glob(pattern)) if not Path(pattern).is_file() else [Path(pattern)]
        if not matched:
            matched = sorted(Path("/").glob(pattern.lstrip("/")))
        dumps.extend(matched)
    dumps = [d for d in dumps if d.is_file()]
    if not dumps:
        raise SystemExit(f"no dumps matched {args.dumps}")

    rows = collect_rows(dumps)
    if args.limit:
        rows = rows[: args.limit]
    todo = [r for r in rows if not (args.strict_wrong_only and r["strict"])]
    print(
        f"dumps={len(dumps)} rows={len(rows)} to-judge={len(todo)} "
        f"prompt_sha={PROMPT_SHA}",
        flush=True,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Cache(Path(args.cache) if args.cache else out_dir / "cache.jsonl")

    pending = []
    for row in todo:
        key = triple_key(row["question"], row["gt"], row["pred"])
        hit = cache.get(key)
        if hit is not None:
            row["judge"] = hit["judge"]
            row["judge_official"] = hit.get("judge_official")
            row["judge_raw"] = hit.get("judge_raw", "")
        else:
            pending.append((key, row))
    unique: dict[str, dict] = {}
    for key, row in pending:
        unique.setdefault(key, row)
    print(
        f"cache hits={len(todo) - len(pending)}  pending rows={len(pending)}  "
        f"unique triples={len(unique)}",
        flush=True,
    )

    if unique:
        done = 0
        started = time.time()

        def work(item):
            key, row = item
            reply = judge_call(
                args.endpoint,
                args.model,
                row["question"],
                row["gt"],
                row["pred"],
                max_tokens=args.max_tokens,
                enable_thinking=args.enable_thinking,
                timeout=args.timeout,
                retries=args.retries,
            )
            score, official = parse_score(reply)
            return key, {"judge": score, "judge_official": official, "judge_raw": reply}

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(work, item) for item in unique.items()]
            for future in as_completed(futures):
                key, payload = future.result()
                cache.put(key, payload)
                done += 1
                if done % 100 == 0 or done == len(unique):
                    rate = done / max(time.time() - started, 1e-6)
                    print(
                        f"  judged {done}/{len(unique)}  ({rate:.1f}/s)",
                        flush=True,
                    )
        for key, row in pending:
            hit = cache.get(key)
            row["judge"] = hit["judge"]
            row["judge_official"] = hit.get("judge_official")
            row["judge_raw"] = hit.get("judge_raw", "")
    cache.close()

    with (out_dir / "judged.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize(rows)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"prompt_sha": PROMPT_SHA, "model": args.model, "arms": summary},
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n{'arm':<66}{'n':>5}{'strict':>8}{'judge':>8}{'cascade':>9}{'S+J-':>6}{'S-J+':>6}")
    for arm, stats in summary.items():
        judge_text = f"{stats['judge_acc']:8.2f}" if stats["judge_acc"] is not None else f"{'--':>8}"
        print(
            f"{arm[:64]:<66}{stats['n']:>5}{stats['strict_acc']:8.2f}{judge_text}"
            f"{stats['cascade_acc']:9.2f}{stats['strict_yes_judge_no']:6d}"
            f"{stats['strict_no_judge_yes']:6d}"
        )
    print(f"\nwrote {out_dir}/judged.jsonl and summary.json")


if __name__ == "__main__":
    main()
