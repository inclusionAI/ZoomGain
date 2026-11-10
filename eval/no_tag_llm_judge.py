"""Opt-in LLM judge for answer-tag rows that carry no <answer> tag (``--judge_when_no_ans_tag``).

Off by default: without the flag ``evaluate.py`` behaves exactly as before. With it, once a benchmark
has finished generating and has been scored by its own protocol (including the VisualProbe judge on the tag
content), every row whose response has no complete ``<answer>`` tag is sent -- the whole response -- to an LLM in
one batch (default Qwen3.8-Flash-Next through an OpenAI-compatible judge endpoint, API keys round-robin over a thread
pool). Rows with a tag keep the benchmark's own verdict.

The official ``accuracy`` is never touched. The second score is ``accuracy_judged``, aggregated exactly like
``accuracy`` (CV-Bench source-balanced, MMVP all-or-nothing pairs, otherwise per question); recomputing the
official score with the same aggregation must reproduce ``accuracy``, otherwise ``accuracy_judged`` is withheld.
A row whose judge call fails keeps its official verdict and is counted in ``no_tag_judge.failed``.

Prompt and client mirror Caption-OPD scripts/llm_judge.py (README section 7.44), where the "final" prompt was
validated: 100% on 320 known-answer rows, 99.2% agreement with a regex rescue on untagged rows that stopped by
themselves, 3.5% Yes on responses cut off mid-reasoning. API keys are read from the key file, never printed.
"""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

DEFAULT_KEYS = "api_keys.txt"
DEFAULT_ENDPOINT = "http://127.0.0.1:8031/v1/chat/completions"
DEFAULT_MODEL = "Qwen3.8-Flash-Next"
DEFAULT_TEMPLATE = "final"

PROMPTS = {
    # Vision-OPD eval/judge_qwenlm.py PROMPT_TEMPLATE, verbatim
    "visionopd": (
        "Your task is to judge whether the response expresses the same meaning "
        "as the answer of a question.\n"
        "The question is: {question}\n"
        "The answer is: {gt}\n"
        "The response is: {response}\n"
        "Please check and compare them and then judge. "
        "If the response is correct, your output should be Yes. "
        "Otherwise, your output should be No. Directly give me your output."
    ),
    "final": (
        "Your task is to judge whether the response gives the correct final answer to a question.\n"
        "The question is: {question}\n"
        "The correct answer is: {gt}\n"
        "The response is: {response}\n"
        "Judge only the final answer the response commits to, e.g. the option it ends with. "
        "If the response never commits to a final answer (for example it is cut off mid-reasoning), "
        "or its final answer differs from the correct answer, your output should be No. "
        "If its final answer matches the correct answer, your output should be Yes. Directly give me your output."
    ),
}
RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504}


def add_arguments(parser) -> None:
    parser.add_argument(
        "--judge_when_no_ans_tag", "--judge-when-no-ans-tag", dest="judge_when_no_ans_tag", action="store_true",
        help="After each benchmark, LLM-judge the full response of every row without an <answer> tag and report "
             "accuracy_judged next to the unchanged official accuracy (default: off).")
    parser.add_argument("--no-tag-judge-model", default=DEFAULT_MODEL)
    parser.add_argument("--no-tag-judge-template", choices=sorted(PROMPTS), default=DEFAULT_TEMPLATE)
    parser.add_argument("--no-tag-judge-endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--no-tag-judge-keys", default=DEFAULT_KEYS)
    parser.add_argument("--no-tag-judge-workers", type=int, default=32)


def load_keys(path: str = DEFAULT_KEYS) -> list[str]:
    keys = [line.strip() for line in open(path, encoding="utf-8") if line.strip() and not line.lstrip().startswith("#")]
    if not keys:
        raise RuntimeError(f"no API keys in {path}")
    return keys


def preflight(args) -> None:
    """Fail before any GPU work when the flag cannot be honoured. No-op without the flag."""
    if not getattr(args, "judge_when_no_ans_tag", False):
        return
    if args.answer_format != "answer_tag":
        raise ValueError("--judge_when_no_ans_tag requires --answer-format answer_tag")
    if args.no_tag_judge_workers < 1:
        raise ValueError("--no-tag-judge-workers must be positive")
    n = len(load_keys(args.no_tag_judge_keys))
    print(f"judge_when_no_ans_tag: untagged rows -> {args.no_tag_judge_model} "
          f"(prompt={args.no_tag_judge_template}, {n} keys, {args.no_tag_judge_workers} workers)", flush=True)


def gt_text(row: dict) -> str:
    """The reference answer as 'C. 16': the gt letter plus its option text when it can be recovered."""
    gt = str(row.get("gt", "")).strip()
    if not (len(gt) == 1 and gt.isalpha()):
        return gt[:-2] if gt.endswith(".0") else gt  # numeric / free-form answer
    text = row.get("gt_choice")
    if not text:
        prompt = str(row.get("prompt") or row.get("question") or "")
        letter = re.escape(gt.upper())
        for pat, flags in ((rf"(?m)^\s*\(?{letter}\)?[\.\):]\s*(.+?)\s*$", 0),  # "C. 16" / "(C) 16" / "C: 16" on its own line
                           (rf"\b{letter}:\s*([^,\n]+)", 0),                    # "Options: A: x, B: y" (MMStar)
                           (rf"\({letter}\)\s*([^()\n]+)", re.I)):              # "(a) Left (b) Right" (MMVP)
            m = re.search(pat, prompt, flags=flags)
            if m:
                text = m.group(1).strip()
                break
    return f"{gt.upper()}. {text}" if text else gt.upper()


def build_prompt(row: dict, template: str = DEFAULT_TEMPLATE, response: str | None = None) -> str:
    question = str(row.get("prompt") or row.get("question") or "").replace("<image>", "").strip()
    resp = row.get("generated", "") if response is None else response
    return PROMPTS[template].format(question=question, gt=gt_text(row), response=(resp or "").strip())


def parse_verdict(text: str) -> bool | None:
    t = (text or "").strip().strip("*").strip().lower()
    if t.startswith("yes"):
        return True
    if t.startswith("no"):
        return False
    has_yes, has_no = re.search(r"\byes\b", t), re.search(r"\bno\b", t)
    if has_yes and not has_no:
        return True
    if has_no and not has_yes:
        return False
    return None


def judge_prompts(prompts: list[str], keys: list[str], endpoint: str = DEFAULT_ENDPOINT, model: str = DEFAULT_MODEL,
                  workers: int = 32, max_tokens: int = 64, timeout: int = 120, attempts: int = 5,
                  progress: bool = True) -> list[dict]:
    """One chat request per prompt, keys round-robin; returns [{raw, verdict, status}] in order. A reply that is
    neither Yes nor No (the judge explained instead) is asked once more for a one-word answer."""
    local = threading.local()
    done = [0]
    lock = threading.Lock()

    def call(i: int, prompt: str) -> dict:
        first = ask(i, prompt)
        if first["status"] != "unparsed":
            return first
        again = ask(i, prompt + "\n\nAnswer with exactly one word: Yes or No.")
        return {**again, "raw": first["raw"] + " || " + again["raw"]} if again["verdict"] is not None else first

    def ask(i: int, prompt: str) -> dict:
        if getattr(local, "session", None) is None:
            local.session = requests.Session()
        key = keys[i % len(keys)]
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0,
                   "max_tokens": max_tokens}
        if model.startswith("Qwen3"):
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        last = "unknown"
        for attempt in range(1, attempts + 1):
            try:
                r = local.session.post(endpoint, json=payload, timeout=timeout,
                                       headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
                if r.status_code >= 400:
                    last = f"HTTP {r.status_code}: {r.text[:200]}".replace(key, "[REDACTED]")
                    if r.status_code in RETRYABLE and attempt < attempts:
                        time.sleep(min(30, 2 ** attempt))
                        continue
                    break
                raw = (r.json()["choices"][0]["message"].get("content") or "").strip()
                v = parse_verdict(raw)
                return {"raw": raw, "verdict": v, "status": "ok" if v is not None else "unparsed"}
            except Exception as exc:  # noqa: BLE001 - keep request diagnostics, never the key
                last = f"{type(exc).__name__}: {exc}".replace(key, "[REDACTED]")
                if attempt < attempts:
                    time.sleep(min(30, 2 ** attempt))
        return {"raw": "", "verdict": None, "status": f"error: {last[:200]}"}

    def wrapped(i, p):
        out = call(i, p)
        with lock:
            done[0] += 1
            if progress and (done[0] % 200 == 0 or done[0] == len(prompts)):
                print(f"   no-tag judge {done[0]}/{len(prompts)}", flush=True)
        return out

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(wrapped, range(len(prompts)), prompts))


def is_tagged(record: dict) -> bool:
    # add_boxed_record_fields sets this on every answer_tag row: a complete, non-empty <answer> tag was found.
    return bool(record.get("answer_tag_format_valid", True))


def aggregate(benchmark: str, records: list[dict], field: str, summary: dict, vclient) -> float | None:
    """The benchmark's own headline aggregation, applied to ``record[field]`` instead of ``record['correct']``."""
    if any(r.get(field) is None for r in records):
        return None
    rows = [dict(r, correct=bool(r[field])) for r in records]  # copies: the summarizers annotate rows
    if benchmark == "cvbench":
        return vclient.summarize_cvbench(rows, full_n=summary.get("full_n", len(rows)))["accuracy"]
    if benchmark == "mmvp":
        return vclient.summarize_mmvp(rows, full_question_n=summary.get("full_question_n", len(rows)))["accuracy"]
    return sum(r["correct"] for r in rows) / len(rows)


def apply(benchmark: str, summary: dict, records: list[dict], args, vclient) -> None:
    """Judge the untagged rows of one finished benchmark; add ``correct_judged`` / ``no_tag_judge`` to records and
    ``accuracy_judged`` / ``no_tag_judge`` to the summary, in place. ``accuracy`` and ``correct`` stay untouched."""
    started = time.perf_counter()
    template, model = args.no_tag_judge_template, args.no_tag_judge_model
    todo = [i for i, r in enumerate(records) if not is_tagged(r) and str(r.get("generated") or "").strip()]
    verdicts: dict[int, dict] = {}
    if todo:
        print(f"{benchmark}: judging {len(todo)} untagged rows with {model} (prompt={template})", flush=True)
        results = judge_prompts([build_prompt(records[i], template) for i in todo], load_keys(args.no_tag_judge_keys),
                                endpoint=args.no_tag_judge_endpoint, model=model, workers=args.no_tag_judge_workers)
        verdicts = dict(zip(todo, results))
    stats = dict(rows=0, yes=0, no=0, failed=0, empty_response=0, official_correct=0, flipped_to_correct=0,
                 flipped_to_wrong=0)
    for i, record in enumerate(records):
        official = record.get("correct")
        if is_tagged(record):
            record["correct_judged"] = official if official is None else bool(official)
            continue
        stats["rows"] += 1
        stats["official_correct"] += bool(official)
        if i in verdicts:
            res = verdicts[i]
        else:  # nothing to judge: an empty response cannot carry an answer
            res = {"raw": "", "verdict": False, "status": "empty_response"}
            stats["empty_response"] += 1
        record["no_tag_judge"] = {"verdict": res["verdict"], "status": res["status"], "raw": res["raw"],
                                  "model": model, "template": template, "gt_text": gt_text(record)}
        if res["verdict"] is None:
            stats["failed"] += 1
            judged = official if official is None else bool(official)
        else:
            stats["yes" if res["verdict"] else "no"] += 1
            judged = bool(res["verdict"])
        record["correct_judged"] = judged
        if official is not None and judged is not None and judged != bool(official):
            stats["flipped_to_correct" if judged else "flipped_to_wrong"] += 1
    info = dict(model=model, template=template, endpoint=args.no_tag_judge_endpoint, **stats,
                elapsed_sec=round(time.perf_counter() - started, 1))
    accuracy_judged = None
    if summary.get("accuracy") is None:
        info["note"] = "official score pending (deferred judge); accuracy_judged not computed"
    else:
        recomputed = aggregate(benchmark, records, "correct", summary, vclient)
        if recomputed is None or abs(recomputed - summary["accuracy"]) > 1e-9:
            info["error"] = f"aggregation mismatch: recomputed {recomputed} vs official {summary['accuracy']}"
            print(f"WARNING {benchmark}: {info['error']}; accuracy_judged withheld", flush=True)
        else:
            accuracy_judged = aggregate(benchmark, records, "correct_judged", summary, vclient)
    if stats["failed"]:
        print(f"WARNING {benchmark}: {stats['failed']} untagged rows got no judge verdict and keep the official one",
              flush=True)
    summary.update(judge_when_no_ans_tag=True, accuracy_judged=accuracy_judged, no_tag_judge=info)


def _pct(v) -> str:
    return "  -   " if v is None else f"{100 * v:6.2f}"


def report_line(run_name: str, step, benchmark: str, summary: dict) -> str:
    info = summary.get("no_tag_judge") or {}
    off, jud = summary.get("accuracy"), summary.get("accuracy_judged")
    delta = "" if off is None or jud is None else f" ({100 * (jud - off):+.2f})"
    return (f"[{run_name}:{step}] {benchmark}: official {_pct(off).strip()}% | judged {_pct(jud).strip()}%{delta} "
            f"[untagged {info.get('rows', 0)}: {info.get('model')} yes {info.get('yes', 0)}, no {info.get('no', 0)}, "
            f"failed {info.get('failed', 0)}]")


def print_table(summary_path, run_name: str, step, benchmarks: list[str]) -> None:
    """Official vs judged accuracy for every requested benchmark of this checkpoint (latest summary row each)."""
    latest: dict[str, dict] = {}
    path = Path(summary_path)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("arm") == run_name and row.get("step") == step and row.get("benchmark") in benchmarks:
                latest[row["benchmark"]] = row
    model = next((r["no_tag_judge"]["model"] for r in latest.values() if r.get("no_tag_judge")), DEFAULT_MODEL)
    print(f"\n=== {run_name} @{step}: official vs judged (untagged rows -> {model}) ===", flush=True)
    print(f"{'benchmark':16s}{'n':>6s}{'official':>10s}{'judged':>9s}{'delta':>8s}{'untagged':>10s}{'yes':>6s}{'failed':>8s}")
    both = []
    for b in benchmarks:
        row = latest.get(b)
        if row is None:
            print(f"{b:16s}  (no result)")
            continue
        info = row.get("no_tag_judge") or {}
        off, jud = row.get("accuracy"), row.get("accuracy_judged")
        delta = "    -   " if off is None or jud is None else f"{100 * (jud - off):+8.2f}"
        note = "" if row.get("judge_when_no_ans_tag") else "  (scored without --judge_when_no_ans_tag)"
        print(f"{b:16s}{row.get('n', 0):6d}{_pct(off):>10s}{_pct(jud):>9s}{delta}{info.get('rows', 0):10d}"
              f"{info.get('yes', 0):6d}{info.get('failed', 0):8d}{note}")
        if off is not None and jud is not None:
            both.append((off, jud))
    if both:
        mo, mj = (sum(x) / len(both) for x in zip(*both))
        print(f"{'mean (' + str(len(both)) + ')':16s}{'':6s}{_pct(mo):>10s}{_pct(mj):>9s}{100 * (mj - mo):+8.2f}", flush=True)
