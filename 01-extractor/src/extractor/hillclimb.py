"""Per-(case, rep) recording for the hillclimb loop.

Wraps the existing extractor, scorer and judge without changing them, and keeps what the
loop needs: the full transcript, model, token usage and the grade dict for each attempt.
"""
from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from extractor.extract import DEFAULT_MAX_RETRIES, DEFAULT_MODEL, MAX_OUTPUT_TOKENS
from extractor.judge import (
    DEFAULT_JUDGE_MODEL,
    JUDGE_MAX_RETRIES,
    JUDGE_MAX_TOKENS,
    RecordJudgment,
    verdict_score,
)
from extractor.judge import (
    _build_prompt as build_judge_prompt,
)
from extractor.prompts import load_prompt
from extractor.schema import PaperExtraction
from extractor.scoring import score_record

TARGET_FIELDS = ("limitations", "method")


def target_score(grade: dict[str, float]) -> float:
    """Headline metric: mean of the two fields with headroom."""
    return sum(grade[f] for f in TARGET_FIELDS) / len(TARGET_FIELDS)


def _usage(completion) -> dict[str, int]:
    u = completion.usage
    return {
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
    }


def _sum_usage(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    return {k: a.get(k, 0) + b.get(k, 0) for k in {*a, *b}}


def run_case(record: dict, rep: int, *, client, prompt_version: str) -> tuple[dict, list[dict]]:
    """Extract, score and judge one record once. Returns (results row, trace turns)."""
    started = time.monotonic()
    expected = PaperExtraction(**record["expected"])
    prompt = load_prompt("extract_paper", prompt_version).replace(
        "{document}", record["source_text"]
    )
    predicted, completion = client.messages.create_with_completion(
        model=DEFAULT_MODEL,
        max_tokens=MAX_OUTPUT_TOKENS,
        max_retries=DEFAULT_MAX_RETRIES,
        response_model=PaperExtraction,
        messages=[{"role": "user", "content": prompt}],
    )
    judge_prompt = build_judge_prompt(record["source_text"], expected, predicted)
    judgment, judge_completion = client.messages.create_with_completion(
        model=DEFAULT_JUDGE_MODEL,
        max_tokens=JUDGE_MAX_TOKENS,
        max_retries=JUDGE_MAX_RETRIES,
        response_model=RecordJudgment,
        messages=[{"role": "user", "content": judge_prompt}],
    )
    grade = {
        **score_record(expected, predicted),
        "problem": verdict_score(judgment.problem.verdict),
        "method": verdict_score(judgment.method.verdict),
    }
    grade = {**grade, "target": target_score(grade)}
    row = {
        "prompt_id": record["arxiv_id"],
        "rep": rep,
        "prompt": prompt,
        "grade": grade,
        "explanation": {
            "problem": judgment.problem.reason,
            "method": judgment.method.reason,
        },
        "model": completion.model,
        "judge_model": judge_completion.model,
        "latency_s": round(time.monotonic() - started, 2),
        "usage": _usage(completion),
        "judge_usage": _usage(judge_completion),
    }
    trace = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": predicted.model_dump_json(indent=2)},
        {"role": "user", "content": f"[judge prompt]\n{judge_prompt}"},
        {"role": "assistant", "content": judgment.model_dump_json(indent=2)},
        {"role": "user", "content": f"[expected]\n{expected.model_dump_json(indent=2)}"},
    ]
    return row, trace


def load_done(results_path: Path) -> set[tuple[str, int]]:
    """(case, rep) pairs already recorded, so an interrupted run can resume."""
    if not results_path.is_file():
        return set()
    rows = (json.loads(line) for line in results_path.read_text().splitlines() if line.strip())
    return {(r["prompt_id"], r["rep"]) for r in rows}


def record_case(
    out_dir: Path, row: dict, trace: list[dict]
) -> None:
    """Append the row and write the trace file as soon as a case completes."""
    (out_dir / "traces").mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "traces" / f"{row['prompt_id']}_rep{row['rep']}.json"
    trace_path.write_text(json.dumps(trace, indent=2) + "\n")
    with (out_dir / "results.jsonl").open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def run_round(
    records: list[dict],
    out_dir: Path,
    *,
    reps: int,
    run_one: Callable[[dict, int], tuple[dict, list[dict]]],
    concurrency: int = 1,
) -> int:
    """Run every missing (case, rep). Returns the number of cases newly recorded.

    A failed case raises after the others finish; whatever completed stays on disk, so
    re-running the same command resumes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    done = load_done(out_dir / "results.jsonl")
    todo = [(r, k) for r in records for k in range(reps) if (r["arxiv_id"], k) not in done]
    lock = threading.Lock()

    def work(item: tuple[dict, int]) -> None:
        row, trace = run_one(*item)
        with lock:
            record_case(out_dir, row, trace)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(work, todo))
    return len(todo)
