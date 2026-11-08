"""Run an extractor over reviewed golden records and score it."""
from collections.abc import Callable
from pathlib import Path

import anthropic
from instructor.core import IncompleteOutputException, InstructorRetryException

from extractor.golden import load_record
from extractor.schema import PaperExtraction
from extractor.scoring import mean_scores, score_record

EXTRACTION_ERRORS = (
    ValueError,
    anthropic.APIError,
    InstructorRetryException,
    IncompleteOutputException,
)
SCORE_FIELDS = ("title", "authors", "datasets", "key_results", "limitations")


def load_reviewed(directory: Path) -> list[dict]:
    records = [load_record(p) for p in sorted(Path(directory).glob("*.json"))]
    reviewed = [r for r in records if r["reviewed"]]
    if not reviewed:
        raise ValueError(f"no reviewed golden records in {directory}")
    return reviewed


def evaluate(
    records: list[dict], extract_fn: Callable[[str], PaperExtraction]
) -> dict:
    """Score extract_fn on each record. Extraction failures score zero and are listed."""
    per_record: dict[str, dict[str, float]] = {}
    failures: list[dict[str, str]] = []
    for record in records:
        arxiv_id = record["arxiv_id"]
        try:
            predicted = extract_fn(record["source_text"])
        except EXTRACTION_ERRORS as exc:
            failures = [*failures, {"arxiv_id": arxiv_id, "error": str(exc)}]
            per_record = {**per_record, arxiv_id: dict.fromkeys(SCORE_FIELDS, 0.0)}
            continue
        scores = score_record(PaperExtraction(**record["expected"]), predicted)
        per_record = {**per_record, arxiv_id: scores}
    return {
        "per_record": per_record,
        "mean": mean_scores(list(per_record.values())),
        "failures": failures,
    }
