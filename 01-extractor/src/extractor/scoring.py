"""Rule-based field scoring of a prediction against a golden extraction.

All scores are in [0, 1]. Free-text fields (problem, method) are left to the
LLM judge and are intentionally not scored here.
"""
import re

from extractor.schema import PaperExtraction

_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace."""
    return " ".join(_NON_ALNUM.sub("", text.lower()).split())


def set_f1(predicted: list[str], expected: list[str]) -> float:
    """F1 over normalized strings; two empty lists count as a perfect match."""
    pred = {normalize(p) for p in predicted}
    gold = {normalize(g) for g in expected}
    if not pred and not gold:
        return 1.0
    hits = len(pred & gold)
    if hits == 0:
        return 0.0
    precision, recall = hits / len(pred), hits / len(gold)
    return 2 * precision * recall / (precision + recall)


def number_recall(expected: list[str], predicted: list[str]) -> float:
    """Fraction of numbers in the expected results that appear in the prediction."""
    gold = set(_NUMBER.findall(" ".join(expected)))
    if not gold:
        return 1.0
    found = set(_NUMBER.findall(" ".join(predicted)))
    return len(gold & found) / len(gold)


def score_record(expected: PaperExtraction, predicted: PaperExtraction) -> dict[str, float]:
    return {
        "title": float(normalize(expected.title) == normalize(predicted.title)),
        "authors": set_f1(predicted.authors, expected.authors),
        "datasets": set_f1(predicted.datasets, expected.datasets),
        "key_results": number_recall(expected.key_results, predicted.key_results),
        "limitations": set_f1(predicted.limitations, expected.limitations),
    }


def mean_scores(rows: list[dict[str, float]]) -> dict[str, float]:
    """Per-field mean across records."""
    if not rows:
        raise ValueError("cannot average zero score rows")
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
