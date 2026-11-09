"""Retrieval test set: questions labelled by verbatim evidence quotes, not by chunk.

A retrieved chunk counts as a hit when it contains the evidence quote, so the labels stay
valid when the chunking strategy changes.
"""
import re
from typing import Literal

from pydantic import BaseModel, field_validator

ARXIV_ID = re.compile(r"^\d{4}\.\d{4,5}$")
_IGNORED = re.compile(r"[\s\-]+")
EXPECTED_EVIDENCE = {"single": 1, "cross": 2, "unanswerable": 0}


class Evidence(BaseModel):
    arxiv_id: str
    quote: str

    @field_validator("arxiv_id")
    @classmethod
    def _valid_id(cls, v: str) -> str:
        if not ARXIV_ID.match(v):
            raise ValueError(f"not a valid arXiv id: {v!r}")
        return v


class Question(BaseModel):
    id: str
    type: Literal["single", "cross", "unanswerable"]
    question: str
    evidence: list[Evidence]
    notes: str = ""


def normalize(text: str) -> str:
    """Matching key: drop whitespace and hyphens, so PDF line breaks and split words
    ("multi-\nhead" vs "multi-head") compare equal."""
    return _IGNORED.sub("", text)


def quote_in_text(quote: str, text: str) -> bool:
    return normalize(quote) in normalize(text)


def _evidence_problems(question: Question, texts: dict[str, str]) -> list[str]:
    expected = EXPECTED_EVIDENCE[question.type]
    problems: list[str] = []
    if len(question.evidence) != expected:
        word = "exactly 1" if expected == 1 else str(expected)
        if question.type == "unanswerable":
            problems.append("unanswerable must have no evidence")
        else:
            problems.append(f"{question.type} needs {word} evidence item(s)")
    for ev in question.evidence:
        if ev.arxiv_id not in texts:
            problems.append(f"unknown paper {ev.arxiv_id}")
        elif not quote_in_text(ev.quote, texts[ev.arxiv_id]):
            problems.append(f"quote not found in {ev.arxiv_id}: {ev.quote[:60]!r}")
    return problems


def validate_questions(questions: list[Question], texts: dict[str, str]) -> dict[str, list[str]]:
    """Return {question id: problems}. Empty dict means every question is valid."""
    result: dict[str, list[str]] = {}
    seen: set[str] = set()
    for q in questions:
        problems = _evidence_problems(q, texts)
        if q.id in seen:
            problems = [*problems, "duplicate id"]
        seen.add(q.id)
        if problems:
            result = {**result, q.id: problems}
    return result
