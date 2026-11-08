"""LLM-as-judge for the free-text fields (problem, method)."""
from typing import Literal

from pydantic import BaseModel

from extractor.prompts import load_prompt
from extractor.schema import PaperExtraction

# A different, stronger model than the extractor, so it does not grade itself.
DEFAULT_JUDGE_MODEL = "claude-opus-5-5"
JUDGE_PROMPT_VERSION = "v1"
JUDGE_MAX_TOKENS = 1024
JUDGE_MAX_RETRIES = 2

Verdict = Literal["correct", "partial", "incorrect"]
_SCORES: dict[str, float] = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


class FieldJudgment(BaseModel):
    reason: str  # declared first so the model reasons before it decides
    verdict: Verdict


class RecordJudgment(BaseModel):
    problem: FieldJudgment
    method: FieldJudgment


def verdict_score(verdict: str) -> float:
    return _SCORES[verdict]


def _build_prompt(source: str, expected: PaperExtraction, predicted: PaperExtraction) -> str:
    values = {
        "{source}": source,
        "{ref_problem}": expected.problem,
        "{ref_method}": expected.method,
        "{cand_problem}": predicted.problem,
        "{cand_method}": predicted.method,
    }
    prompt = load_prompt("judge", JUDGE_PROMPT_VERSION)
    for key, value in values.items():
        prompt = prompt.replace(key, value)
    return prompt


def judge_record(
    source: str,
    expected: PaperExtraction,
    predicted: PaperExtraction,
    *,
    client,
    model: str = DEFAULT_JUDGE_MODEL,
) -> dict[str, float]:
    result = client.messages.create(
        model=model,
        max_tokens=JUDGE_MAX_TOKENS,
        max_retries=JUDGE_MAX_RETRIES,
        response_model=RecordJudgment,
        messages=[{"role": "user", "content": _build_prompt(source, expected, predicted)}],
    )
    return {
        "problem": verdict_score(result.problem.verdict),
        "method": verdict_score(result.method.verdict),
    }
