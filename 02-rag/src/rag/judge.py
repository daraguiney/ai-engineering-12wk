"""LLM judge for answer correctness against the dataset's reference answer and evidence quotes."""
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

JUDGE_MODEL = "claude-opus-5-5"
JUDGE_MAX_TOKENS = 512
JUDGE_MAX_RETRIES = 2
PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "judge_v1.md"
SCORES = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


class Judgment(BaseModel):
    verdict: Literal["correct", "partial", "incorrect"]
    reason: str


def verdict_score(verdict: str) -> float:
    return SCORES[verdict]


def build_judge_prompt(question: str, expected: str, quotes: list[str], answer: str) -> str:
    if not expected.strip():
        raise ValueError("a reference answer is required to judge")
    quote_block = "\n".join(f'- "{q}"' for q in quotes)
    return (
        PROMPT_PATH.read_text(encoding="utf-8")
        .replace("{question}", question)
        .replace("{expected}", expected)
        .replace("{quotes}", quote_block)
        .replace("{answer}", answer)
    )


def judge_answer(question: str, expected: str, quotes: list[str], answer: str, *, client) -> dict:
    judgment, completion = client.messages.create_with_completion(
        model=JUDGE_MODEL,
        max_tokens=JUDGE_MAX_TOKENS,
        max_retries=JUDGE_MAX_RETRIES,
        response_model=Judgment,
        messages=[{"role": "user", "content": build_judge_prompt(question, expected, quotes, answer)}],
    )
    return {
        "verdict": judgment.verdict,
        "score": verdict_score(judgment.verdict),
        "reason": judgment.reason,
        "usage": {"input_tokens": completion.usage.input_tokens, "output_tokens": completion.usage.output_tokens},
    }
