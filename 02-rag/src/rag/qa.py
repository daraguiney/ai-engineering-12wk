"""Cited question answering over retrieved chunks, with explicit abstention."""
from pathlib import Path

import instructor
from anthropic import Anthropic
from pydantic import BaseModel

ANSWER_MODEL = "claude-sonnet-5-5"
MAX_OUTPUT_TOKENS = 1024
MAX_RETRIES = 2
PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "answer_v1.md"


class Answer(BaseModel):
    answerable: bool
    answer: str
    citations: list[int]  # 1-based source numbers


def make_client():
    return instructor.from_anthropic(Anthropic(), mode=instructor.Mode.ANTHROPIC_JSON)


def build_prompt(question: str, hits: list[dict]) -> str:
    sources = "\n\n".join(
        f"[{i}] (paper {h['arxiv_id']})\n{h['text']}" for i, h in enumerate(hits, start=1)
    )
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.replace("{sources}", sources).replace("{question}", question)


def validate_citations(answer: Answer, n_sources: int) -> None:
    if not answer.answerable and answer.citations:
        raise ValueError("an answer that abstains must not cite sources")
    if answer.answerable and not answer.citations:
        raise ValueError("an answer must cite at least one source")
    bad = [c for c in answer.citations if not 1 <= c <= n_sources]
    if bad:
        raise ValueError(f"citation to missing source(s) {bad}; only 1..{n_sources} were provided")


def answer_question(question: str, hits: list[dict], *, client, model: str = ANSWER_MODEL) -> dict:
    answer, completion = client.messages.create_with_completion(
        model=model,
        max_tokens=MAX_OUTPUT_TOKENS,
        max_retries=MAX_RETRIES,
        response_model=Answer,
        messages=[{"role": "user", "content": build_prompt(question, hits)}],
    )
    validate_citations(answer, n_sources=len(hits))
    return {
        "answerable": answer.answerable,
        "answer": answer.answer,
        "cited": [
            {"arxiv_id": hits[c - 1]["arxiv_id"], "chunk_index": hits[c - 1]["chunk_index"]}
            for c in answer.citations
        ],
        "usage": {
            "input_tokens": completion.usage.input_tokens,
            "output_tokens": completion.usage.output_tokens,
        },
    }
