from types import SimpleNamespace

import pytest

from rag.judge import Judgment, build_judge_prompt, judge_answer, verdict_score


def test_prompt_contains_question_reference_quotes_and_answer():
    p = build_judge_prompt("How many?", "80 sentences", ["a minibatch of 80 sentences"], "It is 80.")
    for part in ("How many?", "80 sentences", '- "a minibatch of 80 sentences"', "It is 80."):
        assert part in p
    assert "{question}" not in p and "{answer}" not in p


def test_prompt_requires_a_reference_answer():
    with pytest.raises(ValueError, match="reference"):
        build_judge_prompt("q", "  ", [], "a")


def test_verdict_scores():
    assert [verdict_score(v) for v in ("correct", "partial", "incorrect")] == [1.0, 0.5, 0.0]


def test_judge_answer_returns_score_and_usage():
    class Client:
        messages = SimpleNamespace(create_with_completion=lambda **kw: (
            Judgment(verdict="partial", reason="missed the second number"),
            SimpleNamespace(usage=SimpleNamespace(input_tokens=300, output_tokens=20)),
        ))

    out = judge_answer("q", "ref", ["quote"], "ans", client=Client())
    assert out["score"] == 0.5 and out["verdict"] == "partial" and out["usage"]["input_tokens"] == 300
