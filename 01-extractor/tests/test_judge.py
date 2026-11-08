import pytest

from extractor.judge import FieldJudgment, RecordJudgment, judge_record, verdict_score
from extractor.prompts import load_prompt
from extractor.schema import PaperExtraction

PAPER = PaperExtraction(
    title="T", authors=["A"], problem="p", method="m",
    datasets=[], key_results=[], limitations=[],
)


class FakeClient:
    def __init__(self, result):
        self.result, self.calls = result, []
        self.messages = self
        self.create = self._create

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def judgment(problem="correct", method="partial"):
    return RecordJudgment(
        problem=FieldJudgment(reason="r", verdict=problem),
        method=FieldJudgment(reason="r", verdict=method),
    )


def test_verdict_score_mapping():
    assert verdict_score("correct") == 1.0
    assert verdict_score("partial") == 0.5
    assert verdict_score("incorrect") == 0.0


def test_invalid_verdict_rejected():
    with pytest.raises(ValueError):
        FieldJudgment(reason="r", verdict="great")


def test_judge_record_returns_scores_and_uses_judge_model():
    client = FakeClient(judgment())
    scores = judge_record("source", PAPER, PAPER, client=client, model="judge-m")
    assert scores == {"problem": 1.0, "method": 0.5}
    call = client.calls[0]
    assert call["model"] == "judge-m"
    assert call["response_model"] is RecordJudgment


def test_judge_prompt_contains_source_reference_and_candidate():
    client = FakeClient(judgment())
    ref = PAPER.model_copy(update={"problem": "REFERENCE-PROBLEM"})
    cand = PAPER.model_copy(update={"problem": "CANDIDATE-PROBLEM"})
    judge_record("SOURCE-TEXT", ref, cand, client=client, model="m")
    content = client.calls[0]["messages"][0]["content"]
    for needle in ("SOURCE-TEXT", "REFERENCE-PROBLEM", "CANDIDATE-PROBLEM"):
        assert needle in content


def test_judge_prompt_file_has_all_placeholders():
    prompt = load_prompt("judge", "v1")
    for key in ("{source}", "{ref_problem}", "{ref_method}", "{cand_problem}", "{cand_method}"):
        assert key in prompt
