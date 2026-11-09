import json
from types import SimpleNamespace

import pytest

from extractor.hillclimb import load_done, run_case, run_round, target_score
from extractor.judge import FieldJudgment, RecordJudgment
from extractor.schema import PaperExtraction

EXPECTED = {
    "title": "T",
    "authors": ["A"],
    "problem": "p",
    "method": "m",
    "datasets": ["D"],
    "key_results": ["1.5 BLEU"],
    "limitations": ["L"],
}
RECORD = {"arxiv_id": "1111.1111", "source_text": "Title: T", "expected": EXPECTED}


def _completion(model: str):
    usage = SimpleNamespace(input_tokens=100, output_tokens=20)
    return SimpleNamespace(model=model, usage=usage)


class FakeClient:
    def __init__(self, verdict: str = "correct"):
        self.verdict = verdict
        self.messages = self

    def create_with_completion(self, *, response_model, model, **_):
        if response_model is PaperExtraction:
            return PaperExtraction(**EXPECTED), _completion(model)
        field = FieldJudgment(reason="r", verdict=self.verdict)
        return RecordJudgment(problem=field, method=field), _completion(model)


def test_target_score_is_mean_of_limitations_and_method():
    assert target_score({"limitations": 1.0, "method": 0.5}) == 0.75


def test_run_case_records_grade_usage_and_models():
    row, trace = run_case(RECORD, 0, client=FakeClient(), prompt_version="v2")
    assert row["prompt_id"] == "1111.1111" and row["rep"] == 0
    assert row["grade"]["target"] == 1.0 and row["grade"]["datasets"] == 1.0
    assert row["usage"]["input_tokens"] == 100 and row["judge_usage"]["output_tokens"] == 20
    assert row["model"] == "claude-sonnet-5-5" and row["judge_model"] == "claude-opus-5-5"
    assert row["explanation"]["method"] == "r"
    assert [t["role"] for t in trace][:2] == ["user", "assistant"]


def test_run_case_partial_verdict_lowers_target():
    row, _ = run_case(RECORD, 0, client=FakeClient("partial"), prompt_version="v2")
    assert row["grade"]["method"] == 0.5
    assert row["grade"]["target"] == pytest.approx(0.75)


def test_run_round_resumes_without_repeating_done_cases(tmp_path):
    calls = []

    def run_one(record, rep):
        calls.append((record["arxiv_id"], rep))
        return run_case(record, rep, client=FakeClient(), prompt_version="v2")

    assert run_round([RECORD], tmp_path, reps=2, run_one=run_one) == 2
    assert run_round([RECORD], tmp_path, reps=3, run_one=run_one) == 1
    assert calls == [("1111.1111", 0), ("1111.1111", 1), ("1111.1111", 2)]
    assert load_done(tmp_path / "results.jsonl") == {("1111.1111", k) for k in range(3)}
    assert (tmp_path / "traces" / "1111.1111_rep2.json").is_file()
    rows = [json.loads(x) for x in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 3
