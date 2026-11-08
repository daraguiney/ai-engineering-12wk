import json

import pytest

from extractor.evaluate import evaluate, load_reviewed
from extractor.schema import PaperExtraction

GOLD = {
    "title": "T", "authors": ["A"], "problem": "p", "method": "m",
    "datasets": ["GSM8K"], "key_results": ["10 points"], "limitations": [],
}


def write(tmp_path, arxiv_id, reviewed, expected=GOLD):
    record = {"arxiv_id": arxiv_id, "source_text": f"text {arxiv_id}",
              "expected": expected, "reviewed": reviewed}
    (tmp_path / f"{arxiv_id}.json").write_text(json.dumps(record))


def test_load_reviewed_skips_unreviewed(tmp_path):
    write(tmp_path, "1111.1111", True)
    write(tmp_path, "2222.2222", False)
    assert [r["arxiv_id"] for r in load_reviewed(tmp_path)] == ["1111.1111"]


def test_load_reviewed_empty_raises(tmp_path):
    write(tmp_path, "2222.2222", False)
    with pytest.raises(ValueError, match="reviewed"):
        load_reviewed(tmp_path)


def test_evaluate_scores_each_record_and_means(tmp_path):
    write(tmp_path, "1111.1111", True)
    write(tmp_path, "2222.2222", True)
    perfect = PaperExtraction(**GOLD)
    wrong = PaperExtraction(**{**GOLD, "datasets": ["Other"]})
    answers = {"text 1111.1111": perfect, "text 2222.2222": wrong}

    result = evaluate(load_reviewed(tmp_path), lambda text: answers[text])

    assert result["per_record"]["1111.1111"]["datasets"] == 1.0
    assert result["per_record"]["2222.2222"]["datasets"] == 0.0
    assert result["mean"]["datasets"] == 0.5
    assert result["failures"] == []


def test_evaluate_counts_extraction_failure_as_zero(tmp_path):
    write(tmp_path, "1111.1111", True)

    def boom(text):
        raise ValueError("bad output")

    result = evaluate(load_reviewed(tmp_path), boom)
    assert result["failures"] == [{"arxiv_id": "1111.1111", "error": "bad output"}]
    assert all(v == 0.0 for v in result["mean"].values())


def test_evaluate_merges_judge_scores(tmp_path):
    write(tmp_path, "1111.1111", True)
    perfect = PaperExtraction(**GOLD)

    result = evaluate(
        load_reviewed(tmp_path),
        lambda text: perfect,
        judge_fn=lambda source, expected, predicted: {"problem": 1.0, "method": 0.5},
    )
    assert result["per_record"]["1111.1111"]["method"] == 0.5
    assert result["mean"]["problem"] == 1.0
