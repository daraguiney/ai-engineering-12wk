import pytest

from extractor.schema import PaperExtraction
from extractor.scoring import (
    mean_scores,
    normalize,
    number_recall,
    score_record,
    set_f1,
)


def make(**overrides):
    base = {
        "title": "T",
        "authors": ["Ada L", "Bo K"],
        "problem": "p",
        "method": "m",
        "datasets": ["GSM8K"],
        "key_results": ["28.4 BLEU on WMT"],
        "limitations": [],
    }
    return PaperExtraction(**{**base, **overrides})


def test_normalize_ignores_case_punctuation_whitespace():
    assert normalize("  GSM-8K,  Test! ") == "gsm8k test"


def test_set_f1_perfect_and_partial():
    assert set_f1(["a", "b"], ["b", "a"]) == 1.0
    assert set_f1(["a"], ["a", "b"]) == pytest.approx(2 / 3)


def test_set_f1_both_empty_is_perfect_one_empty_is_zero():
    assert set_f1([], []) == 1.0
    assert set_f1(["a"], []) == 0.0
    assert set_f1([], ["a"]) == 0.0


def test_number_recall_finds_gold_numbers_in_prediction():
    assert number_recall(["28.4 BLEU", "41.8 BLEU"], ["got 28.4 on EN-DE"]) == 0.5


def test_number_recall_no_gold_numbers_is_perfect():
    assert number_recall(["state of the art"], ["anything"]) == 1.0


def test_score_record_perfect_match():
    scores = score_record(make(), make())
    assert set(scores) == {"title", "authors", "datasets", "key_results", "limitations"}
    assert all(v == 1.0 for v in scores.values())


def test_score_record_flags_hallucinated_limitation():
    scores = score_record(make(), make(limitations=["small model"]))
    assert scores["limitations"] == 0.0
    assert scores["datasets"] == 1.0


def test_mean_scores_averages_per_field_without_mutating():
    rows = [{"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 0.0}]
    assert mean_scores(rows) == {"a": 0.5, "b": 0.0}
    assert rows[0] == {"a": 1.0, "b": 0.0}


def test_mean_scores_empty_raises():
    with pytest.raises(ValueError):
        mean_scores([])
