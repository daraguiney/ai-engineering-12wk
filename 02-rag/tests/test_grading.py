import pytest

from rag.grading import agreement, build_grading_sample
from rag.review import save_verdict

QUESTIONS = [
    {"id": f"q{i}", "type": "single", "question": f"Q{i}?", "notes": f"ref{i}",
     "evidence": [{"arxiv_id": "1111.1111", "quote": f"quote{i}"}]}
    for i in range(1, 7)
]
TITLES = {"1111.1111": "A Paper"}


def _rec(qid, verdict, answer=None):
    return {"id": qid, "type": "single", "answerable": True, "answer": answer or f"ans-{qid}",
            "judge": {"verdict": verdict, "score": {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}[verdict]}}


RECORDS = {
    "dense": [_rec("q1", "correct"), _rec("q2", "partial"), {"id": "q3", "type": "single", "answerable": False, "answer": "no"}],
    "ctx": [_rec("q1", "correct"), _rec("q2", "correct", "different"), _rec("q4", "correct"), _rec("q5", "correct"), _rec("q6", "partial")],
}


def test_sample_takes_every_partial_and_caps_the_correct_ones():
    items, key = build_grading_sample(RECORDS, QUESTIONS, TITLES, seed=1, n_correct=2)
    judged = [k["judge"] for k in key.values()]
    assert judged.count("partial") == 2 and judged.count("correct") == 2


def test_items_are_blind_and_deduplicated():
    items, key = build_grading_sample(RECORDS, QUESTIONS, TITLES, seed=1, n_correct=10)
    text = str(items)
    assert "judge" not in text and "partial" not in text and "dense" not in text and "ctx" not in text
    pairs = [(k["question_id"], i["answer"]) for i, k in ((i, key[i["id"]]) for i in items)]
    assert len(pairs) == len(set(pairs))  # q1 correct in both variants appears once
    assert not any(i["answer"] == "no" for i in items)  # abstentions are never sampled


def test_sample_is_reproducible_for_a_seed():
    a = build_grading_sample(RECORDS, QUESTIONS, TITLES, seed=7, n_correct=2)
    b = build_grading_sample(RECORDS, QUESTIONS, TITLES, seed=7, n_correct=2)
    assert a == b


def test_agreement_counts_matches_and_lists_disagreements():
    key = {"g01": {"question_id": "q1", "judge": "correct"}, "g02": {"question_id": "q2", "judge": "partial"}}
    verdicts = {"g01": {"verdict": "correct", "note": ""}, "g02": {"verdict": "correct", "note": "fine to me"}}
    out = agreement(key, verdicts)
    assert out["graded"] == 2 and out["agree"] == 1
    assert out["disagreements"] == [{"id": "g02", "question_id": "q2", "judge": "partial", "human": "correct", "note": "fine to me"}]


def test_agreement_requires_some_human_grades():
    with pytest.raises(ValueError, match="no human grades"):
        agreement({"g01": {"question_id": "q1", "judge": "correct"}}, {})


def test_save_verdict_accepts_custom_allowed_labels(tmp_path):
    import json
    q = tmp_path / "q.json"
    q.write_text(json.dumps([{"id": "g01"}]))
    out = save_verdict(q, tmp_path / "v.json", "g01", "partial", "", allowed=("correct", "partial", "incorrect"))
    assert out["g01"]["verdict"] == "partial"
    with pytest.raises(ValueError):
        save_verdict(q, tmp_path / "v.json", "g01", "ok", "", allowed=("correct", "partial", "incorrect"))
