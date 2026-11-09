import json

import pytest

from rag.review import load_state, save_verdict

QUESTIONS = [
    {"id": "q01", "type": "single", "question": "Q?", "evidence": [], "notes": ""},
    {"id": "q02", "type": "single", "question": "Q2?", "evidence": [], "notes": ""},
]


@pytest.fixture
def files(tmp_path):
    q = tmp_path / "questions.json"
    q.write_text(json.dumps(QUESTIONS))
    return q, tmp_path / "verdicts.json"


def test_load_state_without_verdicts_file(files):
    q, v = files
    state = load_state(q, v)
    assert [x["id"] for x in state["questions"]] == ["q01", "q02"]
    assert state["verdicts"] == {}


def test_save_verdict_persists_and_overwrites(files):
    q, v = files
    save_verdict(q, v, "q01", "ok", "")
    save_verdict(q, v, "q01", "fix", "reword the question")
    assert load_state(q, v)["verdicts"] == {"q01": {"verdict": "fix", "note": "reword the question"}}


def test_save_verdict_keeps_other_questions(files):
    q, v = files
    save_verdict(q, v, "q01", "ok", "")
    save_verdict(q, v, "q02", "drop", "dup")
    assert set(load_state(q, v)["verdicts"]) == {"q01", "q02"}


@pytest.mark.parametrize(
    "qid,verdict,note",
    [("q99", "ok", ""), ("q01", "maybe", ""), ("q01", "ok", "x" * 5001), ("q01", "ok", 5)],
)
def test_save_verdict_rejects_bad_input(files, qid, verdict, note):
    q, v = files
    with pytest.raises(ValueError):
        save_verdict(q, v, qid, verdict, note)
    assert not v.exists()


def test_corrupt_verdicts_file_is_an_error_not_silently_reset(files):
    q, v = files
    v.write_text("{not json")
    with pytest.raises(ValueError, match="verdicts"):
        load_state(q, v)
