import pytest

from rag.testset import Evidence, Question, normalize, quote_in_text, validate_questions

TEXT = {"1111.1111": "The model uses multi-\nhead attention with\n  8 heads.  Results: 28.4 BLEU."}


def test_normalize_ignores_whitespace_and_hyphens():
    assert normalize("multi-\nhead   attention\nwith") == normalize("multi-head attention with")


def test_quote_in_text_tolerates_pdf_line_breaks():
    assert quote_in_text("uses multi-head attention with 8 heads", TEXT["1111.1111"])
    assert not quote_in_text("uses 16 heads", TEXT["1111.1111"])


def _q(**kw):
    base = {
        "id": "q1", "type": "single", "question": "How many heads?",
        "evidence": [Evidence(arxiv_id="1111.1111", quote="with 8 heads")],
    }
    return Question(**{**base, **kw})


def test_valid_question_has_no_problems():
    assert validate_questions([_q()], TEXT) == {}


def test_missing_quote_and_unknown_paper_are_reported():
    bad = _q(id="q2", evidence=[Evidence(arxiv_id="1111.1111", quote="not in paper")])
    unknown = _q(id="q3", evidence=[Evidence(arxiv_id="9999.9999", quote="x")])
    problems = validate_questions([bad, unknown], TEXT)
    assert "quote not found" in problems["q2"][0]
    assert "unknown paper" in problems["q3"][0]


def test_type_rules():
    assert "needs exactly 1" in validate_questions([_q(evidence=[])], TEXT)["q1"][0]
    cross = _q(id="c", type="cross")
    assert "needs 2" in validate_questions([cross], TEXT)["c"][0]
    unans = _q(id="u", type="unanswerable")
    assert "must have no evidence" in validate_questions([unans], TEXT)["u"][0]
    assert validate_questions([_q(id="u", type="unanswerable", evidence=[])], TEXT) == {}


def test_duplicate_ids_rejected():
    assert "duplicate id" in validate_questions([_q(), _q()], TEXT)["q1"][-1]


def test_rejects_unsafe_arxiv_id():
    with pytest.raises(ValueError):
        Evidence(arxiv_id="../x", quote="q")
