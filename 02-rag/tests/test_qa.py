from types import SimpleNamespace

import pytest

from rag.qa import Answer, answer_question, build_prompt, validate_citations

HITS = [
    {"arxiv_id": "1111.1111", "chunk_index": 3, "text": "The model uses 8 heads.", "score": 0.9},
    {"arxiv_id": "2222.2222", "chunk_index": 0, "text": "Training took 5 days.", "score": 0.8},
]


class FakeClient:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []
        self.messages = self

    def create_with_completion(self, **kwargs):
        self.calls.append(kwargs)
        usage = SimpleNamespace(input_tokens=500, output_tokens=40)
        return self.answer, SimpleNamespace(usage=usage)


def test_prompt_numbers_sources_and_includes_question():
    prompt = build_prompt("How many heads?", HITS)
    assert "[1] (paper 1111.1111)" in prompt and "[2] (paper 2222.2222)" in prompt
    assert "The model uses 8 heads." in prompt and "Question: How many heads?" in prompt


def test_answer_question_returns_cited_chunks():
    client = FakeClient(Answer(answerable=True, answer="8 heads [1].", citations=[1]))
    result = answer_question("How many heads?", HITS, client=client)
    assert result["answerable"] and result["cited"] == [{"arxiv_id": "1111.1111", "chunk_index": 3}]
    assert client.calls[0]["response_model"] is Answer
    assert result["usage"] == {"input_tokens": 500, "output_tokens": 40}


def test_abstention_has_no_citations():
    client = FakeClient(Answer(answerable=False, answer="The papers do not state it.", citations=[]))
    result = answer_question("Unknowable?", HITS, client=client)
    assert not result["answerable"] and result["cited"] == []


@pytest.mark.parametrize("citations", [[3], [0], [1, 9]])
def test_citation_to_missing_source_is_rejected(citations):
    with pytest.raises(ValueError, match="source"):
        validate_citations(Answer(answerable=True, answer="x", citations=citations), n_sources=2)


def test_answerable_without_citations_is_rejected():
    with pytest.raises(ValueError, match="cite"):
        validate_citations(Answer(answerable=True, answer="x", citations=[]), n_sources=2)


def test_abstention_with_citations_is_rejected():
    with pytest.raises(ValueError, match="abstain"):
        validate_citations(Answer(answerable=False, answer="x", citations=[1]), n_sources=2)
