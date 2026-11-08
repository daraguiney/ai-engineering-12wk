import pytest

from extractor.extract import extract_paper
from extractor.schema import PaperExtraction


class FakeClient:
    """Stands in for an instructor client; records the call kwargs."""

    def __init__(self, result):
        self.result = result
        self.calls = []
        self.messages = self
        self.create = self._create

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


RESULT = PaperExtraction(
    title="T", authors=["A"], problem="p", method="m",
    datasets=[], key_results=[], limitations=[],
)


def test_returns_validated_model_and_enables_retries():
    client = FakeClient(RESULT)
    out = extract_paper("some text", client=client, model="m", max_retries=2)
    assert out == RESULT
    call = client.calls[0]
    assert call["response_model"] is PaperExtraction
    assert call["max_retries"] == 2


def test_document_is_injected_into_prompt():
    client = FakeClient(RESULT)
    extract_paper("UNIQUE-DOC-TEXT", client=client, model="m")
    assert "UNIQUE-DOC-TEXT" in client.calls[0]["messages"][0]["content"]


def test_rejects_empty_document():
    with pytest.raises(ValueError):
        extract_paper("   ", client=FakeClient(RESULT), model="m")
