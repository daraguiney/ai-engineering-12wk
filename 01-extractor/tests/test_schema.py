import pytest
from pydantic import ValidationError

from extractor.schema import PaperExtraction


def valid_payload(**overrides):
    base = {
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer"],
        "problem": "Sequence transduction relies on recurrent networks.",
        "method": "A Transformer using only self-attention.",
        "datasets": ["WMT 2014 EN-DE"],
        "key_results": ["28.4 BLEU on EN-DE"],
        "limitations": [],
    }
    return {**base, **overrides}


def test_valid_payload_parses():
    assert PaperExtraction(**valid_payload()).title.startswith("Attention")


def test_requires_at_least_one_author():
    with pytest.raises(ValidationError):
        PaperExtraction(**valid_payload(authors=[]))


def test_rejects_blank_title():
    with pytest.raises(ValidationError):
        PaperExtraction(**valid_payload(title="  "))


def test_model_is_frozen():
    paper = PaperExtraction(**valid_payload())
    with pytest.raises(ValidationError):
        paper.title = "other"
