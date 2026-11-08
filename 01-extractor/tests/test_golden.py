import json

import pytest

from extractor.golden import load_record, save_record
from extractor.schema import PaperExtraction

EXTRACTION = PaperExtraction(
    title="T", authors=["A"], problem="p", method="m",
    datasets=[], key_results=[], limitations=[],
)


def test_save_then_load_roundtrip(tmp_path):
    save_record(tmp_path, "1234.5678", "source text", EXTRACTION)
    record = load_record(tmp_path / "1234.5678.json")
    assert record["arxiv_id"] == "1234.5678"
    assert record["source_text"] == "source text"
    assert record["reviewed"] is False
    assert PaperExtraction(**record["expected"]) == EXTRACTION


def test_save_does_not_overwrite_reviewed_record(tmp_path):
    path = save_record(tmp_path, "1234.5678", "src", EXTRACTION)
    data = json.loads(path.read_text())
    path.write_text(json.dumps({**data, "reviewed": True}))
    with pytest.raises(FileExistsError):
        save_record(tmp_path, "1234.5678", "src", EXTRACTION)
