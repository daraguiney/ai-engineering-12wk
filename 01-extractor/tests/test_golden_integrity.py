"""Every golden record must be well-formed, whether or not it is reviewed."""
from pathlib import Path

import pytest

from extractor.golden import load_record
from extractor.schema import PaperExtraction

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "evals" / "golden"
RECORDS = sorted(GOLDEN_DIR.glob("*.json"))


@pytest.mark.parametrize("path", RECORDS, ids=lambda p: p.stem)
def test_record_is_well_formed(path):
    record = load_record(path)
    assert record["arxiv_id"] == path.stem
    assert record["source_text"].strip()
    assert isinstance(record["reviewed"], bool)
    PaperExtraction(**record["expected"])
