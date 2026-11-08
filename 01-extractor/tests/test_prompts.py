import pytest

from extractor.prompts import load_prompt


def test_loads_versioned_prompt():
    assert "{document}" in load_prompt("extract_paper", "v1")


def test_missing_prompt_raises_clear_error():
    with pytest.raises(FileNotFoundError, match="extract_paper"):
        load_prompt("extract_paper", "v999")
