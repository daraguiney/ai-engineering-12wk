import pytest

from extractor.prompts import load_prompt


def test_loads_versioned_prompt():
    assert "{document}" in load_prompt("extract_paper", "v1")


def test_missing_prompt_raises_clear_error():
    with pytest.raises(FileNotFoundError, match="extract_paper"):
        load_prompt("extract_paper", "v999")


def test_v2_prompt_encodes_rubric_rules():
    prompt = load_prompt("extract_paper", "v2")
    assert "{document}" in prompt
    assert "explicitly named" in prompt
    assert "Never infer" in prompt
