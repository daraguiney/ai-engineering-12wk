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


def test_v3_prompt_encodes_new_rules():
    prompt = load_prompt("extract_paper", "v3")
    assert "{document}" in prompt
    assert "Models" in prompt and "NOT datasets" in prompt
    assert "about 40 words" in prompt
    assert "failure cases" in prompt
