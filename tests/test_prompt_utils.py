import pytest

from generate.prompt_utils import build_prompts, merge_generations


def test_build_prompts_extracts_prefix_field():
    rows = [{"doc_id": 0, "prefix": "hello"}, {"doc_id": 1, "prefix": "world"}]
    assert build_prompts(rows) == ["hello", "world"]


def test_merge_generations_adds_generated_continuation_field():
    rows = [{"doc_id": 0, "prefix": "hello"}]
    result = merge_generations(rows, ["generated text"])
    assert result == [{"doc_id": 0, "prefix": "hello", "generated_continuation": "generated text"}]


def test_merge_generations_raises_on_length_mismatch():
    rows = [{"doc_id": 0, "prefix": "hello"}, {"doc_id": 1, "prefix": "world"}]
    with pytest.raises(ValueError):
        merge_generations(rows, ["only one"])
