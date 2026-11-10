import json
from types import SimpleNamespace

import pytest

from rag.chunking import Chunk
from rag.contextualize import (
    build_request,
    contextual_text,
    contextualize_paper,
    load_contexts,
)


class FakeClient:
    def __init__(self):
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        usage = SimpleNamespace(
            input_tokens=10, output_tokens=5, cache_read_input_tokens=100, cache_creation_input_tokens=0
        )
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=" Context sentence. ")], usage=usage)


CHUNKS = [Chunk(index=0, start=0, text="alpha"), Chunk(index=1, start=5, text="beta")]


def test_request_caches_the_document_and_puts_the_chunk_after_it():
    req = build_request("FULL PAPER", "alpha")
    first, second = req["messages"][0]["content"]
    assert "FULL PAPER" in first["text"] and first["cache_control"] == {"type": "ephemeral"}
    assert "alpha" in second["text"] and "cache_control" not in second
    assert req["thinking"] == {"type": "disabled"}


def test_contextualize_paper_records_a_stripped_context_per_chunk(tmp_path):
    out = tmp_path / "ctx.jsonl"
    usage = contextualize_paper(FakeClient(), "p1", "FULL", CHUNKS, out)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [(r["arxiv_id"], r["chunk_index"], r["context"]) for r in rows] == [
        ("p1", 0, "Context sentence."), ("p1", 1, "Context sentence."),
    ]
    assert usage["cache_read_input_tokens"] == 200 and usage["output_tokens"] == 10


def test_rerun_skips_chunks_already_recorded(tmp_path):
    out = tmp_path / "ctx.jsonl"
    contextualize_paper(FakeClient(), "p1", "FULL", CHUNKS, out)
    again = FakeClient()
    contextualize_paper(again, "p1", "FULL", CHUNKS, out)
    assert again.calls == []
    assert len(load_contexts(out)) == 2


def test_empty_context_is_an_error(tmp_path):
    client = FakeClient()
    client.messages.create = lambda **kw: SimpleNamespace(
        content=[SimpleNamespace(type="text", text="  ")],
        usage=SimpleNamespace(input_tokens=1, output_tokens=0, cache_read_input_tokens=0, cache_creation_input_tokens=0),
    )
    with pytest.raises(ValueError, match="empty context"):
        contextualize_paper(client, "p1", "FULL", CHUNKS, tmp_path / "c.jsonl")


def test_contextual_text_prepends_context():
    assert contextual_text("ctx", "body") == "ctx\n\nbody"
