"""Embed chunks and store them in LanceDB; search by vector."""
from pathlib import Path
from typing import Protocol

import lancedb

from rag.chunking import DEFAULT_OVERLAP, DEFAULT_SIZE, chunk_text
from rag.contextualize import contextual_text

TABLE = "chunks"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_BATCH = 64


class Embedder(Protocol):
    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedder:
    """Local ONNX embedding model via fastembed (no API key, no network after first download)."""

    def __init__(self, model_name: str = EMBED_MODEL):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.passage_embed(texts, batch_size=EMBED_BATCH)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


def build_index(
    texts: dict[str, str],
    embedder: Embedder,
    db_path: Path,
    size: int = DEFAULT_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    contexts: dict[tuple[str, int], str] | None = None,
):
    """Chunk every paper, embed the chunks, and (re)create the table. Returns the table."""
    notes = contexts or {}
    rows = []
    for arxiv_id in sorted(texts):
        chunks = chunk_text(texts[arxiv_id], size=size, overlap=overlap)
        embed_texts = [
            contextual_text(notes[(arxiv_id, c.index)], c.text) if (arxiv_id, c.index) in notes else c.text
            for c in chunks
        ]
        vectors = embedder.embed_passages(embed_texts)
        rows += [
            {"arxiv_id": arxiv_id, "chunk_index": c.index, "start": c.start, "text": c.text,
             "contextual_text": et, "vector": v}
            for c, et, v in zip(chunks, embed_texts, vectors, strict=True)
        ]
    if not rows:
        raise ValueError("no text to index")
    db = lancedb.connect(str(db_path))
    return db.create_table(TABLE, data=rows, mode="overwrite")


def open_index(db_path: Path):
    return lancedb.connect(str(db_path)).open_table(TABLE)


def search(table, embedder: Embedder, query: str, k: int = 5) -> list[dict]:
    """Top-k chunks by vector similarity; each hit carries `score` (higher is better)."""
    hits = table.search(embedder.embed_query(query)).metric("cosine").limit(k).to_list()
    return [
        {
            "arxiv_id": h["arxiv_id"],
            "chunk_index": h["chunk_index"],
            "text": h["text"],
            "score": 1.0 - h["_distance"],
        }
        for h in hits
    ]
