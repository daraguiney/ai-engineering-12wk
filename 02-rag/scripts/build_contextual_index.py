"""Build the contextual-chunk index (chunk text prefixed with its LLM-written note before embedding).

Usage: uv run --active python scripts/build_contextual_index.py
Needs results/contexts.jsonl from contextualize_corpus.py. Writes index/lancedb_ctx.
"""
import json
import time
from pathlib import Path

from rag.contextualize import load_contexts
from rag.index import FastEmbedder, build_index

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    kept = json.loads((ROOT / "corpus" / "manifest.json").read_text())["kept"]
    texts = {k["arxiv_id"]: (ROOT / "corpus" / "text" / f"{k['arxiv_id']}.txt").read_text() for k in kept}
    contexts = load_contexts(ROOT / "results" / "contexts.jsonl")
    started = time.monotonic()
    table = build_index(texts, FastEmbedder(), ROOT / "index" / "lancedb_ctx", contexts=contexts)
    print(f"{table.count_rows()} chunks indexed in {time.monotonic() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
