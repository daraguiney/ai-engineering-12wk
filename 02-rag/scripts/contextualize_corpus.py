"""Generate a context note for every chunk (resumable) and report real token usage and cost.

Usage: uv run --active --env-file ../.env python scripts/contextualize_corpus.py [--papers N]
Writes results/contexts.jsonl. Prices: Haiku 5.5, prompts <=100K tokens ($ per MTok).
"""
import argparse
import json
from pathlib import Path

from anthropic import Anthropic

from rag.chunking import chunk_text
from rag.contextualize import USAGE_KEYS, contextualize_paper

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "contexts.jsonl"
PRICE = {"input_tokens": 0.10, "output_tokens": 0.50,
         "cache_read_input_tokens": 0.01, "cache_creation_input_tokens": 0.125}


def cost(usage: dict[str, int]) -> float:
    return sum(usage[k] * PRICE[k] / 1e6 for k in USAGE_KEYS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers", type=int, default=None, help="first N papers only (canary)")
    parser.add_argument("--max-cost", type=float, default=3.0, help="stop after the paper that passes this spend ($)")
    args = parser.parse_args()

    kept = json.loads((ROOT / "corpus" / "manifest.json").read_text())["kept"][: args.papers]
    client, total, n_chunks = Anthropic(), dict.fromkeys(USAGE_KEYS, 0), 0
    for k in kept:
        arxiv_id = k["arxiv_id"]
        text = (ROOT / "corpus" / "text" / f"{arxiv_id}.txt").read_text()
        chunks = chunk_text(text)
        usage = contextualize_paper(client, arxiv_id, text, chunks, OUT)
        total = {key: total[key] + usage[key] for key in USAGE_KEYS}
        n_chunks += len(chunks)
        print(f"{arxiv_id}: {len(chunks)} chunks, ${cost(usage):.4f}", flush=True)
        if cost(total) > args.max_cost:
            print(f"STOPPED: spend ${cost(total):.2f} passed the cap ${args.max_cost:.2f}; re-run to resume")
            return 1
    print(f"total: {n_chunks} chunks, usage {total}, ${cost(total):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
