"""Fetch arXiv abstracts by ID and run the extractor on each.

Usage: uv run --env-file ../.env python scripts/run_sample.py 1706.03762 2005.11401
"""
import sys

from extractor.arxiv import fetch_arxiv_text
from extractor.extract import extract_paper, make_client


def main(ids: list[str]) -> int:
    if not ids:
        print(__doc__, file=sys.stderr)
        return 2
    client = make_client()
    for arxiv_id in ids:
        print(f"--- {arxiv_id}")
        result = extract_paper(fetch_arxiv_text(arxiv_id), client=client)
        print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
