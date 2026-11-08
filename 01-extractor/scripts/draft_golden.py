"""Draft unreviewed golden records (model output) for the given arXiv IDs.

Usage: uv run --env-file ../.env python scripts/draft_golden.py 1706.03762 ...
Review each file in evals/golden/ by hand, fix `expected`, set "reviewed": true.
"""
import sys
import time
from pathlib import Path

import anthropic
from instructor.core import IncompleteOutputException, InstructorRetryException

from extractor.arxiv import fetch_arxiv_text
from extractor.extract import extract_paper, make_client
from extractor.golden import save_record

ARXIV_DELAY_SECONDS = 3  # arXiv API etiquette
GOLDEN_DIR = Path(__file__).resolve().parents[1] / "evals" / "golden"


def main(ids: list[str]) -> int:
    if not ids:
        print(__doc__, file=sys.stderr)
        return 2
    client = make_client()
    failures = 0
    for index, arxiv_id in enumerate(ids):
        if index:
            time.sleep(ARXIV_DELAY_SECONDS)
        try:
            text = fetch_arxiv_text(arxiv_id)
            path = save_record(GOLDEN_DIR, arxiv_id, text, extract_paper(text, client=client))
            print(f"drafted {path.name}")
        except (
            ValueError,
            OSError,
            anthropic.APIError,
            InstructorRetryException,
            IncompleteOutputException,
        ) as exc:
            failures += 1
            print(f"FAILED {arxiv_id}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
