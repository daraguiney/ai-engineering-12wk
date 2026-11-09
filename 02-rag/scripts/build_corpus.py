"""Download the RAG corpus (PDFs + extracted text) for the given arXiv ids.

Usage: uv run python scripts/build_corpus.py <id> [<id> ...]
       uv run python scripts/build_corpus.py --from-golden   # the 50 extractor papers
"""
import sys
from pathlib import Path

from rag.corpus import build_corpus

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT.parent / "01-extractor" / "evals" / "golden"


def main(argv: list[str]) -> int:
    ids = sorted(p.stem for p in GOLDEN.glob("*.json")) if argv == ["--from-golden"] else argv
    if not ids:
        print(__doc__, file=sys.stderr)
        return 2
    failures = build_corpus(ids, ROOT / "corpus" / "pdfs", ROOT / "corpus" / "text")
    print(f"{len(ids) - len(failures)}/{len(ids)} ok")
    for arxiv_id, err in failures.items():
        print(f"  FAILED {arxiv_id}: {err}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
