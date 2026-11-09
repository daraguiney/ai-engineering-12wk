"""Download arXiv PDFs and turn them into plain text."""
import re
import time
import urllib.request
from collections.abc import Iterable
from pathlib import Path

import pymupdf

ARXIV_ID = re.compile(r"^\d{4}\.\d{4,5}$")
PDF_URL = "https://arxiv.org/pdf/{id}"
TIMEOUT_SECONDS = 60
POLITE_DELAY_SECONDS = 3.0  # arXiv asks automated clients to pause between requests


def validate_arxiv_id(arxiv_id: str) -> str:
    if not ARXIV_ID.match(arxiv_id):
        raise ValueError(f"not a valid arXiv id: {arxiv_id!r}")
    return arxiv_id


def pdf_url(arxiv_id: str) -> str:
    return PDF_URL.format(id=validate_arxiv_id(arxiv_id))


def extract_text(pdf_path: Path) -> str:
    """All page text joined with blank lines; raises if the PDF has no text layer."""
    with pymupdf.open(pdf_path) as doc:
        pages = [page.get_text() for page in doc]
    text = "\n\n".join(p.strip() for p in pages if p.strip())
    if not text:
        raise ValueError(f"no text extracted from {pdf_path}")
    return text


def download_pdf(arxiv_id: str, dest: Path) -> Path:
    """Fetch one PDF unless already on disk. Sleeps only when it actually downloads."""
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    request = urllib.request.Request(
        pdf_url(arxiv_id), headers={"User-Agent": "ai-engineering-12wk (learning project)"}
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as resp:
        data = resp.read()
    if not data.startswith(b"%PDF"):
        raise ValueError(f"{arxiv_id}: response is not a PDF")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    time.sleep(POLITE_DELAY_SECONDS)
    return dest


def build_corpus(ids: Iterable[str], pdf_dir: Path, text_dir: Path) -> dict[str, str]:
    """Download and extract each id. Returns {id: error} for the ones that failed."""
    text_dir.mkdir(parents=True, exist_ok=True)
    failures: dict[str, str] = {}
    for arxiv_id in ids:
        try:
            pdf = download_pdf(arxiv_id, pdf_dir / f"{arxiv_id}.pdf")
            (text_dir / f"{arxiv_id}.txt").write_text(extract_text(pdf), encoding="utf-8")
        except (ValueError, OSError) as exc:
            failures = {**failures, arxiv_id: str(exc)}
    return failures
