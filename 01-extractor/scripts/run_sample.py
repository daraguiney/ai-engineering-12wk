"""Fetch arXiv abstracts by ID and run the extractor on each.

Usage: uv run --env-file ../.env python scripts/run_sample.py 1706.03762 2005.11401
"""
import sys
import urllib.request
import xml.etree.ElementTree as ET

from extractor.extract import extract_paper, make_client

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_API = "https://export.arxiv.org/api/query?id_list={id}"
TIMEOUT_SECONDS = 30


def fetch_arxiv_text(arxiv_id: str) -> str:
    with urllib.request.urlopen(ARXIV_API.format(id=arxiv_id), timeout=TIMEOUT_SECONDS) as resp:
        root = ET.fromstring(resp.read())
    entry = root.find(f"{ATOM}entry")
    if entry is None:
        raise ValueError(f"No arXiv entry for {arxiv_id}")
    title = " ".join(entry.findtext(f"{ATOM}title", "").split())
    authors = [a.findtext(f"{ATOM}name", "") for a in entry.findall(f"{ATOM}author")]
    summary = " ".join(entry.findtext(f"{ATOM}summary", "").split())
    return f"Title: {title}\nAuthors: {', '.join(authors)}\n\nAbstract: {summary}"


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
